from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from app.services.model_client import ModelClient, ModelClientError, ModelResponse

from .models import (
    RECORD_FILENAME,
    ProvenanceError,
    RunRecord,
    canonical_json,
    read_json,
    sha256_file,
    sha256_json,
    slug_identifier,
    utc_now,
    verify_artifacts,
    write_json_atomic,
)
from .production_statistics import load_provenance_summary


REJUDGE_SCHEMA_VERSION = "1.0"
REJUDGED_SUMMARY_VERSION = "1.0"
SIDECAR_BATCH_FILENAME = "rejudge_batch.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_IMAGE_LABELS = ("combined", "combined_figure", "render")
_TRUNCATED_STOP_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "token_limit",
}

VISUAL_FORM_RUBRIC: Dict[str, Any] = {
    "rubric_version": "visual-form-v1",
    "task": "post_hoc_visual_form_only",
    "constraints": [
        "Judge only readability, layout, typography, clutter, overlap, and perceptual hierarchy.",
        "Do not infer data fidelity, correctness, or provenance from pixels.",
        "Do not provide generation feedback or editing instructions.",
    ],
    "output_schema": {
        "visual_form": "finite number in [0,1]",
        "diagnostics": "array of concise evidence statements",
    },
    "anchors": {
        "0.0": "Unreadable or structurally broken.",
        "0.5": "Usable but with material readability or layout defects.",
        "1.0": "Publication-ready visual form with no material readability defects.",
    },
}
VISUAL_FORM_RUBRIC_HASH = sha256_json(VISUAL_FORM_RUBRIC)
VISUAL_FORM_PROMPT = (
    "Return exactly one JSON object conforming to this fixed rubric. "
    "Do not return markdown or editing feedback.\n"
    + canonical_json(VISUAL_FORM_RUBRIC)
)


class RejudgeError(ProvenanceError):
    """Raised when post-hoc visual rejudging cannot remain provenance-safe."""


@dataclass(frozen=True)
class CodeGitState:
    commit: str
    dirty: bool


@dataclass(frozen=True)
class RejudgeTarget:
    run_name: str
    run_dir: Path
    expected_record_hash: str | None


@dataclass(frozen=True)
class SealedRender:
    target: RejudgeTarget
    record: RunRecord
    artifact_label: str
    render_path: Path
    render_sha256: str


@dataclass(frozen=True)
class RejudgeBatchResult:
    output_dir: Path
    batch_path: Path
    completed: tuple[str, ...]
    resumed: tuple[str, ...]
    failures: tuple[Dict[str, str], ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "batch_path": str(self.batch_path),
            "completed": list(self.completed),
            "resumed": list(self.resumed),
            "failures": [dict(item) for item in self.failures],
            "exit_code": self.exit_code,
        }


def judge_slug(model: str) -> str:
    return slug_identifier(model).lower()


def _current_git_state() -> CodeGitState:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RejudgeError(f"Cannot resolve rejudge code git state: {exc}") from exc
    return CodeGitState(commit=commit, dirty=bool(status.strip()))


def _resolve_git_state(
    git_state: CodeGitState | Mapping[str, Any] | None,
    *,
    allow_dirty: bool,
) -> CodeGitState:
    if git_state is None:
        resolved = _current_git_state()
    elif isinstance(git_state, CodeGitState):
        resolved = git_state
    elif isinstance(git_state, Mapping):
        if set(git_state) != {"commit", "dirty"}:
            raise RejudgeError(
                "Injected git_state must contain exactly commit and dirty"
            )
        resolved = CodeGitState(
            commit=str(git_state["commit"]),
            dirty=git_state["dirty"],
        )
    else:
        raise RejudgeError("git_state must be CodeGitState, mapping, or None")
    if not _GIT_COMMIT_RE.fullmatch(resolved.commit):
        raise RejudgeError("Rejudge code commit is missing or malformed")
    if not isinstance(resolved.dirty, bool):
        raise RejudgeError("Rejudge code dirty flag must be boolean")
    if resolved.dirty and not allow_dirty:
        raise RejudgeError(
            "Rejudge code worktree is dirty; refusing uncommitted metric output"
        )
    return resolved


def _default_output_dir(source: Path, model: str) -> Path:
    slug = judge_slug(model)
    resolved = source.expanduser().resolve()
    if resolved.is_file():
        return (
            resolved.parent.parent
            / f"{resolved.parent.name}-rejudge"
            / slug
        )
    return resolved.parent / f"{resolved.name}-rejudge" / slug


def _discover_targets(source: Path) -> list[RejudgeTarget]:
    resolved = source.expanduser().resolve()
    if resolved.is_dir():
        targets = []
        found_records = False
        for child in sorted(resolved.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            record_path = child / RECORD_FILENAME
            if not record_path.is_file():
                raise RejudgeError(
                    f"Run directory has no {RECORD_FILENAME}: {child}"
                )
            found_records = True
            record = RunRecord.read(record_path)
            record.validate_provenance()
            if record.status == "failed":
                if (record.error or {}).get("attribution") == "method":
                    continue
                raise RejudgeError(
                    "Cannot skip a non-method failed run during rejudge: "
                    f"{record.run_name}"
                )
            if record.status != "completed":
                raise RejudgeError(
                    f"Cannot rejudge non-terminal run: {record.run_name}"
                )
            targets.append(
                RejudgeTarget(
                    run_name=child.name,
                    run_dir=child,
                    expected_record_hash=None,
                )
            )
        if not found_records:
            raise RejudgeError(f"No run records found under {resolved}")
        return targets

    if not resolved.is_file():
        raise RejudgeError(f"Rejudge input does not exist: {resolved}")
    summary = load_provenance_summary(resolved)
    payload = read_json(resolved)
    source_root_raw = payload.get("source_root")
    if not isinstance(source_root_raw, str) or not source_root_raw.strip():
        raise RejudgeError("Summary has no source_root")
    source_root = Path(source_root_raw).expanduser()
    if not source_root.is_absolute():
        source_root = (resolved.parent / source_root).resolve()
    else:
        source_root = source_root.resolve()
    return [
        RejudgeTarget(
            run_name=str(row["run_name"]),
            run_dir=source_root / str(row["run_name"]),
            expected_record_hash=str(row["record_hash"]),
        )
        for row in summary.rows
        if row.get("status") == "completed"
    ]


def _safe_artifact_path(run_dir: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise RejudgeError(f"Unsafe candidate artifact path: {relative_text}")
    root = run_dir.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RejudgeError(
            f"Candidate artifact escaped run directory: {relative_text}"
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise RejudgeError(f"Candidate render is not a regular file: {relative_text}")
    return candidate


def _load_sealed_render(target: RejudgeTarget) -> SealedRender:
    record_path = target.run_dir / RECORD_FILENAME
    record = RunRecord.read(record_path)
    record.validate_provenance(require_completed=True)
    if record.test_only:
        raise RejudgeError(f"Test-only run cannot be rejudged: {record.run_name}")
    if bool(record.experiment_spec.get("git_dirty")):
        raise RejudgeError(
            f"Dirty-worktree run cannot be rejudged: {record.run_name}"
        )
    if record.run_name != target.run_name:
        raise RejudgeError(
            f"Run directory/name mismatch: {target.run_name} != {record.run_name}"
        )
    if (
        target.expected_record_hash is not None
        and record.record_hash != target.expected_record_hash
    ):
        raise RejudgeError(
            f"Summary record_hash disagrees for {record.run_name}"
        )
    verify_artifacts(record, target.run_dir)

    best = next(
        (
            candidate
            for candidate in record.candidates
            if candidate.get("candidate_id") == record.best_candidate_id
        ),
        None,
    )
    if not isinstance(best, Mapping):
        raise RejudgeError(
            f"Best candidate is missing from RunRecord: {record.run_name}"
        )
    paths = best.get("artifact_paths")
    hashes = best.get("artifact_hashes")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise RejudgeError(
            f"Best candidate has no sealed artifact map: {record.run_name}"
        )

    artifact_label = next((label for label in _IMAGE_LABELS if label in paths), None)
    if artifact_label is None:
        raise RejudgeError(
            f"Best candidate has no combined/render image: {record.run_name}"
        )
    relative_text = paths.get(artifact_label)
    expected_hash = hashes.get(artifact_label)
    if not isinstance(relative_text, str) or not isinstance(expected_hash, str):
        raise RejudgeError(
            f"Best candidate render binding is malformed: {record.run_name}"
        )
    record_key = f"{record.best_candidate_id}.{artifact_label}"
    if (
        record.artifact_paths.get(record_key) != relative_text
        or record.artifact_hashes.get(record_key) != expected_hash
    ):
        raise RejudgeError(
            f"Best candidate render disagrees with RunRecord artifacts: "
            f"{record.run_name}"
        )
    render_path = _safe_artifact_path(target.run_dir, relative_text)
    actual_hash = sha256_file(render_path)
    if actual_hash != expected_hash:
        raise RejudgeError(
            f"Best candidate render hash mismatch: {record.run_name}"
        )
    return SealedRender(
        target=target,
        record=record,
        artifact_label=artifact_label,
        render_path=render_path,
        render_sha256=actual_hash,
    )


def _sidecar_path(output_dir: Path, run_name: str) -> Path:
    return output_dir / f"{slug_identifier(run_name)}.json"


def _seal_payload(payload: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
    sealed = dict(payload)
    sealed.pop(hash_field, None)
    sealed[hash_field] = sha256_json(sealed)
    return sealed


def _read_sidecar(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    sidecar_hash = payload.get("sidecar_hash")
    if not isinstance(sidecar_hash, str) or not _SHA256_RE.fullmatch(sidecar_hash):
        raise RejudgeError(f"Sidecar has no valid hash: {path}")
    unhashed = dict(payload)
    unhashed.pop("sidecar_hash", None)
    if sha256_json(unhashed) != sidecar_hash:
        raise RejudgeError(f"Sidecar hash mismatch: {path}")
    if payload.get("schema_version") != REJUDGE_SCHEMA_VERSION:
        raise RejudgeError(f"Unsupported sidecar schema: {path}")
    return payload


def _binding_fields(
    sealed: SealedRender,
    *,
    request_model: str,
    code_git_state: CodeGitState,
) -> Dict[str, Any]:
    return {
        "run_name": sealed.record.run_name,
        "record_hash": sealed.record.record_hash,
        "best_candidate_id": sealed.record.best_candidate_id,
        "artifact_label": sealed.artifact_label,
        "render_sha256": sealed.render_sha256,
        "judge_request_model": request_model,
        "judge_slug": judge_slug(request_model),
        "rubric_hash": VISUAL_FORM_RUBRIC_HASH,
        "code_git_commit": code_git_state.commit,
        "code_git_dirty": code_git_state.dirty,
    }


def _validate_resume_binding(
    payload: Mapping[str, Any],
    sealed: SealedRender,
    *,
    request_model: str,
    code_git_state: CodeGitState,
) -> None:
    expected = _binding_fields(
        sealed,
        request_model=request_model,
        code_git_state=code_git_state,
    )
    mismatches = [
        name for name, value in expected.items() if payload.get(name) != value
    ]
    if mismatches:
        raise RejudgeError(
            f"Resume sidecar binding mismatch for {sealed.record.run_name}: "
            f"{', '.join(mismatches)}"
        )


def _validate_model_response(response: ModelResponse) -> tuple[float, list[str]]:
    if not isinstance(response.model, str) or not response.model.strip():
        raise RejudgeError("Vision response did not identify the served model")
    stop_reason = (response.stop_reason or "").strip().casefold()
    if stop_reason in _TRUNCATED_STOP_REASONS:
        raise RejudgeError(
            f"Vision response was truncated (stop_reason={response.stop_reason})"
        )
    value = response.value
    if set(value) != {"visual_form", "diagnostics"}:
        raise RejudgeError(
            "Vision response must contain exactly visual_form and diagnostics"
        )
    score = value["visual_form"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RejudgeError("visual_form must be numeric")
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RejudgeError("visual_form must be finite and in [0,1]")
    diagnostics = value["diagnostics"]
    if (
        not isinstance(diagnostics, list)
        or not diagnostics
        or any(not isinstance(item, str) or not item.strip() for item in diagnostics)
    ):
        raise RejudgeError("diagnostics must be a non-empty array of strings")
    return score, [item.strip() for item in diagnostics]


def _validate_completed_sidecar(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "completed":
        raise RejudgeError(
            f"Sidecar is not completed: {payload.get('run_name')!r}"
        )
    for name in ("record_hash", "render_sha256", "rubric_hash"):
        value = payload.get(name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise RejudgeError(
                f"Completed sidecar has invalid {name}: {payload.get('run_name')!r}"
            )
    for name in (
        "run_name",
        "best_candidate_id",
        "artifact_label",
        "judge_request_model",
        "judge_slug",
        "judge_served_model",
        "timestamp",
    ):
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RejudgeError(
                f"Completed sidecar has invalid {name}: {payload.get('run_name')!r}"
            )
    commit = payload.get("code_git_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise RejudgeError("Completed sidecar has invalid code_git_commit")
    if not isinstance(payload.get("code_git_dirty"), bool):
        raise RejudgeError("Completed sidecar has invalid code_git_dirty")
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise RejudgeError("Completed sidecar usage must be an object")
    _validate_model_response(
        ModelResponse(
            value={
                "visual_form": payload.get("score"),
                "diagnostics": payload.get("diagnostics"),
            },
            model=str(payload["judge_served_model"]),
            request_id=payload.get("judge_request_id"),
            usage=dict(usage),
            stop_reason=payload.get("stop_reason"),
            latency_seconds=0.0,
        )
    )


def _success_sidecar(
    sealed: SealedRender,
    response: ModelResponse,
    *,
    request_model: str,
    code_git_state: CodeGitState,
) -> Dict[str, Any]:
    score, diagnostics = _validate_model_response(response)
    payload = {
        "schema_version": REJUDGE_SCHEMA_VERSION,
        "status": "completed",
        **_binding_fields(
            sealed,
            request_model=request_model,
            code_git_state=code_git_state,
        ),
        "judge_served_model": response.model,
        "judge_request_id": response.request_id,
        "usage": dict(response.usage),
        "stop_reason": response.stop_reason,
        "score": score,
        "diagnostics": diagnostics,
        "timestamp": utc_now(),
    }
    canonical_json(payload)
    return _seal_payload(payload, "sidecar_hash")


def _failure_sidecar(
    sealed: SealedRender,
    *,
    request_model: str,
    code_git_state: CodeGitState,
    error: Exception,
) -> Dict[str, Any]:
    payload = {
        "schema_version": REJUDGE_SCHEMA_VERSION,
        "status": "failed",
        **_binding_fields(
            sealed,
            request_model=request_model,
            code_git_state=code_git_state,
        ),
        "judge_served_model": None,
        "usage": {},
        "stop_reason": None,
        "score": None,
        "diagnostics": [],
        "timestamp": utc_now(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    return _seal_payload(payload, "sidecar_hash")


def rejudge_batch(
    source: Path,
    *,
    judge_model: str,
    output_dir: Path | None = None,
    resume: bool = False,
    model_client: Any = None,
    allow_dirty: bool = False,
    git_state: CodeGitState | Mapping[str, Any] | None = None,
) -> RejudgeBatchResult:
    request_model = judge_model.strip()
    if not request_model:
        raise RejudgeError("judge_model must be non-empty")
    code_git_state = _resolve_git_state(
        git_state,
        allow_dirty=allow_dirty,
    )
    source = source.expanduser().resolve()
    targets = _discover_targets(source)
    sidecar_names = [
        _sidecar_path(Path("."), target.run_name).name
        for target in targets
    ]
    if len(sidecar_names) != len(set(sidecar_names)):
        raise RejudgeError("Run names collide after sidecar filename slugging")
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _default_output_dir(source, request_model)
    )
    destination.mkdir(parents=True, exist_ok=True)

    client_error: Exception | None = None
    client = model_client
    if client is None and targets:
        try:
            client = ModelClient.from_env(model=request_model)
        except Exception as exc:
            client_error = exc

    completed: list[str] = []
    resumed: list[str] = []
    failures: list[Dict[str, str]] = []
    sidecar_hashes: Dict[str, str] = {}
    started_at = utc_now()

    for target in targets:
        sidecar_path = _sidecar_path(destination, target.run_name)
        sealed: SealedRender | None = None
        try:
            sealed = _load_sealed_render(target)
            if sidecar_path.exists():
                if not resume:
                    raise RejudgeError(
                        f"Sidecar already exists; use --resume: {sidecar_path}"
                    )
                existing = _read_sidecar(sidecar_path)
                _validate_resume_binding(
                    existing,
                    sealed,
                    request_model=request_model,
                    code_git_state=code_git_state,
                )
                if existing.get("status") == "completed":
                    _validate_completed_sidecar(existing)
                    resumed.append(target.run_name)
                    sidecar_hashes[target.run_name] = str(
                        existing["sidecar_hash"]
                    )
                    continue

            if client_error is not None:
                raise RejudgeError(f"Cannot initialize judge client: {client_error}")
            if client is None:
                raise RejudgeError("Judge client is unavailable")
            response = client.evaluate_image_json(
                VISUAL_FORM_PROMPT,
                sealed.render_path,
                model=request_model,
            )
            if not isinstance(response, ModelResponse):
                raise RejudgeError("ModelClient returned an invalid response type")
            payload = _success_sidecar(
                sealed,
                response,
                request_model=request_model,
                code_git_state=code_git_state,
            )
            write_json_atomic(sidecar_path, payload)
            completed.append(target.run_name)
            sidecar_hashes[target.run_name] = str(payload["sidecar_hash"])
        except Exception as exc:
            failures.append(
                {
                    "run_name": target.run_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            if sealed is None:
                continue
            if sidecar_path.exists():
                try:
                    existing = _read_sidecar(sidecar_path)
                    _validate_resume_binding(
                        existing,
                        sealed,
                        request_model=request_model,
                        code_git_state=code_git_state,
                    )
                except Exception:
                    continue
                if existing.get("status") == "completed":
                    continue
            payload = _failure_sidecar(
                sealed,
                request_model=request_model,
                code_git_state=code_git_state,
                error=exc,
            )
            write_json_atomic(sidecar_path, payload)
            sidecar_hashes[target.run_name] = str(payload["sidecar_hash"])

    batch_payload = {
        "schema_version": REJUDGE_SCHEMA_VERSION,
        "source": str(source),
        "output_dir": str(destination),
        "judge_request_model": request_model,
        "judge_slug": judge_slug(request_model),
        "rubric_hash": VISUAL_FORM_RUBRIC_HASH,
        "code_git_commit": code_git_state.commit,
        "code_git_dirty": code_git_state.dirty,
        "started_at": started_at,
        "finished_at": utc_now(),
        "status": "failed" if failures else "completed",
        "completed": completed,
        "resumed": resumed,
        "failures": failures,
        "sidecar_hashes": sidecar_hashes,
    }
    batch_payload = _seal_payload(batch_payload, "batch_hash")
    batch_path = destination / SIDECAR_BATCH_FILENAME
    write_json_atomic(batch_path, batch_payload)
    return RejudgeBatchResult(
        output_dir=destination,
        batch_path=batch_path,
        completed=tuple(completed),
        resumed=tuple(resumed),
        failures=tuple(failures),
    )


def _load_completed_sidecars(
    sidecar_dir: Path,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not sidecar_dir.is_dir():
        raise RejudgeError(f"Sidecar directory does not exist: {sidecar_dir}")
    batch_path = sidecar_dir / SIDECAR_BATCH_FILENAME
    if not batch_path.is_file():
        raise RejudgeError(f"Sidecar directory has no batch record: {batch_path}")
    batch = read_json(batch_path)
    batch_hash = batch.get("batch_hash")
    if not isinstance(batch_hash, str) or not _SHA256_RE.fullmatch(batch_hash):
        raise RejudgeError("Rejudge batch has no valid batch_hash")
    unhashed_batch = dict(batch)
    unhashed_batch.pop("batch_hash", None)
    if sha256_json(unhashed_batch) != batch_hash:
        raise RejudgeError("Rejudge batch hash mismatch")
    if (
        batch.get("schema_version") != REJUDGE_SCHEMA_VERSION
        or batch.get("status") != "completed"
        or batch.get("failures")
    ):
        raise RejudgeError("Only a completed failure-free rejudge batch can merge")
    expected_hashes = batch.get("sidecar_hashes")
    if not isinstance(expected_hashes, Mapping):
        raise RejudgeError("Rejudge batch has no sidecar_hashes")
    batch_commit = batch.get("code_git_commit")
    if not isinstance(batch_commit, str) or not _GIT_COMMIT_RE.fullmatch(
        batch_commit
    ):
        raise RejudgeError("Rejudge batch has invalid code_git_commit")
    batch_dirty = batch.get("code_git_dirty")
    if not isinstance(batch_dirty, bool):
        raise RejudgeError("Rejudge batch has invalid code_git_dirty")
    batch_slug = batch.get("judge_slug")
    if not isinstance(batch_slug, str) or not batch_slug:
        raise RejudgeError("Rejudge batch has invalid judge_slug")
    batch_rubric_hash = batch.get("rubric_hash")
    if (
        not isinstance(batch_rubric_hash, str)
        or not _SHA256_RE.fullmatch(batch_rubric_hash)
    ):
        raise RejudgeError("Rejudge batch has invalid rubric_hash")

    sidecars: Dict[str, Dict[str, Any]] = {}
    for path in sorted(sidecar_dir.glob("*.json")):
        if path.name == SIDECAR_BATCH_FILENAME:
            continue
        payload = _read_sidecar(path)
        run_name = payload.get("run_name")
        if not isinstance(run_name, str) or not run_name:
            raise RejudgeError(f"Sidecar has no run_name: {path}")
        if run_name in sidecars:
            raise RejudgeError(f"Duplicate sidecar run_name: {run_name}")
        _validate_completed_sidecar(payload)
        if (
            payload.get("code_git_commit") != batch_commit
            or payload.get("code_git_dirty") != batch_dirty
            or payload.get("judge_slug") != batch_slug
            or payload.get("rubric_hash") != batch_rubric_hash
        ):
            raise RejudgeError(
                f"Batch/sidecar provenance mismatch: {run_name}"
            )
        if expected_hashes.get(run_name) != payload.get("sidecar_hash"):
            raise RejudgeError(f"Batch/sidecar hash mismatch: {run_name}")
        sidecars[run_name] = payload
    if set(expected_hashes) != set(sidecars):
        raise RejudgeError("Batch sidecar_hashes do not exactly cover sidecars")
    return sidecars, batch


def merge_rejudged_summary(
    summary_path: Path,
    sidecar_dir: Path,
    *,
    output_path: Path | None = None,
) -> tuple[Path, str]:
    summary_path = summary_path.expanduser().resolve()
    summary = load_provenance_summary(summary_path)
    original = read_json(summary_path)
    sidecars, batch = _load_completed_sidecars(
        sidecar_dir.expanduser().resolve()
    )
    completed_run_names = {
        str(row["run_name"])
        for row in summary.rows
        if row.get("status") == "completed"
    }
    if set(sidecars) != completed_run_names:
        raise RejudgeError(
            "Sidecar coverage does not exactly match completed summary runs; "
            f"missing={sorted(completed_run_names - set(sidecars))}, "
            f"extra={sorted(set(sidecars) - completed_run_names)}"
        )

    slug = str(batch["judge_slug"])
    metric_name = f"metric.visual_form.{slug}"

    rows = []
    failed_zero_runs: list[str] = []
    for raw_row in summary.rows:
        row = dict(raw_row)
        run_name = str(row["run_name"])
        if row.get("status") == "failed":
            if (
                row.get("failure_attribution") != "method"
                or float(row.get("execution_success", -1.0)) != 0.0
            ):
                raise RejudgeError(
                    f"Failed row is not a zero-valued method outcome: {run_name}"
                )
            row[metric_name] = 0.0
            rows.append(row)
            failed_zero_runs.append(run_name)
            continue
        sidecar = sidecars[run_name]
        if sidecar.get("record_hash") != row.get("record_hash"):
            raise RejudgeError(f"Sidecar record_hash mismatch: {run_name}")
        score = sidecar.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise RejudgeError(f"Sidecar score is invalid: {run_name}")
        row[metric_name] = float(score)
        rows.append(row)

    destination = (
        output_path.expanduser().resolve()
        if output_path is not None
        else summary_path.parent / "rejudged_summary.json"
    )
    if destination == summary_path:
        raise RejudgeError("Merged summary must not overwrite the source summary")
    if destination.exists():
        raise RejudgeError(f"Merged summary already exists: {destination}")

    old_summary_hash = str(original.pop("summary_hash"))
    old_generated_at = original.get("generated_at")
    columns = list(original.get("columns") or [])
    if metric_name not in columns:
        columns.append(metric_name)
    merged = {
        **original,
        "generated_at": utc_now(),
        "columns": columns,
        "runs": rows,
        "original_summary_hash": old_summary_hash,
        "original_generated_at": old_generated_at,
        "rejudge": {
            "schema_version": REJUDGED_SUMMARY_VERSION,
            "judge_slug": slug,
            "metric": metric_name,
            "rubric_hash": batch["rubric_hash"],
            "code_git_commit": batch["code_git_commit"],
            "code_git_dirty": batch["code_git_dirty"],
            "sidecar_hashes": {
                run_name: sidecars[run_name]["sidecar_hash"]
                for run_name in sorted(sidecars)
            },
            "failed_zero_runs": sorted(failed_zero_runs),
        },
    }
    merged["summary_hash"] = sha256_json(merged)
    write_json_atomic(destination, merged)
    return destination, metric_name

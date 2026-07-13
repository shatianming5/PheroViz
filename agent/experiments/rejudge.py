from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from app.services.model_client import (
    ModelClient,
    ModelClientError,
    ModelConfig,
    ModelResponse,
)

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


REJUDGE_SCHEMA_VERSION = "2.0"
REJUDGED_SUMMARY_VERSION = "2.0"
SIDECAR_BATCH_FILENAME = "rejudge_batch.json"
MODEL_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_registry.yml"
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
VISUAL_FORM_PROMPT_HASH = hashlib.sha256(
    VISUAL_FORM_PROMPT.encode("utf-8")
).hexdigest()
_REQUIRED_JUDGE_FIELDS = {
    "judge_id",
    "request_model",
    "served_model",
    "protocol",
    "endpoint_class",
    "base_url_env",
    "api_key_env",
    "max_tokens",
    "timeout_seconds",
    "connect_timeout_seconds",
    "retries",
    "rubric_version",
    "rubric_hash",
    "prompt_hash",
    "model_cutoff",
}
_EXPECTED_SERVED_BY_REQUEST = {
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "gemini-3.5-flash": "gemini-3.5-flash",
}


class RejudgeError(ProvenanceError):
    """Raised when post-hoc visual rejudging cannot remain provenance-safe."""


@dataclass(frozen=True)
class CodeGitState:
    commit: str
    dirty: bool


@dataclass(frozen=True)
class JudgeConfig:
    role: str
    judge_id: str
    request_model: str
    served_model: str
    protocol: str
    endpoint_class: str
    base_url_env: str
    api_key_env: str
    max_tokens: int
    timeout_seconds: float
    connect_timeout_seconds: float
    retries: int
    rubric_version: str
    rubric_hash: str
    prompt_hash: str
    registry_sha256: str
    config_hash: str


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


def _load_judge_config(request_model: str) -> JudgeConfig:
    try:
        registry_bytes = MODEL_REGISTRY_PATH.read_bytes()
        registry = yaml.safe_load(registry_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise RejudgeError(f"Cannot load model registry: {exc}") from exc
    judges = registry.get("visual_judges") if isinstance(registry, Mapping) else None
    if not isinstance(judges, Mapping) or set(judges) != {"primary", "secondary"}:
        raise RejudgeError("Registry must define exactly primary and secondary judges")
    if not all(isinstance(raw, Mapping) for raw in judges.values()):
        raise RejudgeError("Every visual judge registry entry must be an object")
    if (
        len({raw.get("judge_id") for raw in judges.values()}) != 2
        or len({raw.get("request_model") for raw in judges.values()}) != 2
    ):
        raise RejudgeError("C5 judge IDs and requested identities must be distinct")

    matches = [
        (str(role), raw)
        for role, raw in judges.items()
        if isinstance(raw, Mapping) and raw.get("request_model") == request_model
    ]
    if len(matches) != 1:
        raise RejudgeError(
            f"Judge model is not an exactly registered C5 identity: {request_model!r}"
        )
    role, raw = matches[0]
    if set(raw) != _REQUIRED_JUDGE_FIELDS:
        missing = sorted(_REQUIRED_JUDGE_FIELDS - set(raw))
        extra = sorted(set(raw) - _REQUIRED_JUDGE_FIELDS)
        raise RejudgeError(
            f"Judge registry fields are not exact; missing={missing}, extra={extra}"
        )
    if (
        raw["protocol"] != "anthropic_messages"
        or raw["endpoint_class"] != "anthropic_compatibility_gateway"
        or raw["max_tokens"] != 1024
        or raw["rubric_version"] != VISUAL_FORM_RUBRIC["rubric_version"]
        or raw["rubric_hash"] != VISUAL_FORM_RUBRIC_HASH
        or raw["prompt_hash"] != VISUAL_FORM_PROMPT_HASH
        or raw["model_cutoff"] is not None
    ):
        raise RejudgeError("Judge registry disagrees with the frozen C5 protocol")
    for field in ("judge_id", "request_model", "served_model", "base_url_env", "api_key_env"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise RejudgeError(f"Judge registry {field} must be a non-empty string")
    if raw["served_model"] != _EXPECTED_SERVED_BY_REQUEST.get(
        str(raw["request_model"])
    ):
        raise RejudgeError(
            "Frozen served_model does not match the registered gateway identity"
        )
    if (
        isinstance(raw["retries"], bool)
        or not isinstance(raw["retries"], int)
        or raw["retries"] < 0
    ):
        raise RejudgeError("Judge registry retries must be a non-negative integer")
    for field in ("timeout_seconds", "connect_timeout_seconds"):
        if (
            isinstance(raw[field], bool)
            or not isinstance(raw[field], (int, float))
            or not math.isfinite(float(raw[field]))
            or float(raw[field]) <= 0
        ):
            raise RejudgeError(f"Judge registry {field} must be positive and finite")

    registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    normalized = {"role": role, **dict(raw), "registry_sha256": registry_sha256}
    return JudgeConfig(
        role=role,
        judge_id=str(raw["judge_id"]),
        request_model=str(raw["request_model"]),
        served_model=str(raw["served_model"]),
        protocol=str(raw["protocol"]),
        endpoint_class=str(raw["endpoint_class"]),
        base_url_env=str(raw["base_url_env"]),
        api_key_env=str(raw["api_key_env"]),
        max_tokens=int(raw["max_tokens"]),
        timeout_seconds=float(raw["timeout_seconds"]),
        connect_timeout_seconds=float(raw["connect_timeout_seconds"]),
        retries=int(raw["retries"]),
        rubric_version=str(raw["rubric_version"]),
        rubric_hash=str(raw["rubric_hash"]),
        prompt_hash=str(raw["prompt_hash"]),
        registry_sha256=registry_sha256,
        config_hash=sha256_json(normalized),
    )


def _model_client_from_config(config: JudgeConfig) -> ModelClient:
    base_url = (os.getenv(config.base_url_env) or "").strip()
    api_key = (os.getenv(config.api_key_env) or "").strip()
    missing = [
        name
        for name, value in (
            (config.base_url_env, base_url),
            (config.api_key_env, api_key),
        )
        if not value
    ]
    if missing:
        raise RejudgeError(
            f"Missing exact judge registry environment: {', '.join(missing)}"
        )
    return ModelClient(
        ModelConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=config.request_model,
            timeout=config.timeout_seconds,
            connect_timeout=config.connect_timeout_seconds,
            retries=config.retries,
            max_tokens=config.max_tokens,
        )
    )


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
        raise RejudgeError(
            "C5 rejudge requires a strict hashed summary, not a run directory"
        )

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


def _build_input_manifest(
    source: Path,
    sealed_renders: Sequence[SealedRender],
    *,
    code_git_state: CodeGitState,
) -> Dict[str, Any]:
    summary = load_provenance_summary(source)
    payload = read_json(source)
    completed = {
        str(row["run_name"]): row
        for row in summary.rows
        if row.get("status") == "completed"
    }
    sealed_by_name = {
        sealed.record.run_name: sealed for sealed in sealed_renders
    }
    if set(sealed_by_name) != set(completed):
        raise RejudgeError("Selected renders do not exactly cover completed summary rows")
    for name, sealed in sealed_by_name.items():
        row = completed[name]
        if (
            row.get("best_candidate_id") != sealed.record.best_candidate_id
            or row.get("spec_hash") != sealed.record.spec_hash
            or row.get("record_hash") != sealed.record.record_hash
        ):
            raise RejudgeError(
                f"Summary selected-candidate provenance is stale: {name}"
            )
    method_failed = []
    for row in summary.rows:
        if row.get("status") != "failed":
            continue
        if (
            row.get("failure_attribution") != "method"
            or float(row.get("execution_success", -1.0)) != 0.0
        ):
            raise RejudgeError(
                f"C5 source contains a non-method failure: {row.get('run_name')!r}"
            )
        method_failed.append(
            {
                "run_name": row["run_name"],
                "record_hash": row["record_hash"],
                "spec_hash": row["spec_hash"],
                "failure_attribution": "method",
                "execution_success": 0.0,
                "score_policy": "zero_without_image_call",
            }
        )
    manifest = {
        "schema_version": "1.0",
        "source_summary_sha256": sha256_file(source),
        "source_summary_hash": summary.summary_hash,
        "source_run_count": len(summary.rows),
        "source_completed_run_count": len(completed),
        "source_method_failed_run_count": len(method_failed),
        "input_record_hashes": dict(payload["input_record_hashes"]),
        "code_git_commit": code_git_state.commit,
        "code_git_dirty": code_git_state.dirty,
        "rubric_hash": VISUAL_FORM_RUBRIC_HASH,
        "prompt_hash": VISUAL_FORM_PROMPT_HASH,
        "presentation_policy": {
            "one_sealed_image_per_independent_call": True,
            "request_fields": ["image", "fixed_visual_form_prompt"],
            "excluded_fields": [
                "source_data",
                "method",
                "prior_scores",
                "generation_feedback",
                "editing_feedback",
            ],
        },
        "completed_runs": [
            {
                "run_name": name,
                "spec_hash": completed[name]["spec_hash"],
                "record_hash": completed[name]["record_hash"],
                "best_candidate_id": sealed_by_name[name].record.best_candidate_id,
                "artifact_label": sealed_by_name[name].artifact_label,
                "render_sha256": sealed_by_name[name].render_sha256,
            }
            for name in sorted(completed)
        ],
        "method_failed_runs": sorted(
            method_failed,
            key=lambda item: str(item["run_name"]),
        ),
    }
    manifest["input_manifest_hash"] = sha256_json(manifest)
    return manifest


def _assert_source_unchanged(source: Path, manifest: Mapping[str, Any]) -> None:
    if sha256_file(source) != manifest.get("source_summary_sha256"):
        raise RejudgeError("C5 source summary changed after input sealing")
    current = load_provenance_summary(source)
    if current.summary_hash != manifest.get("source_summary_hash"):
        raise RejudgeError("C5 source summary hash became stale")


def _assert_no_request_leakage(
    sealed: SealedRender,
    source: Path,
) -> None:
    forbidden = {
        sealed.record.run_name,
        sealed.record.method,
        sealed.record.case_id,
        str(source),
        "best_of_n",
        "flat_iterative",
        "pheroviz_full",
        "prior_scores",
        "source_data",
        "generation_feedback",
        "editing_feedback",
    }
    leaked = sorted(
        token for token in forbidden if token and token in VISUAL_FORM_PROMPT
    )
    if leaked:
        raise RejudgeError(
            f"C5 request prompt contains forbidden source/method fields: {leaked}"
        )


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
    judge: JudgeConfig,
    input_manifest: Mapping[str, Any],
    code_git_state: CodeGitState,
) -> Dict[str, Any]:
    return {
        "run_name": sealed.record.run_name,
        "record_hash": sealed.record.record_hash,
        "best_candidate_id": sealed.record.best_candidate_id,
        "artifact_label": sealed.artifact_label,
        "render_sha256": sealed.render_sha256,
        "judge_id": judge.judge_id,
        "judge_role": judge.role,
        "judge_request_model": judge.request_model,
        "judge_expected_served_model": judge.served_model,
        "judge_slug": judge_slug(judge.request_model),
        "judge_protocol": judge.protocol,
        "judge_endpoint_class": judge.endpoint_class,
        "judge_max_tokens": judge.max_tokens,
        "judge_config_hash": judge.config_hash,
        "model_registry_sha256": judge.registry_sha256,
        "rubric_hash": VISUAL_FORM_RUBRIC_HASH,
        "prompt_hash": VISUAL_FORM_PROMPT_HASH,
        "input_manifest_hash": input_manifest["input_manifest_hash"],
        "source_summary_hash": input_manifest["source_summary_hash"],
        "source_summary_sha256": input_manifest["source_summary_sha256"],
        "code_git_commit": code_git_state.commit,
        "code_git_dirty": code_git_state.dirty,
    }


def _validate_resume_binding(
    payload: Mapping[str, Any],
    sealed: SealedRender,
    *,
    judge: JudgeConfig,
    input_manifest: Mapping[str, Any],
    code_git_state: CodeGitState,
) -> None:
    expected = _binding_fields(
        sealed,
        judge=judge,
        input_manifest=input_manifest,
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
    for name in (
        "record_hash",
        "render_sha256",
        "rubric_hash",
        "prompt_hash",
        "input_manifest_hash",
        "source_summary_hash",
        "source_summary_sha256",
        "judge_config_hash",
        "model_registry_sha256",
    ):
        value = payload.get(name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise RejudgeError(
                f"Completed sidecar has invalid {name}: {payload.get('run_name')!r}"
            )
    for name in (
        "run_name",
        "best_candidate_id",
        "artifact_label",
        "judge_id",
        "judge_role",
        "judge_request_model",
        "judge_expected_served_model",
        "judge_slug",
        "judge_served_model",
        "judge_protocol",
        "judge_endpoint_class",
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
    if payload.get("code_git_dirty"):
        raise RejudgeError("Completed C5 sidecar cannot come from dirty code")
    if payload.get("judge_max_tokens") != 1024:
        raise RejudgeError("Completed sidecar has non-frozen judge_max_tokens")
    if payload.get("judge_served_model") != payload.get(
        "judge_expected_served_model"
    ):
        raise RejudgeError("Completed sidecar served identity mismatch")
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
    judge: JudgeConfig,
    input_manifest: Mapping[str, Any],
    code_git_state: CodeGitState,
) -> Dict[str, Any]:
    score, diagnostics = _validate_model_response(response)
    if response.model != judge.served_model:
        raise RejudgeError(
            "Served judge identity mismatch: "
            f"expected {judge.served_model!r}, got {response.model!r}"
        )
    payload = {
        "schema_version": REJUDGE_SCHEMA_VERSION,
        "status": "completed",
        **_binding_fields(
            sealed,
            judge=judge,
            input_manifest=input_manifest,
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
    judge: JudgeConfig,
    input_manifest: Mapping[str, Any],
    code_git_state: CodeGitState,
    error: Exception,
) -> Dict[str, Any]:
    payload = {
        "schema_version": REJUDGE_SCHEMA_VERSION,
        "status": "failed",
        **_binding_fields(
            sealed,
            judge=judge,
            input_manifest=input_manifest,
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
    judge = _load_judge_config(request_model)
    code_git_state = _resolve_git_state(
        git_state,
        allow_dirty=allow_dirty,
    )
    if code_git_state.dirty:
        raise RejudgeError("C5 rejudge never permits dirty code")
    source = source.expanduser().resolve()
    targets = _discover_targets(source)
    try:
        sealed_renders = [_load_sealed_render(target) for target in targets]
    except Exception as exc:
        raise RejudgeError(f"C5 input sealing failed: {exc}") from exc
    input_manifest = _build_input_manifest(
        source,
        sealed_renders,
        code_git_state=code_git_state,
    )
    sealed_by_name = {
        sealed.record.run_name: sealed for sealed in sealed_renders
    }
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
            client = _model_client_from_config(judge)
        except Exception as exc:
            client_error = exc

    completed: list[str] = []
    resumed: list[str] = []
    failures: list[Dict[str, str]] = []
    sidecar_hashes: Dict[str, str] = {}
    started_at = utc_now()

    for target in targets:
        sidecar_path = _sidecar_path(destination, target.run_name)
        sealed = sealed_by_name[target.run_name]
        try:
            _assert_source_unchanged(source, input_manifest)
            if sha256_file(sealed.render_path) != sealed.render_sha256:
                raise RejudgeError(
                    f"Selected render changed after input sealing: {target.run_name}"
                )
            if sidecar_path.exists():
                if not resume:
                    raise RejudgeError(
                        f"Sidecar already exists; use --resume: {sidecar_path}"
                    )
                existing = _read_sidecar(sidecar_path)
                _validate_resume_binding(
                    existing,
                    sealed,
                    judge=judge,
                    input_manifest=input_manifest,
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
            _assert_no_request_leakage(sealed, source)
            response = client.evaluate_image_json(
                VISUAL_FORM_PROMPT,
                sealed.render_path,
                model=judge.request_model,
                max_tokens=judge.max_tokens,
            )
            if not isinstance(response, ModelResponse):
                raise RejudgeError("ModelClient returned an invalid response type")
            payload = _success_sidecar(
                sealed,
                response,
                judge=judge,
                input_manifest=input_manifest,
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
            if sidecar_path.exists():
                try:
                    existing = _read_sidecar(sidecar_path)
                    _validate_resume_binding(
                        existing,
                        sealed,
                        judge=judge,
                        input_manifest=input_manifest,
                        code_git_state=code_git_state,
                    )
                except Exception:
                    continue
                if existing.get("status") == "completed":
                    continue
            payload = _failure_sidecar(
                sealed,
                judge=judge,
                input_manifest=input_manifest,
                code_git_state=code_git_state,
                error=exc,
            )
            write_json_atomic(sidecar_path, payload)
            sidecar_hashes[target.run_name] = str(payload["sidecar_hash"])

    try:
        _assert_source_unchanged(source, input_manifest)
    except Exception as exc:
        failures.append(
            {
                "run_name": "__source_summary__",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )

    covered = set(completed) | set(resumed)
    expected = {target.run_name for target in targets}
    if not failures and covered != expected:
        failures.append(
            {
                "run_name": "__coverage__",
                "type": "RejudgeError",
                "message": (
                    "Failure-free batch lacks exact target coverage; "
                    f"missing={sorted(expected - covered)}, "
                    f"extra={sorted(covered - expected)}"
                ),
            }
        )

    batch_payload = {
        "schema_version": REJUDGE_SCHEMA_VERSION,
        "source": str(source),
        "output_dir": str(destination),
        "judge_id": judge.judge_id,
        "judge_role": judge.role,
        "judge_request_model": judge.request_model,
        "judge_expected_served_model": judge.served_model,
        "judge_served_models": (
            [judge.served_model] if not failures and targets else []
        ),
        "judge_slug": judge_slug(judge.request_model),
        "judge_protocol": judge.protocol,
        "judge_endpoint_class": judge.endpoint_class,
        "judge_max_tokens": judge.max_tokens,
        "judge_config_hash": judge.config_hash,
        "model_registry_sha256": judge.registry_sha256,
        "rubric_hash": VISUAL_FORM_RUBRIC_HASH,
        "prompt_hash": VISUAL_FORM_PROMPT_HASH,
        "input_manifest": input_manifest,
        "input_manifest_hash": input_manifest["input_manifest_hash"],
        "source_summary_hash": input_manifest["source_summary_hash"],
        "source_summary_sha256": input_manifest["source_summary_sha256"],
        "selected_render_hashes": {
            item["run_name"]: item["render_sha256"]
            for item in input_manifest["completed_runs"]
        },
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
    if batch_dirty is not False:
        raise RejudgeError("Rejudge batch must bind clean code")
    batch_slug = batch.get("judge_slug")
    if not isinstance(batch_slug, str) or not batch_slug:
        raise RejudgeError("Rejudge batch has invalid judge_slug")
    batch_rubric_hash = batch.get("rubric_hash")
    if (
        not isinstance(batch_rubric_hash, str)
        or not _SHA256_RE.fullmatch(batch_rubric_hash)
    ):
        raise RejudgeError("Rejudge batch has invalid rubric_hash")
    for name in (
        "prompt_hash",
        "input_manifest_hash",
        "source_summary_hash",
        "source_summary_sha256",
        "judge_config_hash",
        "model_registry_sha256",
    ):
        value = batch.get(name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise RejudgeError(f"Rejudge batch has invalid {name}")
    manifest = batch.get("input_manifest")
    if not isinstance(manifest, Mapping):
        raise RejudgeError("Rejudge batch has no input_manifest")
    unhashed_manifest = dict(manifest)
    manifest_hash = unhashed_manifest.pop("input_manifest_hash", None)
    if (
        manifest_hash != batch["input_manifest_hash"]
        or sha256_json(unhashed_manifest) != manifest_hash
    ):
        raise RejudgeError("Rejudge input manifest hash mismatch")
    if (
        manifest.get("source_summary_hash") != batch["source_summary_hash"]
        or manifest.get("source_summary_sha256") != batch["source_summary_sha256"]
        or manifest.get("prompt_hash") != batch["prompt_hash"]
        or manifest.get("rubric_hash") != batch["rubric_hash"]
    ):
        raise RejudgeError("Rejudge batch/input-manifest provenance mismatch")
    selected_render_hashes = batch.get("selected_render_hashes")
    expected_render_hashes = {
        item["run_name"]: item["render_sha256"]
        for item in manifest.get("completed_runs", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("run_name"), str)
        and isinstance(item.get("render_sha256"), str)
    }
    if selected_render_hashes != expected_render_hashes:
        raise RejudgeError("Rejudge batch selected-render hashes are inconsistent")
    for name in (
        "judge_id",
        "judge_role",
        "judge_request_model",
        "judge_expected_served_model",
        "judge_protocol",
        "judge_endpoint_class",
    ):
        if not isinstance(batch.get(name), str) or not str(batch[name]).strip():
            raise RejudgeError(f"Rejudge batch has invalid {name}")
    if batch.get("judge_max_tokens") != 1024:
        raise RejudgeError("Rejudge batch has non-frozen judge_max_tokens")
    if batch.get("judge_served_models") not in (
        [batch["judge_expected_served_model"]],
        [],
    ):
        raise RejudgeError("Rejudge batch served identities are inconsistent")

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
            or payload.get("prompt_hash") != batch["prompt_hash"]
            or payload.get("input_manifest_hash") != batch["input_manifest_hash"]
            or payload.get("source_summary_hash") != batch["source_summary_hash"]
            or payload.get("source_summary_sha256") != batch["source_summary_sha256"]
            or payload.get("judge_id") != batch["judge_id"]
            or payload.get("judge_request_model") != batch["judge_request_model"]
            or payload.get("judge_expected_served_model")
            != batch["judge_expected_served_model"]
            or payload.get("judge_protocol") != batch["judge_protocol"]
            or payload.get("judge_endpoint_class") != batch["judge_endpoint_class"]
            or payload.get("judge_max_tokens") != batch["judge_max_tokens"]
            or payload.get("judge_config_hash") != batch["judge_config_hash"]
            or payload.get("model_registry_sha256")
            != batch["model_registry_sha256"]
        ):
            raise RejudgeError(
                f"Batch/sidecar provenance mismatch: {run_name}"
            )
        if expected_hashes.get(run_name) != payload.get("sidecar_hash"):
            raise RejudgeError(f"Batch/sidecar hash mismatch: {run_name}")
        sidecars[run_name] = payload
    if set(expected_hashes) != set(sidecars):
        raise RejudgeError("Batch sidecar_hashes do not exactly cover sidecars")
    expected_runs = {
        str(item["run_name"])
        for item in manifest.get("completed_runs", [])
        if isinstance(item, Mapping) and isinstance(item.get("run_name"), str)
    }
    if set(sidecars) != expected_runs:
        raise RejudgeError("Batch sidecars do not exactly cover input manifest")
    if set(batch.get("completed", [])) | set(batch.get("resumed", [])) != expected_runs:
        raise RejudgeError("Batch completed/resumed lists lack exact coverage")
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
    if "rejudge" in original:
        raise RejudgeError(
            "Legacy single-judge provenance is ambiguous and cannot be extended"
        )
    sidecars, batch = _load_completed_sidecars(
        sidecar_dir.expanduser().resolve()
    )
    judge = _load_judge_config(str(batch["judge_request_model"]))
    if (
        judge.judge_id != batch["judge_id"]
        or judge.served_model != batch["judge_expected_served_model"]
        or judge.config_hash != batch["judge_config_hash"]
        or judge.registry_sha256 != batch["model_registry_sha256"]
    ):
        raise RejudgeError("Rejudge batch no longer matches the exact judge registry")

    existing_c5 = original.get("c5_rejudge")
    if existing_c5 is None:
        original_summary_hash = summary.summary_hash
        original_summary_sha256 = sha256_file(summary_path)
        original_generated_at = original.get("generated_at")
        judges: Dict[str, Any] = {}
        judge_order: list[str] = []
        merge_parent_hashes: list[str] = []
        common_input_manifest_hash = batch["input_manifest_hash"]
    else:
        if (
            not isinstance(existing_c5, Mapping)
            or existing_c5.get("schema_version") != REJUDGED_SUMMARY_VERSION
        ):
            raise RejudgeError("Existing C5 summary provenance is malformed")
        original_summary_hash = existing_c5.get("original_summary_hash")
        original_summary_sha256 = existing_c5.get("original_summary_sha256")
        original_generated_at = existing_c5.get("original_generated_at")
        common_input_manifest_hash = existing_c5.get("input_manifest_hash")
        raw_judges = existing_c5.get("judges")
        raw_order = existing_c5.get("judge_order")
        raw_parents = existing_c5.get("merge_parent_summary_hashes")
        if (
            not isinstance(original_summary_hash, str)
            or not _SHA256_RE.fullmatch(original_summary_hash)
            or not isinstance(original_summary_sha256, str)
            or not _SHA256_RE.fullmatch(original_summary_sha256)
            or not isinstance(common_input_manifest_hash, str)
            or not _SHA256_RE.fullmatch(common_input_manifest_hash)
            or not isinstance(raw_judges, Mapping)
            or not isinstance(raw_order, list)
            or not isinstance(raw_parents, list)
        ):
            raise RejudgeError("Existing C5 lineage is incomplete")
        judges = {str(key): dict(value) for key, value in raw_judges.items()}
        judge_order = [str(value) for value in raw_order]
        merge_parent_hashes = [str(value) for value in raw_parents]
        if (
            len(judge_order) != len(set(judge_order))
            or set(judge_order) != set(judges)
            or existing_c5.get("rubric_hash") != batch["rubric_hash"]
            or existing_c5.get("prompt_hash") != batch["prompt_hash"]
        ):
            raise RejudgeError("Existing C5 judge lineage is inconsistent")
        current_rows = {
            str(row["run_name"]): row for row in summary.rows
        }
        for existing_id, existing in judges.items():
            metric = existing.get("metric")
            values_hash = existing.get("metric_values_hash")
            if (
                existing.get("judge_id") != existing_id
                or existing.get("input_manifest_hash")
                != common_input_manifest_hash
                or existing.get("source_summary_hash") != original_summary_hash
                or existing.get("code_git_dirty") is not False
                or not isinstance(metric, str)
                or not isinstance(values_hash, str)
                or not _SHA256_RE.fullmatch(values_hash)
            ):
                raise RejudgeError("Existing C5 judge provenance is inconsistent")
            current_values = {
                name: current_rows[name].get(metric)
                for name in sorted(current_rows)
            }
            if sha256_json(current_values) != values_hash:
                raise RejudgeError(
                    f"Existing C5 metric values are stale: {existing_id}"
                )

    if (
        batch["source_summary_hash"] != original_summary_hash
        or batch["source_summary_sha256"] != original_summary_sha256
        or batch["input_manifest_hash"] != common_input_manifest_hash
    ):
        raise RejudgeError(
            "Judge batch was not produced from the immutable original C5 summary"
        )
    manifest = batch["input_manifest"]
    if manifest.get("input_record_hashes") != original.get("input_record_hashes"):
        raise RejudgeError("Judge input manifest record hashes disagree with summary")
    if batch["judge_id"] in judges:
        raise RejudgeError(f"Duplicate C5 judge batch: {batch['judge_id']}")
    existing_models = {
        item.get("judge_request_model")
        for item in judges.values()
        if isinstance(item, Mapping)
    }
    if batch["judge_request_model"] in existing_models:
        raise RejudgeError("Duplicate C5 requested judge identity")

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
    if metric_name in original.get("columns", []):
        raise RejudgeError(f"C5 metric already exists: {metric_name}")

    manifest_completed = {
        str(item["run_name"]): item
        for item in manifest["completed_runs"]
    }

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
        selected = manifest_completed[run_name]
        if (
            sidecar.get("render_sha256") != selected.get("render_sha256")
            or sidecar.get("best_candidate_id")
            != selected.get("best_candidate_id")
        ):
            raise RejudgeError(f"Sidecar selected-render binding mismatch: {run_name}")
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

    parent_summary_hash = str(original.pop("summary_hash"))
    columns = list(original.get("columns") or [])
    if metric_name not in columns:
        columns.append(metric_name)
    metric_values_hash = sha256_json(
        {
            str(row["run_name"]): row[metric_name]
            for row in sorted(rows, key=lambda item: str(item["run_name"]))
        }
    )
    judges[str(batch["judge_id"])] = {
        "judge_id": batch["judge_id"],
        "judge_role": batch["judge_role"],
        "judge_request_model": batch["judge_request_model"],
        "judge_expected_served_model": batch["judge_expected_served_model"],
        "judge_served_models": batch["judge_served_models"],
        "judge_protocol": batch["judge_protocol"],
        "judge_endpoint_class": batch["judge_endpoint_class"],
        "judge_max_tokens": batch["judge_max_tokens"],
        "judge_config_hash": batch["judge_config_hash"],
        "model_registry_sha256": batch["model_registry_sha256"],
        "metric": metric_name,
        "metric_values_hash": metric_values_hash,
        "rubric_hash": batch["rubric_hash"],
        "prompt_hash": batch["prompt_hash"],
        "input_manifest_hash": batch["input_manifest_hash"],
        "source_summary_hash": batch["source_summary_hash"],
        "source_summary_sha256": batch["source_summary_sha256"],
        "batch_hash": batch["batch_hash"],
        "code_git_commit": batch["code_git_commit"],
        "code_git_dirty": batch["code_git_dirty"],
        "sidecar_hashes": {
            run_name: sidecars[run_name]["sidecar_hash"]
            for run_name in sorted(sidecars)
        },
        "selected_render_hashes": dict(batch["selected_render_hashes"]),
        "served_identity_by_run": {
            run_name: sidecars[run_name]["judge_served_model"]
            for run_name in sorted(sidecars)
        },
        "failed_zero_runs": sorted(failed_zero_runs),
    }
    judge_order.append(str(batch["judge_id"]))
    merge_parent_hashes.append(parent_summary_hash)
    merged = {
        **original,
        "generated_at": utc_now(),
        "columns": columns,
        "runs": rows,
        "original_summary_hash": original_summary_hash,
        "original_generated_at": original_generated_at,
        "c5_rejudge": {
            "schema_version": REJUDGED_SUMMARY_VERSION,
            "original_summary_hash": original_summary_hash,
            "original_summary_sha256": original_summary_sha256,
            "original_generated_at": original_generated_at,
            "input_manifest_hash": common_input_manifest_hash,
            "rubric_hash": batch["rubric_hash"],
            "prompt_hash": batch["prompt_hash"],
            "judge_order": judge_order,
            "judges": judges,
            "merge_parent_summary_hashes": merge_parent_hashes,
        },
    }
    merged["summary_hash"] = sha256_json(merged)
    write_json_atomic(destination, merged)
    return destination, metric_name

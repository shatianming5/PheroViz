from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .manifest import (
    load_dataset_manifest,
    select_case,
    verify_case_metadata,
)
from .models import (
    RECORD_FILENAME,
    ExperimentSpec,
    ProvenanceError,
    RunRecord,
    sha256_file,
    sha256_path,
    utc_now,
    verify_artifacts,
    write_json_atomic,
)
from .providers import (
    ExperimentProvider,
    ProviderUnavailableError,
    load_provider,
)
from .scheduler import run_schedule


class ExistingRunError(ProvenanceError):
    """Raised when a run would overwrite prior evidence."""


ProviderLoader = Callable[
    [str, Mapping[str, object]],
    ExperimentProvider,
]


def _failure_attribution(exc: Exception) -> str:
    explicit = getattr(exc, "failure_attribution", None)
    if explicit in {"method", "infrastructure"}:
        return str(explicit)
    if isinstance(exc, ProviderUnavailableError):
        return "infrastructure"
    return "unclassified"


@dataclass(frozen=True)
class RunOutcome:
    record: RunRecord
    skipped: bool


def _freeze_dataset_manifest(
    spec: ExperimentSpec,
    *,
    run_dir: Path,
    record: RunRecord,
) -> Path:
    source = Path(spec.dataset_manifest_path)
    if sha256_file(source) != spec.dataset_manifest_hash:
        raise ProvenanceError("Dataset manifest changed after matrix expansion")

    suffix = source.suffix.lower()
    destination = run_dir / f"dataset_manifest.frozen{suffix}"
    if destination.exists():
        if sha256_file(destination) != spec.dataset_manifest_hash:
            raise ProvenanceError(
                "Existing frozen dataset manifest has an unexpected hash"
            )
    else:
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
                for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                    output_handle.write(block)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    cases = load_dataset_manifest(
        destination,
        dataset_mode=spec.dataset_mode,
    )
    selected = select_case(cases, spec.case_id)
    verify_case_metadata(
        selected,
        panel_count=spec.panel_count,
        split=spec.split,
    )
    record.artifact_paths["dataset_manifest"] = destination.relative_to(
        run_dir
    ).as_posix()
    record.artifact_hashes["dataset_manifest"] = sha256_path(destination)
    return destination


def _current_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(
            f"Cannot verify git commit at execution time: {exc}"
        ) from exc
    return result.stdout.strip().lower()


def _prepare_attempt(
    spec: ExperimentSpec,
    *,
    resume: bool,
) -> tuple[Path, Path, int, Path | None]:
    run_dir = Path(spec.artifact_root) / spec.run_name
    record_path = run_dir / RECORD_FILENAME
    preserved_record: Path | None = None
    attempt = 1

    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    elif not record_path.is_file():
        raise ExistingRunError(
            f"Run directory exists without {RECORD_FILENAME}; refusing legacy or "
            f"untracked contents: {run_dir}"
        )
    else:
        existing = RunRecord.read(record_path)
        existing.validate_provenance()
        if existing.spec_hash != spec.spec_hash:
            raise ExistingRunError(
                f"Existing run_name has a different ExperimentSpec: {spec.run_name}"
            )
        if not resume:
            raise ExistingRunError(
                f"Run already exists; use --resume to skip or retry: {spec.run_name}"
            )
        if existing.status == "completed":
            existing.validate_provenance(require_completed=True)
            verify_artifacts(existing, run_dir)
            return run_dir, record_path, existing.attempt, record_path

        attempt = existing.attempt + 1
        preserved_record = (
            run_dir
            / "attempt_records"
            / f"attempt_{existing.attempt:03d}.json"
        )
        if preserved_record.exists():
            raise ExistingRunError(
                f"Preserved attempt record already exists: {preserved_record}"
            )
        write_json_atomic(preserved_record, existing.to_dict())

    attempt_dir = run_dir / f"attempt_{attempt:03d}"
    if attempt_dir.exists():
        raise ExistingRunError(
            f"Attempt directory already exists: {attempt_dir}"
        )
    attempt_dir.mkdir(parents=True)
    return run_dir, attempt_dir, attempt, preserved_record


def execute_experiment(
    spec: ExperimentSpec,
    *,
    resume: bool = False,
    provider_loader: ProviderLoader = load_provider,
    monotonic: Callable[[], float] = time.monotonic,
) -> RunOutcome:
    prepared = _prepare_attempt(spec, resume=resume)
    run_dir, attempt_path, attempt, preserved_record = prepared
    record_path = run_dir / RECORD_FILENAME

    if (
        resume
        and attempt_path == record_path
        and record_path.is_file()
    ):
        return RunOutcome(record=RunRecord.read(record_path), skipped=True)

    record = RunRecord.start(spec, attempt=attempt)
    if preserved_record is not None:
        relative = preserved_record.relative_to(run_dir).as_posix()
        key = f"previous_attempt_{attempt - 1:03d}"
        record.artifact_paths[key] = relative
        record.artifact_hashes[key] = sha256_path(preserved_record)
    record.write(record_path)

    execution_started = monotonic()
    deadline_monotonic = (
        execution_started + float(spec.budget_value)
        if spec.budget_type == "wall_clock_seconds"
        else None
    )

    def persist() -> None:
        record.write(record_path)

    try:
        frozen_manifest_path = _freeze_dataset_manifest(
            spec,
            run_dir=run_dir,
            record=record,
        )
        persist()
        if spec.git_dirty:
            raise ProvenanceError(
                "ExperimentSpec was expanded from a dirty worktree; commit or stash "
                "source changes before running production experiments"
            )
        current_commit = _current_git_commit(Path(spec.repo_root))
        if current_commit != spec.git_commit:
            raise ProvenanceError(
                f"Git commit changed after matrix expansion: "
                f"{spec.git_commit} -> {current_commit}"
            )

        provider = provider_loader(spec.provider, spec.provider_options)
        record.provider_name = provider.name
        record.test_only = provider.test_only
        persist()
        provider.check_available()
        run_schedule(
            spec,
            provider,
            run_dir=run_dir,
            attempt_dir=attempt_path,
            dataset_manifest_path=frozen_manifest_path,
            record=record,
            persist=persist,
            monotonic=monotonic,
            deadline_monotonic=deadline_monotonic,
        )
        record.status = "completed"
        record.finished_at = utc_now()
        record.error = None
        record.write(record_path)
        record.validate_provenance(require_completed=True)
        verify_artifacts(record, run_dir)
        return RunOutcome(record=record, skipped=False)
    except Exception as exc:  # noqa: BLE001 - every failure must be persisted
        record.status = "failed"
        record.finished_at = utc_now()
        record.wall_clock_seconds = max(
            record.wall_clock_seconds,
            monotonic() - execution_started,
            0.0,
        )
        record.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "attribution": _failure_attribution(exc),
        }
        record.write(record_path)
        record.validate_provenance()
        return RunOutcome(record=record, skipped=False)


def execute_matrix(
    specs: Sequence[ExperimentSpec],
    *,
    resume: bool = False,
    provider_loader: ProviderLoader = load_provider,
) -> list[RunOutcome]:
    outcomes: list[RunOutcome] = []
    for spec in specs:
        outcomes.append(
            execute_experiment(
                spec,
                resume=resume,
                provider_loader=provider_loader,
            )
        )
    return outcomes

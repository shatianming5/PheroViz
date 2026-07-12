from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

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
from .providers import ExperimentProvider, load_provider
from .scheduler import run_schedule


class ExistingRunError(ProvenanceError):
    """Raised when a run would overwrite prior evidence."""


ProviderLoader = Callable[
    [str, Mapping[str, object]],
    ExperimentProvider,
]


@dataclass(frozen=True)
class RunOutcome:
    record: RunRecord
    skipped: bool


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

    def persist() -> None:
        record.write(record_path)

    try:
        if spec.git_dirty:
            raise ProvenanceError(
                "ExperimentSpec was expanded from a dirty worktree; commit or stash "
                "source changes before running production experiments"
            )
        manifest_path = Path(spec.dataset_manifest_path)
        if sha256_file(manifest_path) != spec.dataset_manifest_hash:
            raise ProvenanceError(
                "Dataset manifest changed after matrix expansion"
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
            record=record,
            persist=persist,
            monotonic=monotonic,
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

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .models import (
    ExperimentSpec,
    ProvenanceError,
    RunRecord,
    sha256_path,
    write_json_atomic,
)
from .providers import (
    ExperimentProvider,
    GenerationRequest,
    ProviderBatch,
    ProviderExecutionError,
)


class BudgetError(RuntimeError):
    """Raised when a provider violates or underuses a strict budget."""


def selection_score(
    metrics: Mapping[str, float],
    metric_config: Mapping[str, Any],
) -> tuple[float, str]:
    selection = metric_config.get("selection")
    if not isinstance(selection, Mapping):
        raise ProvenanceError("Metric config has no selection object")
    direction = str(selection.get("direction", "maximize"))
    if direction not in {"maximize", "minimize"}:
        raise ProvenanceError(f"Unsupported selection direction: {direction}")

    metric_name = selection.get("metric")
    weights = selection.get("weights")
    if isinstance(metric_name, str):
        if metric_name not in metrics:
            raise ProviderExecutionError(
                f"Candidate is missing selection metric {metric_name!r}"
            )
        score = float(metrics[metric_name])
    elif isinstance(weights, Mapping):
        score = 0.0
        for name, weight in weights.items():
            if name not in metrics:
                raise ProviderExecutionError(
                    f"Candidate is missing weighted metric {name!r}"
                )
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ProvenanceError(f"Weight for {name!r} must be numeric")
            score += float(weight) * float(metrics[name])
    else:
        raise ProvenanceError(
            "Metric selection must define either metric or weights"
        )
    if not math.isfinite(score):
        raise ProviderExecutionError("Candidate selection score is not finite")
    return score, direction


def _is_better(
    score: float,
    best_score: Optional[float],
    direction: str,
) -> bool:
    if best_score is None:
        return True
    if direction == "maximize":
        return score > best_score
    return score < best_score


def _normalize_artifact(
    raw_path: str,
    *,
    run_dir: Path,
) -> tuple[str, str]:
    path = Path(raw_path).expanduser()
    if path.is_symlink():
        raise ProvenanceError(f"Symlink artifacts are not accepted: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProvenanceError(f"Artifact does not exist: {path}") from exc
    try:
        relative = resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ProvenanceError(
            f"Provider artifact escaped its allocated run directory: {resolved}"
        ) from exc
    return relative.as_posix(), sha256_path(resolved)


def _candidate_summary(
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "metrics": candidate["metrics"],
        "selection_score": candidate["selection_score"],
        "artifact_paths": candidate["artifact_paths"],
    }


def run_schedule(
    spec: ExperimentSpec,
    provider: ExperimentProvider,
    *,
    run_dir: Path,
    attempt_dir: Path,
    dataset_manifest_path: Path,
    record: RunRecord,
    persist: Callable[[], None],
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Execute one fair schedule and update ``record`` in place."""

    record.provider_name = provider.name
    record.test_only = provider.test_only
    persist()

    max_calls_raw = spec.method_config.get("max_provider_calls", 10_000)
    if isinstance(max_calls_raw, bool) or not isinstance(max_calls_raw, int):
        raise ProvenanceError("method_config.max_provider_calls must be an integer")
    if max_calls_raw < 1:
        raise ProvenanceError("method_config.max_provider_calls must be positive")

    execution_started = monotonic()
    call_index = 0
    best_score: Optional[float] = None
    best_candidate: Optional[Dict[str, Any]] = None
    last_candidate: Optional[Dict[str, Any]] = None
    history: list[Dict[str, Any]] = []
    stopped = False

    while True:
        elapsed = monotonic() - execution_started
        record.wall_clock_seconds = max(elapsed, 0.0)
        if spec.budget_type == "renders":
            budget = int(spec.budget_value)
            remaining_renders = budget - record.render_count
            remaining_seconds = None
            deadline = None
            if remaining_renders == 0:
                break
            if remaining_renders < 0:
                raise BudgetError("Render budget was exceeded")
        else:
            budget_seconds = float(spec.budget_value)
            remaining_seconds = budget_seconds - elapsed
            remaining_renders = None
            deadline = execution_started + budget_seconds
            if remaining_seconds <= 0:
                break

        if call_index >= max_calls_raw:
            raise BudgetError(
                "Provider call limit reached before the budget completed"
            )
        call_index += 1
        call_dir = attempt_dir / "provider" / f"call_{call_index:04d}"
        call_dir.mkdir(parents=True, exist_ok=False)
        request = GenerationRequest(
            spec=spec,
            dataset_manifest_path=dataset_manifest_path,
            output_dir=call_dir,
            call_index=call_index,
            remaining_renders=remaining_renders,
            remaining_seconds=remaining_seconds,
            deadline_monotonic=deadline,
            history=tuple(history) if spec.schedule == "iterative" else (),
            previous_candidate=(
                _candidate_summary(last_candidate)
                if spec.schedule == "iterative" and last_candidate
                else None
            ),
        )

        try:
            raw_batch = provider.generate(request)
        except ProviderExecutionError as exc:
            for label, raw_path in sorted(exc.artifacts.items()):
                relative, digest = _normalize_artifact(
                    raw_path,
                    run_dir=run_dir,
                )
                key = f"failure.call_{call_index:04d}.{label}"
                record.artifact_paths[key] = relative
                record.artifact_hashes[key] = digest
            record.wall_clock_seconds = max(
                monotonic() - execution_started,
                0.0,
            )
            persist()
            raise
        batch = (
            raw_batch
            if isinstance(raw_batch, ProviderBatch)
            else ProviderBatch(candidates=(raw_batch,))
        )
        if spec.schedule == "best_of_n":
            if len(batch.candidates) != 1:
                raise ProviderExecutionError(
                    "best_of_n providers must return exactly one independent candidate "
                    "per call"
                )
            if batch.candidates[0].render_count != 1:
                raise ProviderExecutionError(
                    "Each best_of_n candidate must consume exactly one render"
                )

        batch_render_count = sum(
            candidate.render_count for candidate in batch.candidates
        )
        call_elapsed = monotonic() - execution_started
        if spec.budget_type == "renders":
            if batch_render_count > int(remaining_renders or 0):
                record.render_count += batch_render_count
                record.wall_clock_seconds = max(call_elapsed, 0.0)
                persist()
                raise BudgetError(
                    f"Provider exceeded render budget by "
                    f"{batch_render_count - int(remaining_renders or 0)}"
                )
        elif call_elapsed > float(spec.budget_value):
            record.render_count += batch_render_count
            record.wall_clock_seconds = max(call_elapsed, 0.0)
            persist()
            raise BudgetError("Provider exceeded wall-clock budget")

        for result in batch.candidates:
            candidate_number = len(record.candidates) + 1
            candidate_id = f"candidate_{candidate_number:04d}"
            artifact_paths: Dict[str, str] = {}
            artifact_hashes: Dict[str, str] = {}
            for label, raw_path in sorted(result.artifacts.items()):
                if not isinstance(label, str) or not label:
                    raise ProvenanceError("Artifact labels must be non-empty strings")
                relative, digest = _normalize_artifact(
                    raw_path,
                    run_dir=run_dir,
                )
                artifact_paths[label] = relative
                artifact_hashes[label] = digest
                record_key = f"{candidate_id}.{label}"
                record.artifact_paths[record_key] = relative
                record.artifact_hashes[record_key] = digest

            score, direction = selection_score(
                result.metrics,
                spec.metric_config,
            )
            candidate = {
                "candidate_id": candidate_id,
                "call_index": call_index,
                "metrics": {
                    name: float(value) for name, value in result.metrics.items()
                },
                "selection_score": score,
                "selection_direction": direction,
                "render_count": result.render_count,
                "artifact_paths": artifact_paths,
                "artifact_hashes": artifact_hashes,
                "provider_metadata": result.metadata,
                "test_only": bool(result.test_only or batch.test_only),
            }
            candidate_path = (
                attempt_dir / "candidates" / f"{candidate_id}.json"
            )
            write_json_atomic(candidate_path, candidate)
            candidate_relative, candidate_digest = _normalize_artifact(
                str(candidate_path),
                run_dir=run_dir,
            )
            record.artifact_paths[f"{candidate_id}.metadata"] = candidate_relative
            record.artifact_hashes[f"{candidate_id}.metadata"] = candidate_digest

            record.candidates.append(candidate)
            record.render_count += result.render_count
            record.test_only = bool(
                record.test_only or batch.test_only or result.test_only
            )
            if _is_better(score, best_score, direction):
                best_score = score
                best_candidate = candidate
                record.best_candidate_id = candidate_id
                record.metrics = dict(candidate["metrics"])
            if best_candidate is None:
                raise ProvenanceError("Best-so-far archive was not initialized")
            record.best_history.append(
                {
                    "after_candidate_id": candidate_id,
                    "best_candidate_id": best_candidate["candidate_id"],
                    "best_selection_score": best_score,
                }
            )
            history.append(_candidate_summary(candidate))
            last_candidate = candidate

            best_path = run_dir / "best_so_far.json"
            write_json_atomic(
                best_path,
                {
                    "run_name": spec.run_name,
                    "best_candidate_id": best_candidate["candidate_id"],
                    "metrics": best_candidate["metrics"],
                    "selection_score": best_score,
                    "history": record.best_history,
                    "artifact_paths": best_candidate["artifact_paths"],
                    "test_only": record.test_only,
                },
            )
            best_relative, best_digest = _normalize_artifact(
                str(best_path),
                run_dir=run_dir,
            )
            record.artifact_paths["best_so_far"] = best_relative
            record.artifact_hashes["best_so_far"] = best_digest
            record.wall_clock_seconds = max(
                monotonic() - execution_started,
                0.0,
            )
            persist()

        if batch.stop:
            stopped = True
            break

    if spec.budget_type == "renders" and record.render_count != int(
        spec.budget_value
    ):
        reason = "provider stopped early" if stopped else "budget did not complete"
        raise BudgetError(
            f"Render budget requires {int(spec.budget_value)} renders, "
            f"recorded {record.render_count}: {reason}"
        )
    if not record.candidates:
        raise ProviderExecutionError("Schedule completed without any candidates")
    record.wall_clock_seconds = max(monotonic() - execution_started, 0.0)

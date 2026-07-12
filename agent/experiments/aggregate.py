from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .manifest import (
    DatasetCase,
    _normalize_doi,
    load_dataset_manifest,
    select_case,
    verify_case_metadata,
)
from .models import (
    RECORD_FILENAME,
    SCHEMA_VERSION,
    ProvenanceError,
    RunRecord,
    sha256_file,
    sha256_json,
    utc_now,
    verify_artifacts,
    write_json_atomic,
)


class AggregationError(ProvenanceError):
    """Raised when experiment outputs cannot be safely aggregated."""


def _discover_records(run_root: Path, excluded_dir: Path | None) -> list[Path]:
    if not run_root.is_dir():
        raise AggregationError(f"Run root does not exist: {run_root}")
    records: list[Path] = []
    excluded = excluded_dir.resolve() if excluded_dir is not None else None
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if excluded is not None and child.resolve() == excluded:
            continue
        record_path = child / RECORD_FILENAME
        if not record_path.is_file():
            raise AggregationError(
                f"Legacy or untracked run directory has no {RECORD_FILENAME}: "
                f"{child}"
            )
        records.append(record_path)
    return records


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(stream.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _summary_row(
    record: RunRecord,
    *,
    doi: str | None = None,
    failed_metric_names: Sequence[str] = (),
) -> Dict[str, Any]:
    completed = record.status == "completed"
    failure_attribution = (
        str((record.error or {}).get("attribution") or "")
        if not completed
        else ""
    )
    row: Dict[str, Any] = {
        "run_name": record.run_name,
        "method": record.method,
        "backbone": record.backbone,
        "case_id": record.case_id,
        "doi": doi or "",
        "panel_count": record.panel_count,
        "split": record.split,
        "seed": record.seed,
        "budget_type": record.budget_type,
        "budget_value": record.budget_value,
        "dataset_manifest_hash": record.dataset_manifest_hash,
        "git_commit": record.git_commit,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "status": record.status,
        "execution_success": 1.0 if completed else 0.0,
        "failure_attribution": failure_attribution,
        "failure_type": (
            str((record.error or {}).get("type") or "")
            if not completed
            else ""
        ),
        "test_only": record.test_only,
        "render_count": record.render_count,
        "wall_clock_seconds": record.wall_clock_seconds,
        "metric_version": record.metric_version,
        "metric_config_hash": record.metric_config_hash,
        "provider": record.provider,
        "provider_name": record.provider_name,
        "best_candidate_id": record.best_candidate_id,
        "spec_hash": record.spec_hash,
        "record_hash": record.record_hash,
    }
    metrics = (
        dict(record.metrics)
        if completed
        else {name: 0.0 for name in failed_metric_names}
    )
    metrics["execution_success"] = 1.0 if completed else 0.0
    for name, value in sorted(metrics.items()):
        row[f"metric.{name}"] = value
    return row


def _verified_frozen_manifest_case(
    record: RunRecord,
    run_dir: Path,
) -> tuple[Path, DatasetCase]:
    relative_text = record.artifact_paths.get("dataset_manifest")
    if not relative_text:
        raise AggregationError(
            f"Run has no frozen dataset manifest: {record.run_name}"
        )
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise AggregationError(
            f"Frozen dataset manifest path is unsafe: {relative_text}"
        )
    frozen = (run_dir.resolve() / relative).resolve()
    try:
        frozen.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise AggregationError(
            f"Frozen dataset manifest escaped run directory: {record.run_name}"
        ) from exc
    if sha256_file(frozen) != record.dataset_manifest_hash:
        raise AggregationError(
            f"Frozen dataset manifest hash mismatch: {record.run_name}"
        )
    cases = load_dataset_manifest(
        frozen,
        dataset_mode=str(
            record.experiment_spec.get("dataset_mode", "legacy")
        ),
    )
    selected = select_case(cases, record.case_id)
    verify_case_metadata(
        selected,
        panel_count=record.panel_count,
        split=record.split,
    )
    return frozen, selected


def verify_frozen_manifest(record: RunRecord, run_dir: Path) -> Path:
    frozen, _ = _verified_frozen_manifest_case(record, run_dir)
    return frozen


def _record_field(record: RunRecord | Mapping[str, Any], name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name)


def _is_analysis_ready_record(
    record: RunRecord | Mapping[str, Any],
) -> bool:
    status = _record_field(record, "status")
    if status == "completed":
        return True
    if status != "failed":
        return False
    if isinstance(record, Mapping):
        return record.get("failure_attribution") == "method"
    return (record.error or {}).get("attribution") == "method"


def assert_paired_ready(
    records: Sequence[RunRecord | Mapping[str, Any]],
    *,
    methods: Sequence[str],
    backbone: str,
    seed: int,
    budget_type: str,
    budget_value: float,
) -> set[str]:
    method_names = list(methods)
    if len(method_names) < 2 or len(method_names) != len(set(method_names)):
        raise AggregationError(
            "Paired comparison requires at least two unique methods"
        )

    by_method: Dict[
        str,
        Dict[str, RunRecord | Mapping[str, Any]],
    ] = {
        method: {} for method in method_names
    }
    for record in records:
        if (
            not _is_analysis_ready_record(record)
            or _record_field(record, "method") not in by_method
            or _record_field(record, "backbone") != backbone
            or _record_field(record, "seed") != seed
            or _record_field(record, "budget_type") != budget_type
            or _record_field(record, "budget_value") != budget_value
        ):
            continue
        method = str(_record_field(record, "method"))
        raw_case_id = _record_field(record, "case_id")
        if not isinstance(raw_case_id, str) or not raw_case_id.strip():
            raise AggregationError(
                f"Paired record for method {method!r} has no case_id"
            )
        case_id = raw_case_id
        method_cases = by_method[method]
        if case_id in method_cases:
            raise AggregationError(
                f"Duplicate paired case {case_id!r} for method "
                f"{method!r}"
            )
        method_cases[case_id] = record

    reference_method = method_names[0]
    reference = by_method[reference_method]
    if not reference:
        raise AggregationError(
            f"No analysis-ready records for paired method {reference_method!r}"
        )
    reference_cases = set(reference)
    reference_hashes = {
        _record_field(record, "dataset_manifest_hash")
        for record in reference.values()
    }
    if len(reference_hashes) != 1:
        raise AggregationError(
            f"Method {reference_method!r} mixes dataset manifests"
        )
    reference_configs = {
        (
            _record_field(record, "metric_config_hash"),
            _record_field(record, "metric_version"),
        )
        for record in reference.values()
    }
    if len(reference_configs) != 1:
        raise AggregationError(
            f"Method {reference_method!r} mixes metric configurations"
        )

    for method in method_names[1:]:
        candidates = by_method[method]
        if not candidates:
            raise AggregationError(
                f"No analysis-ready records for paired method {method!r}"
            )
        case_ids = set(candidates)
        if case_ids != reference_cases:
            missing = sorted(reference_cases - case_ids)
            extra = sorted(case_ids - reference_cases)
            raise AggregationError(
                f"Paired case mismatch for method {method!r}; "
                f"missing={missing}, extra={extra}"
            )
        manifest_hashes = {
            _record_field(record, "dataset_manifest_hash")
            for record in candidates.values()
        }
        if manifest_hashes != reference_hashes:
            raise AggregationError(
                f"Paired methods use different dataset manifests: {method!r}"
            )
        metric_configs = {
            (
                _record_field(record, "metric_config_hash"),
                _record_field(record, "metric_version"),
            )
            for record in candidates.values()
        }
        if metric_configs != reference_configs:
            raise AggregationError(
                f"Paired methods use different metric configurations: {method!r}"
            )
        for case_id in sorted(reference_cases):
            expected = reference[case_id]
            actual = candidates[case_id]
            if (
                _record_field(actual, "panel_count")
                != _record_field(expected, "panel_count")
                or _record_field(actual, "split")
                != _record_field(expected, "split")
            ):
                raise AggregationError(
                    f"Paired case metadata mismatch for {case_id!r}"
                )
    return reference_cases


def _configured_metric_names(record: RunRecord) -> set[str]:
    names = set(record.metrics)
    selection = record.metric_config.get("selection")
    if not isinstance(selection, Mapping):
        return names
    metric = selection.get("metric")
    if isinstance(metric, str) and metric:
        names.add(metric)
    weights = selection.get("weights")
    if isinstance(weights, Mapping):
        names.update(
            str(name) for name in weights if isinstance(name, str) and name
        )
    return names


def aggregate_runs(
    run_root: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    run_root = run_root.expanduser().resolve()
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else run_root
    )
    record_paths = _discover_records(
        run_root,
        output_dir if output_dir != run_root else None,
    )
    accepted: list[tuple[RunRecord, str | None]] = []
    for record_path in record_paths:
        try:
            record = RunRecord.read(record_path)
            record.validate_provenance()
        except (ProvenanceError, TypeError) as exc:
            raise AggregationError(
                f"Invalid provenance in {record_path}: {exc}"
            ) from exc
        if record.test_only:
            raise AggregationError(
                f"Test-only output cannot enter paper aggregation: {record.run_name}"
            )
        if bool(record.experiment_spec.get("git_dirty")):
            raise AggregationError(
                f"Completed run used a dirty worktree and lacks reproducible source "
                f"provenance: {record.run_name}"
            )
        if record.status == "running":
            raise AggregationError(
                f"Confirmatory aggregation is blocked by a non-terminal run: "
                f"{record.run_name}"
            )
        if record.status == "failed":
            attribution = str(
                (record.error or {}).get("attribution") or "unclassified"
            )
            if attribution != "method":
                raise AggregationError(
                    "Confirmatory aggregation is blocked by a failed run that "
                    "cannot be scored as a method outcome: "
                    f"{record.run_name} (attribution={attribution!r})"
                )
            selection = record.metric_config.get("selection")
            if (
                not isinstance(selection, Mapping)
                or selection.get("direction", "maximize") != "maximize"
            ):
                raise AggregationError(
                    "Zero-outcome scoring is only defined for explicitly "
                    f"maximized metrics: {record.run_name}"
                )
        try:
            if record.status == "completed":
                record.validate_provenance(require_completed=True)
                recorded_success = record.metrics.get("execution_success")
                if (
                    recorded_success is not None
                    and float(recorded_success) != 1.0
                ):
                    raise AggregationError(
                        "Completed run reports execution_success != 1: "
                        f"{record.run_name}"
                    )
            verify_artifacts(record, record_path.parent)
            _, selected_case = _verified_frozen_manifest_case(
                record,
                record_path.parent,
            )
        except (OSError, KeyError, ProvenanceError) as exc:
            raise AggregationError(
                f"Incomplete analysis-ready run {record.run_name}: {exc}"
            ) from exc
        doi = _normalize_doi(selected_case.payload.get("doi"))
        if (
            doi is None
            and record.experiment_spec.get("dataset_mode") == "sealed_benchmark"
        ):
            raise AggregationError(
                f"Sealed run {record.run_name} has no valid DOI in its frozen "
                "manifest"
            )
        accepted.append((record, doi))

    if not accepted:
        raise AggregationError(
            "No analysis-ready, provenance-valid production runs were found"
        )

    metric_names_by_config: Dict[tuple[str, str], set[str]] = {}
    for record, _ in accepted:
        key = (record.metric_config_hash, record.metric_version)
        metric_names_by_config.setdefault(key, set()).update(
            _configured_metric_names(record)
        )
        metric_names_by_config[key].add("execution_success")
    rows = [
        _summary_row(
            record,
            doi=doi,
            failed_metric_names=sorted(
                metric_names_by_config[
                    (record.metric_config_hash, record.metric_version)
                ]
            ),
        )
        for record, doi in accepted
    ]
    fixed_columns = [
        "run_name",
        "method",
        "backbone",
        "case_id",
        "doi",
        "panel_count",
        "split",
        "seed",
        "budget_type",
        "budget_value",
        "dataset_manifest_hash",
        "git_commit",
        "started_at",
        "finished_at",
        "status",
        "execution_success",
        "failure_attribution",
        "failure_type",
        "test_only",
        "render_count",
        "wall_clock_seconds",
        "metric_version",
        "metric_config_hash",
        "provider",
        "provider_name",
        "best_candidate_id",
        "spec_hash",
        "record_hash",
    ]
    metric_columns = sorted(
        {key for row in rows for key in row if key.startswith("metric.")}
    )
    columns = fixed_columns + metric_columns

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    _write_csv_atomic(csv_path, rows, columns)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_root": str(run_root),
        "run_count": len(rows),
        "completed_run_count": sum(
            record.status == "completed" for record, _ in accepted
        ),
        "method_failed_run_count": sum(
            record.status == "failed" for record, _ in accepted
        ),
        "columns": columns,
        "runs": rows,
        "input_record_hashes": {
            record.run_name: record.record_hash for record, _ in accepted
        },
    }
    payload["summary_hash"] = sha256_json(payload)
    write_json_atomic(json_path, payload)
    return csv_path, json_path

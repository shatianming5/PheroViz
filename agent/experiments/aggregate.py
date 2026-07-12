from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any, Dict, Sequence

from .models import (
    RECORD_FILENAME,
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


def _summary_row(record: RunRecord) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_name": record.run_name,
        "method": record.method,
        "backbone": record.backbone,
        "seed": record.seed,
        "budget_type": record.budget_type,
        "budget_value": record.budget_value,
        "dataset_manifest_hash": record.dataset_manifest_hash,
        "git_commit": record.git_commit,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "status": record.status,
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
    for name, value in sorted(record.metrics.items()):
        row[f"metric.{name}"] = value
    return row


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
    accepted: list[RunRecord] = []
    for record_path in record_paths:
        try:
            record = RunRecord.read(record_path)
            record.validate_provenance()
        except (ProvenanceError, TypeError) as exc:
            raise AggregationError(
                f"Invalid provenance in {record_path}: {exc}"
            ) from exc
        if record.status != "completed":
            continue
        if record.test_only:
            raise AggregationError(
                f"Test-only output cannot enter paper aggregation: {record.run_name}"
            )
        if bool(record.experiment_spec.get("git_dirty")):
            raise AggregationError(
                f"Completed run used a dirty worktree and lacks reproducible source "
                f"provenance: {record.run_name}"
            )
        try:
            record.validate_provenance(require_completed=True)
            verify_artifacts(record, record_path.parent)
            manifest_path = Path(
                str(record.experiment_spec["dataset_manifest_path"])
            )
            if sha256_file(manifest_path) != record.dataset_manifest_hash:
                raise ProvenanceError("Dataset manifest hash no longer matches")
        except (OSError, KeyError, ProvenanceError) as exc:
            raise AggregationError(
                f"Incomplete completed run {record.run_name}: {exc}"
            ) from exc
        accepted.append(record)

    if not accepted:
        raise AggregationError(
            "No completed, provenance-valid production runs were found"
        )

    rows = [_summary_row(record) for record in accepted]
    fixed_columns = [
        "run_name",
        "method",
        "backbone",
        "seed",
        "budget_type",
        "budget_value",
        "dataset_manifest_hash",
        "git_commit",
        "started_at",
        "finished_at",
        "status",
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
    for row in rows:
        for column in columns:
            row.setdefault(column, "")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    _write_csv_atomic(csv_path, rows, columns)
    payload = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "source_root": str(run_root),
        "run_count": len(rows),
        "columns": columns,
        "runs": rows,
        "input_record_hashes": {
            record.run_name: record.record_hash for record in accepted
        },
    }
    payload["summary_hash"] = sha256_json(payload)
    write_json_atomic(json_path, payload)
    return csv_path, json_path

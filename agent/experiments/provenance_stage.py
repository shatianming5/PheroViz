from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import (
    ProvenanceError,
    RunRecord,
    read_json,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from .production_statistics import load_provenance_summary


D90_COMMIT = "d90d655b96bfb3b95bfdc37665692957969d8968"
TIERS = ("frontier", "mid", "open")
METHODS = ("best_of_n", "flat_iterative", "pheroviz_full")
JUDGES = ("primary", "secondary")
MODEL_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_registry.yml"


class ProvenanceStageError(ProvenanceError):
    """Raised when a final provenance index cannot be validated."""


def _binding(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "file_sha256": sha256_file(resolved)}


def _without_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def _explicit_values(value: Any, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, (str, int, float)) and str(item).strip():
                found.add(str(item))
            elif key in keys and isinstance(item, list):
                found.update(
                    str(entry)
                    for entry in item
                    if isinstance(entry, (str, int, float)) and str(entry).strip()
                )
            found.update(_explicit_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.update(_explicit_values(item, keys))
    return found


def _one_explicit(value: Any, keys: set[str]) -> str | None:
    values = _explicit_values(value, keys)
    return next(iter(values)) if len(values) == 1 else None


def _identity_row(
    record: RunRecord,
    *,
    tier: str,
    record_path: Path,
    source_record_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    best = next(
        (
            candidate
            for candidate in record.candidates
            if candidate.get("candidate_id") == record.best_candidate_id
        ),
        None,
    )
    metadata = best.get("provider_metadata", {}) if isinstance(best, Mapping) else {}
    fields = {
        "served_model": _one_explicit(
            metadata,
            {
                "served_model",
                "served_models",
                "served_model_name",
                "model_id",
                "model",
            },
        ),
        "snapshot_or_revision": _one_explicit(
            metadata, {"snapshot", "snapshot_id", "model_revision", "revision"}
        ),
        "endpoint_class": _one_explicit(
            metadata, {"endpoint_class", "endpoint_type"}
        ),
        "hosting_engine": _one_explicit(
            metadata, {"hosting_engine", "inference_engine", "server_engine"}
        ),
        "precision": _one_explicit(
            metadata, {"precision", "dtype", "torch_dtype"}
        ),
        "access_timestamp": _one_explicit(
            metadata, {"access_timestamp", "accessed_at", "request_timestamp"}
        ),
    }
    missing = [
        name
        for name in (
            "endpoint_class",
            "hosting_engine",
            "precision",
            "access_timestamp",
        )
        if fields[name] is None
    ]
    if fields["served_model"] is None and fields["snapshot_or_revision"] is None:
        missing.extend(("served_model", "snapshot_or_revision"))
    row = {
        "tier": tier,
        "run_name": record.run_name,
        "method": record.method,
        "seed": record.seed,
        "case_id": record.case_id,
        "record_path": str(
            source_record_path if source_record_path is not None else record_path.resolve()
        ),
        "record_file_sha256": sha256_file(record_path),
        "record_hash": record.record_hash,
        "spec_hash": record.spec_hash,
        "request_alias": record.backbone,
        **fields,
        "execution_started_at": record.started_at,
        "execution_finished_at": record.finished_at,
        "missing_fields": missing,
    }
    return row, missing


def build_generator_identity_index(
    completion_reports: Mapping[str, Path],
    *,
    summary_paths: Mapping[str, Path] | None = None,
    record_roots: Mapping[str, Path] | None = None,
    record_indexes: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    if set(completion_reports) != set(TIERS):
        raise ProvenanceStageError("Generator index requires exactly frontier/mid/open")

    sources: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    needed: list[dict[str, str]] = []
    run_names: set[str] = set()

    for tier in TIERS:
        report_path = completion_reports[tier].expanduser().resolve()
        report = read_json(report_path)
        experiment = report.get("experiment")
        if not isinstance(experiment, Mapping):
            raise ProvenanceStageError(f"{tier} completion report has no experiment")
        if (
            experiment.get("git_commit") != D90_COMMIT
            or experiment.get("git_dirty") is not False
        ):
            raise ProvenanceStageError(f"{tier} does not bind clean d90d655")
        summary = report.get("summary")
        if not isinstance(summary, Mapping):
            raise ProvenanceStageError(f"{tier} completion report has no summary")
        raw_path = summary.get("remote_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ProvenanceStageError(f"{tier} summary path is missing")
        source_summary_path = Path(raw_path).expanduser()
        summary_path = (
            summary_paths[tier].expanduser().resolve()
            if summary_paths is not None and tier in summary_paths
            else source_summary_path
        )
        sources[tier] = {
            "completion_report": _binding(report_path),
            "summary": {
                "source_path": str(source_summary_path),
                "archive_path": str(summary_path),
                "expected_file_sha256": summary.get("file_sha256"),
                "expected_summary_hash": summary.get("summary_hash"),
                "expected_record_bindings_hash": summary.get("record_bindings_hash"),
                "available": summary_path.is_file(),
            },
            "source_level_identity": report.get("model"),
        }
        if not summary_path.is_file():
            needed.append(
                {
                    "artifact": f"{tier}.summary",
                    "path": str(source_summary_path),
                    "reason": "sealed summary is unavailable locally",
                }
            )
            continue
        if sha256_file(summary_path) != summary.get("file_sha256"):
            raise ProvenanceStageError(f"{tier} summary file hash mismatch")
        loaded = load_provenance_summary(summary_path)
        if loaded.summary_hash != summary.get("summary_hash"):
            raise ProvenanceStageError(f"{tier} summary semantic hash mismatch")
        source_payload = read_json(summary_path)
        source_root_raw = source_payload.get("source_root")
        if not isinstance(source_root_raw, str) or not source_root_raw:
            raise ProvenanceStageError(f"{tier} summary source_root is missing")
        source_root = Path(source_root_raw).expanduser()
        if not source_root.is_absolute():
            source_root = (summary_path.parent / source_root).resolve()
        recovery_root = (
            record_roots[tier].expanduser().resolve()
            if record_roots is not None and tier in record_roots
            else source_root
        )
        tier_rows = list(loaded.rows)
        if len(tier_rows) != 171:
            raise ProvenanceStageError(f"{tier} summary must contain exactly 171 rows")
        if record_indexes is not None and tier in record_indexes:
            compact_path = record_indexes[tier].expanduser().resolve()
            compact = read_json(compact_path)
            compact_hash = compact.get("record_index_hash")
            if (
                compact_hash
                != sha256_json(_without_hash(compact, "record_index_hash"))
                or compact.get("tier") != tier
                or compact.get("experiment_commit") != D90_COMMIT
                or compact.get("source_summary_hash") != loaded.summary_hash
                or compact.get("record_count") != 171
                or not isinstance(compact.get("rows"), list)
                or len(compact["rows"]) != 171
            ):
                raise ProvenanceStageError(f"{tier} compact record index is invalid")
            compact_rows = {
                str(row.get("run_name")): row
                for row in compact["rows"]
                if isinstance(row, Mapping)
            }
            if len(compact_rows) != 171:
                raise ProvenanceStageError(f"{tier} compact record index has duplicates")
            for summary_row in tier_rows:
                run_name = str(summary_row.get("run_name", ""))
                row = compact_rows.get(run_name)
                if (
                    row is None
                    or row.get("record_hash") != summary_row.get("record_hash")
                    or row.get("spec_hash") != summary_row.get("spec_hash")
                    or row.get("method") != summary_row.get("method")
                    or row.get("seed") != summary_row.get("seed")
                    or row.get("case_id") != summary_row.get("case_id")
                    or row.get("tier") != tier
                    or row.get("request_alias") != summary_row.get("backbone")
                    or row.get("execution_started_at")
                    != summary_row.get("started_at")
                    or row.get("execution_finished_at")
                    != summary_row.get("finished_at")
                    or not isinstance(row.get("record_file_sha256"), str)
                    or len(row["record_file_sha256"]) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in row["record_file_sha256"]
                    )
                    or run_name in run_names
                ):
                    raise ProvenanceStageError(
                        f"{tier} compact record binding mismatch: {run_name}"
                    )
                missing = row.get("missing_fields")
                expected_missing = [
                    field
                    for field in (
                        "endpoint_class",
                        "hosting_engine",
                        "precision",
                        "access_timestamp",
                    )
                    if row.get(field) is None
                ]
                if row.get("served_model") is None and row.get(
                    "snapshot_or_revision"
                ) is None:
                    expected_missing.extend(("served_model", "snapshot_or_revision"))
                if missing != expected_missing:
                    raise ProvenanceStageError(
                        f"{tier} compact identity disclosure mismatch: {run_name}"
                    )
                run_names.add(run_name)
                rows.append(dict(row))
                for field in missing:
                    needed.append(
                        {
                            "artifact": f"{tier}.run_record.{field}",
                            "path": str(row.get("record_path")),
                            "reason": f"explicit {field} is absent or ambiguous",
                        }
                    )
            sources[tier]["compact_record_index"] = _binding(compact_path)
            continue
        for summary_row in tier_rows:
            run_name = str(summary_row.get("run_name", ""))
            if not run_name or run_name in run_names:
                raise ProvenanceStageError(f"Duplicate or empty run_name: {run_name!r}")
            run_names.add(run_name)
            source_record_path = source_root / run_name / "run_record.json"
            record_path = recovery_root / run_name / "run_record.json"
            if not record_path.is_file():
                needed.append(
                    {
                        "artifact": f"{tier}.run_record",
                        "path": str(source_record_path),
                        "reason": f"sealed record unavailable for {run_name}",
                    }
                )
                continue
            record = RunRecord.read(record_path)
            record.validate_provenance(require_completed=True)
            if (
                record.record_hash != summary_row.get("record_hash")
                or record.spec_hash != summary_row.get("spec_hash")
                or record.experiment_spec.get("git_commit") != D90_COMMIT
                or record.experiment_spec.get("git_dirty") is not False
            ):
                raise ProvenanceStageError(f"Stale or dirty record binding: {run_name}")
            row, missing = _identity_row(
                record,
                tier=tier,
                record_path=record_path,
                source_record_path=source_record_path,
            )
            rows.append(row)
            for field in missing:
                needed.append(
                    {
                        "artifact": f"{tier}.run_record.{field}",
                        "path": str(source_record_path),
                        "reason": f"explicit {field} is absent or ambiguous",
                    }
                )

    structurally_complete = len(rows) == 513
    status = (
        "INCOMPLETE_SOURCE_ARTIFACTS"
        if not structurally_complete
        else ("PARTIAL_DISCLOSURE" if needed else "COMPLETE")
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "generator_identity_provenance",
        "status": status,
        "experiment_commit": D90_COMMIT,
        "experiment_source_clean": True,
        "expected_coverage": 513,
        "observed_coverage": len(rows),
        "sources": sources,
        "rows": sorted(rows, key=lambda row: (row["tier"], row["run_name"])),
        "needed": needed,
    }
    if structurally_complete:
        counts = Counter((row["tier"], row["method"]) for row in rows)
        expected = {(tier, method): 57 for tier in TIERS for method in METHODS}
        if counts != expected:
            raise ProvenanceStageError("Generator tier/method coverage is not exact")
    payload["index_hash"] = sha256_json(payload)
    return payload


def validate_generator_identity_index(payload: Mapping[str, Any]) -> None:
    expected_hash = payload.get("index_hash")
    if expected_hash != sha256_json(_without_hash(payload, "index_hash")):
        raise ProvenanceStageError("Generator index hash mismatch")
    rows = payload.get("rows")
    needed = payload.get("needed")
    if not isinstance(rows, list) or not isinstance(needed, list):
        raise ProvenanceStageError("Generator index rows/needed must be arrays")
    status = payload.get("status")
    if status in {"COMPLETE", "PARTIAL_DISCLOSURE"}:
        if len(rows) != 513 or payload.get("observed_coverage") != 513:
            raise ProvenanceStageError("Complete generator index lacks exact coverage")
        names = [row.get("run_name") for row in rows if isinstance(row, Mapping)]
        if len(names) != len(set(names)):
            raise ProvenanceStageError("Complete generator index has duplicate rows")
        counts = Counter(
            (row.get("tier"), row.get("method"))
            for row in rows
            if isinstance(row, Mapping)
        )
        expected = {(tier, method): 57 for tier in TIERS for method in METHODS}
        if counts != expected:
            raise ProvenanceStageError("Complete generator index has inexact cells")
        has_missing = any(
            row.get("missing_fields") for row in rows if isinstance(row, Mapping)
        )
        if status == "COMPLETE" and (needed or has_missing):
            raise ProvenanceStageError("Complete generator index has missing fields")
        if status == "PARTIAL_DISCLOSURE" and (not needed or not has_missing):
            raise ProvenanceStageError(
                "Partial generator index requires explicit disclosure needs"
            )
    elif status != "INCOMPLETE_SOURCE_ARTIFACTS" or not needed:
        raise ProvenanceStageError("Incomplete generator index must state exact needs")


def build_c5_provenance_index(
    final_report_path: Path,
    *,
    batch_paths: Mapping[str, Path | None],
    tier_artifacts: Mapping[str, Mapping[str, Path | None]],
) -> dict[str, Any]:
    expected_batch_keys = {
        f"{tier}.{judge}" for tier in TIERS for judge in JUDGES
    }
    if set(batch_paths) != expected_batch_keys or set(tier_artifacts) != set(TIERS):
        raise ProvenanceStageError("C5 index requires six batches and three tiers")
    report_path = final_report_path.expanduser().resolve()
    report = read_json(report_path)
    expected_hashes = report.get("batch_hashes")
    analyses = report.get("analysis")
    if not isinstance(expected_hashes, Mapping) or not isinstance(analyses, Mapping):
        raise ProvenanceStageError("Final C5 report is malformed")

    needed: list[dict[str, str]] = []
    batches: dict[str, Any] = {}
    common: dict[str, set[str]] = {
        "rubric_hash": set(),
        "prompt_hash": set(),
    }
    tier_hashes = {
        tier: {
            "input_manifest_hash": set(),
            "source_summary_hash": set(),
            "selected_render_hashes_hash": set(),
        }
        for tier in TIERS
    }
    batch_payloads: dict[str, Mapping[str, Any]] = {}
    registry_bytes = MODEL_REGISTRY_PATH.read_bytes()
    registry = yaml.safe_load(registry_bytes)
    frozen_judges = registry.get("visual_judges")
    if not isinstance(frozen_judges, Mapping):
        raise ProvenanceStageError("Model registry has no frozen visual judges")
    total_attempts = 0
    for key in sorted(expected_batch_keys):
        path = batch_paths[key]
        if path is None or not Path(path).expanduser().is_file():
            needed.append(
                {
                    "artifact": f"batch.{key}",
                    "expected_hash": str(expected_hashes[key]),
                    "reason": (
                        "valid batch path is unavailable; observed requested/served "
                        "identity, endpoint, rubric/prompt/input/render hashes, and "
                        "attempt ledger cannot be verified"
                    ),
                }
            )
            batches[key] = {
                "path": None if path is None else str(Path(path).expanduser()),
                "expected_batch_hash": expected_hashes[key],
                "available": False,
            }
            continue
        resolved = Path(path).expanduser().resolve()
        batch = read_json(resolved)
        if (
            batch.get("batch_hash") != expected_hashes[key]
            or sha256_json(_without_hash(batch, "batch_hash")) != batch.get("batch_hash")
            or batch.get("status") != "completed"
            or batch.get("failures") != []
        ):
            raise ProvenanceStageError(f"C5 batch is mismatched or incomplete: {key}")
        role = key.split(".", 1)[1]
        expectation = frozen_judges.get(role)
        if (
            not isinstance(expectation, Mapping)
            or expectation.get("max_tokens") != 1024
            or expectation.get("validation_attempts") != 3
        ):
            raise ProvenanceStageError(f"C5 registry is not frozen for {role}")
        expected_identity = {
            "judge_id": expectation.get("judge_id"),
            "judge_request_model": expectation.get("request_model"),
            "judge_expected_served_model": expectation.get("served_model"),
            "judge_served_models": [expectation.get("served_model")],
            "judge_endpoint_class": expectation.get("endpoint_class"),
            "rubric_hash": expectation.get("rubric_hash"),
            "prompt_hash": expectation.get("prompt_hash"),
        }
        if any(batch.get(field) != value for field, value in expected_identity.items()):
            raise ProvenanceStageError(f"C5 batch identity mismatch: {key}")
        for field in common:
            value = batch.get(field)
            if not isinstance(value, str):
                raise ProvenanceStageError(f"C5 batch lacks {field}: {key}")
            common[field].add(value)
        tier = key.split(".", 1)[0]
        selected_hash = sha256_json(batch.get("selected_render_hashes"))
        for field in ("input_manifest_hash", "source_summary_hash"):
            value = batch.get(field)
            if not isinstance(value, str):
                raise ProvenanceStageError(f"C5 batch lacks {field}: {key}")
            tier_hashes[tier][field].add(value)
        tier_hashes[tier]["selected_render_hashes_hash"].add(selected_hash)
        attempts = batch.get("total_attempts")
        if not isinstance(attempts, int):
            raise ProvenanceStageError(f"C5 batch lacks attempt total: {key}")
        total_attempts += attempts
        batch_payloads[key] = batch
        sidecar_hashes = batch.get("sidecar_hashes")
        attempt_counts = batch.get("attempt_count_by_run")
        if (
            not isinstance(sidecar_hashes, Mapping)
            or len(sidecar_hashes) != 171
            or not isinstance(attempt_counts, Mapping)
            or len(attempt_counts) != 171
        ):
            raise ProvenanceStageError(f"C5 batch sidecar index is incomplete: {key}")
        batches[key] = {
            **_binding(resolved),
            "batch_hash": batch["batch_hash"],
            "judge_id": batch.get("judge_id"),
            "requested_judge_id": batch.get("judge_request_model"),
            "expected_served_judge_id": batch.get("judge_expected_served_model"),
            "observed_served_judge_ids": batch.get("judge_served_models"),
            "endpoint_class": batch.get("judge_endpoint_class"),
            "rubric_hash": batch.get("rubric_hash"),
            "prompt_hash": batch.get("prompt_hash"),
            "input_manifest_hash": batch.get("input_manifest_hash"),
            "source_summary_hash": batch.get("source_summary_hash"),
            "selected_render_hashes_hash": selected_hash,
            "selected_render_count": len(batch.get("selected_render_hashes", {})),
            "sidecar_hashes_hash": sha256_json(sidecar_hashes),
            "sidecar_count": len(sidecar_hashes),
            "attempt_count_by_run_hash": sha256_json(attempt_counts),
            "attempt_ledger_run_count": len(attempt_counts),
            "total_attempts": attempts,
        }

    artifacts: dict[str, Any] = {}
    for tier in TIERS:
        supplied = tier_artifacts[tier]
        if set(supplied) != {"merged_summary", "analysis"}:
            raise ProvenanceStageError(f"{tier} requires merged_summary and analysis")
        artifacts[tier] = {}
        for kind in ("merged_summary", "analysis"):
            path = supplied[kind]
            if path is None or not Path(path).expanduser().is_file():
                expected = (
                    analyses.get(f"{tier}_file_sha256")
                    if kind == "analysis"
                    else None
                )
                needed.append(
                    {
                        "artifact": f"{tier}.{kind}",
                        "expected_hash": (
                            str(expected) if expected is not None else "NOT_BOUND"
                        ),
                        "reason": f"final {kind} path is unavailable",
                    }
                )
                artifacts[tier][kind] = {
                    "path": None if path is None else str(Path(path).expanduser()),
                    "available": False,
                    "expected_file_sha256": expected,
                }
                continue
            resolved = Path(path).expanduser().resolve()
            binding = _binding(resolved)
            payload = read_json(resolved)
            if kind == "analysis":
                if (
                    binding["file_sha256"] != analyses.get(f"{tier}_file_sha256")
                    or payload.get("analysis_hash") != analyses.get(f"{tier}_hash")
                ):
                    raise ProvenanceStageError(f"{tier} analysis hash mismatch")
                analysis_provenance = payload.get("c5_provenance")
                expected_tier_batches = {
                    "visual-form-primary-v1": expected_hashes[f"{tier}.primary"],
                    "visual-form-secondary-v1": expected_hashes[f"{tier}.secondary"],
                }
                merged_binding = artifacts[tier].get("merged_summary")
                if (
                    not isinstance(analysis_provenance, Mapping)
                    or analysis_provenance.get("batch_hashes")
                    != expected_tier_batches
                    or not isinstance(merged_binding, Mapping)
                    or analysis_provenance.get("input_summary_hash")
                    != merged_binding.get("summary_hash")
                ):
                    raise ProvenanceStageError(
                        f"{tier} analysis provenance does not bind merged summary"
                    )
                binding["analysis_hash"] = payload["analysis_hash"]
            else:
                merged = load_provenance_summary(resolved)
                binding["summary_hash"] = merged.summary_hash
                provenance = payload.get("c5_rejudge")
                if not isinstance(provenance, Mapping):
                    raise ProvenanceStageError(f"{tier} merged C5 provenance is absent")
                judges = provenance.get("judges")
                expected_by_judge = {
                    "visual-form-primary-v1": f"{tier}.primary",
                    "visual-form-secondary-v1": f"{tier}.secondary",
                }
                if not isinstance(judges, Mapping) or set(judges) != set(
                    expected_by_judge
                ):
                    raise ProvenanceStageError(f"{tier} merged judges are not exact")
                for judge_id, batch_key in expected_by_judge.items():
                    judge = judges[judge_id]
                    batch = batch_payloads.get(batch_key)
                    if (
                        not isinstance(judge, Mapping)
                        or not isinstance(batch, Mapping)
                        or judge.get("batch_hash") != expected_hashes[batch_key]
                        or judge.get("sidecar_hashes") != batch.get("sidecar_hashes")
                        or judge.get("selected_render_hashes")
                        != batch.get("selected_render_hashes")
                        or judge.get("judge_request_model")
                        != batch.get("judge_request_model")
                        or judge.get("judge_served_models")
                        != batch.get("judge_served_models")
                    ):
                        raise ProvenanceStageError(
                            f"{tier} merged judge provenance mismatch: {judge_id}"
                        )
                binding["c5_rejudge_hash"] = sha256_json(provenance)
            artifacts[tier][kind] = binding

    if any(len(values) > 1 for values in common.values()) or any(
        len(values) > 1
        for fields in tier_hashes.values()
        for values in fields.values()
    ):
        raise ProvenanceStageError("C5 valid batches disagree on frozen provenance")
    if not needed and total_attempts != report.get("generation", {}).get("total_calls"):
        raise ProvenanceStageError("C5 attempt total disagrees with final report")
    payload = {
        "schema_version": "1.0",
        "stage": "c5_provenance",
        "status": "COMPLETE" if not needed else "INCOMPLETE_SOURCE_ARTIFACTS",
        "final_report": {
            **_binding(report_path),
            "report_commit": "49369f3305d2af8901c1a9a4939c951d46bbd2cd",
            "generation_commit": report.get("generation", {}).get("code_commit"),
            "analysis_commit": analyses.get("code_commit"),
        },
        "expected_selected_renders": report.get("generation", {}).get(
            "selected_renders"
        ),
        "expected_total_attempts": report.get("generation", {}).get("total_calls"),
        "observed_total_attempts": total_attempts,
        "model_registry": {
            "path": str(MODEL_REGISTRY_PATH),
            "file_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        },
        "frozen_judge_expectations": {
            role: {
                "judge_id": config.get("judge_id"),
                "requested_judge_id": config.get("request_model"),
                "expected_served_judge_id": config.get("served_model"),
                "endpoint_class": config.get("endpoint_class"),
                "max_tokens": config.get("max_tokens"),
                "validation_attempts": config.get("validation_attempts"),
                "rubric_hash": config.get("rubric_hash"),
                "prompt_hash": config.get("prompt_hash"),
            }
            for role, config in frozen_judges.items()
            if isinstance(config, Mapping)
        },
        "batches": batches,
        "tier_artifacts": artifacts,
        "common_hashes": {
            key: next(iter(values)) if len(values) == 1 else None
            for key, values in common.items()
        },
        "tier_input_hashes": {
            tier: {
                key: next(iter(values)) if len(values) == 1 else None
                for key, values in fields.items()
            }
            for tier, fields in tier_hashes.items()
        },
        "needed": needed,
    }
    payload["index_hash"] = sha256_json(payload)
    return payload


def validate_c5_provenance_index(payload: Mapping[str, Any]) -> None:
    if payload.get("index_hash") != sha256_json(
        _without_hash(payload, "index_hash")
    ):
        raise ProvenanceStageError("C5 index hash mismatch")
    needed = payload.get("needed")
    batches = payload.get("batches")
    if not isinstance(needed, list) or not isinstance(batches, Mapping):
        raise ProvenanceStageError("C5 index needed/batches are malformed")
    if set(batches) != {f"{tier}.{judge}" for tier in TIERS for judge in JUDGES}:
        raise ProvenanceStageError("C5 index does not name exactly six batches")
    if payload.get("status") == "COMPLETE":
        if needed:
            raise ProvenanceStageError("Complete C5 index has unresolved needs")
        if payload.get("observed_total_attempts") != payload.get(
            "expected_total_attempts"
        ):
            raise ProvenanceStageError("Complete C5 index attempt total mismatch")
        if any(value is None for value in payload.get("common_hashes", {}).values()):
            raise ProvenanceStageError("Complete C5 index has missing common hashes")
        if any(
            value is None
            for fields in payload.get("tier_input_hashes", {}).values()
            for value in fields.values()
        ):
            raise ProvenanceStageError("Complete C5 index has missing tier hashes")
        for key, batch in batches.items():
            role = key.split(".", 1)[1]
            expectation = payload.get("frozen_judge_expectations", {}).get(role, {})
            if (
                not isinstance(batch, Mapping)
                or not batch.get("path")
                or not batch.get("batch_hash")
                or not batch.get("requested_judge_id")
                or not batch.get("observed_served_judge_ids")
                or not batch.get("endpoint_class")
                or batch.get("sidecar_count") != 171
                or batch.get("selected_render_count") != 171
                or batch.get("attempt_ledger_run_count") != 171
                or batch.get("judge_id") != expectation.get("judge_id")
                or batch.get("requested_judge_id")
                != expectation.get("requested_judge_id")
                or batch.get("expected_served_judge_id")
                != expectation.get("expected_served_judge_id")
                or batch.get("observed_served_judge_ids")
                != [expectation.get("expected_served_judge_id")]
                or batch.get("endpoint_class") != expectation.get("endpoint_class")
            ):
                raise ProvenanceStageError(
                    f"Complete C5 index lacks batch identity provenance: {key}"
                )
        tier_artifacts = payload.get("tier_artifacts")
        if not isinstance(tier_artifacts, Mapping) or set(tier_artifacts) != set(TIERS):
            raise ProvenanceStageError("Complete C5 index lacks exact tier artifacts")
        for tier, artifacts in tier_artifacts.items():
            if not isinstance(artifacts, Mapping) or set(artifacts) != {
                "merged_summary",
                "analysis",
            }:
                raise ProvenanceStageError(
                    f"Complete C5 index lacks exact {tier} artifacts"
                )
            if any(
                not isinstance(binding, Mapping)
                or not binding.get("path")
                or not binding.get("file_sha256")
                for binding in artifacts.values()
            ):
                raise ProvenanceStageError(
                    f"Complete C5 index has incomplete {tier} artifact bindings"
                )
    elif payload.get("status") != "INCOMPLETE_SOURCE_ARTIFACTS" or not needed:
        raise ProvenanceStageError("Incomplete C5 index must state exact needs")


def write_provenance_index(payload: Mapping[str, Any], path: Path) -> Path:
    path = path.expanduser().resolve()
    write_json_atomic(path, payload)
    return path

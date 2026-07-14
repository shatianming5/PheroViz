from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validate

from experiments.cli import main
from experiments.models import sha256_file, sha256_json
from experiments.provenance_stage import (
    D90_COMMIT,
    ProvenanceStageError,
    build_c5_provenance_index,
    build_generator_identity_index,
    validate_c5_provenance_index,
    validate_generator_identity_index,
)


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "experiments" / "preflight"
SCHEMAS = ROOT / "experiments" / "schemas"


def _generator_reports() -> dict[str, Path]:
    return {
        "frontier": PREFLIGHT / "frontier_complete_d90d655.json",
        "mid": PREFLIGHT / "c4_mid_complete_d90d655.json",
        "open": PREFLIGHT / "open_complete_teardown_d90d655.json",
    }


def _missing_c5_paths() -> tuple[dict[str, None], dict[str, dict[str, None]]]:
    batches = {
        f"{tier}.{judge}": None
        for tier in ("frontier", "mid", "open")
        for judge in ("primary", "secondary")
    }
    artifacts = {
        tier: {"merged_summary": None, "analysis": None}
        for tier in ("frontier", "mid", "open")
    }
    return batches, artifacts


def test_generator_index_reports_unavailable_sealed_summaries() -> None:
    payload = build_generator_identity_index(_generator_reports())
    assert payload["status"] == "INCOMPLETE_SOURCE_ARTIFACTS"
    assert payload["experiment_commit"] == D90_COMMIT
    assert payload["expected_coverage"] == 513
    assert payload["observed_coverage"] == 0
    assert {item["artifact"] for item in payload["needed"]} == {
        "frontier.summary",
        "mid.summary",
        "open.summary",
    }
    validate_generator_identity_index(payload)
    schema = json.loads(
        (SCHEMAS / "generator_identity_provenance_index.schema.json").read_text()
    )
    validate(payload, schema)


def test_generator_index_fails_on_summary_hash_mismatch(
    tmp_path: Path,
) -> None:
    reports = _generator_reports()
    copied = json.loads(reports["frontier"].read_text())
    copied["summary"]["remote_path"] = str(tmp_path / "summary.json")
    (tmp_path / "summary.json").write_text("{}")
    report_path = tmp_path / "frontier.json"
    report_path.write_text(json.dumps(copied))
    reports["frontier"] = report_path
    with pytest.raises(ProvenanceStageError, match="file hash mismatch"):
        build_generator_identity_index(reports)


def test_generator_validator_rejects_forged_complete() -> None:
    payload = build_generator_identity_index(_generator_reports())
    payload["status"] = "COMPLETE"
    payload["needed"] = []
    payload["index_hash"] = sha256_json(
        {key: value for key, value in payload.items() if key != "index_hash"}
    )
    with pytest.raises(ProvenanceStageError, match="exact coverage"):
        validate_generator_identity_index(payload)


def test_c5_index_reports_every_unavailable_artifact() -> None:
    batches, artifacts = _missing_c5_paths()
    payload = build_c5_provenance_index(
        PREFLIGHT / "c5_final_analysis_1005229.json",
        batch_paths=batches,
        tier_artifacts=artifacts,
    )
    assert payload["status"] == "INCOMPLETE_SOURCE_ARTIFACTS"
    assert payload["expected_selected_renders"] == 513
    assert payload["expected_total_attempts"] == 1065
    assert payload["observed_total_attempts"] == 0
    assert len(payload["batches"]) == 6
    assert len(payload["needed"]) == 12
    validate_c5_provenance_index(payload)
    schema = json.loads((SCHEMAS / "c5_provenance_index.schema.json").read_text())
    validate(payload, schema)


def test_c5_index_fails_closed_on_wrong_batch(tmp_path: Path) -> None:
    batches, artifacts = _missing_c5_paths()
    bad = tmp_path / "rejudge_batch.json"
    bad.write_text(json.dumps({"status": "completed", "batch_hash": "0" * 64}))
    batches["frontier.primary"] = bad
    with pytest.raises(ProvenanceStageError, match="mismatched or incomplete"):
        build_c5_provenance_index(
            PREFLIGHT / "c5_final_analysis_1005229.json",
            batch_paths=batches,
            tier_artifacts=artifacts,
        )


def test_c5_validator_rejects_forged_complete() -> None:
    batches, artifacts = _missing_c5_paths()
    payload = build_c5_provenance_index(
        PREFLIGHT / "c5_final_analysis_1005229.json",
        batch_paths=batches,
        tier_artifacts=artifacts,
    )
    payload["status"] = "COMPLETE"
    payload["needed"] = []
    payload["index_hash"] = sha256_json(
        {key: value for key, value in payload.items() if key != "index_hash"}
    )
    with pytest.raises(ProvenanceStageError, match="attempt total mismatch"):
        validate_c5_provenance_index(payload)


def test_cli_writes_honest_incomplete_generator_index(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "generator",
                "completion_reports": {
                    tier: str(path) for tier, path in _generator_reports().items()
                },
            }
        )
    )
    output = tmp_path / "index.json"
    assert main(["provenance-stage", str(manifest), "--out", str(output)]) == 3
    assert json.loads(output.read_text())["status"] == "INCOMPLETE_SOURCE_ARTIFACTS"


def test_generator_validator_rejects_duplicate_complete_rows() -> None:
    rows = []
    for tier in ("frontier", "mid", "open"):
        for method in ("best_of_n", "flat_iterative", "pheroviz_full"):
            for index in range(57):
                rows.append(
                    {
                        "tier": tier,
                        "method": method,
                        "run_name": f"{tier}-{method}-{index}",
                        "missing_fields": [],
                    }
                )
    rows[-1]["run_name"] = rows[0]["run_name"]
    payload = {
        "status": "COMPLETE",
        "observed_coverage": 513,
        "rows": rows,
        "needed": [],
    }
    payload["index_hash"] = sha256_json(payload)
    with pytest.raises(ProvenanceStageError, match="duplicate"):
        validate_generator_identity_index(payload)


def test_recovered_generator_index_has_exact_partial_disclosure() -> None:
    payload = json.loads(
        (PREFLIGHT / "c1_c4_generator_identity_provenance_7ea079d.json").read_text()
    )
    assert payload["status"] == "PARTIAL_DISCLOSURE"
    assert payload["observed_coverage"] == 513
    assert len(payload["rows"]) == 513
    validate_generator_identity_index(payload)
    schema = json.loads(
        (SCHEMAS / "generator_identity_provenance_index.schema.json").read_text()
    )
    validate(payload, schema)


def test_recovered_generator_compact_sources_rebuild_exact_index() -> None:
    base = PREFLIGHT / "provenance_inputs" / "c1_c4"
    rebuilt = build_generator_identity_index(
        _generator_reports(),
        summary_paths={
            tier: base / tier / "summary.json"
            for tier in ("frontier", "mid", "open")
        },
        record_indexes={
            tier: base / tier / "record_identities.json"
            for tier in ("frontier", "mid", "open")
        },
    )
    committed = json.loads(
        (PREFLIGHT / "c1_c4_generator_identity_provenance_7ea079d.json").read_text()
    )
    assert rebuilt == committed


def test_recovered_c5_index_and_sources_are_complete() -> None:
    base = PREFLIGHT / "provenance_inputs" / "c5"
    batches = {
        f"{tier}.{judge}": base / "batches" / f"{tier}.{judge}.json"
        for tier in ("frontier", "mid", "open")
        for judge in ("primary", "secondary")
    }
    artifacts = {
        tier: {
            "merged_summary": base / "merged" / f"{tier}.json",
            "analysis": base / "analyses" / f"{tier}.json",
        }
        for tier in ("frontier", "mid", "open")
    }
    rebuilt = build_c5_provenance_index(
        PREFLIGHT / "c5_final_analysis_1005229.json",
        batch_paths=batches,
        tier_artifacts=artifacts,
    )
    assert rebuilt["status"] == "COMPLETE"
    assert rebuilt["observed_total_attempts"] == 1065
    validate_c5_provenance_index(rebuilt)
    committed = json.loads(
        (PREFLIGHT / "c5_provenance_index_7ea079d.json").read_text()
    )
    assert committed == rebuilt
    schema = json.loads((SCHEMAS / "c5_provenance_index.schema.json").read_text())
    validate(committed, schema)


def test_recovery_manifest_verifies_every_archived_file() -> None:
    path = PREFLIGHT / "provenance_inputs" / "recovery_manifest.json"
    manifest = json.loads(path.read_text())
    manifest_hash = manifest.pop("manifest_hash")
    assert manifest_hash == sha256_json(manifest)
    bindings = list(manifest["generator"]["summaries"].values())
    bindings.extend(manifest["generator"]["record_identity_archives"].values())
    for group in ("batches", "merged", "analyses"):
        bindings.extend(manifest["c5"][group].values())
    for binding in bindings:
        archive_path = ROOT.parent / (binding.get("archive_path") or binding["path"])
        assert archive_path.is_file()
        assert sha256_file(archive_path) == binding["file_sha256"]
    recovery_schema = json.loads(
        (SCHEMAS / "generator_record_identity_recovery.schema.json").read_text()
    )
    for binding in manifest["generator"]["record_identity_archives"].values():
        validate(json.loads((ROOT.parent / binding["path"]).read_text()), recovery_schema)
    assert manifest["transport"]["services_started"] is False
    assert manifest["transport"]["tunnels_started"] is False
    assert manifest["transport"]["gpu_tasks_started"] is False

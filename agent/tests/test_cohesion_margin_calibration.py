from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from experiments.cohesion_margin_calibration import (
    DEFAULT_FIXTURE_BANK,
    EXPECTED_BANK_SPEC_HASH,
    EXPECTED_EVALUATOR_CONFIG_SHA256,
    EXPECTED_EXPECTATION_SHA256,
    EXPECTED_FIXTURE_BANK_FILE_SHA256,
    EXPECTED_FIXTURE_SPEC_HASHES,
    EXPECTED_RESULTS,
    EXPECTED_SOURCE_SHA256,
    REPORT_FILENAME,
    compare_reports,
    main,
    run_calibration,
)
from experiments.models import sha256_file, sha256_json, write_json_atomic


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "preflight"
    )
    path = parent / f"cohesion_margin_test_{label}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _payload(report: dict) -> dict:
    return report["semantic_payload"]


def test_frozen_fixture_bank_hashes_are_exact() -> None:
    bank = json.loads(DEFAULT_FIXTURE_BANK.read_text(encoding="utf-8"))

    assert sha256_file(DEFAULT_FIXTURE_BANK) == EXPECTED_FIXTURE_BANK_FILE_SHA256
    assert bank["bank_spec_hash"] == EXPECTED_BANK_SPEC_HASH
    assert bank["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert bank["expectation_sha256"] == EXPECTED_EXPECTATION_SHA256
    assert bank["evaluator_config_sha256"] == EXPECTED_EVALUATOR_CONFIG_SHA256
    assert {
        item["fixture_id"]: item["spec_sha256"] for item in bank["fixtures"]
    } == EXPECTED_FIXTURE_SPEC_HASHES
    assert len(bank["run_order"]) == 15


def test_calibration_has_exact_15_results_and_freezes_delta() -> None:
    with _workspace("exact") as workspace:
        report = run_calibration(
            workspace / "calibration",
            generated_at="2026-07-13T00:00:00+00:00",
        )
        payload = _payload(report)

        assert payload["status"] == "passed"
        assert payload["run_count"] == 15
        assert payload["candidate_delta_C"] == 0.25
        assert payload["final_delta_C"] == 0.25
        assert payload["delta_C_status"] == "FROZEN"
        assert all(payload["acceptance_checks"].values())
        assert payload["leakage_declaration"] == {
            "uses_human_ratings": False,
            "uses_method_outputs": False,
            "uses_model_outputs": False,
            "uses_network": False,
            "uses_randomness": False,
            "uses_test_inputs": False,
            "uses_test_metrics": False,
        }

        fixture_counts: dict[str, int] = {}
        for result in payload["results"]:
            fixture_id = result["fixture_id"]
            fixture_counts[fixture_id] = fixture_counts.get(fixture_id, 0) + 1
            expected = EXPECTED_RESULTS[fixture_id]
            for key, value in expected.items():
                assert result[key] == value
            assert set(result["artifact_hashes"]) == {
                "render.png",
                "figure_manifest.json",
                "cohesion_result.json",
                "evaluation.json",
                "fixture_result.json",
            }
            assert all(
                len(value) == 64
                for value in result["artifact_hashes"].values()
            )
        assert fixture_counts == {
            fixture_id: 3 for fixture_id in EXPECTED_RESULTS
        }
        assert payload["class_drops"] == {
            "shared_scale": [0.25, 0.25, 0.25],
            "shared_unit": [0.25, 0.25, 0.25],
            "legend_deduplication": [0.25, 0.25, 0.25],
            "palette_mapping": [0.25, 0.25, 0.25],
        }

        report_path = workspace / "calibration" / REPORT_FILENAME
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        unhashed = dict(persisted)
        report_hash = unhashed.pop("report_hash")
        assert report_hash == sha256_json(unhashed)
        assert persisted["semantic_hash"] == sha256_json(payload)


def test_semantic_payload_is_repeatable_with_generated_at_excluded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("repeat") as workspace:
        first = run_calibration(
            workspace / "first",
            generated_at="2026-07-13T00:00:00+00:00",
        )
        second = run_calibration(
            workspace / "second",
            generated_at="2026-07-13T00:01:00+00:00",
        )

        assert first["report_hash"] != second["report_hash"]
        assert first["semantic_hash"] == second["semantic_hash"]
        assert first["semantic_payload"] == second["semantic_payload"]
        comparison = compare_reports(
            workspace / "first" / REPORT_FILENAME,
            workspace / "second" / REPORT_FILENAME,
        )
        assert comparison["semantic_payload_stable"] is True
        assert comparison["generated_at_excluded"] is True

        exit_code = main(
            [
                "compare",
                str(workspace / "first" / REPORT_FILENAME),
                str(workspace / "second" / REPORT_FILENAME),
            ]
        )
        assert exit_code == 0
        cli = json.loads(capsys.readouterr().out)
        assert cli["semantic_payload_stable"] is True


def test_tampered_fixture_bank_fails_closed_and_leaves_delta_unset() -> None:
    with _workspace("tampered") as workspace:
        bank = json.loads(DEFAULT_FIXTURE_BANK.read_text(encoding="utf-8"))
        bank["fixtures"][-1]["expected"]["ratio"] = 0.5
        tampered = workspace / "tampered.json"
        write_json_atomic(tampered, bank)

        report = run_calibration(
            workspace / "blocked",
            fixture_bank_path=tampered,
            generated_at="2026-07-13T00:00:00+00:00",
        )
        payload = _payload(report)

        assert payload["status"] == "blocked"
        assert payload["run_count"] == 0
        assert payload["candidate_delta_C"] is None
        assert payload["final_delta_C"] is None
        assert payload["delta_C_status"] == "UNSET"
        assert payload["error"]["type"] == "CalibrationError"
        assert "Expected-count table changed" in payload["error"]["message"]

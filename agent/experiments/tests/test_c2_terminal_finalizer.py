from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

import experiments.c2_terminal_finalizer as c2_terminal_finalizer
from experiments.c2_terminal_finalizer import (
    CHUNK_IDS,
    REPLACEMENT_CHUNK_IDS,
    TERMINAL_OUTCOME_STATUSES,
    C2AdmissionError,
    finalize_manifest,
    validate_final_report,
)
from experiments.cli import main as cli_main
from experiments.models import sha256_file, sha256_json


SYNTHETIC_CODE_COMMIT = "a" * 40
SYNTHETIC_UNIVERSE_SHA256 = "f" * 64
EXPECTED_CHUNK_IDS = (
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
)
EXPECTED_REPLACEMENT_CHUNK_IDS = ("009", "010", "011", "012")
EXPECTED_TERMINAL_OUTCOME_STATUSES = (
    "DOWNLOADED",
    "NO_SOURCE_DATA",
    "NO_FIGURES",
    "NO_USABLE_CONTENT",
    "POLICY_REJECTED",
    "DOWNLOAD_FAILED",
    "RETRY_EXHAUSTED",
)
STRATA_BY_CHUNK = {
    "001": "P=1",
    "002": "P=1",
    "003": "P=2",
    "004": "P=2",
    "005": "P=3-4",
    "006": "P=3-4",
    "007": "P=5+",
    "008": "P=5+",
    "009": "P=1",
    "010": "P=2",
    "011": "P=3-4",
    "012": "P=5+",
    "013": "P=1",
}


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".c2_terminal_finalizer_test_work"
    path = parent / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seal_report(report: dict[str, Any]) -> None:
    report["report_hash"] = sha256_json(
        {
            key: value
            for key, value in report.items()
            if key not in {"report_hash", "seal"}
        }
    )
    seal = {
        "status": "TERMINAL",
        "sealed_report_hash": report["report_hash"],
    }
    seal["seal_hash"] = sha256_json(seal)
    report["seal"] = seal


def _write_report(path: Path, report: dict[str, Any]) -> None:
    _seal_report(report)
    _write_json(path, report)


def _seal_manifest(manifest: dict[str, Any]) -> None:
    manifest["manifest_hash"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _seal_manifest(manifest)
    _write_json(path, manifest)


def _entry(manifest: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    return next(item for item in manifest["chunks"] if item["chunk_id"] == chunk_id)


def _refresh_manifest_bindings(
    manifest: dict[str, Any],
    manifest_path: Path,
    report_paths: dict[str, Path],
) -> None:
    for chunk_id, report_path in report_paths.items():
        report = _read_json(report_path)
        entry = _entry(manifest, chunk_id)
        entry["expected_report_file_sha256"] = sha256_file(report_path)
        entry["expected_report_hash"] = report["report_hash"]
        entry["input_total"] = report["chunk"]["input_total"]
        entry["input_doi_ids_sha256"] = report["chunk"]["input_doi_ids_sha256"]
    _write_manifest(manifest_path, manifest)


def _rebind_frozen_universe(
    manifest: dict[str, Any],
    manifest_path: Path,
    report_paths: dict[str, Path],
) -> None:
    dois: list[str] = []
    for chunk_id in EXPECTED_CHUNK_IDS:
        dois.extend(_read_json(report_paths[chunk_id])["chunk"]["input_doi_ids"])
    manifest["frozen_universe"]["input_total"] = len(dois)
    manifest["frozen_universe"]["doi_ids_sha256"] = sha256_json(dois)
    for report_path in report_paths.values():
        report = _read_json(report_path)
        report["frozen_universe"] = dict(manifest["frozen_universe"])
        _write_report(report_path, report)
    _refresh_manifest_bindings(manifest, manifest_path, report_paths)


def _build_admission(
    workspace: Path,
    *,
    p5plus_single_cluster: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    source_dois = [
        f"10.9000/synthetic-{chunk_id}" for chunk_id in EXPECTED_CHUNK_IDS
    ]
    frozen_universe = {
        "sha256": SYNTHETIC_UNIVERSE_SHA256,
        "input_total": len(source_dois),
        "doi_ids_sha256": sha256_json(source_dois),
    }
    code = {"commit": SYNTHETIC_CODE_COMMIT, "dirty": False}
    report_paths: dict[str, Path] = {}
    chunks: list[dict[str, Any]] = []
    excluded_roots: list[dict[str, str]] = []
    for chunk_id, doi in zip(EXPECTED_CHUNK_IDS, source_dois, strict=True):
        if chunk_id in EXPECTED_REPLACEMENT_CHUNK_IDS:
            root = {
                "root_id": f"replacement-root-{chunk_id}",
                "root_kind": "replacement",
                "supersedes_root_id": f"superseded-root-{chunk_id}",
                "partial_root": False,
            }
            excluded_roots.append(
                {
                    "chunk_id": chunk_id,
                    "root_id": root["supersedes_root_id"],
                }
            )
        else:
            root = {
                "root_id": f"canonical-root-{chunk_id}",
                "root_kind": "canonical",
                "supersedes_root_id": None,
                "partial_root": False,
            }
        stratum = STRATA_BY_CHUNK[chunk_id]
        cluster_id = (
            "cluster-p5plus-only"
            if p5plus_single_cluster and stratum == "P=5+"
            else f"cluster-{chunk_id}"
        )
        report = {
            "schema_version": "1.0",
            "report_type": "c2_terminal_chunk_report",
            "chunk": {
                "chunk_id": chunk_id,
                "input_total": 1,
                "input_doi_ids": [doi],
                "input_doi_ids_sha256": sha256_json([doi]),
            },
            "code": dict(code),
            "frozen_universe": dict(frozen_universe),
            "root": root,
            "execution": {
                "attempts_per_input": 3,
                "all_inputs_terminal": True,
                "outcomes": [
                    {
                        "doi_id": doi,
                        "attempt_count": 3,
                        "terminal": True,
                        "terminal_status": "NO_SOURCE_DATA",
                    }
                ],
            },
            "evidence": {
                "selection": {
                    "scope": "ALL_TERMINAL_INPUTS",
                    "partial": False,
                    "selective": False,
                    "model_result_selected": False,
                },
                "source_doi_ids": [doi],
                "source_doi_ids_sha256": sha256_json([doi]),
                "source_doi_strata": [
                    {
                        "doi_id": doi,
                        "stratum": stratum,
                        "independent_doi_cluster_id": cluster_id,
                    }
                ],
            },
        }
        report_path = workspace / "sealed" / f"chunk-{chunk_id}.json"
        _write_report(report_path, report)
        report_paths[chunk_id] = report_path
        chunks.append(
            {
                "chunk_id": chunk_id,
                "root": root,
                "report_path": report_path.relative_to(workspace).as_posix(),
                "expected_report_file_sha256": sha256_file(report_path),
                "expected_report_hash": report["report_hash"],
                "input_total": 1,
                "input_doi_ids_sha256": report["chunk"]["input_doi_ids_sha256"],
            }
        )

    manifest = {
        "schema_version": "1.0",
        "frozen_universe": frozen_universe,
        "code": code,
        "replacement_policy": {
            "replacement_chunk_ids": list(EXPECTED_REPLACEMENT_CHUNK_IDS),
            "excluded_superseded_roots": excluded_roots,
        },
        "chunks": chunks,
    }
    manifest_path = workspace / "admission-manifest.json"
    _write_manifest(manifest_path, manifest)
    return manifest_path, manifest, report_paths


def _save_and_refresh(
    report_path: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    report_paths: dict[str, Path],
) -> None:
    _write_report(report_path, report)
    _refresh_manifest_bindings(manifest, manifest_path, report_paths)


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value)) if value else set()
    return set()


def test_exported_rosters_match_independent_literal_oracles() -> None:
    assert CHUNK_IDS == EXPECTED_CHUNK_IDS
    assert REPLACEMENT_CHUNK_IDS == EXPECTED_REPLACEMENT_CHUNK_IDS
    assert set(STRATA_BY_CHUNK) == set(EXPECTED_CHUNK_IDS)


def test_literal_roster_oracle_detects_an_altered_production_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("altered-roster") as workspace:
        manifest_path, _, _ = _build_admission(workspace)
        monkeypatch.setattr(
            c2_terminal_finalizer,
            "CHUNK_IDS",
            EXPECTED_CHUNK_IDS[:-1] + ("014",),
        )

        with pytest.raises(C2AdmissionError, match="complete ordered roster"):
            c2_terminal_finalizer.finalize_manifest(manifest_path)


def test_terminal_status_allowlist_is_closed_in_schema_and_validator() -> None:
    schema_path = (
        Path(c2_terminal_finalizer.__file__).resolve().parent
        / "schemas"
        / "c2_terminal_chunk_report.schema.json"
    )
    schema = _read_json(schema_path)
    status_schema = schema["$defs"]["terminalOutcome"]["properties"][
        "terminal_status"
    ]
    assert tuple(status_schema["enum"]) == EXPECTED_TERMINAL_OUTCOME_STATUSES
    assert TERMINAL_OUTCOME_STATUSES == frozenset(
        EXPECTED_TERMINAL_OUTCOME_STATUSES
    )

    input_dois = ("10.9000/synthetic-001",)
    for status in ("QUEUED", "UNKNOWN_STATUS", "PENDING"):
        execution = {
            "attempts_per_input": 3,
            "all_inputs_terminal": True,
            "outcomes": [
                {
                    "doi_id": input_dois[0],
                    "attempt_count": 3,
                    "terminal": True,
                    "terminal_status": status,
                }
            ],
        }
        with pytest.raises(C2AdmissionError, match="unapproved terminal status"):
            c2_terminal_finalizer._validate_terminal_outcomes(
                execution,
                input_dois,
                "001",
            )


@pytest.mark.parametrize("status", ("QUEUED", "UNKNOWN_STATUS", "PENDING"))
def test_schema_rejects_unknown_queued_and_nonterminal_statuses(status: str) -> None:
    with _workspace(f"terminal-status-{status}") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["execution"]["outcomes"][0]["terminal_status"] = status
        _save_and_refresh(
            report_path,
            report,
            manifest,
            manifest_path,
            report_paths,
        )

        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_final_report_hashes_the_exact_manifest_and_report_bytes_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("byte-substitution") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        manifest_bytes = manifest_path.read_bytes()
        report_bytes = report_path.read_bytes()
        expected_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        expected_report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        expected_report_hash = _read_json(report_path)["report_hash"]
        manifest_resolved = manifest_path.resolve()
        report_resolved = report_path.resolve()
        original_reader = c2_terminal_finalizer._read_json_object

        def _read_then_substitute(
            path: Path,
            label: str,
        ) -> tuple[dict[str, Any], str]:
            value, payload_sha256 = original_reader(path, label)
            if path == manifest_resolved:
                path.write_bytes(b'{"substituted_manifest":true}\n')
            elif path == report_resolved:
                path.write_bytes(b'{"substituted_report":true}\n')
            return value, payload_sha256

        monkeypatch.setattr(
            c2_terminal_finalizer,
            "_read_json_object",
            _read_then_substitute,
        )
        final_report = c2_terminal_finalizer.finalize_manifest(manifest_path)

        report_binding = next(
            item for item in final_report["chunks"] if item["chunk_id"] == "013"
        )
        assert final_report["admission_manifest"]["file_sha256"] == (
            expected_manifest_sha256
        )
        assert report_binding["report_file_sha256"] == expected_report_sha256
        assert report_binding["report_hash"] == expected_report_hash
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() != (
            expected_manifest_sha256
        )
        assert hashlib.sha256(report_path.read_bytes()).hexdigest() != (
            expected_report_sha256
        )


def test_admits_complete_sealed_roster_and_emits_terminal_only_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("success") as workspace:
        manifest_path, manifest, _ = _build_admission(workspace)
        report = finalize_manifest(manifest_path)

        assert report["status"] == "ADMITTED"
        assert report["claim_status"] == "SUPPORTED"
        assert report["trend_status"] == "NOT_RUN"
        assert report["equivalence_status"] == "NOT_RUN"
        assert report["source_doi_ids_sha256"] == manifest["frozen_universe"]["doi_ids_sha256"]
        assert all(
            item["independent_doi_cluster_count"] >= 2
            for item in report["strata"]
        )
        assert "metric" not in {key.casefold() for key in _all_keys(report)}
        assert "analysis" not in {key.casefold() for key in _all_keys(report)}
        validate_final_report(report)

        output_path = workspace / "final-report.json"
        assert (
            cli_main(
                [
                    "c2-terminal-finalize",
                    str(manifest_path),
                    "--out",
                    str(output_path),
                ]
            )
            == 0
        )
        assert _read_json(output_path) == report
        assert '"status": "ADMITTED"' in capsys.readouterr().out


def test_p5plus_deficiency_is_explicitly_blocked_without_analysis() -> None:
    with _workspace("p5plus-blocked") as workspace:
        manifest_path, _, _ = _build_admission(
            workspace,
            p5plus_single_cluster=True,
        )
        report = finalize_manifest(manifest_path)

        assert report["status"] == "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
        assert report["deficient_strata"] == ["P=5+"]
        assert report["claim_status"] == "UNSUPPORTED"
        assert report["trend_status"] == "NOT_RUN"
        assert report["equivalence_status"] == "NOT_RUN"
        assert all(
            "metric" not in key.casefold() and "analysis" not in key.casefold()
            for key in _all_keys(report)
        )

        output_path = workspace / "blocked-final-report.json"
        assert (
            cli_main(
                [
                    "c2-terminal-finalize",
                    str(manifest_path),
                    "--out",
                    str(output_path),
                ]
            )
            == 3
        )
        assert _read_json(output_path)["status"] == report["status"]


def test_other_stratum_deficiency_uses_its_own_clear_blocked_status() -> None:
    with _workspace("p1-blocked") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        for chunk_id in ("001", "002", "009", "013"):
            report_path = report_paths[chunk_id]
            report = _read_json(report_path)
            report["evidence"]["source_doi_strata"][0][
                "independent_doi_cluster_id"
            ] = "cluster-p1-only"
            _write_report(report_path, report)
        _refresh_manifest_bindings(manifest, manifest_path, report_paths)

        report = finalize_manifest(manifest_path)
        assert report["status"] == "BLOCKED_INSUFFICIENT_INDEPENDENT_P1"
        assert report["deficient_strata"] == ["P=1"]
        assert report["claim_status"] == "UNSUPPORTED"
        assert report["trend_status"] == "NOT_RUN"
        assert report["equivalence_status"] == "NOT_RUN"


def test_rejects_forged_chunk_report_hash() -> None:
    with _workspace("forged") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["chunk"]["input_total"] = 2
        _write_json(report_path, report)
        _entry(manifest, "013")["expected_report_file_sha256"] = sha256_file(report_path)
        _write_manifest(manifest_path, manifest)

        with pytest.raises(C2AdmissionError, match="report_hash"):
            finalize_manifest(manifest_path)


def test_rejects_mixed_code_commits() -> None:
    with _workspace("mixed-commits") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["code"]["commit"] = "b" * 40
        _save_and_refresh(
            report_path,
            report,
            manifest,
            manifest_path,
            report_paths,
        )

        with pytest.raises(C2AdmissionError, match="code provenance"):
            finalize_manifest(manifest_path)


def test_rejects_dirty_chunk_code() -> None:
    with _workspace("dirty-code") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["code"]["dirty"] = True
        _save_and_refresh(
            report_path,
            report,
            manifest,
            manifest_path,
            report_paths,
        )

        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_rejects_duplicate_doi_inputs_even_when_all_hashes_are_rebound() -> None:
    with _workspace("duplicate-doi") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        duplicate_doi = _read_json(report_paths["001"])["chunk"]["input_doi_ids"][0]
        report["chunk"]["input_doi_ids"] = [duplicate_doi]
        report["chunk"]["input_doi_ids_sha256"] = sha256_json([duplicate_doi])
        report["execution"]["outcomes"][0]["doi_id"] = duplicate_doi
        report["evidence"]["source_doi_ids"] = [duplicate_doi]
        report["evidence"]["source_doi_ids_sha256"] = sha256_json([duplicate_doi])
        report["evidence"]["source_doi_strata"][0]["doi_id"] = duplicate_doi
        _write_report(report_path, report)
        _rebind_frozen_universe(manifest, manifest_path, report_paths)

        with pytest.raises(C2AdmissionError, match="duplicate DOI"):
            finalize_manifest(manifest_path)


def test_rejects_missing_sealed_report_path() -> None:
    with _workspace("missing-report") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        report_paths["013"].unlink()

        with pytest.raises(C2AdmissionError, match="report path is missing"):
            finalize_manifest(manifest_path)


def test_rejects_missing_chunk_roster_entry() -> None:
    with _workspace("missing-chunk") as workspace:
        manifest_path, manifest, _ = _build_admission(workspace)
        manifest["chunks"].pop()
        _write_manifest(manifest_path, manifest)

        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_rejects_missing_doi_input() -> None:
    with _workspace("missing-doi") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["chunk"]["input_doi_ids"] = []
        report["chunk"]["input_doi_ids_sha256"] = sha256_json([])
        _save_and_refresh(
            report_path,
            report,
            manifest,
            manifest_path,
            report_paths,
        )

        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_rejects_stale_sealed_report_file() -> None:
    with _workspace("stale-report") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report_path.write_text(
            report_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        with pytest.raises(C2AdmissionError, match="stale or mismatched"):
            finalize_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("partial", True),
        ("selective", True),
        ("model_result_selected", True),
    ],
)
def test_rejects_partial_selective_or_model_selected_evidence(
    field: str,
    value: bool,
) -> None:
    with _workspace(f"forbidden-selection-{field}") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["evidence"]["selection"][field] = value
        _save_and_refresh(
            report_path,
            report,
            manifest,
            manifest_path,
            report_paths,
        )

        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_rejects_non_three_attempts_and_nonterminal_execution() -> None:
    with _workspace("attempts-and-terminal") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["execution"]["attempts_per_input"] = 2
        report["execution"]["outcomes"][0]["attempt_count"] = 2
        _save_and_refresh(
            report_path,
            report,
            manifest,
            manifest_path,
            report_paths,
        )
        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)

        manifest_path, manifest, report_paths = _build_admission(workspace / "second")
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["execution"]["all_inputs_terminal"] = False
        _save_and_refresh(
            report_path,
            report,
            manifest,
            manifest_path,
            report_paths,
        )
        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_rejects_missing_terminal_seal() -> None:
    with _workspace("missing-seal") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report.pop("seal")
        report["report_hash"] = sha256_json(
            {
                key: value
                for key, value in report.items()
                if key not in {"report_hash", "seal"}
            }
        )
        _write_json(report_path, report)
        _entry(manifest, "013")["expected_report_file_sha256"] = sha256_file(report_path)
        _entry(manifest, "013")["expected_report_hash"] = report["report_hash"]
        _write_manifest(manifest_path, manifest)

        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_rejects_partial_root() -> None:
    with _workspace("partial-root") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report = _read_json(report_path)
        report["root"]["partial_root"] = True
        _write_report(report_path, report)
        _entry(manifest, "013")["root"]["partial_root"] = True
        _refresh_manifest_bindings(manifest, manifest_path, report_paths)

        with pytest.raises(C2AdmissionError, match="schema validation failed"):
            finalize_manifest(manifest_path)


def test_rejects_superseded_replacement_root() -> None:
    with _workspace("superseded-root") as workspace:
        manifest_path, manifest, report_paths = _build_admission(workspace)
        report_path = report_paths["009"]
        report = _read_json(report_path)
        superseded_root = report["root"]["supersedes_root_id"]
        report["root"]["root_id"] = superseded_root
        _write_report(report_path, report)
        _entry(manifest, "009")["root"]["root_id"] = superseded_root
        _refresh_manifest_bindings(manifest, manifest_path, report_paths)

        with pytest.raises(C2AdmissionError, match="superseded root"):
            finalize_manifest(manifest_path)

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

import experiments.c2_full_replacement_evidence as v2_evidence
import experiments.c2_full_replacement_finalizer as v2_finalizer
import experiments.c2_full_replacement_policy as v2_policy
from experiments.c2_full_replacement_evidence import thaw_evidence_value
from experiments.c2_full_replacement_finalizer import (
    C2FullReplacementError,
    prepare_full_replacement_finalization,
    validate_synthetic_final_report_for_testing,
    write_full_replacement_report,
)
from experiments.c2_full_replacement_policy import (
    ATTEMPT_IDS,
    CHUNK_IDS,
    DOI_CASE_AGGREGATION_RULE_VERSION,
    FINAL_CHUNK_INPUT_TOTAL,
    FROZEN_INPUT_TOTAL,
    FULL_REPLACEMENT_INPUT_TOTALS,
    P_DISPOSITIONS,
    C2FullReplacementPolicyError,
    StratifiedSourceClassification,
    aggregate_stratified_source_classifications,
    compile_synthetic_policy_for_testing,
    load_production_policy,
)
from experiments.cli import _build_parser, main as cli_main
from experiments.models import sha256_json


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
EXPECTED_CHUNK_INPUT_TOTALS = (
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    200,
    63,
)
EXPECTED_ATTEMPTS = ("initial", "retry1", "retry2")
EXPECTED_P_DISPOSITIONS = ("P1", "P2", "P3_4", "P5PLUS")


@pytest.fixture(autouse=True)
def _exercise_guarded_full_replacement_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_m1_guard_for_test(monkeypatch)


def _allow_m1_guard_for_test(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (v2_evidence, v2_finalizer, v2_policy):
        monkeypatch.setattr(
            module,
            "require_external_m1_trust_lock",
            lambda: None,
        )


@dataclass
class SyntheticFixture:
    workspace: Path
    evidence_root: Path
    output_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    policy_data: dict[str, Any]
    policy: Any

    def seal_manifest(self) -> None:
        self.manifest["manifest_hash"] = sha256_json(
            {
                key: value
                for key, value in self.manifest.items()
                if key != "manifest_hash"
            }
        )
        _write_json(self.manifest_path, self.manifest)

    def rebind(self, binding: dict[str, Any]) -> None:
        binding["sha256"] = _sha256_file(self.evidence_root / binding["path"])
        self.seal_manifest()

    def refresh_builder_binding(self, doi_id: str) -> None:
        source = next(
            item for item in self.manifest["source_canonical"] if item["doi_id"] == doi_id
        )
        source["canonical_builder"]["sha256"] = _sha256_file(
            self.evidence_root / source["canonical_builder"]["path"]
        )
        disposition = next(
            item
            for item in self.manifest["acquisition_dispositions"]
            if item["doi_id"] == doi_id
        )
        disposition["canonical_builder_binding_or_null"] = dict(
            source["canonical_builder"]
        )
        self.seal_manifest()

    def refresh_source_inventory_binding(self, doi_id: str) -> None:
        source = next(
            item for item in self.manifest["source_canonical"] if item["doi_id"] == doi_id
        )
        source["source_inventory"]["sha256"] = _sha256_file(
            self.evidence_root / source["source_inventory"]["path"]
        )
        disposition = next(
            item
            for item in self.manifest["acquisition_dispositions"]
            if item["doi_id"] == doi_id
        )
        disposition["source_inventory_binding_or_null"] = dict(
            source["source_inventory"]
        )
        builder_path = self.evidence_root / source["canonical_builder"]["path"]
        builder = _read_json(builder_path)
        builder["source_inventory_sha256"] = source["source_inventory"]["sha256"]
        _write_json(builder_path, builder)
        self.refresh_builder_binding(doi_id)


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".c2_full_replacement_v2_test_work"
    path = parent / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, evidence_root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(evidence_root).as_posix(),
        "sha256": _sha256_file(path),
    }


def _source_hashes(value: str) -> dict[str, str]:
    digest = value * 64
    return {
        "source_inventory_sha256": digest,
        "raw_source_evidence_sha256": digest,
        "candidate_manifest_sha256": digest,
        "proposal_manifest_sha256": digest,
        "review_manifest_sha256": digest,
        "canonical_manifest_sha256": digest,
        "canonical_builder_sha256": digest,
    }


def _policy_data() -> dict[str, Any]:
    doi_ids = [
        f"10.9000/v21-synthetic-{ordinal:04d}"
        for ordinal in range(1, FROZEN_INPUT_TOTAL + 1)
    ]
    partition: list[dict[str, Any]] = []
    start = 1
    chunk_hashes: dict[str, str] = {}
    for chunk_id, total in zip(
        EXPECTED_CHUNK_IDS,
        EXPECTED_CHUNK_INPUT_TOTALS,
        strict=True,
    ):
        chunk_dois = doi_ids[start - 1 : start - 1 + total]
        partition.append(
            {
                "chunk_id": chunk_id,
                "first_global_ordinal": start,
                "input_total": total,
                "doi_ids_sha256": sha256_json(chunk_dois),
            }
        )
        chunk_hashes[chunk_id] = hashlib.sha256(
            f"synthetic-universe-chunk-{chunk_id}".encode("utf-8")
        ).hexdigest()
        start += total
    doi_hash = sha256_json(doi_ids)
    return {
        "policy_id": "synthetic-c2-v21-source-classification",
        "policy_version": "stagea-source-test",
        "frozen_universe": {
            "sha256": "a" * 64,
            "doi_ids_sha256": doi_hash,
            "input_total": FROZEN_INPUT_TOTAL,
        },
        "frozen_bindings": {
            "universe_file_sha256": "b" * 64,
            "universe_ordered_doi_ids_sha256": doi_hash,
            "universe_input_total": FROZEN_INPUT_TOTAL,
            "universe_chunk_count": 13,
            "universe_chunk_file_sha256_by_id": chunk_hashes,
            "universe_chunk_concat_sha256": "c" * 64,
            "normalizer_code_commit_full": "d" * 40,
            "normalizer_code_sha256": "e" * 64,
            "canonical_builder_code_commit_full": "f" * 40,
            "canonical_builder_code_sha256": "1" * 64,
            "canonical_builder_rule_version": "CANONICAL_BUILDER_V1",
            "canonical_builder_rule_sha256": "2" * 64,
            "panel_recomputation_rule_version": "PANEL_RECOMPUTATION_V1",
            "panel_recomputation_rule_sha256": "3" * 64,
            "doi_case_aggregation_rule_version": DOI_CASE_AGGREGATION_RULE_VERSION,
            "doi_case_aggregation_rule_sha256": "4" * 64,
            "v2_schema_file_sha256": "5" * 64,
        },
        "partition": partition,
        "replacement_plan": [
            {
                "chunk_id": chunk_id,
                "replacement_root_id": f"synthetic-replacement-{chunk_id}",
                "retired_root_ids": [f"synthetic-retired-{chunk_id}"],
            }
            for chunk_id in EXPECTED_CHUNK_IDS
        ],
        "ordered_doi_ids": doi_ids,
    }


def _canonical_case_binding(case_base: dict[str, Any]) -> str:
    return sha256_json(case_base)


def _write_parent_binding(
    evidence_root: Path,
    ordinal: int,
    kind: str,
    doi_id: str,
) -> dict[str, str]:
    path = evidence_root / "parent-bindings" / f"{ordinal:04d}" / f"{kind}.json"
    _write_json(
        path,
        {
            "artifact_type": f"c2_v21_{kind}_manifest",
            "parent_doi_id": doi_id,
            "complete": True,
            "model_result_selected": False,
        },
    )
    return _artifact(path, evidence_root)


def _write_source_canonical(
    *,
    evidence_root: Path,
    policy: Any,
    ordinal: int,
    panel_counts: list[int] | None,
) -> dict[str, Any]:
    doi_id = policy.ordered_doi_ids[ordinal - 1]
    if panel_counts is None:
        candidate_ids: list[str] = []
    elif not panel_counts:
        candidate_ids = [f"source-{ordinal:04d}-inventory-only"]
    else:
        candidate_ids = sorted(
            f"source-{ordinal:04d}-{case_index:02d}-{panel_index:02d}"
            for case_index, panel_count in enumerate(panel_counts, start=1)
            for panel_index in range(1, panel_count + 1)
        )
    raw_sources: dict[str, dict[str, str]] = {}
    for candidate_id in candidate_ids:
        raw_source_path = (
            evidence_root / "raw-sources" / f"{ordinal:04d}" / f"{candidate_id}.json"
        )
        _write_json(
            raw_source_path,
            {
                "artifact_type": "c2_v21_synthetic_raw_source",
                "parent_doi_id": doi_id,
                "source_candidate_id": candidate_id,
            },
        )
        raw_sources[candidate_id] = _artifact(raw_source_path, evidence_root)
    raw_source_evidence_path = (
        evidence_root / "raw-source-evidence" / f"{ordinal:04d}.json"
    )
    _write_json(
        raw_source_evidence_path,
        {
            "artifact_type": "c2_v21_raw_source_evidence",
            "parent_doi_id": doi_id,
            "complete": True,
            "model_result_selected": False,
            "source_candidates": [
                {
                    "source_candidate_id": candidate_id,
                    "parent_doi_id": doi_id,
                    "raw_source_path": raw_sources[candidate_id]["path"],
                    "raw_source_sha256": raw_sources[candidate_id]["sha256"],
                }
                for candidate_id in candidate_ids
            ],
        },
    )
    raw_source_evidence = _artifact(raw_source_evidence_path, evidence_root)
    inventory_path = evidence_root / "source-inventory" / f"{ordinal:04d}.json"
    _write_json(
        inventory_path,
        {
            "artifact_type": "c2_v21_source_inventory",
            "parent_doi_id": doi_id,
            "complete": True,
            "model_result_selected": False,
            "raw_source_evidence": raw_source_evidence,
            "source_candidate_ids": candidate_ids,
        },
    )
    inventory = _artifact(inventory_path, evidence_root)
    parent_bindings = {
        kind: _write_parent_binding(evidence_root, ordinal, kind, doi_id)
        for kind in ("candidate", "proposal", "review", "canonical")
    }
    source_cases = [
        {
            "source_candidate_id": candidate_id,
            "parent_doi_id": doi_id,
            "raw_source_sha256": raw_sources[candidate_id]["sha256"],
            "verified": True,
            "eligible_single_source": True,
        }
        for candidate_id in candidate_ids
    ]
    cases: list[dict[str, Any]] = []
    if panel_counts is not None:
        offset = 0
        for case_index, panel_count in enumerate(panel_counts, start=1):
            case_id = f"case-{ordinal:04d}-{case_index:02d}"
            panel_ids = [f"panel-{panel_index:02d}" for panel_index in range(1, panel_count + 1)]
            stratum = "P1" if panel_count == 1 else (
                "P2" if panel_count == 2 else ("P3_4" if panel_count <= 4 else "P5PLUS")
            )
            code_label = {
                "P1": "P=1",
                "P2": "P=2",
                "P3_4": "P=3-4",
                "P5PLUS": "P=5+",
            }[stratum]
            case_base = {
                "case_id": case_id,
                "doi_id": doi_id,
                "case_kind": "single" if panel_count == 1 else "multi",
                "curation_status": "verified",
                "eligible_for_experiment": True,
                "asserted_panel_ids": panel_ids,
                "expected_evaluation_panel_ids": panel_ids,
                "asserted_panel_count": panel_count,
                "asserted_public_stratum": stratum,
                "asserted_code_label": code_label,
            }
            canonical_case_binding = _canonical_case_binding(case_base)
            panels: list[dict[str, Any]] = []
            for panel_index in range(1, panel_count + 1):
                candidate_id = candidate_ids[offset]
                offset += 1
                table_path = (
                    evidence_root
                    / "source-tables"
                    / f"{ordinal:04d}"
                    / f"{candidate_id}.json"
                )
                verification_sha = hashlib.sha256(
                    f"verify-{candidate_id}".encode("utf-8")
                ).hexdigest()
                _write_json(
                    table_path,
                    {
                        "artifact_type": "c2_v21_source_table",
                        "parent_doi_id": doi_id,
                        "source_candidate_id": candidate_id,
                        "raw_source_sha256": raw_sources[candidate_id]["sha256"],
                        "candidate_binding_sha256": parent_bindings["candidate"]["sha256"],
                        "proposal_binding_sha256": parent_bindings["proposal"]["sha256"],
                        "canonical_case_binding_sha256": canonical_case_binding,
                        "verification_evidence_sha256": verification_sha,
                    },
                )
                table = _artifact(table_path, evidence_root)
                panels.append(
                    {
                        "panel_id": f"panel-{panel_index:02d}",
                        "source_candidate_id": candidate_id,
                        "parent_doi_id": doi_id,
                        "source_table_path": table["path"],
                        "source_table_sha256": table["sha256"],
                        "sheet_or_null": None,
                        "raw_source_sha256": raw_sources[candidate_id]["sha256"],
                        "candidate_binding_sha256": parent_bindings["candidate"]["sha256"],
                        "proposal_binding_sha256": parent_bindings["proposal"]["sha256"],
                        "canonical_case_binding_sha256": canonical_case_binding,
                        "verification_evidence_sha256": verification_sha,
                    }
                )
            cases.append(
                {
                    **case_base,
                    "verified_panels": panels,
                    "bindings": {
                        "raw_source_evidence_sha256": raw_source_evidence["sha256"],
                        "candidate_binding_sha256": parent_bindings["candidate"]["sha256"],
                        "proposal_binding_sha256": parent_bindings["proposal"]["sha256"],
                        "review_binding_sha256": parent_bindings["review"]["sha256"],
                        "canonical_case_binding_sha256": canonical_case_binding,
                    },
                }
            )
    builder_path = evidence_root / "canonical-builder" / f"{ordinal:04d}.json"
    _write_json(
        builder_path,
        {
            "artifact_type": "c2_v21_canonical_builder_output",
            "parent_doi_id": doi_id,
            "complete": True,
            "model_result_selected": False,
            "code": {
                "commit": policy.frozen_bindings["canonical_builder_code_commit_full"],
                "sha256": policy.frozen_bindings["canonical_builder_code_sha256"],
                "dirty": False,
            },
            "builder_rule": {
                "version": policy.frozen_bindings["canonical_builder_rule_version"],
                "sha256": policy.frozen_bindings["canonical_builder_rule_sha256"],
            },
            "source_inventory_sha256": inventory["sha256"],
            "raw_source_evidence_sha256": raw_source_evidence["sha256"],
            "candidate_manifest": parent_bindings["candidate"],
            "proposal_manifest": parent_bindings["proposal"],
            "review_manifest": parent_bindings["review"],
            "canonical_manifest": parent_bindings["canonical"],
            "parent_doi_ids_sha256": policy.frozen_universe.doi_ids_sha256,
            "source_cases": source_cases,
            "cases": cases,
        },
    )
    builder = _artifact(builder_path, evidence_root)
    eligible_ids = [case["case_id"] for case in sorted(cases, key=lambda item: item["case_id"])]
    if panel_counts is None:
        final_disposition = "NON_STRATIFIED_DOWNLOADED_NO_VERIFIED_SOURCE"
        reason = "DOWNLOADED_EMPTY_VERIFIED_SOURCE_INVENTORY"
    elif not panel_counts:
        final_disposition = "NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE"
        reason = "DOWNLOADED_NO_QUALIFYING_CANONICAL_CASE"
    else:
        final_disposition = "STRATIFIED_SOURCE_CANONICAL"
        reason = "VERIFIED_SOURCE_CANONICAL_CASES"
    return {
        "doi_id": doi_id,
        "source_inventory": inventory,
        "raw_source_evidence": raw_source_evidence,
        "canonical_builder": builder,
        "eligible_ids": eligible_ids,
        "final_disposition": final_disposition,
        "classification_reason": reason,
    }


def _seal_terminal_report(report: dict[str, Any]) -> None:
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key not in {"report_hash", "seal"}}
    )
    seal = {"status": "TERMINAL", "sealed_report_hash": report["report_hash"]}
    seal["seal_hash"] = sha256_json(seal)
    report["seal"] = seal


def _build_fixture(
    workspace: Path,
    *,
    source_plan: dict[int, list[int] | None] | None = None,
) -> SyntheticFixture:
    policy_data = _policy_data()
    policy = compile_synthetic_policy_for_testing(policy_data)
    source_plan = source_plan or {
        1: [2],
        2: [2],
        3: [3],
        4: [3],
        5: [3],
        6: [5],
        7: [],
        8: None,
    }
    evidence_root = workspace / "evidence"
    output_root = workspace / "output"
    evidence_root.mkdir(parents=True, mode=0o700)
    output_root.mkdir(mode=0o700)
    chunks: list[dict[str, Any]] = []
    outcomes_by_doi: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for chunk_id in EXPECTED_CHUNK_IDS:
        partition = policy.partition_for_chunk(chunk_id)
        chunk_dois = policy.dois_for_chunk(chunk_id)
        attempts: list[dict[str, Any]] = []
        retry2_outcomes: list[dict[str, Any]] = []
        for attempt_id in EXPECTED_ATTEMPTS:
            raw_rows: list[dict[str, Any]] = []
            processed_rows: list[dict[str, Any]] = []
            skipped_rows: list[dict[str, Any]] = []
            for local_ordinal, doi_id in enumerate(chunk_dois, start=1):
                ordinal = partition.first_global_ordinal + local_ordinal - 1
                downloaded = ordinal in source_plan
                raw_rows.append(
                    {
                        "attempt_id": attempt_id,
                        "global_ordinal": ordinal,
                        "local_ordinal": local_ordinal,
                        "doi_id": doi_id,
                        "raw_disposition": "PROCESSED" if downloaded else "SKIPPED",
                    }
                )
                if downloaded:
                    processed_rows.append(
                        {
                            "global_ordinal": ordinal,
                            "local_ordinal": local_ordinal,
                            "doi_id": doi_id,
                        }
                    )
                else:
                    skipped_rows.append(
                        {
                            "global_ordinal": ordinal,
                            "local_ordinal": local_ordinal,
                            "doi_id": doi_id,
                            "terminal_status_raw": "no-source-data",
                        }
                    )
            raw_path = evidence_root / "raw" / chunk_id / f"{attempt_id}.jsonl"
            processed_path = evidence_root / "processed" / chunk_id / f"{attempt_id}.json"
            skipped_path = evidence_root / "skipped" / chunk_id / f"{attempt_id}.json"
            _write_jsonl(raw_path, raw_rows)
            _write_json(
                processed_path,
                {
                    "artifact_type": "c2_v2_processed_success",
                    "chunk_id": chunk_id,
                    "attempt_id": attempt_id,
                    "records": processed_rows,
                },
            )
            _write_json(
                skipped_path,
                {
                    "artifact_type": "c2_v2_skipped_status",
                    "chunk_id": chunk_id,
                    "attempt_id": attempt_id,
                    "records": skipped_rows,
                },
            )
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "raw_stream": _artifact(raw_path, evidence_root),
                    "processed_success": _artifact(processed_path, evidence_root),
                    "skipped_status": _artifact(skipped_path, evidence_root),
                }
            )
            if attempt_id == "retry2":
                retry2_outcomes = [
                    {
                        "doi_id": row["doi_id"],
                        "attempt_count": 3,
                        "terminal": True,
                        "terminal_status_raw": (
                            "downloaded"
                            if row["raw_disposition"] == "PROCESSED"
                            else "no-source-data"
                        ),
                        "terminal_status": (
                            "DOWNLOADED"
                            if row["raw_disposition"] == "PROCESSED"
                            else "NO_SOURCE_DATA"
                        ),
                    }
                    for row in raw_rows
                ]
        terminal_path = evidence_root / "terminal-outcomes" / f"{chunk_id}.json"
        _write_json(
            terminal_path,
            {
                "artifact_type": "c2_v21_terminal_outcomes",
                "chunk_id": chunk_id,
                "input_doi_ids_sha256": partition.doi_ids_sha256,
                "outcomes": retry2_outcomes,
            },
        )
        terminal = _artifact(terminal_path, evidence_root)
        sealed_path = evidence_root / "sealed-terminal" / f"{chunk_id}.json"
        sealed = {
            "report_type": "c2_v21_sealed_terminal_report",
            "chunk_id": chunk_id,
            "input_doi_ids_sha256": partition.doi_ids_sha256,
            "terminal_outcomes_file_sha256": terminal["sha256"],
        }
        _seal_terminal_report(sealed)
        _write_json(sealed_path, sealed)
        sealed_binding = _artifact(sealed_path, evidence_root)
        root_plan = policy.root_plan_for_chunk(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "root": {
                    "replacement_root_id": root_plan.replacement_root_id,
                    "retired_root_ids": list(root_plan.retired_root_ids),
                    "partial_root": False,
                },
                "input_total": partition.input_total,
                "input_doi_ids_sha256": partition.doi_ids_sha256,
                "terminal_outcomes": terminal,
                "sealed_terminal_report": sealed_binding,
                "attempts": attempts,
            }
        )
        terminal_binding = {
            "chunk_id": chunk_id,
            "input_doi_ids_sha256": partition.doi_ids_sha256,
            "terminal_outcome_file_sha256": terminal["sha256"],
            "sealed_report_file_sha256": sealed_binding["sha256"],
            "sealed_report_hash": sealed["report_hash"],
            "attempt_count": 3,
            "terminal": True,
        }
        for outcome in retry2_outcomes:
            outcomes_by_doi[outcome["doi_id"]] = (outcome, terminal_binding)
    source_entries = [
        _write_source_canonical(
            evidence_root=evidence_root,
            policy=policy,
            ordinal=ordinal,
            panel_counts=panel_counts,
        )
        for ordinal, panel_counts in source_plan.items()
    ]
    source_by_doi = {item["doi_id"]: item for item in source_entries}
    acquisitions: list[dict[str, Any]] = []
    direct_dispositions = {
        "NO_SOURCE_DATA": (
            "NON_STRATIFIED_NO_SOURCE_DATA",
            "TERMINAL_NO_SOURCE_DATA",
        )
    }
    for doi_id in policy.ordered_doi_ids:
        outcome, terminal_binding = outcomes_by_doi[doi_id]
        if outcome["terminal_status"] == "DOWNLOADED":
            source = source_by_doi[doi_id]
            acquisitions.append(
                {
                    "doi_id": doi_id,
                    "terminal_status_raw": outcome["terminal_status_raw"],
                    "terminal_status": outcome["terminal_status"],
                    "terminal_evidence_binding": terminal_binding,
                    "final_disposition": source["final_disposition"],
                    "source_inventory_binding_or_null": source["source_inventory"],
                    "canonical_builder_binding_or_null": source["canonical_builder"],
                    "all_eligible_case_ids": source["eligible_ids"],
                    "classification_reason": source["classification_reason"],
                }
            )
        else:
            disposition, reason = direct_dispositions[outcome["terminal_status"]]
            acquisitions.append(
                {
                    "doi_id": doi_id,
                    "terminal_status_raw": outcome["terminal_status_raw"],
                    "terminal_status": outcome["terminal_status"],
                    "terminal_evidence_binding": terminal_binding,
                    "final_disposition": disposition,
                    "source_inventory_binding_or_null": None,
                    "canonical_builder_binding_or_null": None,
                    "all_eligible_case_ids": [],
                    "classification_reason": reason,
                }
            )
    manifest = {
        "schema_version": "2.1-stagea",
        "manifest_type": "c2_full_replacement_source_classification",
        "frozen_universe": policy.frozen_universe.to_dict(),
        "frozen_bindings": dict(policy.frozen_bindings),
        "code": {"commit": "6" * 40, "dirty": False},
        "chunks": chunks,
        "source_canonical": [
            {
                "doi_id": item["doi_id"],
                "source_inventory": item["source_inventory"],
                "canonical_builder": item["canonical_builder"],
            }
            for item in source_entries
        ],
        "acquisition_dispositions": acquisitions,
    }
    manifest_path = evidence_root / "manifest.json"
    fixture = SyntheticFixture(
        workspace=workspace,
        evidence_root=evidence_root,
        output_root=output_root,
        manifest_path=manifest_path,
        manifest=manifest,
        policy_data=policy_data,
        policy=policy,
    )
    fixture.seal_manifest()
    return fixture


def _chunk(fixture: SyntheticFixture, chunk_id: str) -> dict[str, Any]:
    return next(item for item in fixture.manifest["chunks"] if item["chunk_id"] == chunk_id)


def _attempt(fixture: SyntheticFixture, chunk_id: str, attempt_id: str) -> dict[str, Any]:
    return next(
        item
        for item in _chunk(fixture, chunk_id)["attempts"]
        if item["attempt_id"] == attempt_id
    )


def _prepare(fixture: SyntheticFixture) -> Any:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            v2_finalizer,
            "load_production_policy",
            lambda: fixture.policy,
        )
        return prepare_full_replacement_finalization(fixture.manifest_path)


def _finalize(
    fixture: SyntheticFixture,
    output_path: Path,
) -> tuple[dict[str, Any], Path]:
    finalized = _prepare(fixture)
    try:
        return thaw_evidence_value(finalized.report), write_full_replacement_report(
            finalized,
            output_path,
        )
    finally:
        finalized.evidence.close()


def test_literal_acquisition_roster_partition_and_enum_oracles() -> None:
    assert CHUNK_IDS == EXPECTED_CHUNK_IDS
    assert FULL_REPLACEMENT_INPUT_TOTALS == EXPECTED_CHUNK_INPUT_TOTALS
    assert FINAL_CHUNK_INPUT_TOTAL == 63
    assert FROZEN_INPUT_TOTAL == 2463
    assert sum(EXPECTED_CHUNK_INPUT_TOTALS) == 2463
    assert ATTEMPT_IDS == EXPECTED_ATTEMPTS
    assert P_DISPOSITIONS == EXPECTED_P_DISPOSITIONS


def test_production_resolver_and_cli_remain_stage_b_blocked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(C2FullReplacementPolicyError, match="Stage-A only"):
        load_production_policy()
    assert list(inspect.signature(load_production_policy).parameters) == []
    parser = _build_parser()
    parsed = parser.parse_args(
        [
            "c2-full-replacement-finalize",
            "unavailable.json",
            "--out",
            "unavailable-output.json",
        ]
    )
    assert not any("policy" in key or "digest" in key or "map" in key for key in vars(parsed))
    exit_code = cli_main(
        [
            "c2-full-replacement-finalize",
            "unavailable.json",
            "--out",
            "unavailable-output.json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE" in captured.err


def test_full_acquisition_coverage_and_stratified_subset_gate_publish() -> None:
    with _workspace("coverage") as workspace:
        fixture = _build_fixture(workspace)
        report, output = _finalize(fixture, fixture.output_root / "final.json")
        assert output.exists()
        assert report["status"] == "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
        assert report["claim_scope"] == "STRATIFIED_SOURCE_TERMINAL_ADMISSION_ONLY"
        assert report["decision"] == "NOT_RUN_COVERAGE_GATE"
        assert len(report["acquisition_dispositions"]) == 2463
        assert len(report["stratified_source_classifications"]) == 6
        assert all(
            "derived_public_stratum" not in row
            and "cluster_id" not in row
            for row in report["acquisition_dispositions"]
        )
        assert report["acquisition_dispositions"][6]["final_disposition"] == (
            "NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE"
        )
        assert report["acquisition_dispositions"][7]["final_disposition"] == (
            "NON_STRATIFIED_DOWNLOADED_NO_VERIFIED_SOURCE"
        )
        assert [
            (row["p_disposition"], row["independent_cluster_count"])
            for row in report["strata"]
        ] == [("P1", 0), ("P2", 2), ("P3_4", 3), ("P5PLUS", 1)]
        assert report["stratified_source_classifications"][-1]["cluster_id"] == (
            report["stratified_source_classifications"][-1]["doi_id"]
        )
        final_chunk = [
            item
            for item in report["canonical_attempt_ledger"]
            if item["chunk_id"] == "013"
        ]
        assert len(final_chunk) == 3
        for item in final_chunk:
            assert [row["local_ordinal"] for row in item["outcomes"]] == list(range(1, 64))
            assert [row["global_ordinal"] for row in item["outcomes"]] == list(range(2401, 2464))
        validate_synthetic_final_report_for_testing(report, fixture.policy)


def test_publication_uses_immutable_evidence_not_a_self_hashed_report() -> None:
    with _workspace("immutable-publication") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            original = thaw_evidence_value(finalized.report)
            with pytest.raises(TypeError):
                finalized.report["status"] = "ADMITTED"
            with pytest.raises(TypeError):
                finalized.report["stratified_source_classifications"][0][
                    "derived_public_stratum"
                ] = "P5PLUS"
            with pytest.raises(TypeError):
                finalized.evidence.acquisition_dispositions[0]["final_disposition"] = (
                    "STRATIFIED_SOURCE_CANONICAL"
                )
            with pytest.raises(TypeError):
                finalized.policy.frozen_bindings["universe_file_sha256"] = "0" * 64

            forged = thaw_evidence_value(finalized.report)
            forged_classification = next(
                item
                for item in forged["stratified_source_classifications"]
                if item["derived_public_stratum"] == "P3_4"
            )
            forged_classification["derived_public_stratum"] = "P5PLUS"
            forged_classification["derived_code_label"] = "P=5+"
            forged_classification["qualified_panel_counts"] = [5]
            forged["stratified_source_classifications_sha256"] = sha256_json(
                forged["stratified_source_classifications"]
            )
            forged["strata"] = [
                {
                    "p_disposition": "P1",
                    "source_doi_count": 0,
                    "independent_cluster_count": 0,
                    "coverage_required": False,
                    "coverage_met": True,
                    "inference_eligible": False,
                    "deficient": False,
                },
                {
                    "p_disposition": "P2",
                    "source_doi_count": 2,
                    "independent_cluster_count": 2,
                    "coverage_required": True,
                    "coverage_met": True,
                    "inference_eligible": False,
                    "deficient": False,
                },
                {
                    "p_disposition": "P3_4",
                    "source_doi_count": 2,
                    "independent_cluster_count": 2,
                    "coverage_required": True,
                    "coverage_met": True,
                    "inference_eligible": False,
                    "deficient": False,
                },
                {
                    "p_disposition": "P5PLUS",
                    "source_doi_count": 2,
                    "independent_cluster_count": 2,
                    "coverage_required": True,
                    "coverage_met": True,
                    "inference_eligible": False,
                    "deficient": False,
                },
            ]
            forged["deficient_strata"] = []
            forged["status"] = "ADMITTED"
            forged["claim_status"] = "SUPPORTED"
            forged["decision"] = "NOT_RUN_NO_ANALYSIS_AUTHORIZED"
            forged["final_report_hash"] = sha256_json(
                {
                    key: value
                    for key, value in forged.items()
                    if key != "final_report_hash"
                }
            )
            validate_synthetic_final_report_for_testing(forged, finalized.policy)
            with pytest.raises(C2FullReplacementError, match="validated full-replacement"):
                write_full_replacement_report(
                    forged,  # type: ignore[arg-type]
                    fixture.output_root / "forged.json",
                )

            output_path = fixture.output_root / "evidence-derived.json"
            assert write_full_replacement_report(finalized, output_path) == output_path
            assert _read_json(output_path) == original
            assert _read_json(output_path)["status"] == (
                "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
            )
            assert any(
                item["derived_public_stratum"] == "P3_4"
                for item in _read_json(output_path)["stratified_source_classifications"]
            )
        finally:
            finalized.evidence.close()


def test_no_prefrozen_p_map_or_acquisition_p_fabrication_is_accepted() -> None:
    policy_data = _policy_data()
    policy_data["doi_p_cluster_map"] = []
    with pytest.raises(C2FullReplacementPolicyError, match="fields must be exactly"):
        compile_synthetic_policy_for_testing(policy_data)
    with _workspace("fabrication") as workspace:
        fixture = _build_fixture(workspace)
        fixture.manifest["acquisition_dispositions"][0]["derived_public_stratum"] = "P5PLUS"
        fixture.seal_manifest()
        with pytest.raises(C2FullReplacementError, match="schema validation failed"):
            _prepare(fixture)


def test_cross_doi_panel_and_asserted_panel_count_are_rejected() -> None:
    with _workspace("cross-doi") as workspace:
        fixture = _build_fixture(workspace)
        doi_id = fixture.policy.ordered_doi_ids[0]
        source = next(item for item in fixture.manifest["source_canonical"] if item["doi_id"] == doi_id)
        builder_path = fixture.evidence_root / source["canonical_builder"]["path"]
        builder = _read_json(builder_path)
        panel = builder["cases"][0]["verified_panels"][0]
        table_path = fixture.evidence_root / panel["source_table_path"]
        table = _read_json(table_path)
        table["parent_doi_id"] = fixture.policy.ordered_doi_ids[1]
        _write_json(table_path, table)
        panel["source_table_sha256"] = _sha256_file(table_path)
        _write_json(builder_path, builder)
        fixture.refresh_builder_binding(doi_id)
        with pytest.raises(C2FullReplacementError, match="cross-DOI"):
            _prepare(fixture)

    with _workspace("panel-recompute") as workspace:
        fixture = _build_fixture(workspace)
        doi_id = fixture.policy.ordered_doi_ids[0]
        source = next(item for item in fixture.manifest["source_canonical"] if item["doi_id"] == doi_id)
        builder_path = fixture.evidence_root / source["canonical_builder"]["path"]
        builder = _read_json(builder_path)
        case = builder["cases"][0]
        case["asserted_panel_count"] = 99
        case_base = {
            name: case[name]
            for name in (
                "case_id",
                "doi_id",
                "case_kind",
                "curation_status",
                "eligible_for_experiment",
                "asserted_panel_ids",
                "expected_evaluation_panel_ids",
                "asserted_panel_count",
                "asserted_public_stratum",
                "asserted_code_label",
            )
        }
        canonical_binding = _canonical_case_binding(case_base)
        case["bindings"]["canonical_case_binding_sha256"] = canonical_binding
        for panel in case["verified_panels"]:
            panel["canonical_case_binding_sha256"] = canonical_binding
            table_path = fixture.evidence_root / panel["source_table_path"]
            table = _read_json(table_path)
            table["canonical_case_binding_sha256"] = canonical_binding
            _write_json(table_path, table)
            panel["source_table_sha256"] = _sha256_file(table_path)
        _write_json(builder_path, builder)
        fixture.refresh_builder_binding(doi_id)
        with pytest.raises(C2FullReplacementError, match="asserted panel membership"):
            _prepare(fixture)


def test_raw_source_bytes_and_parent_chain_are_verified() -> None:
    with _workspace("raw-source-bytes") as workspace:
        fixture = _build_fixture(workspace)
        doi_id = fixture.policy.ordered_doi_ids[0]
        source = next(
            item
            for item in fixture.manifest["source_canonical"]
            if item["doi_id"] == doi_id
        )
        inventory = _read_json(fixture.evidence_root / source["source_inventory"]["path"])
        raw_evidence = _read_json(
            fixture.evidence_root / inventory["raw_source_evidence"]["path"]
        )
        raw_path = (
            fixture.evidence_root
            / raw_evidence["source_candidates"][0]["raw_source_path"]
        )
        raw_path.write_text('{"tampered":true}\n', encoding="utf-8")
        with pytest.raises(C2FullReplacementError, match="raw source candidate"):
            _prepare(fixture)

    with _workspace("raw-source-parent") as workspace:
        fixture = _build_fixture(workspace)
        doi_id = fixture.policy.ordered_doi_ids[0]
        source = next(
            item
            for item in fixture.manifest["source_canonical"]
            if item["doi_id"] == doi_id
        )
        inventory_path = fixture.evidence_root / source["source_inventory"]["path"]
        inventory = _read_json(inventory_path)
        raw_evidence_path = (
            fixture.evidence_root / inventory["raw_source_evidence"]["path"]
        )
        raw_evidence = _read_json(raw_evidence_path)
        raw_evidence["source_candidates"][0]["parent_doi_id"] = (
            fixture.policy.ordered_doi_ids[1]
        )
        _write_json(raw_evidence_path, raw_evidence)
        inventory["raw_source_evidence"]["sha256"] = _sha256_file(raw_evidence_path)
        _write_json(inventory_path, inventory)
        fixture.refresh_source_inventory_binding(doi_id)
        with pytest.raises(C2FullReplacementError, match="cross-DOI"):
            _prepare(fixture)


def test_multiple_same_stratum_cases_are_retained_as_one_doi_cluster() -> None:
    with _workspace("multiple-case") as workspace:
        fixture = _build_fixture(
            workspace,
            source_plan={
                1: [2, 2],
                2: [2],
                3: [3],
                4: [3],
                5: [5],
                6: [5],
            },
        )
        finalized = _prepare(fixture)
        try:
            classification = finalized.report["stratified_source_classifications"][0]
            assert list(classification["qualified_panel_counts"]) == [2, 2]
            assert len(classification["canonical_case_ids"]) == 2
            p2 = finalized.report["strata"][1]
            assert p2["source_doi_count"] == 2
            assert p2["independent_cluster_count"] == 2
        finally:
            finalized.evidence.close()


def test_multi_stratum_cases_and_missing_downloaded_source_coverage_fail() -> None:
    with _workspace("multi-stratum") as workspace:
        fixture = _build_fixture(
            workspace,
            source_plan={1: [2, 3], 2: [2], 3: [3], 4: [5], 5: [5]},
        )
        with pytest.raises(C2FullReplacementError, match="MULTI_STRATUM"):
            _prepare(fixture)
    with _workspace("missing-source") as workspace:
        fixture = _build_fixture(workspace)
        fixture.manifest["source_canonical"].pop()
        fixture.seal_manifest()
        with pytest.raises(C2FullReplacementError, match="exactly cover"):
            _prepare(fixture)


def test_all_eligible_cases_and_derived_dispositions_cannot_be_selected_or_forged() -> None:
    with _workspace("all-cases") as workspace:
        fixture = _build_fixture(
            workspace,
            source_plan={1: [2, 2], 2: [2], 3: [3], 4: [3], 5: [5], 6: [5]},
        )
        disposition = fixture.manifest["acquisition_dispositions"][0]
        disposition["all_eligible_case_ids"] = disposition["all_eligible_case_ids"][:1]
        fixture.seal_manifest()
        with pytest.raises(C2FullReplacementError, match="incomplete, selective"):
            _prepare(fixture)
    with _workspace("forged-disposition") as workspace:
        fixture = _build_fixture(workspace)
        fixture.manifest["acquisition_dispositions"][0]["final_disposition"] = (
            "NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE"
        )
        fixture.seal_manifest()
        with pytest.raises(C2FullReplacementError, match="not derived"):
            _prepare(fixture)


def test_raw_terminal_adapter_and_013_evidence_fail_closed() -> None:
    with _workspace("raw-status") as workspace:
        fixture = _build_fixture(workspace)
        skipped = _attempt(fixture, "001", "retry2")["skipped_status"]
        skipped_path = fixture.evidence_root / skipped["path"]
        payload = _read_json(skipped_path)
        payload["records"][0]["terminal_status_raw"] = "queued"
        _write_json(skipped_path, payload)
        fixture.rebind(skipped)
        with pytest.raises(C2FullReplacementError, match="schema validation failed"):
            _prepare(fixture)
    with _workspace("chunk-013") as workspace:
        fixture = _build_fixture(workspace)
        raw = _attempt(fixture, "013", "initial")["raw_stream"]
        raw_path = fixture.evidence_root / raw["path"]
        rows = raw_path.read_text(encoding="utf-8").splitlines()
        raw_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        fixture.rebind(raw)
        with pytest.raises(C2FullReplacementError, match="invalid row count"):
            _prepare(fixture)


def test_descriptor_single_read_retains_hashed_bytes_after_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("single-read") as workspace:
        fixture = _build_fixture(workspace)
        raw = _attempt(fixture, "001", "initial")["raw_stream"]["path"]
        raw_path = fixture.evidence_root / raw
        original_hash = _sha256_file(raw_path)
        original_reader = v2_evidence._read_no_follow_artifact
        replaced = False

        def read_then_replace(root: Any, path: str, label: str) -> Any:
            nonlocal replaced
            artifact = original_reader(root, path, label)
            if path == raw and not replaced:
                replaced = True
                raw_path.write_text(
                    '{"attempt_id":"initial","global_ordinal":1,"local_ordinal":1,'
                    '"doi_id":"10.9000/forged","raw_disposition":"SKIPPED"}\n',
                    encoding="utf-8",
                )
            return artifact

        monkeypatch.setattr(v2_evidence, "_read_no_follow_artifact", read_then_replace)
        finalized = _prepare(fixture)
        try:
            ledger = next(
                item
                for item in finalized.report["canonical_attempt_ledger"]
                if item["chunk_id"] == "001" and item["attempt_id"] == "initial"
            )
            assert ledger["raw_stream"]["sha256"] == original_hash
            assert _sha256_file(raw_path) != original_hash
        finally:
            finalized.evidence.close()


def test_evidence_and_output_trust_collision_and_no_replace_guards() -> None:
    with _workspace("guardrails") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            manifest_bytes = fixture.manifest_path.read_bytes()
            with pytest.raises(C2FullReplacementError, match="outside the evidence root"):
                write_full_replacement_report(finalized, fixture.evidence_root / "final.json")
            with pytest.raises(C2FullReplacementError, match="non-containing"):
                write_full_replacement_report(finalized, workspace / "ancestor.json")
            existing = fixture.output_root / "existing.json"
            existing.write_bytes(b"do-not-overwrite")
            with pytest.raises(C2FullReplacementError, match="already exists"):
                write_full_replacement_report(finalized, existing)
            assert existing.read_bytes() == b"do-not-overwrite"
            hardlink = fixture.output_root / "hardlink.json"
            os.link(fixture.manifest_path, hardlink)
            with pytest.raises(C2FullReplacementError, match="unsafe hard-link count"):
                write_full_replacement_report(finalized, hardlink)
            assert fixture.manifest_path.read_bytes() == manifest_bytes
        finally:
            finalized.evidence.close()


def test_symlinked_untrusted_evidence_and_output_parents_are_rejected() -> None:
    with _workspace("symlink-evidence") as workspace:
        fixture = _build_fixture(workspace)
        raw = _attempt(fixture, "001", "initial")["raw_stream"]
        raw_path = fixture.evidence_root / raw["path"]
        replacement = workspace / "replacement.jsonl"
        replacement.write_bytes(raw_path.read_bytes())
        raw_path.unlink()
        raw_path.symlink_to(replacement)
        with pytest.raises(C2FullReplacementError, match="Cannot descriptor-read"):
            _prepare(fixture)
    with _workspace("untrusted-output") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        os.chmod(fixture.output_root, 0o777)
        try:
            with pytest.raises(C2FullReplacementError, match="Cannot secure V2.1"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "unsafe.json",
                )
        finally:
            os.chmod(fixture.output_root, 0o700)
            finalized.evidence.close()
    with _workspace("symlink-output-parent") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        alias = workspace / "output-alias"
        alias.symlink_to(fixture.output_root, target_is_directory=True)
        try:
            with pytest.raises(C2FullReplacementError, match="Cannot secure V2.1"):
                write_full_replacement_report(finalized, alias / "unsafe.json")
        finally:
            finalized.evidence.close()


def test_evidence_leaf_and_intermediate_trust_boundaries_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("writable-evidence-leaf") as workspace:
        fixture = _build_fixture(workspace)
        raw_path = fixture.evidence_root / _attempt(
            fixture,
            "001",
            "initial",
        )["raw_stream"]["path"]
        os.chmod(raw_path, 0o666)
        with pytest.raises(C2FullReplacementError, match="group/world writable"):
            _prepare(fixture)

    with _workspace("writable-evidence-intermediate") as workspace:
        fixture = _build_fixture(workspace)
        raw_path = fixture.evidence_root / _attempt(
            fixture,
            "001",
            "initial",
        )["raw_stream"]["path"]
        os.chmod(raw_path.parent, 0o777)
        with pytest.raises(C2FullReplacementError, match="group/world writable"):
            _prepare(fixture)

    with _workspace("hardlinked-evidence-leaf") as workspace:
        fixture = _build_fixture(workspace)
        raw_path = fixture.evidence_root / _attempt(
            fixture,
            "001",
            "initial",
        )["raw_stream"]["path"]
        os.link(raw_path, workspace / "raw-alias.jsonl")
        with pytest.raises(C2FullReplacementError, match="unsafe hard-link count"):
            _prepare(fixture)

    with _workspace("mocked-attacker-owner") as workspace:
        fixture = _build_fixture(workspace)
        original_validator = v2_evidence.validate_trusted_regular_file_descriptor

        def reject_mocked_owner(
            descriptor: int,
            path: Path,
            *,
            label: str = "Trusted evidence artifact",
        ) -> os.stat_result:
            if path.name == "manifest.json":
                raise v2_evidence.ProvenanceError(
                    f"{label} has an unsafe owner: {path}"
                )
            return original_validator(descriptor, path, label=label)

        monkeypatch.setattr(
            v2_evidence,
            "validate_trusted_regular_file_descriptor",
            reject_mocked_owner,
        )
        with pytest.raises(C2FullReplacementError, match="unsafe owner"):
            _prepare(fixture)


def test_prepublication_evidence_revalidation_rejects_mutation_and_replacement() -> None:
    with _workspace("evidence-leaf-mutation") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            raw_path = fixture.evidence_root / _attempt(
                fixture,
                "001",
                "initial",
            )["raw_stream"]["path"]
            raw_path.write_bytes(raw_path.read_bytes() + b"\n")
            output = fixture.output_root / "mutated.json"
            with pytest.raises(
                C2FullReplacementError,
                match="validated evidence artifact changed",
            ):
                write_full_replacement_report(finalized, output)
            assert not output.exists()
        finally:
            finalized.evidence.close()

    with _workspace("evidence-root-mutation") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            (fixture.evidence_root / "attacker-root-marker").write_bytes(b"changed")
            output = fixture.output_root / "root-mutated.json"
            with pytest.raises(
                C2FullReplacementError,
                match="evidence root metadata changed",
            ):
                write_full_replacement_report(finalized, output)
            assert not output.exists()
        finally:
            finalized.evidence.close()

    with _workspace("evidence-leaf-replacement") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            raw_path = fixture.evidence_root / _attempt(
                fixture,
                "001",
                "initial",
            )["raw_stream"]["path"]
            original_bytes = raw_path.read_bytes()
            raw_path.rename(workspace / "replaced-raw.jsonl")
            raw_path.write_bytes(original_bytes)
            output = fixture.output_root / "replaced.json"
            with pytest.raises(
                C2FullReplacementError,
                match="validated evidence artifact changed",
            ):
                write_full_replacement_report(finalized, output)
            assert not output.exists()
        finally:
            finalized.evidence.close()

    with _workspace("evidence-directory-mutation") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            raw_path = fixture.evidence_root / _attempt(
                fixture,
                "001",
                "initial",
            )["raw_stream"]["path"]
            (raw_path.parent / "attacker-marker").write_bytes(b"changed")
            output = fixture.output_root / "directory-mutated.json"
            with pytest.raises(
                C2FullReplacementError,
                match="validated evidence artifact changed",
            ):
                write_full_replacement_report(finalized, output)
            assert not output.exists()
        finally:
            finalized.evidence.close()


def test_evidence_artifact_directory_and_leaf_reject_mutating_acl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("artifact-acl") as workspace:
        fixture = _build_fixture(workspace)
        raw_path = fixture.evidence_root / _attempt(
            fixture,
            "001",
            "initial",
        )["raw_stream"]["path"]
        directory_descriptor = os.open(
            raw_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        file_descriptor = os.open(raw_path, os.O_RDONLY | os.O_NOFOLLOW)
        monkeypatch.setattr(
            v2_evidence._models,
            "_trusted_acl_allows_foreign_mutation",
            lambda _descriptor: True,
        )
        try:
            with pytest.raises(v2_evidence.ProvenanceError, match="mutating ACL"):
                v2_evidence.validate_trusted_directory_descriptor(
                    directory_descriptor,
                    raw_path.parent,
                )
            with pytest.raises(v2_evidence.ProvenanceError, match="mutating ACL"):
                v2_evidence.validate_trusted_regular_file_descriptor(
                    file_descriptor,
                    raw_path,
                )
        finally:
            os.close(file_descriptor)
            os.close(directory_descriptor)


def test_link_race_unsupported_and_staging_reuse_preserve_unrelated_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("link-race") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        output = fixture.output_root / "race.json"
        original_link = os.link

        def competing_leaf(
            source: str,
            destination: str,
            *,
            src_dir_fd: int,
            dst_dir_fd: int,
            follow_symlinks: bool,
        ) -> None:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=dst_dir_fd,
            )
            try:
                os.write(descriptor, b"competing-output")
            finally:
                os.close(descriptor)
            original_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: True)
        monkeypatch.setattr(v2_finalizer.os, "link", competing_leaf)
        try:
            with pytest.raises(C2FullReplacementError, match="existing output leaf"):
                write_full_replacement_report(finalized, output)
            assert output.read_bytes() == b"competing-output"
        finally:
            finalized.evidence.close()
    monkeypatch.undo()
    _allow_m1_guard_for_test(monkeypatch)
    with _workspace("unsupported") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: False)
        try:
            with pytest.raises(C2FullReplacementError, match="unsupported"):
                write_full_replacement_report(finalized, fixture.output_root / "no.json")
        finally:
            finalized.evidence.close()


def test_failed_link_and_output_parent_swap_do_not_report_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("failed-link") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        staged_names: list[str] = []

        def fail_after_staging_reuse(
            source: str,
            _destination: str,
            *,
            src_dir_fd: int,
            dst_dir_fd: int,
            follow_symlinks: bool,
        ) -> None:
            del dst_dir_fd, follow_symlinks
            staged_names.append(source)
            os.unlink(source, dir_fd=src_dir_fd)
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(descriptor, b"unrelated-after-failed-link")
            finally:
                os.close(descriptor)
            raise OSError(errno.EIO, "synthetic link failure")

        monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: True)
        monkeypatch.setattr(v2_finalizer.os, "link", fail_after_staging_reuse)
        try:
            with pytest.raises(C2FullReplacementError, match="no-replace publication failed"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "failed.json",
                )
            assert staged_names
            assert (fixture.output_root / staged_names[0]).read_bytes() == (
                b"unrelated-after-failed-link"
            )
        finally:
            finalized.evidence.close()
    monkeypatch.undo()
    _allow_m1_guard_for_test(monkeypatch)
    with _workspace("output-parent-swap") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        old_output = workspace / "output-original"
        original_link = os.link

        def link_then_swap_parent(
            source: str,
            destination: str,
            *,
            src_dir_fd: int,
            dst_dir_fd: int,
            follow_symlinks: bool,
        ) -> None:
            original_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )
            fixture.output_root.rename(old_output)
            fixture.output_root.symlink_to(workspace / "attacker")

        monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: True)
        monkeypatch.setattr(v2_finalizer.os, "link", link_then_swap_parent)
        try:
            with pytest.raises(C2FullReplacementError, match="output verification failed"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "swapped.json",
                )
            assert (old_output / "swapped.json").exists()
        finally:
            finalized.evidence.close()


def test_parent_and_leaf_swap_fail_without_success_or_input_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("evidence-swap") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        old_evidence = workspace / "evidence-original"
        manifest_bytes = fixture.manifest_path.read_bytes()
        fixture.evidence_root.rename(old_evidence)
        fixture.evidence_root.symlink_to(workspace / "attacker")
        try:
            with pytest.raises(C2FullReplacementError, match="evidence root changed"):
                write_full_replacement_report(finalized, fixture.output_root / "out.json")
            assert (old_evidence / "manifest.json").read_bytes() == manifest_bytes
        finally:
            finalized.evidence.close()
    with _workspace("leaf-swap") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        original_link = os.link

        def link_then_swap(
            source: str,
            destination: str,
            *,
            src_dir_fd: int,
            dst_dir_fd: int,
            follow_symlinks: bool,
        ) -> None:
            original_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )
            os.unlink(destination, dir_fd=dst_dir_fd)
            os.symlink(fixture.manifest_path, destination, dir_fd=dst_dir_fd)

        monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: True)
        monkeypatch.setattr(v2_finalizer.os, "link", link_then_swap)
        try:
            with pytest.raises(C2FullReplacementError, match="does not identify"):
                write_full_replacement_report(finalized, fixture.output_root / "leaf.json")
        finally:
            finalized.evidence.close()


def test_source_coverage_aggregation_excludes_p1_from_gate() -> None:
    classifications = (
        StratifiedSourceClassification(
            doi_id="10.9000/unit-p1",
            canonical_case_set_hash="a" * 64,
            canonical_case_ids=("case-a",),
            verified_panel_descriptors_sha256="b" * 64,
            qualified_panel_counts=(1,),
            derived_public_stratum="P1",
            derived_code_label="P=1",
            source_binding_hashes=_source_hashes("c"),
        ),
        StratifiedSourceClassification(
            doi_id="10.9000/unit-p2-a",
            canonical_case_set_hash="d" * 64,
            canonical_case_ids=("case-b",),
            verified_panel_descriptors_sha256="e" * 64,
            qualified_panel_counts=(2,),
            derived_public_stratum="P2",
            derived_code_label="P=2",
            source_binding_hashes=_source_hashes("f"),
        ),
        StratifiedSourceClassification(
            doi_id="10.9000/unit-p2-b",
            canonical_case_set_hash="1" * 64,
            canonical_case_ids=("case-c",),
            verified_panel_descriptors_sha256="2" * 64,
            qualified_panel_counts=(2,),
            derived_public_stratum="P2",
            derived_code_label="P=2",
            source_binding_hashes=_source_hashes("3"),
        ),
    )
    aggregation = aggregate_stratified_source_classifications(classifications)
    assert aggregation.status == "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
    assert aggregation.strata[0].coverage_required is False
    assert aggregation.strata[0].deficient is False


def test_final_report_rejects_a_forged_source_cluster_alias() -> None:
    with _workspace("forged-cluster") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            forged = thaw_evidence_value(finalized.report)
            forged["stratified_source_classifications"][0]["cluster_id"] = (
                "10.9000/forged-cluster"
            )
            forged["stratified_source_classifications_sha256"] = sha256_json(
                forged["stratified_source_classifications"]
            )
            forged["final_report_hash"] = sha256_json(
                {
                    key: value
                    for key, value in forged.items()
                    if key != "final_report_hash"
                }
            )
            with pytest.raises(C2FullReplacementError, match="cluster_id"):
                validate_synthetic_final_report_for_testing(forged, fixture.policy)
        finally:
            finalized.evidence.close()

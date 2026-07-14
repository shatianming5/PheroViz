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
from experiments.c2_full_replacement_finalizer import (
    C2FullReplacementError,
    finalize_synthetic_to_path_for_testing,
    prepare_full_replacement_finalization_for_testing,
    validate_synthetic_final_report_for_testing,
    write_full_replacement_report,
)
from experiments.c2_full_replacement_policy import (
    ATTEMPT_IDS,
    CHUNK_IDS,
    FINAL_CHUNK_INPUT_TOTAL,
    FROZEN_INPUT_TOTAL,
    FULL_REPLACEMENT_INPUT_TOTALS,
    P1_NONINFERENTIAL_CLUSTER,
    P_DISPOSITIONS,
    C2FullReplacementPolicyError,
    aggregate_policy_rows,
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_policy_data(
    *,
    include_p1: bool = False,
    p5plus_single_cluster: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal in range(1, FROZEN_INPUT_TOTAL + 1):
        if include_p1 and ordinal == 1:
            disposition = "P1"
            cluster = P1_NONINFERENTIAL_CLUSTER
        else:
            disposition = ("P2", "P3_4", "P5PLUS")[(ordinal - 1) % 3]
            if disposition == "P5PLUS" and p5plus_single_cluster:
                cluster = "synthetic-p5plus-one"
            else:
                cluster = f"synthetic-{disposition.lower()}-{(ordinal // 3) % 3}"
        rows.append(
            {
                "global_ordinal": ordinal,
                "doi_id": f"10.9000/v2-synthetic-{ordinal:04d}",
                "p_disposition": disposition,
                "independent_cluster_id": cluster,
            }
        )
    partition: list[dict[str, Any]] = []
    start = 1
    for chunk_id, total in zip(
        EXPECTED_CHUNK_IDS,
        EXPECTED_CHUNK_INPUT_TOTALS,
        strict=True,
    ):
        chunk_dois = [
            row["doi_id"]
            for row in rows[start - 1 : start - 1 + total]
        ]
        partition.append(
            {
                "chunk_id": chunk_id,
                "first_global_ordinal": start,
                "input_total": total,
                "doi_ids_sha256": sha256_json(chunk_dois),
            }
        )
        start += total
    return {
        "policy_id": "synthetic-c2-full-replacement-v2",
        "policy_version": "stagea-test",
        "frozen_universe": {
            "sha256": "a" * 64,
            "doi_ids_sha256": sha256_json([row["doi_id"] for row in rows]),
            "input_total": FROZEN_INPUT_TOTAL,
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
        "doi_p_cluster_map": rows,
        "raw_mapping_source_manifest": [{"kind": "synthetic-stagea-only"}],
        "raw_evidence_commitments": [{"kind": "synthetic-stagea-only"}],
    }


def _artifact(path: Path, evidence_root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(evidence_root).as_posix(),
        "sha256": _sha256_file(path),
    }


def _build_fixture(
    workspace: Path,
    *,
    include_p1: bool = False,
    p5plus_single_cluster: bool = False,
) -> SyntheticFixture:
    policy_data = _synthetic_policy_data(
        include_p1=include_p1,
        p5plus_single_cluster=p5plus_single_cluster,
    )
    policy = compile_synthetic_policy_for_testing(policy_data)
    evidence_root = workspace / "evidence"
    output_root = workspace / "output"
    evidence_root.mkdir(mode=0o700)
    output_root.mkdir(mode=0o700)
    chunks: list[dict[str, Any]] = []
    for chunk_id in EXPECTED_CHUNK_IDS:
        partition = policy.partition_for_chunk(chunk_id)
        rows = policy.rows_for_chunk(chunk_id)
        mapping_path = evidence_root / "mapping" / f"{chunk_id}.jsonl"
        _write_jsonl(
            mapping_path,
            [
                {
                    "global_ordinal": row.global_ordinal,
                    "local_ordinal": local_ordinal,
                    "doi_id": row.doi_id,
                    "p_disposition": row.p_disposition,
                    "independent_cluster_id": row.independent_cluster_id,
                }
                for local_ordinal, row in enumerate(rows, start=1)
            ],
        )
        attempts: list[dict[str, Any]] = []
        for attempt_index, attempt_id in enumerate(EXPECTED_ATTEMPTS):
            raw_rows: list[dict[str, Any]] = []
            processed_rows: list[dict[str, Any]] = []
            skipped_rows: list[dict[str, Any]] = []
            for local_ordinal, row in enumerate(rows, start=1):
                processed = (row.global_ordinal + attempt_index) % 3 != 0
                raw_rows.append(
                    {
                        "attempt_id": attempt_id,
                        "global_ordinal": row.global_ordinal,
                        "local_ordinal": local_ordinal,
                        "doi_id": row.doi_id,
                        "raw_disposition": "PROCESSED" if processed else "SKIPPED",
                    }
                )
                if processed:
                    processed_rows.append(
                        {
                            "global_ordinal": row.global_ordinal,
                            "local_ordinal": local_ordinal,
                            "doi_id": row.doi_id,
                        }
                    )
                else:
                    skipped_rows.append(
                        {
                            "global_ordinal": row.global_ordinal,
                            "local_ordinal": local_ordinal,
                            "doi_id": row.doi_id,
                            "terminal_status": "NO_SOURCE_DATA",
                        }
                    )
            raw_path = evidence_root / "raw" / chunk_id / f"{attempt_id}.jsonl"
            processed_path = (
                evidence_root / "processed" / chunk_id / f"{attempt_id}.json"
            )
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
                "canonical_mapping": _artifact(mapping_path, evidence_root),
                "attempts": attempts,
            }
        )
    manifest = {
        "schema_version": "2.0-stagea",
        "manifest_type": "c2_full_replacement_admission",
        "frozen_universe": policy.frozen_universe.to_dict(),
        "code": {"commit": "b" * 40, "dirty": False},
        "chunks": chunks,
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


def _attempt(
    fixture: SyntheticFixture,
    chunk_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    return next(
        item
        for item in _chunk(fixture, chunk_id)["attempts"]
        if item["attempt_id"] == attempt_id
    )


def _prepare(fixture: SyntheticFixture) -> Any:
    return prepare_full_replacement_finalization_for_testing(
        fixture.manifest_path,
        fixture.policy,
    )


def test_literal_roster_partition_and_enum_oracles() -> None:
    assert CHUNK_IDS == EXPECTED_CHUNK_IDS
    assert FULL_REPLACEMENT_INPUT_TOTALS == EXPECTED_CHUNK_INPUT_TOTALS
    assert FINAL_CHUNK_INPUT_TOTAL == 63
    assert FROZEN_INPUT_TOTAL == 2463
    assert sum(EXPECTED_CHUNK_INPUT_TOTALS) == 2463
    assert ATTEMPT_IDS == EXPECTED_ATTEMPTS
    assert P_DISPOSITIONS == EXPECTED_P_DISPOSITIONS


def test_production_resolver_and_cli_fail_closed_without_stage_b_policy(
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
    assert not any("policy" in key or "digest" in key for key in vars(parsed))
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
    assert "Stage-A only" in captured.err


def test_full_synthetic_universe_is_attested_and_published_no_replace() -> None:
    with _workspace("success") as workspace:
        fixture = _build_fixture(workspace)
        report, output = finalize_synthetic_to_path_for_testing(
            fixture.manifest_path,
            fixture.output_root / "final.json",
            fixture.policy,
        )
        assert output.exists()
        assert report["status"] == "ADMITTED"
        assert report["claim_status"] == "SUPPORTED"
        assert report["source_doi_count"] == 2463
        assert len(report["canonical_attempt_ledger"]) == 39
        assert sum(
            len(entry["outcomes"]) for entry in report["canonical_attempt_ledger"]
        ) == 2463 * 3
        final_chunk_attempts = [
            entry
            for entry in report["canonical_attempt_ledger"]
            if entry["chunk_id"] == "013"
        ]
        assert len(final_chunk_attempts) == 3
        for entry in final_chunk_attempts:
            assert [row["local_ordinal"] for row in entry["outcomes"]] == list(
                range(1, 64)
            )
            assert [row["global_ordinal"] for row in entry["outcomes"]] == list(
                range(2401, 2464)
            )
        serialized = json.loads(output.read_text(encoding="utf-8"))
        assert serialized == report
        validate_synthetic_final_report_for_testing(serialized, fixture.policy)
        assert not list(fixture.output_root.glob(".*.stagea"))


def test_p1_noninferential_model_blocks_and_p5plus_has_precedence() -> None:
    with _workspace("p1") as workspace:
        fixture = _build_fixture(workspace, include_p1=True)
        finalized = _prepare(fixture)
        try:
            report = finalized.report
            assert report["status"] == "BLOCKED_INSUFFICIENT_INDEPENDENT_P1"
            assert report["claim_status"] == "UNSUPPORTED"
            assert report["trend_status"] == "NOT_RUN"
            assert report["equivalence_status"] == "NOT_RUN"
            p1 = report["strata"][0]
            assert p1 == {
                "p_disposition": "P1",
                "doi_count": 1,
                "independent_cluster_count": 0,
                "inference_eligible": False,
                "deficient": True,
            }
        finally:
            finalized.evidence.close()
    with _workspace("p5-precedence") as workspace:
        fixture = _build_fixture(
            workspace,
            include_p1=True,
            p5plus_single_cluster=True,
        )
        finalized = _prepare(fixture)
        try:
            assert finalized.report["status"] == (
                "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
            )
            assert finalized.report["deficient_strata"] == ["P1", "P5PLUS"]
        finally:
            finalized.evidence.close()


def test_policy_rejects_invalid_p1_roster_partition_and_nonfull_mapping() -> None:
    bad_p1 = _synthetic_policy_data(include_p1=True)
    bad_p1["doi_p_cluster_map"][0]["independent_cluster_id"] = "doi-derived-1"
    with pytest.raises(C2FullReplacementPolicyError, match="P1 cluster"):
        compile_synthetic_policy_for_testing(bad_p1)

    bad_roster = _synthetic_policy_data()
    bad_roster["partition"][0]["chunk_id"] = "014"
    with pytest.raises(C2FullReplacementPolicyError, match="ordered roster"):
        compile_synthetic_policy_for_testing(bad_roster)

    bad_partial = _synthetic_policy_data()
    bad_partial["partition"][-1]["input_total"] = 200
    with pytest.raises(C2FullReplacementPolicyError, match="must contain 63"):
        compile_synthetic_policy_for_testing(bad_partial)

    bad_map = _synthetic_policy_data()
    bad_map["doi_p_cluster_map"].pop()
    with pytest.raises(C2FullReplacementPolicyError, match="exactly 2,463"):
        compile_synthetic_policy_for_testing(bad_map)


def test_small_mapping_aggregation_is_unit_only_and_enforces_p1_semantics() -> None:
    from experiments.c2_full_replacement_policy import PolicyRow

    aggregation = aggregate_policy_rows(
        (
            PolicyRow(1, "10.9000/unit-p2-1", "P2", "unit-p2-a"),
            PolicyRow(2, "10.9000/unit-p2-2", "P2", "unit-p2-b"),
        )
    )
    assert aggregation.status == "ADMITTED"
    assert aggregation.strata[0].doi_count == 0
    assert aggregation.strata[0].deficient is False
    assert aggregation.strata[0].inference_eligible is False
    with pytest.raises(C2FullReplacementPolicyError, match="P1 cluster"):
        aggregate_policy_rows(
            (
                PolicyRow(1, "10.9000/unit-p1", "P1", "generated-p1"),
            )
        )


def test_non_p1_gates_and_cross_disposition_cluster_spanning_are_closed() -> None:
    from experiments.c2_full_replacement_policy import PolicyRow

    p3_deficient = aggregate_policy_rows(
        (
            PolicyRow(1, "10.9000/unit-p2-a", "P2", "p2-a"),
            PolicyRow(2, "10.9000/unit-p2-b", "P2", "p2-b"),
            PolicyRow(3, "10.9000/unit-p3-a", "P3_4", "p3-a"),
        )
    )
    assert p3_deficient.status == "BLOCKED_INSUFFICIENT_INDEPENDENT_P3_4"
    assert p3_deficient.deficient_strata == ("P3_4",)

    p5_precedence = aggregate_policy_rows(
        (
            PolicyRow(1, "10.9000/unit-p2-a", "P2", "p2-a"),
            PolicyRow(2, "10.9000/unit-p3-a", "P3_4", "p3-a"),
            PolicyRow(3, "10.9000/unit-p5-a", "P5PLUS", "p5-a"),
        )
    )
    assert p5_precedence.status == "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
    with pytest.raises(C2FullReplacementPolicyError, match="cannot span"):
        aggregate_policy_rows(
            (
                PolicyRow(1, "10.9000/unit-cross-p2", "P2", "shared-cluster"),
                PolicyRow(2, "10.9000/unit-cross-p3", "P3_4", "shared-cluster"),
            )
        )


def test_mapping_is_compared_per_ordinal_not_by_claimed_hash() -> None:
    with _workspace("mapping") as workspace:
        fixture = _build_fixture(workspace)
        binding = _chunk(fixture, "001")["canonical_mapping"]
        mapping_path = fixture.evidence_root / binding["path"]
        original = mapping_path.read_text(encoding="utf-8")

        lines = original.splitlines()
        changed = json.loads(lines[0])
        changed["p_disposition"] = "P3_4"
        lines[0] = json.dumps(changed, sort_keys=True, separators=(",", ":"))
        mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(C2FullReplacementError, match="bytes do not match"):
            _prepare(fixture)

        fixture.rebind(binding)
        with pytest.raises(C2FullReplacementError, match="differs from compiled"):
            _prepare(fixture)

        mapping_path.write_text(original, encoding="utf-8")
        fixture.rebind(binding)
        lines = original.splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fixture.rebind(binding)
        with pytest.raises(C2FullReplacementError, match="differs from compiled"):
            _prepare(fixture)

        mapping_path.write_text(original, encoding="utf-8")
        fixture.rebind(binding)
        lines = original.splitlines()
        duplicate = json.loads(lines[1])
        duplicate["global_ordinal"] = 1
        lines[1] = json.dumps(duplicate, sort_keys=True, separators=(",", ":"))
        mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fixture.rebind(binding)
        with pytest.raises(C2FullReplacementError, match="differs from compiled"):
            _prepare(fixture)


def test_raw_ledgers_recompute_downloaded_partition_and_retry2_outcome() -> None:
    with _workspace("raw-ledger") as workspace:
        fixture = _build_fixture(workspace)
        retry2 = _attempt(fixture, "001", "retry2")
        skipped_path = fixture.evidence_root / retry2["skipped_status"]["path"]
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
        assert skipped["records"]
        skipped["records"][0]["terminal_status"] = "RETRY_EXHAUSTED"
        _write_json(skipped_path, skipped)
        fixture.rebind(retry2["skipped_status"])
        finalized = _prepare(fixture)
        try:
            ledger = next(
                item
                for item in finalized.report["canonical_attempt_ledger"]
                if item["chunk_id"] == "001" and item["attempt_id"] == "retry2"
            )
            assert "RETRY_EXHAUSTED" in {
                item["terminal_status"] for item in ledger["outcomes"]
            }
            assert "DOWNLOADED" in {
                item["terminal_status"] for item in ledger["outcomes"]
            }
        finally:
            finalized.evidence.close()

        skipped["records"][0]["terminal_status"] = "DOWNLOADED"
        _write_json(skipped_path, skipped)
        fixture.rebind(retry2["skipped_status"])
        with pytest.raises(C2FullReplacementError, match="schema validation failed"):
            _prepare(fixture)


def test_raw_partition_and_013_every_attempt_are_fail_closed() -> None:
    with _workspace("chunk-013") as workspace:
        fixture = _build_fixture(workspace)
        attempt = _attempt(fixture, "013", "initial")
        raw_path = fixture.evidence_root / attempt["raw_stream"]["path"]
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        raw_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        fixture.rebind(attempt["raw_stream"])
        with pytest.raises(C2FullReplacementError, match="invalid row count"):
            _prepare(fixture)

        reordered_workspace = workspace / "reordered"
        reordered_workspace.mkdir(mode=0o700)
        fixture = _build_fixture(reordered_workspace)
        attempt = _attempt(fixture, "013", "retry2")
        raw_path = fixture.evidence_root / attempt["raw_stream"]["path"]
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fixture.rebind(attempt["raw_stream"])
        with pytest.raises(C2FullReplacementError, match="order differs"):
            _prepare(fixture)


def test_raw_status_attempt_roster_and_duplicate_artifacts_are_fail_closed() -> None:
    with _workspace("raw-status") as workspace:
        fixture = _build_fixture(workspace)
        retry1 = _attempt(fixture, "001", "retry1")
        skipped_path = fixture.evidence_root / retry1["skipped_status"]["path"]
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
        skipped["records"][0]["terminal_status"] = "QUEUED"
        _write_json(skipped_path, skipped)
        fixture.rebind(retry1["skipped_status"])
        with pytest.raises(C2FullReplacementError, match="schema validation failed"):
            _prepare(fixture)

    with _workspace("attempt-roster") as workspace:
        fixture = _build_fixture(workspace)
        _chunk(fixture, "001")["attempts"].pop()
        fixture.seal_manifest()
        with pytest.raises(C2FullReplacementError, match="schema validation failed"):
            _prepare(fixture)

    with _workspace("duplicate-artifact") as workspace:
        fixture = _build_fixture(workspace)
        attempt = _attempt(fixture, "001", "initial")
        attempt["processed_success"] = dict(attempt["raw_stream"])
        fixture.seal_manifest()
        with pytest.raises(C2FullReplacementError, match="reuses an artifact path"):
            _prepare(fixture)


def test_descriptor_single_read_uses_the_hashed_bytes_despite_later_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("single-read") as workspace:
        fixture = _build_fixture(workspace)
        target = _attempt(fixture, "001", "initial")["raw_stream"]["path"]
        target_path = fixture.evidence_root / target
        original_hash = _sha256_file(target_path)
        original_reader = v2_evidence._read_no_follow_artifact
        substituted = False

        def read_then_substitute(root: Any, relative_path: str, label: str) -> Any:
            nonlocal substituted
            artifact = original_reader(root, relative_path, label)
            if relative_path == target and not substituted:
                substituted = True
                target_path.write_text(
                    '{"attempt_id":"initial","global_ordinal":1,"local_ordinal":1,'
                    '"doi_id":"10.9000/forged","raw_disposition":"SKIPPED"}\n',
                    encoding="utf-8",
                )
            return artifact

        monkeypatch.setattr(v2_evidence, "_read_no_follow_artifact", read_then_substitute)
        finalized = _prepare(fixture)
        try:
            ledger = next(
                item
                for item in finalized.report["canonical_attempt_ledger"]
                if item["chunk_id"] == "001" and item["attempt_id"] == "initial"
            )
            assert ledger["raw_stream"]["sha256"] == original_hash
            assert _sha256_file(target_path) != original_hash
        finally:
            finalized.evidence.close()


def test_symlinked_or_untrusted_evidence_root_is_rejected_before_attestation() -> None:
    with _workspace("evidence-safety") as workspace:
        fixture = _build_fixture(workspace)
        raw = _attempt(fixture, "001", "initial")["raw_stream"]
        raw_path = fixture.evidence_root / raw["path"]
        replacement = workspace / "replacement.jsonl"
        replacement.write_bytes(raw_path.read_bytes())
        raw_path.unlink()
        raw_path.symlink_to(replacement)
        with pytest.raises(C2FullReplacementError, match="Cannot descriptor-read"):
            _prepare(fixture)

    with _workspace("untrusted-evidence") as workspace:
        fixture = _build_fixture(workspace)
        os.chmod(fixture.evidence_root, 0o777)
        try:
            with pytest.raises(C2FullReplacementError, match="trusted V2 evidence root"):
                _prepare(fixture)
        finally:
            os.chmod(fixture.evidence_root, 0o700)


def test_output_rejects_evidence_collisions_hardlinks_symlinks_and_existing_leaf() -> None:
    with _workspace("output-collision") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            manifest_bytes = fixture.manifest_path.read_bytes()
            with pytest.raises(C2FullReplacementError, match="outside the evidence root"):
                write_full_replacement_report(
                    finalized,
                    fixture.evidence_root / "final.json",
                )
            assert fixture.manifest_path.read_bytes() == manifest_bytes
            with pytest.raises(C2FullReplacementError, match="non-containing"):
                write_full_replacement_report(
                    finalized,
                    workspace / "ancestor-parent-final.json",
                )
            assert fixture.manifest_path.read_bytes() == manifest_bytes

            hardlink = fixture.output_root / "hardlink.json"
            os.link(fixture.manifest_path, hardlink)
            with pytest.raises(C2FullReplacementError, match="aliases a validated evidence inode"):
                write_full_replacement_report(finalized, hardlink)
            assert fixture.manifest_path.read_bytes() == manifest_bytes

            symlink = fixture.output_root / "symlink.json"
            symlink.symlink_to(fixture.manifest_path)
            with pytest.raises(C2FullReplacementError, match="Cannot normalize V2"):
                write_full_replacement_report(finalized, symlink)
            assert fixture.manifest_path.read_bytes() == manifest_bytes

            existing = fixture.output_root / "existing.json"
            existing.write_bytes(b"do-not-overwrite")
            with pytest.raises(C2FullReplacementError, match="already exists"):
                write_full_replacement_report(finalized, existing)
            assert existing.read_bytes() == b"do-not-overwrite"
        finally:
            finalized.evidence.close()


def test_untrusted_or_symlinked_output_parent_is_rejected_before_publication() -> None:
    with _workspace("output-parent") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        os.chmod(fixture.output_root, 0o777)
        try:
            with pytest.raises(C2FullReplacementError, match="Cannot secure V2 output"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "unsafe.json",
                )
            assert not (fixture.output_root / "unsafe.json").exists()
        finally:
            os.chmod(fixture.output_root, 0o700)
            finalized.evidence.close()

    with _workspace("output-parent-symlink") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        alias = workspace / "output-alias"
        alias.symlink_to(fixture.output_root, target_is_directory=True)
        try:
            with pytest.raises(C2FullReplacementError, match="Cannot secure V2 output"):
                write_full_replacement_report(finalized, alias / "unsafe.json")
            assert not (fixture.output_root / "unsafe.json").exists()
        finally:
            finalized.evidence.close()


def test_link_publication_is_atomic_no_replace_under_a_precheck_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("link-race") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        output = fixture.output_root / "race.json"
        original_link = os.link

        def create_competing_leaf(
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
        monkeypatch.setattr(v2_finalizer.os, "link", create_competing_leaf)
        try:
            with pytest.raises(C2FullReplacementError, match="existing output leaf"):
                write_full_replacement_report(finalized, output)
            assert output.read_bytes() == b"competing-output"
        finally:
            finalized.evidence.close()


def test_unsupported_link_and_staging_reuse_never_report_success_or_delete_reused_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("link-unsupported") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: False)
            with pytest.raises(C2FullReplacementError, match="unsupported"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "unsupported.json",
                )
        finally:
            finalized.evidence.close()

    with _workspace("staging-reuse") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        original_link = os.link
        staged_names: list[str] = []

        def replace_staging_after_link(
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
            staged_names.append(source)
            os.unlink(source, dir_fd=src_dir_fd)
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(descriptor, b"unrelated-staging-file")
            finally:
                os.close(descriptor)

        monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: True)
        monkeypatch.setattr(v2_finalizer.os, "link", replace_staging_after_link)
        try:
            with pytest.raises(C2FullReplacementError, match="staging name was replaced"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "staging-race.json",
                )
            assert staged_names
            reused = fixture.output_root / staged_names[0]
            assert reused.read_bytes() == b"unrelated-staging-file"
        finally:
            finalized.evidence.close()


def test_failed_link_never_unlinks_a_reused_staging_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("failed-link-reuse") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        staged_names: list[str] = []

        def fail_after_reusing_staging(
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
        monkeypatch.setattr(v2_finalizer.os, "link", fail_after_reusing_staging)
        try:
            with pytest.raises(C2FullReplacementError, match="no-replace publication failed"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "failed-link.json",
                )
            assert staged_names
            reused = fixture.output_root / staged_names[0]
            assert reused.read_bytes() == b"unrelated-after-failed-link"
            assert not (fixture.output_root / "failed-link.json").exists()
        finally:
            finalized.evidence.close()


def test_directory_fsync_failure_produces_no_success_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("fsync-failure") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        original_fsync = os.fsync
        call_count = 0

        def fail_parent_fsync(descriptor: int) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError(errno.EIO, "synthetic directory fsync failure")
            original_fsync(descriptor)

        monkeypatch.setattr(v2_finalizer.os, "fsync", fail_parent_fsync)
        try:
            with pytest.raises(C2FullReplacementError, match="output publication failed"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "fsync-failure.json",
                )
            assert call_count >= 2
        finally:
            finalized.evidence.close()


def test_parent_and_leaf_swaps_fail_without_evidence_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("parent-swap") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        old_evidence = workspace / "evidence-original"
        manifest_bytes = fixture.manifest_path.read_bytes()
        fixture.evidence_root.rename(old_evidence)
        fixture.evidence_root.symlink_to(workspace / "untrusted-target")
        try:
            with pytest.raises(C2FullReplacementError, match="evidence root changed"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "parent-swap.json",
                )
            assert (old_evidence / "manifest.json").read_bytes() == manifest_bytes
            assert not (fixture.output_root / "parent-swap.json").exists()
        finally:
            finalized.evidence.close()

    with _workspace("output-parent-swap") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        manifest_bytes = fixture.manifest_path.read_bytes()
        original_link = os.link
        old_output = workspace / "output-original"

        def link_then_swap_output_parent(
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
            fixture.output_root.symlink_to(workspace / "attacker-parent")

        monkeypatch.setattr(v2_finalizer, "_linkat_no_replace_supported", lambda: True)
        monkeypatch.setattr(v2_finalizer.os, "link", link_then_swap_output_parent)
        try:
            with pytest.raises(C2FullReplacementError, match="output verification failed"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "parent-swap.json",
                )
            assert fixture.manifest_path.read_bytes() == manifest_bytes
            assert (old_output / "parent-swap.json").exists()
        finally:
            finalized.evidence.close()

    monkeypatch.undo()

    with _workspace("leaf-swap") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        manifest_bytes = fixture.manifest_path.read_bytes()
        original_link = os.link

        def link_then_swap_leaf(
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
        monkeypatch.setattr(v2_finalizer.os, "link", link_then_swap_leaf)
        try:
            with pytest.raises(C2FullReplacementError, match="does not identify"):
                write_full_replacement_report(
                    finalized,
                    fixture.output_root / "leaf-swap.json",
                )
            assert fixture.manifest_path.read_bytes() == manifest_bytes
        finally:
            finalized.evidence.close()


def test_report_claims_and_strata_cannot_be_post_hoc_edited() -> None:
    with _workspace("report-edit") as workspace:
        fixture = _build_fixture(workspace)
        finalized = _prepare(fixture)
        try:
            forged = json.loads(json.dumps(finalized.report))
            forged["strata"][1]["independent_cluster_count"] = 999
            forged["final_report_hash"] = sha256_json(
                {
                    key: value
                    for key, value in forged.items()
                    if key != "final_report_hash"
                }
            )
            with pytest.raises(C2FullReplacementError, match="P-stratum results"):
                validate_synthetic_final_report_for_testing(forged, fixture.policy)
        finally:
            finalized.evidence.close()

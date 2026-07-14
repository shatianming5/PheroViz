from __future__ import annotations

import copy
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

from experiments import c2_remediation_preflight as preflight
from tests.test_experiment_support import experiment_workspace


@dataclass
class SyntheticPreflightFixture:
    plan: dict[str, Any]
    bindings: preflight.PreflightBindings
    roots: dict[str, Path]


def _jsonl(records: list[dict[str, str]]) -> bytes:
    return b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


@contextmanager
def _synthetic_fixture() -> Iterator[SyntheticPreflightFixture]:
    with experiment_workspace("c2-remediation-preflight") as workspace:
        workspace.chmod(0o700)
        frozen = workspace / "frozen"
        chunks_dir = frozen / "chunks"
        frozen.mkdir(mode=0o700)
        chunks_dir.mkdir(mode=0o700)

        universe_records = [
            {"doi": f"10.4242/c2-preflight-{ordinal:04d}"}
            for ordinal in range(1, 2_464)
        ]
        universe_path = frozen / "universe.jsonl"
        universe_path.write_bytes(_jsonl(universe_records))
        bindings: list[preflight.FrozenChunkBinding] = []
        roots: dict[str, Path] = {}
        plan_chunks: list[dict[str, Any]] = []
        first_ordinal = 1

        for chunk_number in range(1, 14):
            chunk_id = f"{chunk_number:03d}"
            input_total = 63 if chunk_id == "013" else 200
            records = universe_records[
                first_ordinal - 1 : first_ordinal - 1 + input_total
            ]
            payload = _jsonl(records)
            chunk_path = chunks_dir / f"chunk_{chunk_id}.jsonl"
            chunk_path.write_bytes(payload)
            root = workspace / f"root-{chunk_id}"
            root.mkdir(mode=0o700)
            (root / "accepted.jsonl").write_bytes(payload)
            roots[chunk_id] = root
            bindings.append(
                preflight.FrozenChunkBinding(
                    chunk_id=chunk_id,
                    input_total=input_total,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    first_global_ordinal=first_ordinal,
                    last_global_ordinal=first_ordinal + input_total - 1,
                )
            )
            role = (
                "candidate_v2"
                if chunk_id in {"009", "010", "012"}
                else (
                    "strict_path_derivative"
                    if chunk_id == "011"
                    else (
                        "transparent_63_derivative"
                        if chunk_id == "013"
                        else "legacy"
                    )
                )
            )
            plan_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "frozen_chunk_path": str(chunk_path),
                    "root": {
                        "path": str(root),
                        "role": role,
                        "source_classification_claim": None,
                    },
                }
            )
            first_ordinal += input_total

        yield SyntheticPreflightFixture(
            plan={
                "schema_version": preflight.PREFLIGHT_SCHEMA_VERSION,
                "frozen_universe_path": str(universe_path),
                "chunks": plan_chunks,
            },
            bindings=preflight.PreflightBindings(
                frozen_universe_sha256=hashlib.sha256(
                    universe_path.read_bytes()
                ).hexdigest(),
                chunks=tuple(bindings),
            ),
            roots=roots,
        )


def _chunk(report: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    return next(item for item in report["chunks"] if item["chunk_id"] == chunk_id)


def _violation_codes(chunk: dict[str, Any]) -> set[str]:
    return {item["code"] for item in chunk["violations"]}


def test_inventories_exact_frozen_partition_and_blocks_until_integrated_gates() -> None:
    with _synthetic_fixture() as fixture:
        report = preflight.run_preflight(fixture.plan, bindings=fixture.bindings)
        repeated = preflight.run_preflight(fixture.plan, bindings=fixture.bindings)

    assert report == repeated
    assert report["overall_status"] == "BLOCKED"
    assert report["execution_authorized"] is False
    assert report["frozen_universe"]["status"] == "PASS"
    assert [chunk["chunk_id"] for chunk in report["chunks"]] == list(
        preflight.CHUNK_IDS
    )
    assert [chunk["expected_input_total"] for chunk in report["chunks"]] == [
        *([200] * 12),
        63,
    ]
    assert all(
        chunk["input_inventory_status"] == "PASS" for chunk in report["chunks"]
    )
    assert _chunk(report, "001")["required_action"] == "FRESH_REMEDIATION_REQUIRED"
    assert (
        _chunk(report, "009")["required_action"]
        == "CANDIDATE_V2_VALIDATION_ONLY"
    )
    assert (
        _chunk(report, "011")["required_action"]
        == "STRICT_PATH_ONLY_DERIVATIVE_OR_REMEDIATION_REQUIRED"
    )
    assert (
        _chunk(report, "013")["required_action"]
        == "TRANSPARENT_63_INPUT_DERIVATIVE_OR_FRESH_3X63_REQUIRED"
    )
    assert all(
        len(chunk["root"]["inventory"]["inventory_sha256"]) == 64
        for chunk in report["chunks"]
    )
    assert report["gates_before_real_execution"]["source_extension"][
        "status"
    ] == "BLOCKED"
    assert report["gates_before_real_execution"]["stage_b_policy_resource"][
        "status"
    ] == "BLOCKED"


def test_rejects_padded_accepted_input_without_repartitioning_it() -> None:
    with _synthetic_fixture() as fixture:
        accepted = fixture.roots["001"] / "accepted.jsonl"
        accepted.write_bytes(
            accepted.read_bytes()
            + b'{"doi":"10.4242/c2-preflight-unbound-padding"}\n'
        )

        report = preflight.run_preflight(fixture.plan, bindings=fixture.bindings)

    chunk = _chunk(report, "001")
    assert report["overall_status"] == "BLOCKED"
    assert chunk["input_inventory_status"] == "REJECTED"
    assert "ROOT_INPUT_UNSAFE_OR_UNBOUND" in _violation_codes(chunk)
    assert "padding, repartitioning, and merging are forbidden" in chunk["root"][
        "reason"
    ]


def test_rejects_merged_chunk_root_plan() -> None:
    with _synthetic_fixture() as fixture:
        plan = copy.deepcopy(fixture.plan)
        plan["chunks"][1]["root"]["path"] = plan["chunks"][0]["root"]["path"]

        with pytest.raises(
            preflight.C2RemediationPreflightError,
            match="merging chunks is forbidden",
        ):
            preflight.run_preflight(plan, bindings=fixture.bindings)


def test_rejects_unsafe_root_and_final_or_unverified_source_claims() -> None:
    with _synthetic_fixture() as fixture:
        unsafe_root = fixture.roots["001"].parent / "unsafe-root-001"
        os.symlink(fixture.roots["001"], unsafe_root)
        plan = copy.deepcopy(fixture.plan)
        plan["chunks"][0]["root"].update(
            {
                "path": str(unsafe_root),
                "role": "final",
                "source_classification_claim": {
                    "asserted_stratum": "P2",
                },
            }
        )

        report = preflight.run_preflight(plan, bindings=fixture.bindings)

    chunk = _chunk(report, "001")
    assert chunk["root"]["status"] == "REJECTED"
    assert "ROOT_INPUT_UNSAFE_OR_UNBOUND" in _violation_codes(chunk)
    assert "LEGACY_OR_CANDIDATE_ROOT_PASSED_AS_FINAL" in _violation_codes(chunk)
    assert (
        "SOURCE_CLASSIFICATION_CLAIM_WITHOUT_STRICT_EVIDENCE"
        in _violation_codes(chunk)
    )


def test_rejects_root_swapped_during_immutable_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _synthetic_fixture() as fixture:
        root = fixture.roots["001"]
        replacement = root.parent / "replacement-root-001"
        parked = root.parent / "parked-root-001"
        replacement_payload = (root / "accepted.jsonl").read_bytes()
        original_read = preflight._read_regular_file_at
        swapped = False

        def swap_after_first_inventory_read(
            parent_fd: int,
            name: str,
            *,
            label: str,
        ) -> tuple[bytes, Any]:
            nonlocal swapped
            result = original_read(parent_fd, name, label=label)
            if label == "Evidence root accepted.jsonl" and not swapped:
                replacement.mkdir(mode=0o700)
                (replacement / "accepted.jsonl").write_bytes(replacement_payload)
                root.rename(parked)
                replacement.rename(root)
                swapped = True
            return result

        monkeypatch.setattr(
            preflight,
            "_read_regular_file_at",
            swap_after_first_inventory_read,
        )
        report = preflight.run_preflight(fixture.plan, bindings=fixture.bindings)

    chunk = _chunk(report, "001")
    assert swapped is True
    assert chunk["root"]["status"] == "REJECTED"
    assert "ROOT_INPUT_UNSAFE_OR_UNBOUND" in _violation_codes(chunk)
    assert "changed during immutable inventory" in chunk["root"]["reason"]

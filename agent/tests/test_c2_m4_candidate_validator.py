from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments import c2_m4_candidate_validator as validator
from experiments.c2_m4_candidate_validation_bindings import (
    M4ArtifactBindings,
    M4CandidateBinding,
    M4CandidateValidationBindings,
    M4SnapshotBinding,
    load_m4_candidate_validation_bindings,
)
from experiments.models import canonical_json
from tests.test_experiment_support import experiment_workspace


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in rows)
    )


def _seal(value: dict[str, Any], field: str) -> None:
    value[field] = _sha256(
        canonical_json(
            {key: item for key, item in value.items() if key != field}
        ).encode("utf-8")
    )


def _source_records() -> list[dict[str, Any]]:
    return [
        {
            "article_url": f"https://example.test/article-{index}",
            "doi": f"10.9999/candidate-{index}",
            "policy_accepted": True,
            "download_eligible": True,
            "journal_allowed": True,
            "require_cc_by": True,
            "reject_reasons": [],
        }
        for index in range(1, 3)
    ]


def _provenance(doi: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "doi": doi,
        "download_status": "empty",
        "source_data_origin": "not_present",
        "files": [],
        "rejection_reasons": ["no-source-data"],
    }


def _snapshot_rows(root: Path) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        if Path(directory) == root:
            names[:] = [name for name in names if name != ".pipeline_worktree"]
        for filename in files:
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            rows.append([relative, len(payload), _sha256(payload)])
    return sorted(rows)


def _write_native_inventory(root: Path, profile: str) -> None:
    if profile == "RERUN2_FORENSIC_ACQUISITION_V2":
        excluded = {"root_inventory.json", "root_inventory.sha256"}
        rows = [row for row in _snapshot_rows(root) if row[0] not in excluded]
        inventory: dict[str, Any] = {
            "schema_version": "1.0",
            "inventory_type": "synthetic-rerun2-root-inventory",
            "entries": [
                {"path": path, "size_bytes": size, "sha256": digest}
                for path, size, digest in rows
            ],
        }
        _seal(inventory, "inventory_hash")
        path = root / "root_inventory.json"
        _write_json(path, inventory)
        (root / "root_inventory.sha256").write_text(
            f"{_sha256(path.read_bytes())}  root_inventory.json\n",
            encoding="ascii",
        )
        return

    rows = [
        row
        for row in _snapshot_rows(root)
        if row[0] != "control/root_inventory.json"
        and not row[0].startswith("sealed_report_v1/")
    ]
    inventory = {
        "schema_version": "1.0",
        "root_name": root.name,
        "excludes": [
            ".pipeline_worktree/",
            "control/root_inventory.json",
            "sealed_report_v1/",
        ],
        "files": [
            {"path": path, "bytes": size, "sha256": digest}
            for path, size, digest in rows
        ],
        "total_bytes": sum(size for _, size, _ in rows),
    }
    _seal(inventory, "inventory_hash")
    _write_json(root / "control/root_inventory.json", inventory)


def _build_root(
    root_base: Path,
    *,
    chunk_id: str,
    profile: str,
) -> M4CandidateBinding:
    root_name = f"ccby_sr_npj_chunk{chunk_id}_rerun2_clean_ca98442"
    root = root_base / root_name
    root.mkdir(mode=0o700)
    (root / ".pipeline_worktree").mkdir(mode=0o700)
    records = _source_records()
    accepted_payload = b"".join(
        canonical_json(record).encode("utf-8") + b"\n" for record in records
    )
    (root / "accepted.jsonl").write_bytes(accepted_payload)
    accepted_sha256 = _sha256(accepted_payload)
    article_ids = [f"article-{index}" for index in range(1, 3)]
    dois = [str(record["doi"]) for record in records]
    statuses_by_attempt = {
        "initial": ("no-source-data", "no-figures"),
        "retry1": ("no-figures", "no-source-data"),
        "retry2": ("no-source-data", "no-source-data"),
    }

    provenance_payloads: dict[str, bytes] = {}
    for article_id, doi in zip(article_ids, dois, strict=True):
        path = root / f"content/_provenance/{article_id}.json"
        _write_json(path, _provenance(doi))
        provenance_payloads[f"content/_provenance/{article_id}.json"] = (
            path.read_bytes()
        )

    attempt_entries: list[dict[str, Any]] = []
    rows_by_attempt: dict[str, list[dict[str, Any]]] = {}
    for attempt in validator.ATTEMPTS:
        prefix = root / "control" / attempt
        prefix.mkdir(parents=True)
        (prefix / "accepted.jsonl").write_bytes(accepted_payload)
        rows: list[dict[str, Any]] = []
        for index, (record, article_id, doi, status) in enumerate(
            zip(
                records,
                article_ids,
                dois,
                statuses_by_attempt[attempt],
                strict=True,
            ),
            start=1,
        ):
            row: dict[str, Any] = {
                "article_id": article_id,
                "attempt": attempt,
                "doi": doi,
                "input_index_1based": index,
                "processed": True,
                "status": status,
            }
            if profile == "RERUN2_FORENSIC_ACQUISITION_V2":
                snapshot_path = (
                    f"control/{attempt}/provenance/{article_id}.json"
                )
                provenance_path = root / snapshot_path
                _write_json(provenance_path, _provenance(doi))
                provenance_payload = provenance_path.read_bytes()
                row.update(
                    {
                        "asset_snapshot_bindings": [],
                        "input_record_sha256": _sha256(
                            canonical_json(record).encode("utf-8")
                        ),
                        "provenance": {
                            "captured_live_path": (
                                f"content/_provenance/{article_id}.json"
                            ),
                            "download_status": "empty",
                            "files_declared": 0,
                            "snapshot_path": snapshot_path,
                            "snapshot_sha256": _sha256(provenance_payload),
                        },
                    }
                )
            rows.append(row)
        rows_by_attempt[attempt] = rows
        ledger_path = prefix / "attempt_ledger.jsonl"
        _write_jsonl(ledger_path, rows)
        (prefix / "processed.txt").write_text(
            "".join(f"{article_id}\n" for article_id in reversed(article_ids)),
            encoding="utf-8",
        )
        (prefix / "_skipped.txt").write_text(
            "".join(
                f"{article_id}\t{row['status']}\n"
                for article_id, row in reversed(list(zip(article_ids, rows)))
            ),
            encoding="utf-8",
        )
        (prefix / "postfetch.log").write_text(
            f"synthetic {attempt}\n",
            encoding="utf-8",
        )
        (prefix / "postfetch.exit").write_text("0\n", encoding="ascii")
        status_counts = dict(
            sorted(Counter(str(row["status"]) for row in rows).items())
        )
        summary: dict[str, Any] = {
            "schema_version": (
                "2.0"
                if profile == "RERUN2_FORENSIC_ACQUISITION_V2"
                else "1.0"
            ),
            "attempt": attempt,
            "coverage_exact": True,
            "input_sha256": accepted_sha256,
            "ledger_sha256": _sha256(ledger_path.read_bytes()),
            "processed_sha256": _sha256((prefix / "processed.txt").read_bytes()),
            "skipped_sha256": _sha256((prefix / "_skipped.txt").read_bytes()),
            "postfetch_log_sha256": _sha256(
                (prefix / "postfetch.log").read_bytes()
            ),
            "postfetch_exit": 0,
            "processed_records": 2,
            "processed_unique": 2,
            "skipped_records": 2,
            "statuses": status_counts,
        }
        if profile == "RERUN2_FORENSIC_ACQUISITION_V2":
            summary.update(
                {
                    "ledger_records": 2,
                    "input_records": 2,
                    "input_unique_dois": 2,
                    "asset_snapshot_files": 0,
                    "provenance_snapshots": 2,
                }
            )
        _seal(summary, "summary_hash")
        summary_path = prefix / "attempt_summary.json"
        _write_json(summary_path, summary)
        entry = {
            "attempt": attempt,
            "coverage_exact": True,
            "coverage_records": 2,
            "input_sha256": accepted_sha256,
            "ledger_sha256": _sha256(ledger_path.read_bytes()),
            "postfetch_exit": 0,
            "postfetch_log_sha256": _sha256(
                (prefix / "postfetch.log").read_bytes()
            ),
            "processed_records": 2,
            "processed_unique": 2,
            "statuses": status_counts,
            "summary_file_sha256": _sha256(summary_path.read_bytes()),
            "summary_hash": summary["summary_hash"],
        }
        if profile == "RERUN2_FORENSIC_ACQUISITION_V2":
            entry.update(
                {
                    "asset_snapshot_files": 0,
                    "provenance_snapshots": 2,
                }
            )
        else:
            entry.update(
                {
                    "input_records": 2,
                    "processed_sha256": summary["processed_sha256"],
                    "skipped_sha256": summary["skipped_sha256"],
                }
            )
        attempt_entries.append(entry)

    binding_contract: dict[str, Any] = {
        "schema_version": "2.0",
        "batch": root_name,
        "chunk": int(chunk_id),
        "chunk_records": 2,
        "chunk_start_index_1based": 1,
        "chunk_end_index_1based": 2,
        "attempts": list(validator.ATTEMPTS),
        "attempt_coverage_required_each": 2,
        "code_commit": validator.FROZEN_CODE_COMMIT,
        "code_dirty": False,
        "exact_byte_copy": True,
        "execution_accepted_sha256": accepted_sha256,
        "source_chunk_sha256": accepted_sha256,
        "source_universe_sha256": "a" * 64,
        "source_freeze_summary_sha256": (
            validator.FROZEN_FREEZE_SUMMARY_SHA256
        ),
        "source_freeze_summary_hash": validator.FROZEN_FREEZE_SUMMARY_HASH,
        "outcome_independent": True,
        "review_or_experiment_outcomes_used": False,
    }
    _seal(binding_contract, "summary_hash")
    binding_path = root / "control/pre_download_binding.json"
    _write_json(binding_path, binding_contract)
    (root / "control/pre_download_binding.sha256").write_text(
        f"{_sha256(binding_path.read_bytes())}  pre_download_binding.json\n",
        encoding="ascii",
    )

    coverage = {
        "schema_version": "2.0",
        "all_coverage_exact": True,
        "all_inputs_exact_byte_equal": True,
        "attempt_count": 3,
        "attempts": attempt_entries,
        "records_per_attempt": 2,
        "total_ledger_records": 6,
    }
    _seal(coverage, "summary_hash")
    _write_json(root / "control/attempt_coverage.json", coverage)

    terminal_rows: list[dict[str, Any]] = []
    for index, (article_id, doi) in enumerate(
        zip(article_ids, dois, strict=True),
        start=1,
    ):
        expected_statuses = {
            attempt: rows_by_attempt[attempt][index - 1]["status"]
            for attempt in validator.ATTEMPTS
        }
        if profile == "RERUN2_FORENSIC_ACQUISITION_V2":
            attempts = []
            for attempt in validator.ATTEMPTS:
                provenance_path = (
                    f"control/{attempt}/provenance/{article_id}.json"
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "ledger_path": (
                            f"control/{attempt}/attempt_ledger.jsonl"
                        ),
                        "ledger_record_index_1based": index,
                        "provenance_snapshot_path": provenance_path,
                        "provenance_snapshot_sha256": _sha256(
                            (root / provenance_path).read_bytes()
                        ),
                        "status": expected_statuses[attempt],
                    }
                )
            retry2_path = f"control/retry2/provenance/{article_id}.json"
            terminal = {
                "article_id": article_id,
                "attempts": attempts,
                "doi": doi,
                "input_index_1based": index,
                "terminal_content_artifacts": [],
                "terminal_content_directory": None,
                "terminal_provenance_path": retry2_path,
                "terminal_provenance_sha256": _sha256(
                    (root / retry2_path).read_bytes()
                ),
                "terminal_selection_rule": "retry2 is fixed",
                "terminal_status": expected_statuses["retry2"],
            }
        else:
            provenance_path = f"content/_provenance/{article_id}.json"
            terminal = {
                "article_id": article_id,
                "doi": doi,
                "provenance_path": provenance_path,
                "provenance_sha256": _sha256(
                    provenance_payloads[provenance_path]
                ),
                "rounds": expected_statuses,
                "terminal_status": expected_statuses["retry2"],
            }
        terminal_rows.append(terminal)
    terminal_path = root / "control/terminal_outcomes.jsonl"
    _write_jsonl(terminal_path, terminal_rows)

    (root / "cases_v1").mkdir()
    (root / "cases_v1/candidates.jsonl").write_bytes(b"")
    (root / "proposals_v1").mkdir()
    (root / "proposals_v1/proposed.jsonl").write_bytes(b"")
    if profile == "RERUN2_FORENSIC_ACQUISITION_V2":
        sealed_report = {
            "schema_version": "1.0",
            "report_type": "c2_rerun2_sealed_root_report",
            "result": "PASS",
            "root_id": root_name,
            "source": {
                "records": 2,
                "unique_dois": 2,
                "source_chunk_sha256": accepted_sha256,
            },
            "formal_attempts": {"total_ledger_records": 6},
            "terminal": {
                "terminal_outcomes_sha256": _sha256(terminal_path.read_bytes())
            },
        }
    else:
        sealed_report = {
            "sealed_report_version": "2.0",
            "status": "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES",
            "frozen_input": {
                "chunk": 12,
                "records": 2,
                "sha256": accepted_sha256,
            },
            "attempt_evidence": {"attempts": attempt_entries},
            "p_strata": {"legacy": "ignored"},
        }
    _seal(sealed_report, "report_hash")
    sealed_path = root / "sealed_report_v1/sealed_report.json"
    _write_json(sealed_path, sealed_report)
    (root / "sealed_report_v1/sealed_report.sha256").write_text(
        f"{_sha256(sealed_path.read_bytes())}  sealed_report.json\n",
        encoding="ascii",
    )
    _write_native_inventory(root, profile)

    with validator._DescriptorSnapshotter(root) as snapshotter:
        snapshot = snapshotter.snapshot()
    inventory_path = (
        "root_inventory.json"
        if profile == "RERUN2_FORENSIC_ACQUISITION_V2"
        else "control/root_inventory.json"
    )
    return M4CandidateBinding(
        chunk_id=chunk_id,
        root_name=root_name,
        format_profile=profile,
        required_action="CANDIDATE_V2_VALIDATION_ONLY",
        first_global_ordinal=1,
        last_global_ordinal=2,
        input_total=2,
        accepted_bytes=len(accepted_payload),
        accepted_sha256=accepted_sha256,
        snapshot=M4SnapshotBinding(
            file_count=snapshot.file_count,
            total_bytes=snapshot.total_bytes,
            canonical_bytes=snapshot.canonical_bytes,
            sha256=snapshot.sha256,
        ),
        artifacts=M4ArtifactBindings(
            inventory_path=inventory_path,
            inventory_sha256=snapshot.digest(inventory_path, "inventory"),
            pre_download_binding_sha256=snapshot.digest(
                "control/pre_download_binding.json",
                "binding",
            ),
            attempt_coverage_sha256=snapshot.digest(
                "control/attempt_coverage.json",
                "coverage",
            ),
            terminal_outcomes_sha256=snapshot.digest(
                "control/terminal_outcomes.jsonl",
                "terminal",
            ),
            sealed_report_sha256=snapshot.digest(
                "sealed_report_v1/sealed_report.json",
                "sealed report",
            ),
            candidate_surface_sha256=validator.EMPTY_SHA256,
        ),
    )


def _fake_authorization() -> SimpleNamespace:
    report = {
        "authorization_mode": "OWNER_AUTHORIZED_NON_INDEPENDENT",
        "authorization_id_sha256": "auth",
        "capabilities": [validator.EXPECTED_CAPABILITY],
        "frozen_universe_sha256": "a" * 64,
        "independent_verification": False,
        "execution_authorized": True,
        "admission_authorized": False,
        "publication_authorized": False,
    }
    return SimpleNamespace(
        authorization_id_sha256="auth",
        capabilities=(validator.EXPECTED_CAPABILITY,),
        frozen_universe_sha256="a" * 64,
        to_report_dict=lambda: report,
    )


def _install_fake_fixed_state(
    monkeypatch: pytest.MonkeyPatch,
    bindings: tuple[M4CandidateBinding, ...],
) -> None:
    authorization = _fake_authorization()
    policy_chunks = {
        binding.chunk_id: SimpleNamespace(
            required_action=binding.required_action,
            first_global_ordinal=binding.first_global_ordinal,
            last_global_ordinal=binding.last_global_ordinal,
            input_total=binding.input_total,
            chunk_file_sha256=binding.accepted_sha256,
        )
        for binding in bindings
    }
    policy = SimpleNamespace(
        authorization_id_sha256="auth",
        policy_id_sha256="policy",
        resource_sha256="policy-resource",
        universe_file_sha256="a" * 64,
        chunk=lambda chunk_id: policy_chunks[chunk_id],
    )
    fixed = M4CandidateValidationBindings(
        binding_id_sha256="binding",
        authorization_id_sha256="auth",
        owner_policy_id_sha256="policy",
        owner_policy_resource_sha256="policy-resource",
        frozen_universe_sha256="a" * 64,
        snapshot_algorithm=validator.SNAPSHOT_ALGORITHM,
        excluded_tree=".pipeline_worktree/",
        candidates=bindings,
        resource_sha256="binding-resource",
    )
    monkeypatch.setattr(
        validator,
        "require_owner_authorized_c2_execution",
        lambda: authorization,
    )
    monkeypatch.setattr(
        validator,
        "load_owner_remediation_execution_policy",
        lambda: policy,
    )
    monkeypatch.setattr(
        validator,
        "load_m4_candidate_validation_bindings",
        lambda: fixed,
    )


def test_fixed_binding_resource_compiles_exact_candidate_roster() -> None:
    bindings = load_m4_candidate_validation_bindings()
    assert tuple(binding.chunk_id for binding in bindings.candidates) == (
        "009",
        "010",
        "012",
    )
    assert bindings.candidates[2].format_profile == (
        "LEGACY_COMPACT_ACQUISITION_ONLY"
    )
    assert bindings.candidates[0].snapshot.sha256 == (
        "aff27489af5869b750ad900637b544d60c5ba5ad24bfeecd5ed88ce8de87ab22"
    )


@pytest.mark.parametrize(
    ("chunk_id", "profile"),
    [
        ("009", "RERUN2_FORENSIC_ACQUISITION_V2"),
        ("012", "LEGACY_COMPACT_ACQUISITION_ONLY"),
    ],
)
def test_candidate_profiles_validate_without_mutating_roots(
    monkeypatch: pytest.MonkeyPatch,
    chunk_id: str,
    profile: str,
) -> None:
    with experiment_workspace(f"c2-m4-{chunk_id}") as workspace:
        root_base = workspace / "outputs"
        root_base.mkdir(mode=0o700)
        binding = _build_root(
            root_base,
            chunk_id=chunk_id,
            profile=profile,
        )
        _install_fake_fixed_state(monkeypatch, (binding,))
        before = _snapshot_rows(root_base / binding.root_name)
        first = validator.validate_fixed_m4_candidates(root_base)
        second = validator.validate_fixed_m4_candidates(root_base)
        after = _snapshot_rows(root_base / binding.root_name)

    assert first == second
    assert before == after
    assert first["execution_authorized"] is True
    assert first["admission_authorized"] is False
    assert first["publication_authorized"] is False
    assert first["v2_format_validated"] is False
    assert not {
        "status",
        "claim_status",
        "final_report_hash",
        "admitted",
    }.intersection(first)
    candidate = first["candidates"][0]
    assert candidate["format_profile"] == profile
    assert candidate["source_absence"]["source_bearing_attempt_records"] == 0
    assert candidate["v2_admission_eligible"] is False
    expected_hash = _sha256(
        canonical_json(
            {
                key: value
                for key, value in first.items()
                if key != "report_hash"
            }
        ).encode("utf-8")
    )
    assert first["report_hash"] == expected_hash


def test_source_path_is_rejected_even_when_snapshot_pin_is_updated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-m4-source-injection") as workspace:
        root_base = workspace / "outputs"
        root_base.mkdir(mode=0o700)
        binding = _build_root(
            root_base,
            chunk_id="009",
            profile="RERUN2_FORENSIC_ACQUISITION_V2",
        )
        root = root_base / binding.root_name
        source = root / "content/_sources/article-1/source.json"
        source.parent.mkdir(parents=True)
        source.write_text('{"source":true}\n', encoding="utf-8")
        _write_native_inventory(
            root,
            "RERUN2_FORENSIC_ACQUISITION_V2",
        )
        with validator._DescriptorSnapshotter(root) as snapshotter:
            snapshot = snapshotter.snapshot()
        binding = replace(
            binding,
            snapshot=M4SnapshotBinding(
                file_count=snapshot.file_count,
                total_bytes=snapshot.total_bytes,
                canonical_bytes=snapshot.canonical_bytes,
                sha256=snapshot.sha256,
            ),
            artifacts=replace(
                binding.artifacts,
                inventory_sha256=snapshot.digest(
                    "root_inventory.json",
                    "inventory",
                ),
            ),
        )
        _install_fake_fixed_state(monkeypatch, (binding,))
        with pytest.raises(
            validator.C2M4CandidateValidationError,
            match="source-bearing evidence paths",
        ):
            validator.validate_fixed_m4_candidates(root_base)


def test_symlink_artifact_is_rejected_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-m4-symlink") as workspace:
        root_base = workspace / "outputs"
        root_base.mkdir(mode=0o700)
        binding = _build_root(
            root_base,
            chunk_id="012",
            profile="LEGACY_COMPACT_ACQUISITION_ONLY",
        )
        root = root_base / binding.root_name
        terminal = root / "control/terminal_outcomes.jsonl"
        terminal.unlink()
        os.symlink(root / "accepted.jsonl", terminal)
        _install_fake_fixed_state(monkeypatch, (binding,))
        with pytest.raises(
            validator.C2M4CandidateValidationError,
            match="symlink",
        ):
            validator.validate_fixed_m4_candidates(root_base)


@pytest.mark.parametrize("artifact_type", ["hardlink", "fifo"])
def test_non_private_regular_artifacts_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: str,
) -> None:
    with experiment_workspace(f"c2-m4-{artifact_type}") as workspace:
        root_base = workspace / "outputs"
        root_base.mkdir(mode=0o700)
        binding = _build_root(
            root_base,
            chunk_id="009",
            profile="RERUN2_FORENSIC_ACQUISITION_V2",
        )
        root = root_base / binding.root_name
        rogue = root / "rogue"
        if artifact_type == "hardlink":
            os.link(root / "accepted.jsonl", rogue)
            match = "hard links"
        else:
            os.mkfifo(rogue)
            match = "special artifact"
        _install_fake_fixed_state(monkeypatch, (binding,))
        with pytest.raises(
            validator.C2M4CandidateValidationError,
            match=match,
        ):
            validator.validate_fixed_m4_candidates(root_base)


def test_terminal_reordering_is_rejected_after_repinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-m4-terminal-order") as workspace:
        root_base = workspace / "outputs"
        root_base.mkdir(mode=0o700)
        binding = _build_root(
            root_base,
            chunk_id="012",
            profile="LEGACY_COMPACT_ACQUISITION_ONLY",
        )
        root = root_base / binding.root_name
        terminal_path = root / "control/terminal_outcomes.jsonl"
        rows = validator._jsonl_objects(
            terminal_path.read_bytes(),
            "test terminal",
        )
        _write_jsonl(terminal_path, list(reversed(rows)))
        with validator._DescriptorSnapshotter(root) as snapshotter:
            snapshot = snapshotter.snapshot()
        binding = replace(
            binding,
            snapshot=M4SnapshotBinding(
                file_count=snapshot.file_count,
                total_bytes=snapshot.total_bytes,
                canonical_bytes=snapshot.canonical_bytes,
                sha256=snapshot.sha256,
            ),
            artifacts=replace(
                binding.artifacts,
                terminal_outcomes_sha256=snapshot.digest(
                    "control/terminal_outcomes.jsonl",
                    "terminal",
                ),
            ),
        )
        _install_fake_fixed_state(monkeypatch, (binding,))
        with pytest.raises(
            validator.C2M4CandidateValidationError,
            match="terminal DOI order differs",
        ):
            validator.validate_fixed_m4_candidates(root_base)


def test_legacy_profile_rejects_v2_ledger_relabeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-m4-legacy-relabel") as workspace:
        root_base = workspace / "outputs"
        root_base.mkdir(mode=0o700)
        binding = _build_root(
            root_base,
            chunk_id="012",
            profile="LEGACY_COMPACT_ACQUISITION_ONLY",
        )
        root = root_base / binding.root_name
        ledger = root / "control/initial/attempt_ledger.jsonl"
        rows = [
            {
                **row,
                "asset_snapshot_bindings": [],
            }
            for row in validator._jsonl_objects(
                ledger.read_bytes(),
                "test ledger",
            )
        ]
        _write_jsonl(ledger, rows)
        with validator._DescriptorSnapshotter(root) as snapshotter:
            snapshot = snapshotter.snapshot()
        binding = replace(
            binding,
            snapshot=M4SnapshotBinding(
                file_count=snapshot.file_count,
                total_bytes=snapshot.total_bytes,
                canonical_bytes=snapshot.canonical_bytes,
                sha256=snapshot.sha256,
            ),
        )
        _install_fake_fixed_state(monkeypatch, (binding,))
        with pytest.raises(
            validator.C2M4CandidateValidationError,
            match="ledger row shape differs",
        ):
            validator.validate_fixed_m4_candidates(root_base)


def test_owner_policy_action_cannot_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-m4-policy-action") as workspace:
        root_base = workspace / "outputs"
        root_base.mkdir(mode=0o700)
        binding = _build_root(
            root_base,
            chunk_id="009",
            profile="RERUN2_FORENSIC_ACQUISITION_V2",
        )
        _install_fake_fixed_state(monkeypatch, (binding,))
        bad_binding = replace(
            binding,
            required_action="FRESH_REMEDIATION_REQUIRED",
        )
        fixed = validator.load_m4_candidate_validation_bindings()
        monkeypatch.setattr(
            validator,
            "load_m4_candidate_validation_bindings",
            lambda: replace(fixed, candidates=(bad_binding,)),
        )
        with pytest.raises(
            validator.C2M4CandidateValidationError,
            match="differs from owner policy",
        ):
            validator.validate_fixed_m4_candidates(root_base)


def test_missing_owner_capability_blocks_before_root_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = _fake_authorization()
    authorization.capabilities = ()
    monkeypatch.setattr(
        validator,
        "require_owner_authorized_c2_execution",
        lambda: authorization,
    )
    with pytest.raises(
        validator.C2M4CandidateValidationError,
        match="lacks C2 candidate-validation capability",
    ):
        validator.validate_fixed_m4_candidates(Path("/not/read"))

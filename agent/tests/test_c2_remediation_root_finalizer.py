from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from experiments import c2_remediation_root_finalizer as finalizer
from experiments import cli
from experiments.models import sha256_file
from tests.test_experiment_support import experiment_workspace


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = _sha256(
        _canonical_json({key: item for key, item in value.items() if key != field})
    )
    return value


def _records() -> list[dict[str, str]]:
    return [
        {
            "doi": f"10.9999/c2-{index}",
            "article_url": f"https://example.test/article-{index}",
            "policy_accepted": True,
            "download_eligible": True,
            "journal_allowed": True,
            "require_cc_by": True,
            "reject_reasons": [],
            "license": {
                "license_id": "CC-BY-4.0",
                "content_version": "vor",
                "normalized_url": "https://creativecommons.org/licenses/by/4.0/",
            },
        }
        for index in range(1, 2464)
    ]


def _status_for(index: int, attempt_index: int) -> str:
    if (index + attempt_index) % 4 == 0:
        return "downloaded"
    return ("no-source-data", "no-figures", "fetch-error")[
        (index + attempt_index) % 3
    ]


def _make_fixture(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunk_id: str,
) -> dict[str, Path]:
    workspace.chmod(0o700)
    frozen_root = workspace / "frozen"
    frozen_root.mkdir(mode=0o700)
    universe_records = _records()
    universe_bytes = b"".join(
        _canonical_json(record) + b"\n" for record in universe_records
    )
    universe = frozen_root / "frozen_universe.jsonl"
    universe.write_bytes(universe_bytes)

    partitions = {
        frozen_chunk_id: finalizer.FrozenPartition(
            chunk_id=frozen_chunk_id,
            records=partition.records,
            source_sha256=_sha256(
                b"".join(
                    _canonical_json(record) + b"\n"
                    for record in universe_records[
                        partition.start_index_1based
                        - 1 : partition.end_index_1based
                    ]
                )
            ),
            start_index_1based=partition.start_index_1based,
            end_index_1based=partition.end_index_1based,
        )
        for frozen_chunk_id, partition in finalizer.FROZEN_PARTITIONS.items()
    }
    monkeypatch.setattr(finalizer, "FROZEN_PARTITIONS", partitions)
    partition = partitions[chunk_id]
    source_records = universe_records[
        partition.start_index_1based - 1 : partition.end_index_1based
    ]
    source_bytes = b"".join(
        _canonical_json(record) + b"\n" for record in source_records
    )
    source_chunk = frozen_root / f"chunk_{chunk_id}.jsonl"
    source_chunk.write_bytes(source_bytes)

    chunk_summary = [
        {
            "chunk": index,
            "records": partitions[f"{index:03d}"].records,
            "sha256": partitions[f"{index:03d}"].source_sha256,
            "start_index_1based": partitions[f"{index:03d}"].start_index_1based,
            "end_index_1based": partitions[f"{index:03d}"].end_index_1based,
        }
        for index in range(1, 14)
    ]
    summary = {
        "universe_sha256": _sha256(universe_bytes),
        "universe_records": 2463,
        "chunks": chunk_summary,
    }
    summary["summary_hash"] = _sha256(_canonical_json(summary))
    freeze_summary = frozen_root / "freeze_summary.json"
    _write_json(freeze_summary, summary)
    monkeypatch.setattr(finalizer, "FROZEN_UNIVERSE_SHA256", _sha256(universe_bytes))
    monkeypatch.setattr(
        finalizer,
        "FROZEN_FREEZE_SUMMARY_SHA256",
        sha256_file(freeze_summary),
    )
    monkeypatch.setattr(
        finalizer,
        "FROZEN_FREEZE_SUMMARY_HASH",
        str(summary["summary_hash"]),
    )
    monkeypatch.setattr(
        finalizer,
        "_verify_worktree",
        lambda _: {
            "commit": finalizer.FROZEN_CODE_COMMIT,
            "tree": "synthetic-clean-tree",
            "dirty": False,
        },
    )

    raw_root = workspace / "raw"
    raw_root.mkdir(mode=0o700)
    (raw_root / "accepted.jsonl").write_bytes(source_bytes)
    config = {
        "schema_version": "synthetic-v2",
        "arguments": {
            "require_cc_by": True,
            "sort": "input",
            "max_figs": 12,
            "max_empty_figs": 2,
            "sleep_seconds": 1.0,
            "timeout_seconds": 300,
            "max_retries": 3,
            "workers": 1,
        },
        "attempts": list(finalizer.ATTEMPTS),
        "no_model_calls": True,
        "no_early_stop": True,
        "outcome_independent": True,
    }
    _seal(config, "config_hash")
    _write_json(raw_root / "control/acquisition_config.json", config)
    binding = {
        "schema_version": "synthetic-v2",
        "batch": "synthetic-pre-download-root",
        "chunk": int(chunk_id),
        "chunk_records": partition.records,
        "chunk_start_index_1based": partition.start_index_1based,
        "chunk_end_index_1based": partition.end_index_1based,
        "attempts": list(finalizer.ATTEMPTS),
        "attempt_coverage_required_each": partition.records,
        "code_commit": finalizer.FROZEN_CODE_COMMIT,
        "code_dirty": False,
        "acquisition_config_hash": config["config_hash"],
        "exact_byte_copy": True,
        "execution_accepted_sha256": partition.source_sha256,
        "outcome_independent": True,
        "review_or_experiment_outcomes_used": False,
        "selection_rule": (
            "execute every frozen chunk record in initial and both fixed retries "
            "regardless of every preceding outcome"
        ),
        "source_chunk_sha256": partition.source_sha256,
        "source_universe_sha256": finalizer.FROZEN_UNIVERSE_SHA256,
        "source_freeze_summary_sha256": finalizer.FROZEN_FREEZE_SUMMARY_SHA256,
        "source_freeze_summary_hash": finalizer.FROZEN_FREEZE_SUMMARY_HASH,
    }
    _seal(binding, "summary_hash")
    binding_path = raw_root / "control/pre_download_binding.json"
    _write_json(binding_path, binding)
    (raw_root / "control/pre_download_binding.sha256").write_text(
        f"{sha256_file(binding_path)}  pre_download_binding.json\n",
        encoding="utf-8",
    )
    execution_evidence = {
        "schema_version": "c2-remediation-execution-evidence-v1",
        "status": "PASS",
        "code_commit": finalizer.FROZEN_CODE_COMMIT,
        "code_dirty": False,
        "secret_scan_status": "CLEAN",
        "concurrency": {
            "status": "PASS",
            "max_fresh_roots": 1,
            "fresh_roots_active": 1,
            "retained_active_pids": [],
        },
        "tests": [
            {
                "command": "python -m pytest -q tests/test_c2_remediation_root_finalizer.py",
                "exit_code": 0,
                "output_sha256": "0" * 64,
            }
        ],
    }
    _seal(execution_evidence, "evidence_hash")
    _write_json(raw_root / "control/execution_evidence.json", execution_evidence)

    final_statuses: dict[str, str] = {}
    for attempt_index, attempt in enumerate(finalizer.ATTEMPTS):
        attempt_dir = raw_root / "control" / attempt
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "accepted.jsonl").write_bytes(source_bytes)
        statuses: dict[str, str] = {}
        processed: list[str] = []
        skipped: list[str] = []
        for ordinal, record in enumerate(source_records, start=1):
            article_id = record["article_url"].rsplit("/", 1)[-1]
            status = _status_for(ordinal, attempt_index)
            statuses[article_id] = status
            if status == "downloaded":
                processed.append(article_id)
            else:
                skipped.append(f"{article_id}\t{status}")
        (attempt_dir / "processed.txt").write_text(
            "\n".join(processed) + ("\n" if processed else ""),
            encoding="utf-8",
        )
        (attempt_dir / "_skipped.txt").write_text(
            "\n".join(skipped) + ("\n" if skipped else ""),
            encoding="utf-8",
        )
        (attempt_dir / "postfetch.log").write_text(
            f"synthetic {attempt}\n",
            encoding="utf-8",
        )
        (attempt_dir / "postfetch.exit").write_text("0\n", encoding="utf-8")
        hashes = {
            "input_sha256": sha256_file(attempt_dir / "accepted.jsonl"),
            "config_hash": str(config["config_hash"]),
            "processed_sha256": sha256_file(attempt_dir / "processed.txt"),
            "skipped_sha256": sha256_file(attempt_dir / "_skipped.txt"),
            "postfetch_log_sha256": sha256_file(attempt_dir / "postfetch.log"),
            "postfetch_exit_sha256": sha256_file(attempt_dir / "postfetch.exit"),
            "source_chunk_sha256": partition.source_sha256,
            "source_universe_sha256": finalizer.FROZEN_UNIVERSE_SHA256,
            "source_freeze_summary_sha256": finalizer.FROZEN_FREEZE_SUMMARY_SHA256,
            "source_freeze_summary_hash": finalizer.FROZEN_FREEZE_SUMMARY_HASH,
        }
        receipt = {
            "schema_version": "c2-v2-raw-attempt-receipt-v1",
            "attempt": attempt,
            "attempt_index": attempt_index,
            "start_monotonic_ns": 1_000 * attempt_index,
            "end_monotonic_ns": 1_000 * attempt_index + 500,
            **hashes,
        }
        _seal(receipt, "receipt_hash")
        _write_json(attempt_dir / "raw_attempt_receipt.json", receipt)
        events = [
            {
                "attempt": attempt,
                "attempt_index": attempt_index,
                "article_id": record["article_url"].rsplit("/", 1)[-1],
                "doi": record["doi"],
                "input_ordinal": ordinal,
                "event_monotonic_ns": attempt_index * 1_000 + ordinal,
                "processed": statuses[
                    record["article_url"].rsplit("/", 1)[-1]
                ]
                == "downloaded",
                "status": statuses[record["article_url"].rsplit("/", 1)[-1]],
            }
            for ordinal, record in enumerate(source_records, start=1)
        ]
        (attempt_dir / "raw_attempt_events.jsonl").write_bytes(
            b"".join(_canonical_json(event) + b"\n" for event in events)
        )
        if attempt == "retry2":
            final_statuses = statuses

    provenance_dir = raw_root / "content" / "_provenance"
    provenance_dir.mkdir(parents=True)
    for record in source_records:
        article_id = record["article_url"].rsplit("/", 1)[-1]
        status = final_statuses[article_id]
        provenance: dict[str, Any] = {
            "doi": record["doi"],
            "rejection_reasons": [] if status == "downloaded" else [status],
        }
        if status == "downloaded":
            provenance["download_status"] = "downloaded"
        _write_json(provenance_dir / f"{article_id}.json", provenance)

    target_parent = workspace / "output"
    target_parent.mkdir(mode=0o700)
    target_parent.chmod(0o700)
    protected_root = target_parent / "synthetic-protected-old-root"
    (protected_root / "sealed_report_v1").mkdir(parents=True)
    (protected_root / "accepted.jsonl").write_bytes(source_bytes)
    protected_report = protected_root / "sealed_report_v1/sealed_report.json"
    protected_report.write_text('{"synthetic":"sealed"}\n', encoding="utf-8")
    protected_inventory = finalizer._protected_root_inventory(protected_root)
    monkeypatch.setattr(
        finalizer,
        "OLD_ROOT_PRESERVATION",
        {
            chunk_id: {
                "root_name": protected_root.name,
                **protected_inventory,
                "sealed_report_sha256": sha256_file(protected_report),
            }
        },
    )
    worktree = workspace / "code-worktree"
    worktree.mkdir(mode=0o700)
    return {
        "raw_root": raw_root,
        "target_root": target_parent / finalizer.expected_target_root_name(chunk_id),
        "source_chunk": source_chunk,
        "frozen_universe": universe,
        "freeze_summary": freeze_summary,
        "worktree": worktree,
    }


def _finalize(paths: dict[str, Path], chunk_id: str) -> dict[str, Any]:
    return finalizer.finalize_remediation_root(chunk_id=chunk_id, **paths)


def test_static_partition_covers_the_frozen_2463_record_universe() -> None:
    partitions = finalizer.FROZEN_PARTITIONS
    assert tuple(partitions) == tuple(f"{index:03d}" for index in range(1, 14))
    assert all(partitions[f"{index:03d}"].records == 200 for index in range(1, 13))
    assert partitions["013"].records == 63
    assert partitions["001"].start_index_1based == 1
    assert partitions["013"].end_index_1based == 2463
    assert all(
        len(partition.source_sha256) == 64
        for partition in partitions.values()
    )
    assert finalizer.SUPPORTED_REMEDIATION_CHUNKS == {
        *(f"{index:03d}" for index in range(1, 9)),
        "011",
        "013",
    }
    with pytest.raises(finalizer.C2RemediationError, match="not authorized"):
        finalizer.expected_target_root_name("009")


@pytest.mark.parametrize(("chunk_id", "expected_records"), [("001", 200), ("013", 63)])
def test_finalizes_exact_200_and_63_roots(
    monkeypatch: pytest.MonkeyPatch,
    chunk_id: str,
    expected_records: int,
) -> None:
    with experiment_workspace(f"c2-remediation-{chunk_id}") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id=chunk_id)
        result = _finalize(paths, chunk_id)

        target = paths["target_root"]
        assert result["input_total"] == expected_records
        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES"
        assert len(
            (target / "control/terminal_outcomes.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ) == expected_records
        attestation = json.loads(
            (target / "control/v2/raw_attempt_attestation.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(attestation["entries"]) == expected_records * 3
        assert attestation["N"] == expected_records
        terminal_summary = json.loads(
            (target / "control/postfetch_terminal_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(terminal_summary["terminal"]) == finalizer.STATUS_VALUES
        assert (target / "control/preservation_ledger.json").is_file()
        relocation = json.loads(
            (target / "control/v2/provenance_relocation.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(relocation["entries"]) == expected_records
        assert all(
            entry["target_relative_path"].startswith("content/_provenance/")
            for entry in relocation["entries"]
        )
        assert json.loads(
            (target / "sealed_report_v1/postseal_preservation.json").read_text(
                encoding="utf-8"
            )
        )["status"] == "PASS_UNCHANGED"
        report = target / "sealed_report_v1/sealed_report.json"
        assert (
            target / "sealed_report_v1/sealed_report.sha256"
        ).read_text(encoding="utf-8") == f"{sha256_file(report)}  sealed_report.json\n"


def test_rejects_source_hash_mismatch_before_creating_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-source-mismatch") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        paths["source_chunk"].write_text('{"doi":"10.9/x"}\n', encoding="utf-8")

        with pytest.raises(finalizer.C2RemediationError, match="source chunk hash"):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


def test_rejects_freeze_summary_mismatch_for_any_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-summary-mismatch") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        summary = json.loads(paths["freeze_summary"].read_text(encoding="utf-8"))
        summary["chunks"][1]["sha256"] = "f" * 64
        _seal(summary, "summary_hash")
        _write_json(paths["freeze_summary"], summary)
        monkeypatch.setattr(
            finalizer,
            "FROZEN_FREEZE_SUMMARY_SHA256",
            sha256_file(paths["freeze_summary"]),
        )
        monkeypatch.setattr(
            finalizer,
            "FROZEN_FREEZE_SUMMARY_HASH",
            summary["summary_hash"],
        )

        with pytest.raises(finalizer.C2RemediationError, match="partition binding"):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


def test_rejects_existing_target_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-existing-target") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        paths["target_root"].mkdir()
        marker = paths["target_root"] / "must-not-change"
        marker.write_text("present", encoding="utf-8")

        with pytest.raises(finalizer.C2RemediationError, match="already exists"):
            _finalize(paths, "001")
        assert marker.read_text(encoding="utf-8") == "present"


def test_rejects_symlinked_provenance_before_creating_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-provenance-symlink") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        provenance = paths["raw_root"] / "content/_provenance/article-1.json"
        provenance.unlink()
        os.symlink(
            paths["raw_root"] / "content/_provenance/article-2.json",
            provenance,
        )

        with pytest.raises(finalizer.C2RemediationError, match="symlink"):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


def test_rejects_absolute_provenance_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-absolute-provenance") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        provenance_path = paths["raw_root"] / "content/_provenance/article-1.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["unsafe_local_path"] = "/not/portable"
        _write_json(provenance_path, provenance)

        with pytest.raises(finalizer.C2RemediationError, match="absolute filesystem"):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("downloaded-status", "invalid skipped status"),
        ("overlap", "processed/skipped overlap"),
        ("missing", "partition is incomplete"),
    ],
)
def test_rejects_invalid_processed_skipped_partitions(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    with experiment_workspace(f"c2-remediation-{mutation}") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        attempt_dir = paths["raw_root"] / "control/initial"
        skipped = attempt_dir / "_skipped.txt"
        if mutation == "downloaded-status":
            skipped.write_text("article-1\tdownloaded\n", encoding="utf-8")
        elif mutation == "overlap":
            processed = attempt_dir / "processed.txt"
            processed.write_text(
                processed.read_text(encoding="utf-8") + "article-1\n",
                encoding="utf-8",
            )
        else:
            rows = skipped.read_text(encoding="utf-8").splitlines()
            skipped.write_text("\n".join(rows[1:]) + "\n", encoding="utf-8")

        with pytest.raises(finalizer.C2RemediationError, match=message):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


def test_rejects_symlink_target_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-target-symlink") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        real_parent = workspace / "real-output"
        real_parent.mkdir(mode=0o700)
        link_parent = workspace / "linked-output"
        os.symlink(real_parent, link_parent)
        paths["target_root"] = link_parent / finalizer.expected_target_root_name("001")

        with pytest.raises(finalizer.ProvenanceError):
            _finalize(paths, "001")
        assert not (real_parent / "target").exists()


def test_rejects_target_nested_in_raw_evidence_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-overlapping-roots") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        paths["target_root"] = (
            paths["raw_root"] / finalizer.expected_target_root_name("001")
        )

        with pytest.raises(finalizer.C2RemediationError, match="non-overlapping"):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


def test_rejects_noncanonical_target_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-wrong-target-name") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        paths["target_root"] = paths["target_root"].parent / "wrong-root-name"

        with pytest.raises(finalizer.C2RemediationError, match="naming contract"):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


def test_rejects_changed_protected_old_root_before_target_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-protected-root") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        contract = finalizer.OLD_ROOT_PRESERVATION["001"]
        protected = paths["target_root"].parent / str(contract["root_name"])
        (protected / "accepted.jsonl").write_text("changed\n", encoding="utf-8")

        with pytest.raises(
            finalizer.C2RemediationError,
            match="(total_bytes|inventory_hash) mismatch",
        ):
            _finalize(paths, "001")
        assert not paths["target_root"].exists()


def test_blocks_generated_root_when_secret_scan_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-remediation-secret-scan") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        evidence_path = paths["raw_root"] / "control/execution_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["tests"][0]["command"] = "API_KEY=abcdefghijklmnop"
        _seal(evidence, "evidence_hash")
        _write_json(evidence_path, evidence)

        with pytest.raises(finalizer.C2RemediationError, match="secret scan"):
            _finalize(paths, "001")
        scan = json.loads(
            (paths["target_root"] / "control/secret_scan.json").read_text(
                encoding="utf-8"
            )
        )
        assert scan["status"] == "BLOCKED_SECRET_HIT"


def test_cli_wires_the_dedicated_remediation_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, Any] = {}

    def fake_finalizer(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"status": "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES"}

    monkeypatch.setattr(cli, "finalize_remediation_root", fake_finalizer)
    assert (
        cli.main(
            [
                "c2-remediation-root-finalize",
                "013",
                "--raw-root",
                "/raw",
                "--target-root",
                "/target",
                "--source-chunk",
                "/source",
                "--frozen-universe",
                "/universe",
                "--freeze-summary",
                "/summary",
                "--worktree",
                "/worktree",
            ]
        )
        == 0
    )
    assert received["chunk_id"] == "013"
    assert received["target_root"] == Path("/target")
    assert "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES" in capsys.readouterr().out

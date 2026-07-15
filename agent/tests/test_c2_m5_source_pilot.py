from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import zipfile
import builtins
import dataclasses
import itertools
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 10):
    _original_zip = builtins.zip

    def _compatible_zip(*iterables: object, strict: bool = False):
        if not strict:
            return _original_zip(*iterables)

        def strict_iterator():
            sentinel = object()
            for values in itertools.zip_longest(*iterables, fillvalue=sentinel):
                if any(value is sentinel for value in values):
                    raise ValueError("zip() arguments have unequal lengths")
                yield values

        return strict_iterator()

    _original_dataclass = dataclasses.dataclass

    def _compatible_dataclass(_cls=None, **kwargs: object):
        kwargs.pop("slots", None)
        return _original_dataclass(_cls, **kwargs)

    builtins.zip = _compatible_zip
    dataclasses.dataclass = _compatible_dataclass

import c2_m5_bootstrap_attestation as m5_attestation
import c2_m5_release_test_runner as release_runner
from experiments import c2_m5_source_pilot as pilot
from experiments import c2_remediation_root_finalizer as finalizer
from experiments import c2_source_bearing_extension as source_extension


@lru_cache(maxsize=1)
def _dependency_closure() -> tuple[
    dict[str, object],
    dict[str, bytes],
    bytes,
    bytes,
]:
    manifest = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "resources"
        / "c2_m5_python_dependency_manifest_v2.json"
    ).read_bytes()
    archive = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "resources"
        / "c2_m5_python_dependencies_py39_v1.zip"
    ).read_bytes()
    binding, payloads = m5_attestation._verified_dependency_payloads(
        manifest,
        archive,
    )
    return binding, payloads, manifest, archive


def _refresh_raw_inventory(root: Path) -> None:
    inventory_path = root / "control" / "raw_inventory.json"
    inventory = json.loads(inventory_path.read_bytes())
    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": pilot._sha256(path.read_bytes()),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != inventory_path
    ]
    inventory.update(
        {
            "artifact_count": len(entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "files": entries,
        }
    )
    inventory["inventory_hash"] = pilot._sha256_json(
        {
            key: value
            for key, value in inventory.items()
            if key != "inventory_hash"
        }
    )
    inventory_path.write_bytes(pilot._json_file_bytes(inventory))


def _record(ordinal: int) -> pilot.FrozenRecord:
    article_id = f"s41598-025-{ordinal:05d}-x"
    doi = f"10.1038/{article_id}"
    return pilot.FrozenRecord(
        ordinal=ordinal,
        article_id=article_id,
        doi=doi,
        value={
            "article_url": f"https://www.nature.com/articles/{article_id}",
            "doi": doi,
            "policy_accepted": True,
            "download_eligible": True,
            "journal_allowed": True,
            "require_cc_by": True,
            "reject_reasons": [],
            "license": {
                "license_id": "CC-BY-4.0",
                "content_version": "vor",
                "normalized_url": (
                    "https://creativecommons.org/licenses/by/4.0/"
                ),
            },
        },
    )


def _attempt(
    attempt: str,
    records: list[pilot.FrozenRecord],
    statuses: dict[str, str],
) -> pilot.AttemptResult:
    processed = "".join(
        f"{record.article_id}\n"
        for record in records
        if statuses[record.article_id] == "downloaded"
    ).encode()
    skipped = "".join(
        f"{record.article_id}\t{statuses[record.article_id]}\n"
        for record in records
        if statuses[record.article_id] != "downloaded"
    ).encode()
    return pilot.AttemptResult(
        attempt=attempt,
        start_monotonic_ns=pilot.ATTEMPT_INDEX[attempt] * 10,
        end_monotonic_ns=pilot.ATTEMPT_INDEX[attempt] * 10 + 5,
        statuses=statuses,
        accepted=b"accepted\n",
        operational_processed="".join(
            f"{record.article_id}\n" for record in records
        ).encode(),
        operational_skipped=skipped,
        processed=processed,
        skipped=skipped,
        log=b"real downloader log\n",
        exit_payload=b"0\n",
        network_budget=pilot._json_file_bytes(
            pilot._sealed(
                {
                    "schema_version": "c2-m5-network-budget-v1",
                    "request_count": 1,
                    "response_bytes": 2,
                    "request_cap": pilot.MAX_NETWORK_REQUESTS_PER_ATTEMPT,
                    "response_byte_cap": (
                        pilot.MAX_NETWORK_RESPONSE_BYTES_PER_ATTEMPT
                    ),
                    "wall_timeout_seconds": (
                        pilot.ATTEMPT_WALL_TIMEOUT_SECONDS
                    ),
                },
                "budget_hash",
            )
        ),
        harvest={},
    )


def test_derives_disjoint_strict_partition_from_operational_overlap() -> None:
    records = [_record(1), _record(2), _record(3)]
    operational_processed = "".join(
        f"{record.article_id}\n" for record in records
    ).encode()
    operational_skipped = (
        f"{records[0].article_id}\tno-source-data\n"
        f"{records[2].article_id}\tno-figures\n"
    ).encode()

    statuses, processed, skipped = pilot.derive_strict_statuses(
        records,
        operational_processed,
        operational_skipped,
    )

    assert statuses == {
        records[0].article_id: "no-source-data",
        records[1].article_id: "downloaded",
        records[2].article_id: "no-figures",
    }
    assert processed == f"{records[1].article_id}\n".encode()
    assert skipped == (
        f"{records[0].article_id}\tno-source-data\n"
        f"{records[2].article_id}\tno-figures\n"
    ).encode()
    assert set(processed.decode().splitlines()).isdisjoint(
        line.split("\t", 1)[0] for line in skipped.decode().splitlines()
    )


@pytest.mark.parametrize(
    ("processed", "skipped", "message"),
    [
        (b"", b"", "does not cover"),
        (b"unknown\n", b"", "invalid"),
        (
            f"{_record(1).article_id}\n".encode(),
            f"{_record(1).article_id}\tother\n".encode(),
            "invalid",
        ),
        (
            f"{_record(1).article_id}\n".encode(),
            (
                f"{_record(1).article_id}\tno-figures\n"
                f"{_record(1).article_id}\tno-source-data\n"
            ).encode(),
            "invalid",
        ),
    ],
)
def test_rejects_malformed_operational_status(
    processed: bytes,
    skipped: bytes,
    message: str,
) -> None:
    with pytest.raises(pilot.C2M5SourcePilotError, match=message):
        pilot.derive_strict_statuses([_record(1)], processed, skipped)


def test_asset_registry_deduplicates_across_attempts() -> None:
    writes: dict[str, bytes] = {}
    registry = pilot.AssetRegistry(lambda path, payload: writes.setdefault(path, payload) or path)
    record = _record(1)
    payload = b"a,b\n1,2\n"

    first = registry.add(
        record=record,
        attempt="initial",
        payload=payload,
        declared_kind="source_data",
    )
    second = registry.add(
        record=record,
        attempt="retry2",
        payload=payload,
        declared_kind="source_data",
    )

    assert first == second
    assert first.first_attempt == "initial"
    assert registry.asset_count == 1
    assert registry.total_bytes == len(payload)
    assert writes == {first.relative_path: payload}


def _zip_payload(name: str, payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    return output.getvalue()


def test_asset_registry_rejects_archive_member_over_v2_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import c2_source_bearing_extension as source_extension

    monkeypatch.setattr(source_extension, "MAX_ARCHIVE_MEMBER_BYTES", 16)
    registry = pilot.AssetRegistry(lambda path, payload: path)

    with pytest.raises(pilot.C2M5SourcePilotError, match="not V2-recognized"):
        registry.add(
            record=_record(1),
            attempt="initial",
            payload=_zip_payload("table.csv", b"a,b\n" + b"1,2\n" * 5),
            declared_kind="source_archive",
        )


def test_asset_registry_shares_v2_archive_budget_across_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import c2_source_bearing_extension as source_extension

    monkeypatch.setattr(
        source_extension,
        "MAX_ARCHIVE_RUN_UNCOMPRESSED_BYTES",
        50,
    )
    registry = pilot.AssetRegistry(lambda path, payload: path)
    table = b"a,b\n" + b"1,2\n" * 6
    registry.add(
        record=_record(1),
        attempt="initial",
        payload=_zip_payload("first.csv", table),
        declared_kind="source_archive",
    )

    with pytest.raises(pilot.C2M5SourcePilotError, match="not V2-recognized"):
        registry.add(
            record=_record(2),
            attempt="initial",
            payload=_zip_payload("second.csv", table),
            declared_kind="source_archive",
        )


def test_asset_registry_rejects_declared_kind_mismatch() -> None:
    registry = pilot.AssetRegistry(lambda path, payload: path)

    with pytest.raises(pilot.C2M5SourcePilotError, match="declared kind"):
        registry.add(
            record=_record(1),
            attempt="initial",
            payload=b"a,b\n1,2\n",
            declared_kind="source_archive",
        )


def test_skipped_figure_zip_does_not_consume_retained_archive_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import c2_source_bearing_extension as source_extension

    monkeypatch.setattr(
        source_extension,
        "MAX_ARCHIVE_RUN_UNCOMPRESSED_BYTES",
        50,
    )
    first = _record(1)
    source = tmp_path / "content" / first.article_id / "source_data"
    figures = tmp_path / "content" / first.article_id / "figures"
    source.mkdir(parents=True)
    figures.mkdir()
    (source / "table.csv").write_bytes(b"a,b\n1,2\n")
    (source / "table-copy.csv").write_bytes(b"a,b\n1,2\n")
    (figures / "not-context.zip").write_bytes(
        _zip_payload("ignored.csv", b"a,b\n" + b"1,2\n" * 6)
    )
    registry = pilot.AssetRegistry(lambda path, payload: path)

    pilot._harvest_download(
        workspace=tmp_path,
        record=first,
        attempt="initial",
        registry=registry,
    )
    retained = registry.add(
        record=_record(2),
        attempt="initial",
        payload=_zip_payload("retained.csv", b"a,b\n" + b"1,2\n" * 6),
        declared_kind="source_archive",
    )

    assert retained.declared_format_tuple == ("ZIP_V1", "GENERIC_ZIP_V1")


def test_harvests_only_downloaded_article_source_and_context(tmp_path: Path) -> None:
    record = _record(1)
    article = tmp_path / "content" / record.article_id
    source = article / "source_data"
    figures = article / "figures"
    source.mkdir(parents=True)
    figures.mkdir()
    (source / "table.csv").write_bytes(b"a,b\n1,2\n")
    (figures / "fig_001.png").write_bytes(b"\x89PNG\r\n\x1a\npayload")
    (figures / "fig_001.txt").write_text("Figure caption", encoding="utf-8")
    writes: dict[str, bytes] = {}
    registry = pilot.AssetRegistry(lambda path, payload: writes.setdefault(path, payload) or path)

    observed = pilot._harvest_download(
        workspace=tmp_path,
        record=record,
        attempt="initial",
        registry=registry,
    )

    assert observed == {
        "source_assets_observed": 1,
        "figure_assets_observed": 1,
        "caption_assets_observed": 1,
        "assets": [
            {
                **asset.descriptor_value(),
                "first_attempt": asset.first_attempt,
            }
            for asset in sorted(
                registry.for_article(record.article_id),
                key=lambda item: item.asset_id,
            )
        ],
    }
    assert registry.asset_count == 3
    assert {asset.declared_asset_kind for asset in registry.for_article(record.article_id)} == {
        "source_data",
        "figure",
        "caption",
    }


def test_harvest_rejects_symlinked_source_asset(tmp_path: Path) -> None:
    record = _record(1)
    source = tmp_path / "content" / record.article_id / "source_data"
    source.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"a,b\n1,2\n")
    (source / "table.csv").symlink_to(outside)
    registry = pilot.AssetRegistry(lambda path, payload: path)

    with pytest.raises(pilot.C2M5SourcePilotError, match="symlink"):
        pilot._harvest_download(
            workspace=tmp_path,
            record=record,
            attempt="initial",
            registry=registry,
        )


def test_harvest_rejects_symlinked_article_parent(tmp_path: Path) -> None:
    record = _record(1)
    outside = tmp_path / "outside" / "source_data"
    outside.mkdir(parents=True)
    (outside / "table.csv").write_bytes(b"a,b\n1,2\n")
    content = tmp_path / "content"
    content.mkdir()
    (content / record.article_id).symlink_to(outside.parent, target_is_directory=True)
    registry = pilot.AssetRegistry(lambda path, payload: path)

    with pytest.raises(pilot.C2M5SourcePilotError, match="chain is unsafe"):
        pilot._harvest_download(
            workspace=tmp_path,
            record=record,
            attempt="initial",
            registry=registry,
        )


def test_stable_reader_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    target = real_parent / "source.csv"
    target.write_bytes(b"a,b\n1,2\n")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(pilot.C2M5SourcePilotError, match="opened safely"):
        pilot._read_stable_regular(
            linked_parent / target.name,
            "symlinked-parent source",
            max_bytes=1024,
        )


def test_stable_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)
    started = time.monotonic()

    with pytest.raises(pilot.C2M5SourcePilotError, match="private regular"):
        pilot._read_stable_regular(
            fifo,
            "fifo source",
            max_bytes=1024,
        )

    assert time.monotonic() - started < 1


def test_prior_attempt_source_survives_source_less_retry2() -> None:
    records = [_record(1), _record(2)]
    writes: dict[str, bytes] = {}

    def write(path: str, payload: bytes) -> str:
        assert path not in writes
        writes[path] = payload
        return path

    registry = pilot.AssetRegistry(write)
    registry.add(
        record=records[0],
        attempt="initial",
        payload=b"a,b\n1,2\n",
        declared_kind="source_data",
    )
    attempts = {
        "initial": _attempt(
            "initial",
            records,
            {
                records[0].article_id: "downloaded",
                records[1].article_id: "no-figures",
            },
        ),
        "retry1": _attempt(
            "retry1",
            records,
            {
                records[0].article_id: "no-source-data",
                records[1].article_id: "no-source-data",
            },
        ),
        "retry2": _attempt(
            "retry2",
            records,
            {
                records[0].article_id: "no-source-data",
                records[1].article_id: "fetch-error",
            },
        ),
    }

    source_articles, source_assets = pilot._write_terminal_provenance(
        write_bytes=write,
        records=records,
        attempts=attempts,
        registry=registry,
    )

    assert (source_articles, source_assets) == (1, 1)
    provenance = json.loads(
        writes[f"content/_provenance/{records[0].article_id}.json"]
    )
    assert provenance["download_status"] == "downloaded"
    assert provenance["rejection_reasons"] == ["no-source-data"]
    assert provenance["attempt_statuses"]["initial"] == "downloaded"
    descriptor = json.loads(
        writes[f"content/_source_evidence/{records[0].article_id}.json"]
    )
    assert descriptor["schema_version"] == "c2-source-evidence-v2"
    assert descriptor["assets"][0]["candidate_hints"] == []
    assert descriptor["descriptor_hash"] == hashlib.sha256(
        pilot._canonical_bytes(
            {key: value for key, value in descriptor.items() if key != "descriptor_hash"}
        )
    ).hexdigest()
    source_less = json.loads(
        writes[f"content/_provenance/{records[1].article_id}.json"]
    )
    assert "download_status" not in source_less
    assert "source_evidence" not in source_less
    assert source_less["rejection_reasons"] == ["fetch-error"]


def test_hermetic_200_record_attempt_is_complete_and_ordered(tmp_path: Path) -> None:
    records = [_record(index) for index in range(1, 201)]
    downloaded = records[73]
    workspace = tmp_path / "attempt"
    source = workspace / "content" / downloaded.article_id / "source_data"
    source.mkdir(parents=True)
    (source / "table.csv").write_bytes(b"x,y\n3,4\n")
    accepted = b"".join(
        pilot._canonical_bytes(record.value) + b"\n" for record in records
    )
    (workspace / "accepted.jsonl").write_bytes(accepted)
    (workspace / "operational_processed.txt").write_text(
        "".join(f"{record.article_id}\n" for record in records),
        encoding="utf-8",
    )
    (workspace / "_skipped.txt").write_text(
        "".join(
            f"{record.article_id}\tno-source-data\n"
            for record in records
            if record != downloaded
        ),
        encoding="utf-8",
    )
    (workspace / "postfetch.log").write_text("fake network run\n", encoding="utf-8")
    pilot._initialize_network_budget(workspace)
    writes: dict[str, bytes] = {}
    registry = pilot.AssetRegistry(lambda path, payload: writes.setdefault(path, payload) or path)

    result = pilot._attempt_artifacts(
        attempt="initial",
        source_bytes=accepted,
        records=records,
        workspace=workspace,
        start_monotonic_ns=1,
        end_monotonic_ns=2,
        registry=registry,
    )

    assert len(result.statuses) == 200
    assert Counter(result.statuses.values()) == {
        "no-source-data": 199,
        "downloaded": 1,
    }
    assert result.processed == f"{downloaded.article_id}\n".encode()
    assert len(result.skipped.decode().splitlines()) == 199
    assert registry.article_ids == {downloaded.article_id}


def test_execute_to_raw_root_and_finalizer_validation_is_hermetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir(mode=0o700)
    records = [_record(index) for index in range(1, 201)]
    source_bytes = b"".join(
        pilot._canonical_bytes(record.value) + b"\n" for record in records
    )
    source_chunk = input_root / "chunk.jsonl"
    universe = input_root / "universe.jsonl"
    source_chunk.write_bytes(source_bytes)
    universe.write_bytes(source_bytes)
    summary_body = {"schema_version": "synthetic-m5-freeze-v1"}
    summary = {
        **summary_body,
        "summary_hash": pilot._sha256_json(summary_body),
    }
    freeze_summary = input_root / "freeze.json"
    freeze_summary.write_bytes(pilot._json_file_bytes(summary))
    monkeypatch.setattr(pilot, "CHUNK_SHA256", pilot._sha256(source_bytes))
    monkeypatch.setattr(pilot, "UNIVERSE_SHA256", pilot._sha256(source_bytes))
    monkeypatch.setattr(
        pilot,
        "FREEZE_SUMMARY_SHA256",
        pilot._sha256(freeze_summary.read_bytes()),
    )
    monkeypatch.setattr(
        pilot,
        "FREEZE_SUMMARY_HASH",
        str(summary["summary_hash"]),
    )

    downloader_script = b"""\
import argparse
import json
import os
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--jsonl")
parser.add_argument("--out")
parser.add_argument("--processed-file")
args, _ = parser.parse_known_args()
records = [
    json.loads(line)
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines()
]
article_ids = [record["article_url"].rstrip("/").rsplit("/", 1)[-1] for record in records]
Path(args.processed_file).write_text(
    "".join(f"{article_id}\\n" for article_id in article_ids),
    encoding="utf-8",
)
Path(args.out).parent.joinpath("_skipped.txt").write_text(
    "".join(f"{article_id}\\tno-source-data\\n" for article_id in article_ids[1:]),
    encoding="utf-8",
)
source = Path(args.out) / article_ids[0] / "source_data" / "source.csv"
source.parent.mkdir(parents=True)
source_payload = b"series,value\\nA,1\\nB,2\\n"
source.write_bytes(source_payload)
with open(os.environ["C2_M5_NETWORK_BUDGET_PATH"], "r+b") as budget:
    budget.write(struct.pack(">QQ", len(article_ids), len(source_payload)))
print(f"synthetic guarded acquisition covered {len(article_ids)} records")
"""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    downloader = pilot.DownloaderBinding(
        worktree=worktree,
        tree="a" * 40,
        script=worktree / "nature_all_in_one.py",
        script_blob="b" * 40,
        runtime_files={"frozen_downloader.py": downloader_script},
    )
    guard_path = (
        Path(__file__).resolve().parents[1]
        / "c2_m5_sitecustomize"
        / "sitecustomize.py"
    )
    bootstrap_path = (
        Path(__file__).resolve().parents[1]
        / "c2_m5_downloader_bootstrap.py"
    )
    dependency_binding, dependency_payloads, dependency_manifest, _ = (
        _dependency_closure()
    )
    attestation = pilot.AdapterAttestation(
        implementation_commit="1" * 40,
        attestation_commit="2" * 40,
        head_commit="2" * 40,
        manifest_sha256="3" * 64,
        path_sha256={
            "agent/c2_m5_sitecustomize/sitecustomize.py": pilot._sha256(
                guard_path.read_bytes()
            ),
            "agent/c2_m5_downloader_bootstrap.py": pilot._sha256(
                bootstrap_path.read_bytes()
            ),
        },
        path_payloads={},
        dependency_manifest_sha256=pilot._sha256(dependency_manifest),
        dependency_binding=dependency_binding,
        dependency_payloads=dependency_payloads,
        externally_pinned_bootstrap_sha256="4" * 64,
        externally_pinned_python_sha256=dependency_binding["python"][
            "executable_sha256"
        ],
        externally_pinned_python_library_sha256=dependency_binding["python"][
            "runtime_library_sha256"
        ],
    )
    monkeypatch.setattr(pilot, "verify_adapter_attestation", lambda: attestation)
    monkeypatch.setattr(
        finalizer,
        "_require_active_m5_adapter_attestation",
        lambda: attestation,
    )
    monkeypatch.setattr(pilot, "_verify_downloader_worktree", lambda _: downloader)
    partition = finalizer.FrozenPartition(
        chunk_id="001",
        records=200,
        source_sha256=pilot._sha256(source_bytes),
        start_index_1based=1,
        end_index_1based=200,
    )
    monkeypatch.setattr(finalizer, "FROZEN_PARTITIONS", {"001": partition})
    monkeypatch.setattr(
        finalizer,
        "FROZEN_UNIVERSE_SHA256",
        pilot._sha256(source_bytes),
    )
    monkeypatch.setattr(
        finalizer,
        "FROZEN_FREEZE_SUMMARY_SHA256",
        pilot._sha256(freeze_summary.read_bytes()),
    )
    monkeypatch.setattr(
        finalizer,
        "FROZEN_FREEZE_SUMMARY_HASH",
        str(summary["summary_hash"]),
    )
    monkeypatch.setattr(finalizer, "require_external_m1_trust_lock", lambda: None)
    monkeypatch.setattr(
        finalizer,
        "_require_owner_remediation_policy_for_chunk",
        lambda *_: (
            SimpleNamespace(authorization_id_sha256="test-authorization"),
            SimpleNamespace(policy_id_sha256="test-policy"),
            SimpleNamespace(required_action="FRESH_REMEDIATION_REQUIRED"),
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_verify_protected_old_root",
        lambda *_: {"inventory_hash": "protected"},
    )
    monkeypatch.setattr(
        finalizer,
        "_verify_frozen_inputs",
        lambda *_: (source_bytes, [record.value for record in records]),
    )
    monkeypatch.setattr(
        finalizer,
        "_verify_worktree",
        lambda _: {
            "commit": pilot.FROZEN_DOWNLOADER_COMMIT,
            "tree": downloader.tree,
            "dirty": False,
            "script_blob": downloader.script_blob,
            "script_sha256": pilot.FROZEN_DOWNLOADER_SHA256,
            "runtime_bundle": [
                {
                    "relative_path": relative,
                    "bytes": len(payload),
                    "sha256": pilot._sha256(payload),
                }
                for relative, payload in sorted(
                    downloader.runtime_files.items()
                )
            ],
        },
    )
    test_source_attestation = SimpleNamespace(
        approved_implementation_commit_full="5" * 40,
        attestation_commit_full="6" * 40,
        manifest_sha256="7" * 64,
        code_blob_set_sha256="8" * 64,
        sha256_for=lambda _relative: "9" * 64,
        verify_runtime=lambda **_paths: None,
    )
    monkeypatch.setattr(
        source_extension,
        "_require_stage_b_production_source_extension_trust",
        lambda: test_source_attestation,
    )
    raw_root = output_root / pilot.EXPECTED_RAW_ROOT

    execution = pilot.execute_source_pilot(
        raw_root=raw_root,
        source_chunk=source_chunk,
        frozen_universe=universe,
        freeze_summary=freeze_summary,
        worktree=worktree,
        workers=1,
    )
    validation = pilot.validate_source_pilot(
        raw_root=raw_root,
        source_chunk=source_chunk,
        frozen_universe=universe,
        freeze_summary=freeze_summary,
        worktree=worktree,
    )

    assert execution["source_article_count"] == 1
    assert execution["source_asset_count"] == 1
    assert execution["attempt_status_counts"] == {
        attempt: {"downloaded": 1, "no-source-data": 199}
        for attempt in pilot.ATTEMPTS
    }
    assert validation["status"] == "PASS_SOURCE_BEARING_RAW_ROOT_NON_ADMISSIVE"
    assert validation["records"] == 200
    assert validation["attempt_rows"] == 600
    assert validation["source_article_count"] == 1
    assert validation["source_asset_count"] == 1
    assert validation["admission_authorized"] is False
    assert validation["publication_authorized"] is False
    assert stat.S_IMODE(raw_root.lstat().st_mode) == 0o500
    for path in raw_root.rglob("*"):
        expected_mode = 0o500 if path.is_dir() else 0o400
        assert stat.S_IMODE(path.lstat().st_mode) == expected_mode

    target_root = output_root / pilot.EXPECTED_FINAL_ROOT
    finalization = pilot.finalize_source_pilot(
        raw_root=raw_root,
        target_root=target_root,
        source_chunk=source_chunk,
        frozen_universe=universe,
        freeze_summary=freeze_summary,
        worktree=worktree,
    )
    assert finalization["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
    assert finalization["raw_validation"] == validation
    assert finalization["admission_authorized"] is False
    assert finalization["publication_authorized"] is False
    assert stat.S_IMODE(target_root.lstat().st_mode) == 0o500
    for path in target_root.rglob("*"):
        expected_mode = 0o500 if path.is_dir() else 0o400
        assert stat.S_IMODE(path.lstat().st_mode) == expected_mode
    sealed_report = json.loads(
        (
            target_root
            / "sealed_report_v1"
            / "sealed_report.json"
        ).read_bytes()
    )
    assert sealed_report["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
    assert (
        sealed_report["trust_chain"]["m5_finalized_snapshot_qualification"][
            "status"
        ]
        == "PASS_SOURCE_BEARING_NO_CANDIDATE_HINTS"
    )

    tampered_root = tmp_path / "tampered" / pilot.EXPECTED_RAW_ROOT
    tampered_root.parent.mkdir()
    shutil.copytree(raw_root, tampered_root)
    for path in [tampered_root, *tampered_root.rglob("*")]:
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    article_id = records[0].article_id
    descriptor_path = (
        tampered_root / "content" / "_source_evidence" / f"{article_id}.json"
    )
    descriptor = json.loads(descriptor_path.read_bytes())
    source_path = tampered_root / descriptor["assets"][0]["relative_path"]
    replacement = b"series,value\nA,9\nB,8\n"
    source_path.write_bytes(replacement)
    descriptor["assets"][0]["sha256"] = pilot._sha256(replacement)
    descriptor["assets"][0]["bytes"] = len(replacement)
    descriptor["descriptor_hash"] = pilot._sha256_json(
        {
            key: value
            for key, value in descriptor.items()
            if key != "descriptor_hash"
        }
    )
    descriptor_path.write_bytes(pilot._json_file_bytes(descriptor))
    provenance_path = (
        tampered_root / "content" / "_provenance" / f"{article_id}.json"
    )
    provenance_value = json.loads(provenance_path.read_bytes())
    provenance_value["source_evidence"]["descriptor_sha256"] = pilot._sha256(
        descriptor_path.read_bytes()
    )
    provenance_value["source_evidence"]["descriptor_bytes"] = len(
        descriptor_path.read_bytes()
    )
    provenance_value["retained_assets"][0]["sha256"] = pilot._sha256(replacement)
    provenance_value["retained_assets"][0]["bytes"] = len(replacement)
    provenance_path.write_bytes(pilot._json_file_bytes(provenance_value))
    _refresh_raw_inventory(tampered_root)
    for path in sorted(tampered_root.rglob("*"), reverse=True):
        os.chmod(path, 0o500 if path.is_dir() else 0o400)
    os.chmod(tampered_root, 0o500)

    with pytest.raises(
        pilot.C2M5SourcePilotError,
        match="source harvest asset is invalid",
    ):
        pilot.validate_source_pilot(
            raw_root=tampered_root,
            source_chunk=source_chunk,
            frozen_universe=universe,
            freeze_summary=freeze_summary,
            worktree=worktree,
        )

    claim_tampered_root = (
        tmp_path / "claim-tampered" / pilot.EXPECTED_RAW_ROOT
    )
    claim_tampered_root.parent.mkdir()
    shutil.copytree(raw_root, claim_tampered_root)
    for path in [claim_tampered_root, *claim_tampered_root.rglob("*")]:
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    claim_path = (
        claim_tampered_root
        / "content"
        / "_provenance"
        / f"{article_id}.json"
    )
    claim = json.loads(claim_path.read_bytes())
    claim["attempt_statuses"]["initial"] = "fetch-error"
    claim_path.write_bytes(pilot._json_file_bytes(claim))
    _refresh_raw_inventory(claim_tampered_root)
    for path in sorted(claim_tampered_root.rglob("*"), reverse=True):
        os.chmod(path, 0o500 if path.is_dir() else 0o400)
    os.chmod(claim_tampered_root, 0o500)

    with pytest.raises(
        pilot.C2M5SourcePilotError,
        match="reconstructed claim",
    ):
        pilot.validate_source_pilot(
            raw_root=claim_tampered_root,
            source_chunk=source_chunk,
            frozen_universe=universe,
            freeze_summary=freeze_summary,
            worktree=worktree,
        )


def test_subprocess_environment_drops_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    environment = pilot._safe_subprocess_environment(workspace)

    assert "ANTHROPIC_AUTH_TOKEN" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "HTTPS_PROXY" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["NO_PROXY"] == "*"
    assert environment["C2_M5_NETWORK_GUARD_REQUIRED"] == "1"
    command = pilot._attempt_command(workspace, 4)
    assert command[1:4] == ["-I", "-S", "-B"]
    assert command[4] == str(workspace / "guarded_downloader_bootstrap.py")


def test_attempt_process_enforces_log_and_wall_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overflow_workspace = tmp_path / "overflow"
    overflow_workspace.mkdir()
    monkeypatch.setattr(pilot, "MAX_POSTFETCH_LOG_BYTES", 32)

    overflow_exit, overflow_retained = pilot._run_bounded_process(
        command=[
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import os; os.write(1, b'x' * 4096)",
        ],
        workspace=overflow_workspace,
        environment={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
    )

    assert overflow_exit == 125
    assert overflow_retained == ()
    assert (overflow_workspace / "postfetch.log").stat().st_size == 32

    timeout_workspace = tmp_path / "timeout"
    timeout_workspace.mkdir()
    started = time.monotonic()
    timeout_exit, timeout_retained = pilot._run_bounded_process(
        command=[
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import time; time.sleep(10)",
        ],
        workspace=timeout_workspace,
        environment={"PATH": "/usr/bin:/bin"},
        timeout_seconds=0.1,
    )

    assert timeout_exit == 124
    assert timeout_retained == ()
    assert time.monotonic() - started < 3
    assert (timeout_workspace / "postfetch.log").stat().st_size <= 32

    closed_output_workspace = tmp_path / "closed-output"
    closed_output_workspace.mkdir()
    started = time.monotonic()
    closed_output_exit, closed_output_retained = pilot._run_bounded_process(
        command=[
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import os,time; os.close(1); os.close(2); time.sleep(10)",
        ],
        workspace=closed_output_workspace,
        environment={"PATH": "/usr/bin:/bin"},
        timeout_seconds=0.1,
    )

    assert closed_output_exit == 124
    assert closed_output_retained == ()
    assert time.monotonic() - started < 3

    descendant_workspace = tmp_path / "descendant"
    descendant_workspace.mkdir()
    descendant_exit, descendant_retained = pilot._run_bounded_process(
        command=[
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import pathlib,subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-I','-S','-B','-c',"
                "'import time; time.sleep(30)'],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                "pathlib.Path('child.pid').write_text(str(child.pid))"
            ),
        ],
        workspace=descendant_workspace,
        environment={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
    )

    assert descendant_exit == 0
    assert descendant_retained == ()
    child_pid = int((descendant_workspace / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_attempt_status_completion_drains_log_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_factory = pilot.selectors.DefaultSelector

    class StatusFirstSelector:
        def __init__(self) -> None:
            self._selector = selector_factory()

        def register(self, *args: object) -> object:
            return self._selector.register(*args)

        def unregister(self, *args: object) -> object:
            return self._selector.unregister(*args)

        def select(self, *args: object, **kwargs: object) -> object:
            ready = self._selector.select(*args, **kwargs)
            status = [
                event for event in ready if event[0].data == "status"
            ]
            return status or ready

        def close(self) -> None:
            self._selector.close()

    monkeypatch.setattr(
        pilot.selectors,
        "DefaultSelector",
        StatusFirstSelector,
    )
    workspace = tmp_path / "tail"
    workspace.mkdir()
    exit_code, retained = pilot._run_bounded_process(
        command=[
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import os; os.write(1, b'tail')",
        ],
        workspace=workspace,
        environment={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
    )

    assert exit_code == 0
    assert retained == ()
    assert (workspace / "postfetch.log").read_bytes() == b"tail"

    monkeypatch.setattr(pilot, "MAX_POSTFETCH_LOG_BYTES", 3)
    overflow = tmp_path / "tail-overflow"
    overflow.mkdir()
    exit_code, retained = pilot._run_bounded_process(
        command=[
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import os; os.write(1, b'tail')",
        ],
        workspace=overflow,
        environment={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
    )

    assert exit_code == 125
    assert retained == ()
    assert (overflow / "postfetch.log").read_bytes() == b"tai"


def test_attempt_process_cleans_group_when_selector_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "selector-failure"
    workspace.mkdir()
    observed: dict[str, subprocess.Popen[bytes]] = {}
    real_popen = pilot.subprocess.Popen

    def capture_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        if kwargs.get("start_new_session", False):
            observed["process"] = process
        return process

    class BrokenSelector:
        def register(self, *_: object) -> None:
            raise RuntimeError("selector setup failed")

        def close(self) -> None:
            pass

    monkeypatch.setattr(pilot.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(pilot.selectors, "DefaultSelector", BrokenSelector)
    process: subprocess.Popen[bytes] | None = None
    try:
        with pytest.raises(RuntimeError, match="selector setup failed"):
            pilot._run_bounded_process(
                command=[
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    "import time; time.sleep(60)",
                ],
                workspace=workspace,
                environment={"PATH": "/usr/bin:/bin"},
                timeout_seconds=5,
            )
        process = observed["process"]
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
    finally:
        process = observed.get("process", process)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def test_attempt_supervisor_rejects_early_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "early-supervisor-exit"
    workspace.mkdir()
    monkeypatch.setattr(
        pilot,
        "_ATTEMPT_PROCESS_SUPERVISOR",
        (
            "import os,sys; fd=int(sys.argv[1]); "
            "os.write(fd,b'R0\\n'); os.close(fd)"
        ),
    )

    with pytest.raises(pilot.C2M5SourcePilotError) as raised:
        pilot._run_bounded_process(
            command=[sys.executable, "-I", "-S", "-B", "-c", "pass"],
            workspace=workspace,
            environment={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
        )
    observed: BaseException | None = raised.value
    messages: list[str] = []
    while observed is not None:
        messages.append(str(observed))
        observed = observed.__cause__
    assert any(
        "supervisor exited before group cleanup" in message
        for message in messages
    )


def test_attempt_supervisor_spawn_signal_reaps_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "spawn-signal"
    workspace.mkdir()
    observed: dict[str, subprocess.Popen[bytes]] = {}
    real_popen = pilot.subprocess.Popen

    def capture_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        if kwargs.get("start_new_session", False):
            observed["process"] = process
        return process

    monkeypatch.setattr(pilot.subprocess, "Popen", capture_popen)
    with pytest.raises(
        pilot.C2M5SourcePilotError,
        match="status is malformed",
    ):
        pilot._run_bounded_process(
            command=[
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import os,signal,time; "
                    "os.kill(os.getppid(),signal.SIGTERM); "
                    "time.sleep(60)"
                ),
            ],
            workspace=workspace,
            environment={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
        )

    process = observed["process"]
    assert process.returncode == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


@pytest.mark.parametrize(
    "signum",
    [signal.SIGHUP, signal.SIGINT, signal.SIGTERM],
)
def test_attempt_process_cleans_group_on_signal(
    tmp_path: Path,
    signum: int,
) -> None:
    state_path = tmp_path / f"attempt-{signum}.state"
    workspace = tmp_path / f"attempt-{signum}"
    workspace.mkdir()
    agent_root = Path(__file__).resolve().parents[1]
    child_code = (
        "import os,pathlib,signal,time; "
        "blocked=signal.pthread_sigmask(signal.SIG_BLOCK, []); "
        f"pathlib.Path({str(state_path)!r}).write_text("
        "f'{os.getpid()} {os.getpgrp()} ' + "
        "(','.join(str(int(value)) for value in sorted(blocked)) or '-')); "
        "time.sleep(60)"
    )
    wrapper_code = f"""
import pathlib
import sys
sys.path.insert(0, {str(agent_root)!r})
from experiments import c2_m5_source_pilot as pilot
try:
    pilot._run_bounded_process(
        command=[sys.executable, "-I", "-S", "-B", "-c", {child_code!r}],
        workspace=pathlib.Path({str(workspace)!r}),
        environment={{"PATH": "/usr/bin:/bin"}},
        timeout_seconds=30,
    )
except pilot.C2M5SourcePilotError as exc:
    raise SystemExit(17 if "signal" in str(exc) else 18)
raise SystemExit(19)
"""
    wrapper = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", wrapper_code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    child_group: int | None = None
    try:
        deadline = time.monotonic() + 5
        while child_pid is None:
            if wrapper.poll() is not None:
                stdout, stderr = wrapper.communicate()
                raise AssertionError(
                    f"attempt wrapper exited early: {stdout!r} {stderr!r}"
                )
            try:
                pieces = state_path.read_text(encoding="ascii").split()
                child_pid = int(pieces[0])
                child_group = int(pieces[1])
                blocked = {
                    int(value)
                    for value in pieces[2].split(",")
                    if value != "-"
                }
            except (FileNotFoundError, ValueError, IndexError):
                child_pid = None
            if time.monotonic() >= deadline:
                raise AssertionError("attempt child did not publish its state")
            if child_pid is None:
                time.sleep(0.01)
        assert not blocked.intersection(pilot._WATCHED_SIGNALS)
        os.kill(wrapper.pid, signum)
        wrapper.wait(timeout=10)
        stdout, stderr = wrapper.communicate()
        assert wrapper.returncode == 17, (stdout, stderr)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert child_group is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(child_group, 0)
    finally:
        if child_group is not None:
            try:
                os.killpg(child_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait()


def test_attempt_spawn_failure_restores_signal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "spawn-failure"
    workspace.mkdir()
    original_handlers = {
        signum: signal.getsignal(signum)
        for signum in pilot._WATCHED_SIGNALS
    }
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    def fail_spawn(*_: object, **__: object) -> subprocess.Popen[bytes]:
        raise OSError("fixed attempt spawn failure")

    monkeypatch.setattr(pilot.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="fixed attempt spawn failure"):
        pilot._run_bounded_process(
            command=[sys.executable, "-I", "-S", "-B", "-c", "pass"],
            workspace=workspace,
            environment={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
        )

    assert {
        signum: signal.getsignal(signum)
        for signum in pilot._WATCHED_SIGNALS
    } == original_handlers
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == original_mask


def test_attempt_census_failure_still_kills_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class FakeProcess:
        pid = 701

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            events.append(("kill", 701))

        @staticmethod
        def wait(timeout: float) -> int:
            events.append(("wait", timeout))
            return -signal.SIGKILL

    monkeypatch.setattr(
        pilot.os,
        "killpg",
        lambda group_id, signum: events.append(
            ("killpg", group_id, signum)
        ),
    )
    monkeypatch.setattr(
        pilot,
        "_process_group_members",
        lambda _group_id: (_ for _ in ()).throw(
            pilot.C2M5SourcePilotError("fixed attempt census failure")
        ),
    )

    with pytest.raises(
        pilot.C2M5SourcePilotError,
        match="fixed attempt census failure",
    ):
        pilot._terminate_process_group(FakeProcess())

    assert ("killpg", 701, signal.SIGKILL) in events
    assert ("wait", 2.0) in events


def test_acquisition_lease_initialization_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    monkeypatch.setattr(
        pilot.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(
            OSError("fixed lease fsync failure")
        ),
    )

    with pytest.raises(OSError, match="fixed lease fsync failure"):
        pilot._ExclusiveLease(tmp_path, attestation)

    assert not (tmp_path / pilot.LOCK_NAME).exists()


def test_acquisition_lease_short_write_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    monkeypatch.setattr(pilot.os, "write", lambda _descriptor, payload: len(payload) - 1)

    with pytest.raises(
        pilot.C2M5SourcePilotError,
        match="lease write was incomplete",
    ):
        pilot._ExclusiveLease(tmp_path, attestation)

    assert not (tmp_path / pilot.LOCK_NAME).exists()


def test_unverified_process_cleanup_preserves_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "unverified-process-cleanup"
    workspace.mkdir()
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    lease = pilot._ExclusiveLease(tmp_path, attestation)
    cleanup = pilot._ExecutionCleanup(lambda: None, [], lease)
    real_terminate = pilot._terminate_process_group

    def fail_after_termination(
        process: subprocess.Popen[bytes],
    ) -> tuple[int, ...]:
        real_terminate(process)
        raise pilot.C2M5SourcePilotError(
            "fixed post-termination census failure"
        )

    monkeypatch.setattr(
        pilot,
        "_terminate_process_group",
        fail_after_termination,
    )
    lease_path = tmp_path / pilot.LOCK_NAME
    try:
        with pytest.raises(
            pilot.C2M5SourcePilotError,
            match="post-termination census failure",
        ):
            with lease:
                with cleanup:
                    pilot._run_bounded_process(
                        command=[
                            sys.executable,
                            "-I",
                            "-S",
                            "-B",
                            "-c",
                            "pass",
                        ],
                        workspace=workspace,
                        environment={"PATH": "/usr/bin:/bin"},
                        timeout_seconds=5,
                        _cleanup_failure=cleanup.inhibit_lease_release,
                    )
        assert lease_path.exists()
        assert lease._descriptor == -1
    finally:
        if lease._descriptor != -1:
            os.close(lease._descriptor)
        lease_path.unlink(missing_ok=True)


def test_acquisition_lease_cleanup_preserves_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    lease = pilot._ExclusiveLease(tmp_path, attestation)
    lease.authorize_release()
    real_close = lease.close

    def fail_close() -> None:
        raise OSError("fixed lease close failure")

    monkeypatch.setattr(lease, "close", fail_close)
    try:
        with pytest.raises(ValueError, match="fixed primary failure") as raised:
            with lease:
                raise ValueError("fixed primary failure")
        assert isinstance(raised.value.__cause__, OSError)
    finally:
        monkeypatch.setattr(lease, "close", real_close)
        real_close()


def test_staging_cleanup_finishes_before_lease_release(
    tmp_path: Path,
) -> None:
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    lease_path = tmp_path / pilot.LOCK_NAME
    events: list[str] = []

    class FakeRoot:
        published = False

        @staticmethod
        def discard_unpublished() -> None:
            assert lease_path.exists()
            events.append("discard")

        @staticmethod
        def close() -> None:
            assert lease_path.exists()
            events.append("close")

    root = FakeRoot()
    lease = pilot._ExclusiveLease(tmp_path, attestation)
    with pytest.raises(ValueError, match="fixed execution failure"):
        with lease:
            with pilot._ExecutionCleanup(lambda: root, [], lease):
                raise ValueError("fixed execution failure")

    assert events == ["discard", "close"]
    assert not lease_path.exists()


def test_pre_staging_signal_releases_clean_lease(
    tmp_path: Path,
) -> None:
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    supervisor = pilot._SourcePilotSignalSupervisor()
    supervisor._record(signal.SIGTERM, None)
    lease = pilot._ExclusiveLease(tmp_path, attestation)
    cleanup = pilot._ExecutionCleanup(lambda: None, [], lease)
    raised_error: BaseException = RuntimeError("signal test cleanup")
    try:
        with pytest.raises(
            pilot.C2M5SourcePilotError,
            match="interrupted by signal",
        ) as raised:
            with lease:
                with cleanup:
                    supervisor.checkpoint()
        raised_error = raised.value
        assert not (tmp_path / pilot.LOCK_NAME).exists()
    finally:
        supervisor.finish(raised_error)


def test_staging_cleanup_failure_preserves_lease(
    tmp_path: Path,
) -> None:
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    lease_path = tmp_path / pilot.LOCK_NAME

    class FakeRoot:
        published = False

        @staticmethod
        def discard_unpublished() -> None:
            raise OSError("fixed staging discard failure")

        @staticmethod
        def close() -> None:
            pass

    lease = pilot._ExclusiveLease(tmp_path, attestation)
    try:
        with pytest.raises(ValueError, match="fixed execution failure") as raised:
            with lease:
                with pilot._ExecutionCleanup(
                    lambda: FakeRoot(),
                    [],
                    lease,
                ):
                    raise ValueError("fixed execution failure")
        assert isinstance(raised.value.__cause__, OSError)
        assert lease_path.exists()
        assert lease._descriptor == -1
    finally:
        if lease._descriptor != -1:
            os.close(lease._descriptor)
        lease_path.unlink(missing_ok=True)


def test_target_root_creation_failure_discards_unreturned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / pilot.EXPECTED_RAW_ROOT
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    staging_name = (
        f".{raw_root.name}.c2-remediation-staging-"
        f"{'b' * 32}"
    )

    class Guard:
        def __init__(self) -> None:
            self.parent_fd = os.open(
                tmp_path,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            self.leaf_name = raw_root.name

        def close(self) -> None:
            if self.parent_fd != -1:
                os.close(self.parent_fd)
                self.parent_fd = -1

    guard = Guard()
    monkeypatch.setattr(
        finalizer,
        "_open_private_staging_parent",
        lambda _raw_root: guard,
    )

    def fail_after_mkdir(_raw_root: Path) -> None:
        staging = tmp_path / staging_name
        staging.mkdir(mode=0o700)
        (staging / "partial").write_bytes(b"partial")
        raise OSError("fixed post-mkdir creation failure")

    monkeypatch.setattr(finalizer, "_create_target_root", fail_after_mkdir)
    lease = pilot._ExclusiveLease(tmp_path, attestation)
    cleanup = pilot._ExecutionCleanup(lambda: None, [], lease)

    with pytest.raises(OSError, match="fixed post-mkdir creation failure"):
        with lease:
            with cleanup:
                pilot._create_target_root_transactionally(
                    finalizer=finalizer,
                    raw_root=raw_root,
                    cleanup=cleanup,
                )

    assert not (tmp_path / staging_name).exists()
    assert not (tmp_path / pilot.LOCK_NAME).exists()


def test_preexisting_staging_preserves_fail_closed_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / pilot.EXPECTED_RAW_ROOT
    attestation = SimpleNamespace(attestation_commit="a" * 40)
    staging = tmp_path / (
        f".{raw_root.name}.c2-remediation-staging-"
        f"{'c' * 32}"
    )
    staging.mkdir(mode=0o700)

    class Guard:
        def __init__(self) -> None:
            self.parent_fd = os.open(
                tmp_path,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            self.leaf_name = raw_root.name

        def close(self) -> None:
            if self.parent_fd != -1:
                os.close(self.parent_fd)
                self.parent_fd = -1

    monkeypatch.setattr(
        finalizer,
        "_open_private_staging_parent",
        lambda _raw_root: Guard(),
    )
    lease = pilot._ExclusiveLease(tmp_path, attestation)
    cleanup = pilot._ExecutionCleanup(lambda: None, [], lease)
    lease_path = tmp_path / pilot.LOCK_NAME
    try:
        with pytest.raises(
            pilot.C2M5SourcePilotError,
            match="pre-existing M5 source-pilot staging root",
        ):
            with lease:
                with cleanup:
                    pilot._create_target_root_transactionally(
                        finalizer=finalizer,
                        raw_root=raw_root,
                        cleanup=cleanup,
                    )
        assert staging.exists()
        assert lease_path.exists()
    finally:
        lease_path.unlink(missing_ok=True)
        staging.rmdir()


def test_publication_commit_is_signal_atomic() -> None:
    supervisor = pilot._SourcePilotSignalSupervisor()
    published: list[bool] = []
    try:
        def publish() -> None:
            os.kill(os.getpid(), signal.SIGTERM)
            published.append(True)

        supervisor.commit_publication(publish)
        assert published == [True]
        assert supervisor.publication_committed is True
        assert supervisor.interrupted_signal == signal.SIGTERM
        supervisor.checkpoint()
    finally:
        supervisor.finish(None)


def test_publication_commit_rejects_preexisting_signal() -> None:
    supervisor = pilot._SourcePilotSignalSupervisor()
    published: list[bool] = []
    raised_error: BaseException = RuntimeError("signal test cleanup")
    try:
        supervisor._record(signal.SIGTERM, None)
        with pytest.raises(
            pilot.C2M5SourcePilotError,
            match="interrupted by signal",
        ) as raised:
            supervisor.commit_publication(
                lambda: published.append(True)
            )
        raised_error = raised.value
        assert published == []
        assert supervisor.publication_committed is False
    finally:
        supervisor.finish(raised_error)


def test_dependency_snapshot_is_vendored_pinned_and_closed(tmp_path: Path) -> None:
    binding, payloads, manifest_payload, archive_payload = _dependency_closure()

    assert binding["runtime_file_count"] == len(payloads)
    assert binding["schema_version"] == "c2_m5_python_dependency_binding_v2"
    assert binding["python"]["trust_policy"].startswith("ROOT_OWNED")
    assert "requests/__init__.py" in payloads
    assert "bs4/__init__.py" in payloads
    assert "jsonschema/__init__.py" in payloads
    assert not any(path.startswith("rich/") for path in payloads)

    tampered = json.loads(manifest_payload)
    tampered["runtime_inventory"][0]["sha256"] = "0" * 64
    with pytest.raises(
        m5_attestation.C2M5BootstrapAttestationError,
        match="payload mismatch",
    ):
        m5_attestation._verified_dependency_payloads(
            pilot._canonical_bytes(tampered) + b"\n",
            archive_payload,
        )
    modified_archive = bytearray(archive_payload)
    modified_archive[len(modified_archive) // 2] ^= 1
    with pytest.raises(
        m5_attestation.C2M5BootstrapAttestationError,
        match="archive binding",
    ):
        m5_attestation._verified_dependency_payloads(
            manifest_payload,
            bytes(modified_archive),
        )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pilot._materialize_dependencies(workspace, payloads)
    dependency_root = workspace / "dependencies"
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(dependency_root)!r});"
        "import bs4,jsonschema,requests,rpds;"
        f"assert all(str({str(dependency_root)!r}) in module.__file__ "
        "for module in (bs4,jsonschema,requests,rpds))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_downloader_git_probe_uses_absolute_binary_and_scrubbed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="fixed\n", stderr="")

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)

    assert pilot._git_text(Path("/tmp/worktree"), ("rev-parse", "HEAD"), "git") == "fixed"
    command = captured["command"]
    environment = captured["environment"]
    assert isinstance(command, list) and command[0] == "/usr/bin/git"
    assert "--no-replace-objects" in command
    assert isinstance(environment, dict)
    assert environment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_PROTOCOL_FROM_USER"] == "0"
    assert environment["GIT_ALLOW_PROTOCOL"] == ""
    assert "protocol.allow=never" in command
    assert m5_attestation._GIT_ENV["GIT_NO_LAZY_FETCH"] == "1"
    assert m5_attestation._GIT_ENV["GIT_PROTOCOL_FROM_USER"] == "0"
    assert m5_attestation._GIT_ENV["GIT_ALLOW_PROTOCOL"] == ""
    assert "protocol.allow=never" in m5_attestation._GIT_CONFIG


def test_filter_free_clean_tree_never_runs_checkout_filters(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "M5 Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "m5@example.invalid"],
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("trusted\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    marker = tmp_path / "filter-ran"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "config",
            "filter.m5evil.clean",
            f"/usr/bin/touch {shlex.quote(str(marker))}; /bin/cat",
        ],
        check=True,
    )
    info_attributes = repository / ".git" / "info" / "attributes"
    info_attributes.write_text("*.txt filter=m5evil\n", encoding="utf-8")
    tracked.write_text("changed\n", encoding="utf-8")
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    with pytest.raises(
        m5_attestation.C2M5BootstrapAttestationError,
        match="differs from HEAD",
    ):
        m5_attestation._verify_filter_free_clean_tree(repository, commit)
    assert not marker.exists()


def test_attestation_git_log_disables_signature_program(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "M5 Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "m5@example.invalid"],
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("trusted\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    original = subprocess.check_output(
        ["git", "-C", str(repository), "cat-file", "commit", "HEAD"]
    )
    headers, separator, message = original.partition(b"\n\n")
    assert separator
    signed = (
        headers
        + b"\ngpgsig -----BEGIN PGP SIGNATURE-----\n"
        + b" fake\n"
        + b" -----END PGP SIGNATURE-----\n\n"
        + message
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "hash-object", "-t", "commit", "-w", "--stdin"],
        input=signed,
        text=False,
    ).decode("ascii").strip()
    subprocess.run(
        ["git", "-C", str(repository), "update-ref", "HEAD", commit],
        check=True,
    )
    marker = tmp_path / "gpg-ran"
    gpg_program = tmp_path / "gpg-program"
    gpg_program.write_text(
        f"#!/bin/sh\n/usr/bin/touch {shlex.quote(str(marker))}\nexit 1\n",
        encoding="utf-8",
    )
    gpg_program.chmod(0o700)
    subprocess.run(
        ["git", "-C", str(repository), "config", "log.showSignature", "true"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "gpg.program", str(gpg_program)],
        check=True,
    )

    observed = m5_attestation._git_text(
        repository,
        ("log", "-1", "--format=%H", "--", "tracked.txt"),
        "signed log",
    )

    assert observed == commit
    assert not marker.exists()


def test_final_m5_manifest_verifies_positive_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_repository = Path(__file__).resolve().parents[2]
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "M5 Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "m5@example.invalid"],
        check=True,
    )
    for relative in sorted(m5_attestation.REQUIRED_ATTESTED_PATHS):
        source = source_repository / relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in {
        "apply_fallback.py",
        "apply_stage_updates.py",
        "run_chain.py",
        "run_multi_panel.py",
    }:
        (repository / "agent" / name).write_text(
            '"""Synthetic non-runtime file for clean-tree verification."""\n',
            encoding="utf-8",
        )
    subprocess.run(["git", "-C", str(repository), "add", "agent"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "implementation"],
        check=True,
    )
    implementation_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    attested_paths = []
    for relative in sorted(m5_attestation.REQUIRED_ATTESTED_PATHS):
        payload = (repository / relative).read_bytes()
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", f"HEAD:{relative}"],
            text=True,
        ).strip()
        attested_paths.append(
            {
                "relative_path": relative,
                "git_blob_object_id": blob,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "c2_m5_source_pilot_attestation_v1",
        "approved_implementation_commit_full": implementation_commit,
        "attested_paths": attested_paths,
    }
    manifest_path = repository / m5_attestation.ATTESTATION_RELATIVE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(m5_attestation._canonical_bytes(manifest) + b"\n")
    subprocess.run(
        ["git", "-C", str(repository), "add", m5_attestation.ATTESTATION_RELATIVE],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "attestation"],
        check=True,
    )
    attestation_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dependency_manifest = json.loads(
        (
            repository / m5_attestation.DEPENDENCY_MANIFEST_RELATIVE
        ).read_bytes()
    )
    python_binding = dependency_manifest["python"]
    bootstrap_path = repository / "agent/c2_m5_source_pilot_bootstrap.py"
    monkeypatch.setattr(
        m5_attestation,
        "__file__",
        str(repository / "agent/c2_m5_bootstrap_attestation.py"),
    )

    observed = m5_attestation.verify_adapter_attestation(
        activate=False,
        expected_attestation_commit=attestation_commit,
        expected_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        expected_bootstrap_sha256=hashlib.sha256(
            bootstrap_path.read_bytes()
        ).hexdigest(),
        expected_python_sha256=python_binding["executable_sha256"],
        expected_python_library_sha256=python_binding[
            "runtime_library_sha256"
        ],
    )

    assert observed.implementation_commit == implementation_commit
    assert observed.attestation_commit == attestation_commit
    assert set(observed.path_sha256) == m5_attestation.REQUIRED_ATTESTED_PATHS
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(repository / "agent/c2_m5_source_pilot_bootstrap.py"),
            "--help",
        ],
        cwd=repository,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
            "C2_M5_EXPECTED_ATTESTATION_COMMIT": attestation_commit,
            "C2_M5_EXPECTED_MANIFEST_SHA256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "C2_M5_EXPECTED_BOOTSTRAP_SHA256": hashlib.sha256(
                bootstrap_path.read_bytes()
            ).hexdigest(),
            "C2_M5_EXPECTED_PYTHON_SHA256": python_binding[
                "executable_sha256"
            ],
            "C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256": python_binding[
                "runtime_library_sha256"
            ],
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "execute" in completed.stdout


def test_m5_release_manifest_covers_transitive_test_imports() -> None:
    assert {
        "agent/experiments/aggregate.py",
        "agent/experiments/c2_full_replacement_evidence.py",
        "agent/experiments/c2_full_replacement_finalizer.py",
        "agent/experiments/c2_full_replacement_stageb_policy.py",
        "agent/experiments/c2_terminal_finalizer.py",
        "agent/experiments/decision_report.py",
        "agent/experiments/harness.py",
        "agent/experiments/matrix.py",
        "agent/experiments/production_statistics.py",
        "agent/experiments/provenance_stage.py",
        "agent/experiments/rejudge.py",
        "agent/experiments/scheduler.py",
        "agent/tests/__init__.py",
        "agent/experiments/tests/__init__.py",
        "agent/tests/fixtures/c2_source_bearing_extension_test_attestation.json",
        "agent/sample.xlsx",
    }.issubset(m5_attestation.REQUIRED_ATTESTED_PATHS)


def test_release_process_topology_rejects_unapproved_session_control() -> None:
    repository = Path(__file__).resolve().parents[2]
    relative_paths = (
        "agent/c2_m5_release_test_runner.py",
        "agent/experiments/c2_m5_source_pilot.py",
        "agent/tests/test_c2_m5_source_pilot.py",
    )
    payloads = {
        relative: (repository / relative).read_bytes()
        for relative in relative_paths
    }
    release_runner._verify_release_process_topology(payloads)

    payloads["agent/unapproved.py"] = (
        b"import os\nos." + b"set" + b"sid()\n"
    )
    with pytest.raises(
        release_runner.M5ReleaseTestError,
        match="forbidden session control",
    ):
        release_runner._verify_release_process_topology(payloads)

    payloads = {
        relative: (repository / relative).read_bytes()
        for relative in relative_paths
    }
    payloads["agent/bypass.py"] = (
        b"process = _ORIGINAL_" + b"POPEN(['x'])\n"
    )
    with pytest.raises(
        release_runner.M5ReleaseTestError,
        match="bypasses containment",
    ):
        release_runner._verify_release_process_topology(payloads)


def test_release_runner_refuses_without_external_runner_digest() -> None:
    runner = (
        Path(__file__).resolve().parents[1] / "c2_m5_release_test_runner.py"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(runner)],
        cwd=runner.parent,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "C2_M5_EXPECTED_RELEASE_RUNNER_SHA256" in completed.stderr


def test_release_runner_fixes_pytest_configuration_boundary() -> None:
    child = release_runner._PYTEST_CHILD
    assert '"-c"' in child
    assert '"--rootdir"' in child
    assert '"--confcutdir"' in child
    assert '"--noconftest"' in child
    assert release_runner._PYTEST_CONFIG == b"[pytest]\naddopts =\n"


def test_private_git_copy_never_executes_source_upload_pack_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "M5 Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "m5@example.invalid"],
        check=True,
    )
    (repository / "tracked.txt").write_text("retained\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "retained"],
        check=True,
    )
    marker = tmp_path / "upload-pack-ran"
    hook = tmp_path / "pack-objects-hook"
    hook.write_text(
        "#!/bin/sh\n"
        f"/usr/bin/touch {shlex.quote(str(marker))}\n"
        'exec /usr/bin/git pack-objects "$@"\n',
        encoding="utf-8",
    )
    hook.chmod(0o700)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "config",
            "uploadpack.packObjectsHook",
            str(hook),
        ],
        check=True,
    )

    private_repository = tmp_path / "private"
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    monkeypatch.setattr(release_runner, "_REPOSITORY", repository)
    release_runner._clone_private_git_repository(
        private_repository,
        head,
    )

    assert not marker.exists()
    assert (
        subprocess.check_output(
            ["git", "-C", str(private_repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        == head
    )


def test_m5_topology_helpers_reject_grafts_and_merge_parents(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "M5 Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "m5@example.invalid"],
        check=True,
    )
    (repository / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "base.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "base"],
        check=True,
    )
    base_branch = subprocess.check_output(
        ["git", "-C", str(repository), "symbolic-ref", "--short", "HEAD"],
        text=True,
    ).strip()
    subprocess.run(
        ["git", "-C", str(repository), "checkout", "-q", "-b", "side"],
        check=True,
    )
    (repository / "side.txt").write_text("side\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "side.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "side"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "checkout", "-q", base_branch],
        check=True,
    )
    (repository / "main.txt").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "main.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "main"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "merge", "-q", "--no-ff", "side", "-m", "merge"],
        check=True,
    )
    merge_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    with pytest.raises(
        m5_attestation.C2M5BootstrapAttestationError,
        match="exactly one literal parent",
    ):
        m5_attestation._single_parent_commit(
            repository,
            merge_commit,
            "test M5 merge",
        )
    with pytest.raises(
        release_runner.M5ReleaseTestError,
        match="exactly one literal parent",
    ):
        release_runner._single_parent_commit(
            repository,
            merge_commit,
            "test M5 merge",
        )

    graft_path = repository / ".git/info/grafts"
    graft_path.parent.mkdir(parents=True, exist_ok=True)
    graft_path.write_text(f"{merge_commit} {merge_commit}\n", encoding="ascii")
    with pytest.raises(
        m5_attestation.C2M5BootstrapAttestationError,
        match="graft metadata",
    ):
        m5_attestation._require_no_git_grafts(repository)
    with pytest.raises(
        release_runner.M5ReleaseTestError,
        match="graft metadata",
    ):
        release_runner._require_no_git_grafts(repository)


def test_release_runner_kills_descendants_after_leader_exit(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    leader: subprocess.Popen[bytes] | None = None
    child_pid: int | None = None
    try:
        leader = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import pathlib,subprocess,sys; "
                    "child=subprocess.Popen("
                    "[sys.executable,'-I','-S','-B','-c',"
                    "'import time; time.sleep(60)']); "
                    f"pathlib.Path({str(child_pid_path)!r}).write_text("
                    "str(child.pid))"
                ),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        assert leader.wait(timeout=5) == 0
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        release_runner._terminate_process_group(leader)

        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if leader is not None:
            release_runner._terminate_process_group(leader)


def test_release_containment_keeps_nested_group_in_test_session() -> None:
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import os; "
                    "print(os.getpid(), os.getpgid(0), os.getsid(0))"
                ),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        process_id, group_id, session_id = map(int, stdout.split())
        assert process_id == group_id == process.pid
        expected_session = int(
            os.environ.get(
                release_runner._RELEASE_TEST_SESSION_ENV,
                str(process.pid),
            )
        )
        assert session_id == expected_session
        if release_runner._RELEASE_TEST_CONTAINMENT_INSTALLED:
            registry_path = Path(
                os.environ[release_runner._RELEASE_TEST_REGISTRY_ENV]
            )
            registrations = {
                tuple(map(int, line.split()))
                for line in registry_path.read_text(
                    encoding="ascii"
                ).splitlines()
            }
            group_registrations = {
                registration
                for registration in registrations
                if registration[0] == process.pid
            }
            assert (process.pid, process.pid) in group_registrations
            assert any(
                anchor_pid != process.pid
                for _, anchor_pid in group_registrations
            )
            assert release_runner._CONTAINED_EXEC_TRAMPOLINE.index(
                "os.read(release_fd"
            ) < release_runner._CONTAINED_EXEC_TRAMPOLINE.index(
                "os.setpgid"
            )
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def test_release_root_rejects_ambient_session_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    registry = release_runner._open_test_session_registry(registry_path)
    completion_pipe = os.pipe()
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_SESSION_ENV,
        str(os.getsid(0)),
    )
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_REGISTRY_ENV,
        str(registry_path),
    )
    try:
        with pytest.raises(
            release_runner.M5ReleaseTestError,
            match="rejects ambient capabilities",
        ):
            release_runner._run_release_child(
                [sys.executable, "-I", "-S", "-B", "-c", "pass"],
                cwd=tmp_path,
                environment={"PATH": "/usr/bin:/bin"},
                _completion_pipe=completion_pipe,
                _session_registry=registry,
            )
    finally:
        os.close(registry.descriptor)


def test_release_root_session_is_process_nonreentrant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    registry = release_runner._open_test_session_registry(registry_path)
    completion_pipe = os.pipe()
    monkeypatch.setattr(
        release_runner,
        "_RELEASE_TEST_CONTAINMENT_INSTALLED",
        True,
    )
    for name in (
        release_runner._RELEASE_TEST_SESSION_ENV,
        release_runner._RELEASE_TEST_REGISTRY_ENV,
        release_runner._RELEASE_TEST_REGISTRY_DEVICE_ENV,
        release_runner._RELEASE_TEST_REGISTRY_INODE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    try:
        with pytest.raises(
            release_runner.M5ReleaseTestError,
            match="rejects ambient capabilities",
        ):
            release_runner._run_release_child(
                [sys.executable, "-I", "-S", "-B", "-c", "pass"],
                cwd=tmp_path,
                environment={"PATH": "/usr/bin:/bin"},
                _completion_pipe=completion_pipe,
                _session_registry=registry,
            )
    finally:
        os.close(registry.descriptor)


def test_unacknowledged_contained_process_is_stopped_before_group_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    censuses = iter(
        (
            ((701, 600, "R"),),
            ((701, 701, "T"), (702, 701, "T")),
            ((701, 701, "Z"), (702, 701, "Z")),
        )
    )

    class FakeProcess:
        pid = 701

        @staticmethod
        def kill() -> None:
            events.append(("kill", 701))

        @staticmethod
        def wait(timeout: float) -> int:
            events.append(("wait", timeout))
            return -signal.SIGKILL

    monkeypatch.setattr(
        release_runner,
        "_owned_test_session_members",
        lambda _session_id: next(censuses),
    )
    monkeypatch.setattr(
        release_runner.os,
        "kill",
        lambda process_id, signum: events.append(
            ("signal", process_id, signum)
        ),
    )
    monkeypatch.setattr(
        release_runner.os,
        "killpg",
        lambda group_id, signum: events.append(
            ("killpg", group_id, signum)
        ),
    )
    monkeypatch.setattr(release_runner.time, "sleep", lambda _seconds: None)

    release_runner._terminate_unacknowledged_process(FakeProcess(), 600)

    assert events == [
        ("signal", 701, signal.SIGSTOP),
        ("killpg", 701, signal.SIGKILL),
        ("kill", 701),
        ("wait", release_runner._PROCESS_GROUP_CLEANUP_SECONDS),
    ]


def test_pending_registration_precedes_contained_group_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[int, bytes]] = []

    def record_write(descriptor: int, payload: bytes) -> int:
        writes.append((descriptor, payload))
        return len(payload)

    monkeypatch.setattr(release_runner.os, "write", record_write)
    release_runner._register_and_release_contained_group(11, 12, 701)

    assert writes == [
        (11, b"701 701\n"),
        (12, b"1"),
    ]


def test_contained_spawn_uses_installation_pinned_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned_path = tmp_path / "pinned.log"
    rebound_path = tmp_path / "rebound.log"
    for path in (pinned_path, rebound_path):
        path.write_bytes(b"")
        path.chmod(0o600)
    pinned = pinned_path.stat()
    rebound = rebound_path.stat()
    monkeypatch.setattr(
        release_runner,
        "_RELEASE_TEST_CONTAINMENT_INSTALLED",
        True,
    )
    monkeypatch.setattr(
        release_runner,
        "_RELEASE_TEST_PINNED_REGISTRY",
        (pinned_path, pinned.st_dev, pinned.st_ino),
    )
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_REGISTRY_ENV,
        str(rebound_path),
    )
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_REGISTRY_DEVICE_ENV,
        str(rebound.st_dev),
    )
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_REGISTRY_INODE_ENV,
        str(rebound.st_ino),
    )

    assert release_runner._contained_release_test_registry_binding() == (
        pinned_path,
        pinned.st_dev,
        pinned.st_ino,
    )


def test_unacknowledged_contained_anchor_cannot_survive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    registry = release_runner._open_test_session_registry(registry_path)
    state_path = tmp_path / "unacknowledged-pids"
    session_id = os.getsid(0)
    for name, value in (
        (release_runner._RELEASE_TEST_SESSION_ENV, str(session_id)),
        (release_runner._RELEASE_TEST_REGISTRY_ENV, str(registry.path)),
        (
            release_runner._RELEASE_TEST_REGISTRY_DEVICE_ENV,
            str(registry.device),
        ),
        (
            release_runner._RELEASE_TEST_REGISTRY_INODE_ENV,
            str(registry.inode),
        ),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        release_runner,
        "_CONTAINED_SPAWN_TIMEOUT_SECONDS",
        0.5,
    )
    monkeypatch.setattr(
        release_runner,
        "_CONTAINED_EXEC_TRAMPOLINE",
        """\
import os
import signal
import sys

ready_fd = int(sys.argv[1])
registry_fd = int(sys.argv[2])
release_fd = int(sys.argv[3])
state_path = sys.argv[4]
if os.read(release_fd, 1) != b"1":
    raise SystemExit(2)
os.close(release_fd)
os.setpgid(0, 0)
anchor_pid = os.fork()
if anchor_pid == 0:
    os.close(ready_fd)
    os.close(registry_fd)
    while True:
        signal.pause()
with open(state_path, "w", encoding="ascii") as stream:
    stream.write(f"{os.getpid()} {anchor_pid}\\n")
    stream.flush()
    os.fsync(stream.fileno())
while True:
    signal.pause()
""",
    )
    try:
        with pytest.raises(
            release_runner.M5ReleaseTestError,
            match="did not acknowledge",
        ):
            release_runner._start_contained_popen(
                getattr(release_runner, "_ORIGINAL_" + "POPEN"),
                ([str(state_path)],),
                {
                    "env": dict(os.environ),
                    "stdin": subprocess.DEVNULL,
                },
            )
        leader_pid, anchor_pid = map(
            int,
            state_path.read_text(encoding="ascii").split(),
        )
        assert leader_pid != anchor_pid
        assert not [
            (process_id, state)
            for process_id, group_id, state in (
                release_runner._owned_test_session_members(session_id)
            )
            if group_id == leader_pid and state[:1] != "Z"
        ]
    finally:
        os.close(registry.descriptor)


def test_release_completion_requires_trusted_supervisor_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    contained = release_runner._RELEASE_TEST_CONTAINMENT_INSTALLED
    if not contained:
        for name in (
            release_runner._RELEASE_TEST_SESSION_ENV,
            release_runner._RELEASE_TEST_REGISTRY_ENV,
            release_runner._RELEASE_TEST_REGISTRY_DEVICE_ENV,
            release_runner._RELEASE_TEST_REGISTRY_INODE_ENV,
        ):
            monkeypatch.delenv(name, raising=False)

    def run_completion(child_code: str, timeout: float = 5) -> int:
        registry = release_runner._open_test_session_registry(registry_path)
        read_fd, write_fd = os.pipe()
        monkeypatch.setattr(
            release_runner,
            "_TEST_TIMEOUT_SECONDS",
            timeout,
        )
        try:
            return release_runner._run_release_child(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    child_code,
                    str(write_fd),
                ],
                cwd=tmp_path,
                environment={"PATH": "/usr/bin:/bin"},
                _completion_pipe=(read_fd, write_fd),
                _session_registry=None if contained else registry,
            )
        finally:
            os.close(registry.descriptor)

    valid = (
        "import os,signal,sys; fd=int(sys.argv[1]); "
        "os.write(fd,b'0\\n'); os.close(fd); signal.pause()"
    )
    assert run_completion(valid) == 0

    trailing = (
        "import os,signal,sys,time; fd=int(sys.argv[1]); "
        "os.write(fd,b'0\\n'); time.sleep(0.05); "
        "os.write(fd,b'junk'); os.close(fd); signal.pause()"
    )
    with pytest.raises(
        release_runner.M5ReleaseTestError,
        match="completion status is malformed",
    ):
        run_completion(trailing)

    for payload, message in (
        (b"", "malformed"),
        (b"0", "malformed"),
        (b"256\n", "invalid"),
        (b"12345678901234567\n", "oversized"),
    ):
        invalid = (
            "import os,signal,sys; fd=int(sys.argv[1]); "
            f"os.write(fd,{payload!r}); os.close(fd); signal.pause()"
        )
        with pytest.raises(
            release_runner.M5ReleaseTestError,
            match=message,
        ):
            run_completion(invalid)

    held_open = (
        "import os,signal,sys; fd=int(sys.argv[1]); "
        "os.write(fd,b'0\\n'); signal.pause()"
    )
    with pytest.raises(
        release_runner.M5ReleaseTestError,
        match="fixed timeout",
    ):
        run_completion(held_open, timeout=0.1)


def test_pytest_worker_cannot_inherit_completion_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    contained = release_runner._RELEASE_TEST_CONTAINMENT_INSTALLED
    if not contained:
        for name in (
            release_runner._RELEASE_TEST_SESSION_ENV,
            release_runner._RELEASE_TEST_REGISTRY_ENV,
            release_runner._RELEASE_TEST_REGISTRY_DEVICE_ENV,
            release_runner._RELEASE_TEST_REGISTRY_INODE_ENV,
        ):
            monkeypatch.delenv(name, raising=False)
    registry = release_runner._open_test_session_registry(registry_path)
    read_fd, write_fd = os.pipe()
    worker_code = (
        "import os,sys; fd=int(sys.argv[1]); "
        "\ntry: os.fstat(fd)\n"
        "except OSError: raise SystemExit(0)\n"
        "raise SystemExit(9)"
    )
    try:
        result = release_runner._run_release_child(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                release_runner._PYTEST_SESSION_SUPERVISOR,
                str(write_fd),
                worker_code,
                str(write_fd),
            ],
            cwd=tmp_path,
            environment={
                "PATH": "/usr/bin:/bin",
                release_runner._RELEASE_TEST_REGISTRY_ENV: str(
                    registry_path
                ),
                release_runner._RELEASE_TEST_REGISTRY_DEVICE_ENV: str(
                    registry.device
                ),
                release_runner._RELEASE_TEST_REGISTRY_INODE_ENV: str(
                    registry.inode
                ),
            },
            _completion_pipe=(read_fd, write_fd),
            _session_registry=None if contained else registry,
        )
    finally:
        os.close(registry.descriptor)
    assert result == 0


def test_release_registry_is_pinned_and_nonblocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    registry = release_runner._open_test_session_registry(registry_path)
    registry_path.unlink()
    os.mkfifo(registry_path, 0o600)
    try:
        registrations, invalid = release_runner._registered_test_groups(
            registry,
            os.getsid(0),
        )
        assert registrations == set()
        assert invalid
    finally:
        os.close(registry.descriptor)
        registry_path.unlink()

    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    original = release_runner._open_test_session_registry(registry_path)
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_REGISTRY_ENV,
        str(registry_path),
    )
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_REGISTRY_DEVICE_ENV,
        str(original.device),
    )
    monkeypatch.setenv(
        release_runner._RELEASE_TEST_REGISTRY_INODE_ENV,
        str(original.inode),
    )
    registry_path.unlink()
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    try:
        with pytest.raises(
            release_runner.M5ReleaseTestError,
            match="identity changed",
        ):
            release_runner._open_release_test_registry_for_append()
    finally:
        os.close(original.descriptor)


def test_registered_group_cleanup_attempts_every_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations = {(301, 302), (401, 402), (501, 502)}
    group_by_anchor = {anchor: group for group, anchor in registrations}
    attempted: list[int] = []
    monkeypatch.setattr(
        release_runner.os,
        "getsid",
        lambda _process_id: 101,
    )
    monkeypatch.setattr(
        release_runner.os,
        "getpgid",
        lambda process_id: group_by_anchor[process_id],
    )

    def fail_first(group_id: int, _signum: int) -> None:
        attempted.append(group_id)
        if group_id == 301:
            raise PermissionError("fixed first-group failure")

    monkeypatch.setattr(release_runner.os, "killpg", fail_first)

    invalid, errors = release_runner._signal_registered_groups(
        registrations,
        101,
        signal.SIGKILL,
    )

    assert not invalid
    assert attempted == [301, 401, 501]
    assert len(errors) == 1


def test_release_runner_terminates_every_test_session_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signaled: list[tuple[set[tuple[int, int]], int]] = []
    waited: list[float] = []
    direct_signals: list[tuple[int, int]] = []
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    registry = release_runner._open_test_session_registry(registry_path)

    class FakeProcess:
        pid = 101

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            direct_signals.append((101, signal.SIGKILL))

        @staticmethod
        def wait(timeout: float) -> int:
            waited.append(timeout)
            return -signal.SIGKILL

    monkeypatch.setattr(
        release_runner,
        "_freeze_test_session",
        lambda _process, _registry, _deadline: (
            ((101, 101, "T"), (304, 303, "T"), (306, 305, "T")),
            {(303, 304), (305, 306)},
            [],
        ),
    )
    monkeypatch.setattr(
        release_runner,
        "_registered_test_groups",
        lambda _registry, _session_id: (
            {(303, 304), (305, 306)},
            False,
        ),
    )
    monkeypatch.setattr(
        release_runner,
        "_signal_registered_groups",
        lambda registrations, _session_id, signum: (
            signaled.append((set(registrations), signum)) or False,
            [],
        ),
    )
    monkeypatch.setattr(
        release_runner,
        "_owned_test_session_members",
        lambda _session_id: ((101, 101, "T"),),
    )
    monkeypatch.setattr(
        release_runner.os,
        "getsid",
        lambda _process_id: 101,
    )
    monkeypatch.setattr(
        release_runner.os,
        "getpgid",
        lambda _process_id: 101,
    )
    monkeypatch.setattr(
        release_runner.os,
        "kill",
        lambda process_id, signum: direct_signals.append(
            (process_id, signum)
        ),
    )
    monkeypatch.setattr(
        release_runner.os,
        "killpg",
        lambda group_id, signum: direct_signals.append(
            (-group_id, signum)
        ),
    )

    try:
        release_runner._terminate_test_session(FakeProcess(), registry)
    finally:
        os.close(registry.descriptor)

    assert len(signaled) == 3
    assert all(
        registrations == {(303, 304), (305, 306)}
        and signum == signal.SIGKILL
        for registrations, signum in signaled
    )
    assert waited == [release_runner._PROCESS_GROUP_CLEANUP_SECONDS]
    assert (-101, signal.SIGKILL) in direct_signals
    assert (101, signal.SIGKILL) in direct_signals


def test_release_session_census_failure_still_finalizes_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "groups.log"
    registry_path.write_bytes(b"")
    registry_path.chmod(0o600)
    registry = release_runner._open_test_session_registry(registry_path)
    registered_killed: list[tuple[set[tuple[int, int]], int]] = []
    direct_signals: list[tuple[int, int]] = []
    waited: list[float] = []

    class FakeProcess:
        pid = 401

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            direct_signals.append((401, signal.SIGKILL))

        @staticmethod
        def wait(timeout: float) -> int:
            waited.append(timeout)
            return -signal.SIGKILL

    def fail_census(_session_id: int) -> tuple[tuple[int, int, str], ...]:
        raise release_runner.M5ReleaseTestError("fixed census failure")

    monkeypatch.setattr(
        release_runner,
        "_registered_test_groups",
        lambda _registry, _session_id: ({(503, 504)}, False),
    )
    monkeypatch.setattr(
        release_runner,
        "_signal_registered_groups",
        lambda registrations, _session_id, signum: (
            registered_killed.append(
                (set(registrations), signum)
            )
            or False,
            [],
        ),
    )
    monkeypatch.setattr(
        release_runner,
        "_freeze_test_session",
        lambda *_: (_ for _ in ()).throw(
            release_runner.M5ReleaseTestError("fixed census failure")
        ),
    )
    monkeypatch.setattr(
        release_runner,
        "_owned_test_session_members",
        fail_census,
    )
    monkeypatch.setattr(
        release_runner.os,
        "getsid",
        lambda _process_id: 401,
    )
    monkeypatch.setattr(
        release_runner.os,
        "getpgid",
        lambda _process_id: 401,
    )
    monkeypatch.setattr(
        release_runner.os,
        "kill",
        lambda process_id, signum: direct_signals.append(
            (process_id, signum)
        ),
    )
    monkeypatch.setattr(
        release_runner.os,
        "killpg",
        lambda group_id, signum: direct_signals.append(
            (-group_id, signum)
        ),
    )

    try:
        with pytest.raises(
            release_runner.M5ReleaseTestError,
            match="fixed census failure",
        ):
            release_runner._terminate_test_session(FakeProcess(), registry)
    finally:
        os.close(registry.descriptor)

    assert registered_killed == [
        ({(503, 504)}, signal.SIGKILL),
        ({(503, 504)}, signal.SIGKILL),
    ]
    assert (-401, signal.SIGKILL) in direct_signals
    assert (401, signal.SIGKILL) in direct_signals
    assert waited == [release_runner._PROCESS_GROUP_CLEANUP_SECONDS]


def test_release_runner_cleans_process_group_on_sigterm(tmp_path: Path) -> None:
    leader_pid_path = tmp_path / "signal-leader.pid"
    process_state_path = tmp_path / "signal-process-state.txt"
    agent_root = Path(__file__).resolve().parents[1]
    child_code = (
        "import os,pathlib,signal,subprocess,sys,time; "
        "blocked=signal.pthread_sigmask(signal.SIG_BLOCK, []); "
        "child=subprocess.Popen("
        "[sys.executable,'-I','-S','-B','-c',"
        "'import time; time.sleep(60)']); "
        f"pathlib.Path({str(process_state_path)!r}).write_text("
        "f'{os.getpid()} {child.pid} ' + "
        "','.join(str(int(value)) for value in sorted(blocked))); "
        "time.sleep(60)"
    )
    wrapper_code = f"""
import os
import pathlib
import signal
import sys
sys.path.insert(0, {str(agent_root)!r})
import c2_m5_release_test_runner as runner
signal.signal(signal.SIGUSR1, lambda *_: sys.exit(20))
real_popen = runner.subprocess.Popen
def observed_popen(*args, **kwargs):
    process = real_popen(*args, **kwargs)
    try:
        with pathlib.Path({str(leader_pid_path)!r}).open("x") as handle:
            handle.write(str(process.pid))
        return process
    except FileExistsError:
        return process
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
runner.subprocess.Popen = observed_popen
signal.pthread_sigmask(
    signal.SIG_BLOCK,
    (signal.SIGHUP, signal.SIGINT, signal.SIGTERM),
)
try:
    runner._run_release_child(
        [sys.executable, "-I", "-S", "-B", "-c", {child_code!r}],
        cwd=pathlib.Path({str(tmp_path)!r}),
        environment={{
            "PATH": "/usr/bin:/bin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
        }},
    )
except runner.M5ReleaseTestError as exc:
    raise SystemExit(17 if "signal" in str(exc) else 18)
raise SystemExit(19)
"""
    wrapper = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", wrapper_code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    leader_pid: int | None = None
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while child_pid is None:
            try:
                published_leader = int(
                    leader_pid_path.read_text(encoding="ascii")
                )
                if leader_pid is None:
                    leader_pid = published_leader
                else:
                    assert leader_pid == published_leader
            except (FileNotFoundError, ValueError):
                pass
            if wrapper.poll() is not None:
                raise AssertionError(
                    f"release wrapper exited early: {wrapper.returncode}"
                )
            try:
                pieces = process_state_path.read_text(
                    encoding="utf-8"
                ).split()
                if len(pieces) >= 2:
                    observed_leader, child_pid = map(int, pieces[:2])
                    if leader_pid is None:
                        leader_pid = observed_leader
                    else:
                        assert leader_pid == observed_leader
                    blocked_signals = {
                        int(value)
                        for value in pieces[2].split(",")
                        if value
                    } if len(pieces) == 3 else set()
                    break
            except FileNotFoundError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError("release wrapper did not start its child")
            time.sleep(0.01)

        assert leader_pid is not None
        assert child_pid is not None
        assert not blocked_signals.intersection(
            {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
        )
        os.kill(wrapper.pid, signal.SIGTERM)
        wrapper.wait(timeout=15)
        stdout, stderr = wrapper.communicate(timeout=1)

        assert wrapper.returncode == 17, (stdout, stderr)
        with pytest.raises(ProcessLookupError):
            os.killpg(leader_pid, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if leader_pid is None:
            try:
                leader_pid = int(
                    leader_pid_path.read_text(encoding="ascii")
                )
            except (FileNotFoundError, ValueError):
                pass
        if wrapper.poll() is None:
            os.kill(wrapper.pid, signal.SIGUSR1)
            try:
                wrapper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if leader_pid is not None:
                    try:
                        os.killpg(leader_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                wrapper.kill()
                wrapper.wait()
        if leader_pid is not None:
            try:
                os.killpg(leader_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_release_runner_cleans_group_for_signal_at_cleanup_entry(
    tmp_path: Path,
) -> None:
    leader_pid_path = tmp_path / "cleanup-entry-leader.pid"
    process_state_path = tmp_path / "cleanup-entry-process-state.txt"
    agent_root = Path(__file__).resolve().parents[1]
    child_code = (
        "import os,pathlib,subprocess,sys; "
        "child=subprocess.Popen("
        "[sys.executable,'-I','-S','-B','-c',"
        "'import time; time.sleep(60)']); "
        f"pathlib.Path({str(process_state_path)!r}).write_text("
        "f'{os.getpid()} {child.pid}')"
    )
    wrapper_code = f"""
import os
import pathlib
import signal
import sys
sys.path.insert(0, {str(agent_root)!r})
import c2_m5_release_test_runner as runner
signal.signal(signal.SIGUSR1, lambda *_: sys.exit(20))
real_popen = runner.subprocess.Popen
def observed_popen(*args, **kwargs):
    process = real_popen(*args, **kwargs)
    try:
        with pathlib.Path({str(leader_pid_path)!r}).open("x") as handle:
            handle.write(str(process.pid))
        return process
    except FileExistsError:
        return process
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
runner.subprocess.Popen = observed_popen
real_exc_info = sys.exc_info
signal_sent = False
class RunnerSys:
    def __getattr__(self, name):
        return getattr(sys, name)
    @staticmethod
    def exc_info():
        global signal_sent
        if not signal_sent:
            signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return real_exc_info()
runner.sys = RunnerSys()
try:
    runner._run_release_child(
        [sys.executable, "-I", "-S", "-B", "-c", {child_code!r}],
        cwd=pathlib.Path({str(tmp_path)!r}),
        environment={{
            "PATH": "/usr/bin:/bin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
        }},
    )
except runner.M5ReleaseTestError as exc:
    raise SystemExit(17 if "signal" in str(exc) else 18)
raise SystemExit(19)
"""
    wrapper = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", wrapper_code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    leader_pid: int | None = None
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while child_pid is None:
            try:
                published_leader = int(
                    leader_pid_path.read_text(encoding="ascii")
                )
                if leader_pid is None:
                    leader_pid = published_leader
                else:
                    assert leader_pid == published_leader
            except (FileNotFoundError, ValueError):
                pass
            if wrapper.poll() is not None:
                raise AssertionError(
                    f"cleanup-entry wrapper exited early: {wrapper.returncode}"
                )
            try:
                pieces = process_state_path.read_text(
                    encoding="utf-8"
                ).split()
                if len(pieces) == 2:
                    observed_leader, child_pid = map(int, pieces)
                    if leader_pid is None:
                        leader_pid = observed_leader
                    else:
                        assert leader_pid == observed_leader
                    break
            except FileNotFoundError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "cleanup-entry child did not publish process state"
                )
            time.sleep(0.01)

        assert leader_pid is not None
        assert child_pid is not None
        wrapper.wait(timeout=15)
        assert wrapper.returncode == 17
        with pytest.raises(ProcessLookupError):
            os.killpg(leader_pid, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if leader_pid is None:
            try:
                leader_pid = int(
                    leader_pid_path.read_text(encoding="ascii")
                )
            except (FileNotFoundError, ValueError):
                pass
        if wrapper.poll() is None:
            os.kill(wrapper.pid, signal.SIGUSR1)
            try:
                wrapper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if leader_pid is not None:
                    try:
                        os.killpg(leader_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                wrapper.kill()
                wrapper.wait()
        if leader_pid is not None:
            try:
                os.killpg(leader_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_release_runner_preserves_spawn_error_during_signal_restore(
    tmp_path: Path,
) -> None:
    agent_root = Path(__file__).resolve().parents[1]
    wrapper_code = f"""
import os
import pathlib
import signal
import sys
sys.path.insert(0, {str(agent_root)!r})
import c2_m5_release_test_runner as runner
def original_handler(*_):
    raise RuntimeError("original handler escaped")
signal.signal(signal.SIGTERM, original_handler)
real_mask = signal.pthread_sigmask
mask_calls = 0
signal_injected = False
class SignalProxy:
    def __getattr__(self, name):
        return getattr(signal, name)
    def pthread_sigmask(self, operation, signals):
        global mask_calls, signal_injected
        mask_calls += 1
        result = real_mask(operation, signals)
        if mask_calls == 5:
            signal_injected = True
            os.kill(os.getpid(), signal.SIGTERM)
        return result
runner.signal = SignalProxy()
def fail_spawn(*_, **__):
    raise OSError("fixed spawn failure")
runner.subprocess.Popen = fail_spawn
try:
    runner._run_release_child(
        [sys.executable, "-I", "-S", "-B", "-c", "pass"],
        cwd=pathlib.Path({str(tmp_path)!r}),
        environment={{
            "PATH": "/usr/bin:/bin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
        }},
    )
except OSError as exc:
    raise SystemExit(
        17 if str(exc) == "fixed spawn failure" and signal_injected else 18
    )
except RuntimeError:
    raise SystemExit(19)
raise SystemExit(20)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", wrapper_code],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 17, (completed.stdout, completed.stderr)


def test_release_signal_supervisor_covers_workspace_cleanup(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "supervised-workspace"
    agent_root = Path(__file__).resolve().parents[1]
    wrapper_code = f"""
import os
import pathlib
import shutil
import signal
import sys
sys.path.insert(0, {str(agent_root)!r})
import c2_m5_release_test_runner as runner
workspace = pathlib.Path({str(workspace)!r})
def operation(_supervisor):
    workspace.mkdir()
    try:
        return 0
    finally:
        os.kill(os.getpid(), signal.SIGTERM)
        shutil.rmtree(workspace)
try:
    runner._run_with_signal_supervision(operation)
except runner.M5ReleaseTestError as exc:
    raise SystemExit(
        17 if "signal" in str(exc) and not workspace.exists() else 18
    )
raise SystemExit(19)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", wrapper_code],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 17, (completed.stdout, completed.stderr)
    assert not workspace.exists()


@pytest.mark.parametrize(
    "module_name,error_name",
    (
        ("c2_m5_release_test_runner", "M5ReleaseTestError"),
        (
            "experiments.c2_m5_source_pilot",
            "C2M5SourcePilotError",
        ),
    ),
)
def test_signal_supervisors_restore_state_after_interruption(
    module_name: str,
    error_name: str,
) -> None:
    agent_root = Path(__file__).resolve().parents[1]
    wrapper_code = f"""
import importlib
import os
import signal
import sys
sys.path.insert(0, {str(agent_root)!r})
module = importlib.import_module({module_name!r})
watched = module._WATCHED_SIGNALS
handlers = {{signum: signal.getsignal(signum) for signum in watched}}
mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
def operation(supervisor):
    os.kill(os.getpid(), signal.SIGTERM)
    supervisor.checkpoint()
try:
    module._run_with_signal_supervision(operation)
except getattr(module, {error_name!r}) as exc:
    restored = (
        {{signum: signal.getsignal(signum) for signum in watched}} == handlers
        and signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask
    )
    raise SystemExit(17 if "signal" in str(exc) and restored else 18)
raise SystemExit(19)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", wrapper_code],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 17, (
        completed.stdout,
        completed.stderr,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin ACL regression")
def test_runtime_acl_checks_reject_mutating_allow_entry(tmp_path: Path) -> None:
    target = tmp_path / "runtime.py"
    target.write_text("trusted = True\n", encoding="utf-8")
    username = subprocess.check_output(
        ["/usr/bin/id", "-un"],
        text=True,
    ).strip()
    subprocess.run(
        [
            "/bin/chmod",
            "+a",
            f"user:{username} allow write",
            str(target),
        ],
        check=True,
    )
    try:
        with pytest.raises(
            m5_attestation.C2M5BootstrapAttestationError,
            match="Darwin ACL",
        ):
            m5_attestation._require_nonmutating_acl(target, "test runtime")
        with pytest.raises(
            release_runner.M5ReleaseTestError,
            match="Darwin ACL",
        ):
            release_runner._require_nonmutating_acl(target, "test runtime")
    finally:
        subprocess.run(["/bin/chmod", "-N", str(target)], check=True)


def test_fixed_python_binding_requires_stdlib_zip_absence() -> None:
    dependency_binding, _, manifest_payload, _ = _dependency_closure()
    python_binding = dependency_binding["python"]
    assert python_binding["stdlib_zip_status"] == "ABSENT"
    runtime_binding = json.loads(manifest_payload)["python"]
    assert runtime_binding["stdlib_zip_path"].endswith("/lib/python39.zip")
    assert not Path(runtime_binding["stdlib_zip_path"]).exists()


def test_git_object_reads_never_invoke_partial_clone_transport(tmp_path: Path) -> None:
    repository = tmp_path / "partial"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "M5 Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "m5@example.invalid"],
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("trusted\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    object_id = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD:tracked.txt"],
        text=True,
    ).strip()
    marker = tmp_path / "transport-ran"
    subprocess.run(
        ["git", "-C", str(repository), "config", "core.repositoryFormatVersion", "1"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "extensions.partialClone", "origin"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "remote.origin.promisor", "true"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "config",
            "remote.origin.url",
            f"ext::/usr/bin/touch {marker}",
        ],
        check=True,
    )
    (repository / ".git" / "objects" / object_id[:2] / object_id[2:]).unlink()

    with pytest.raises(
        m5_attestation.C2M5BootstrapAttestationError,
        match="cannot establish missing object",
    ):
        m5_attestation._git_bytes(
            repository,
            ("show", "HEAD:tracked.txt"),
            "missing object",
        )
    assert not marker.exists()


def test_materialized_downloader_runs_behind_network_guard(tmp_path: Path) -> None:
    script = (
        b"from concurrent.futures import ProcessPoolExecutor\n"
        b"def guarded(_):\n"
        b"    import requests\n"
        b"    return requests.get.__module__\n"
        b"if __name__ == '__main__':\n"
        b"    with ProcessPoolExecutor(max_workers=2) as pool:\n"
        b"        assert list(pool.map(guarded, range(2))) == ['sitecustomize'] * 2\n"
        b"    print('guarded runtime bundle')\n"
    )
    binding = pilot.DownloaderBinding(
        worktree=tmp_path,
        tree="a" * 40,
        script=tmp_path / "unused.py",
        script_blob="b" * 40,
        runtime_files={"frozen_downloader.py": script},
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "accepted.jsonl").write_bytes(b"")
    guard_payload = (
        Path(__file__).resolve().parents[1]
        / "c2_m5_sitecustomize"
        / "sitecustomize.py"
    ).read_bytes()
    _, dependency_payloads, _, _ = _dependency_closure()

    exit_code, retained_active_pids = pilot._run_attempt_subprocess(
        downloader=binding,
        workspace=workspace,
        workers=1,
        network_guard_payload=guard_payload,
        downloader_bootstrap_payload=(
            Path(__file__).resolve().parents[1]
            / "c2_m5_downloader_bootstrap.py"
        ).read_bytes(),
        dependency_payloads=dependency_payloads,
    )

    assert exit_code == 0
    assert retained_active_pids == ()
    assert (workspace / "postfetch.log").read_text().strip() == "guarded runtime bundle"
    assert (workspace / "frozen_downloader.py").read_bytes() == script
    assert not list(workspace.rglob("*.pyc"))


@pytest.mark.parametrize(
    "relative",
    (
        "c2_m5_source_pilot_bootstrap.py",
        "c2_m5_downloader_bootstrap.py",
    ),
)
@pytest.mark.parametrize(
    "missing_flag",
    ("isolated", "no_site", "dont_write_bytecode"),
)
def test_bootstraps_require_all_isolation_flags(
    relative: str,
    missing_flag: str,
) -> None:
    bootstrap = Path(__file__).resolve().parents[1] / relative
    module_name = bootstrap.stem
    agent_root = bootstrap.parent
    code = f"""
import sys
from types import SimpleNamespace
sys.path.insert(0, {str(agent_root)!r})
module = __import__({module_name!r})
values = {{
    "isolated": True,
    "no_site": True,
    "dont_write_bytecode": True,
}}
values[{missing_flag!r}] = False
module._require_isolated_python(SimpleNamespace(**values))
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
        },
    )

    assert completed.returncode == 2
    assert "invoke Python with -I -S -B" in completed.stderr


def test_source_bootstrap_requires_external_release_pins() -> None:
    bootstrap = (
        Path(__file__).resolve().parents[1] / "c2_m5_source_pilot_bootstrap.py"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("C2_M5_EXPECTED_")
    }

    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(bootstrap)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 2
    assert "requires exact external release pins" in completed.stderr


def test_source_bootstrap_executes_retained_helper_bytes(
    tmp_path: Path,
) -> None:
    bootstrap = (
        Path(__file__).resolve().parents[1]
        / "c2_m5_source_pilot_bootstrap.py"
    )
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 'attested'\n", encoding="utf-8")
    code = """
import importlib.util
import pathlib
import sys

bootstrap_path = pathlib.Path(sys.argv[1])
helper_path = pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("m5_source_bootstrap", bootstrap_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
retained = helper_path.read_bytes()
helper_path.write_text("VALUE = 'swapped'\\n", encoding="utf-8")
loaded = module._load_verified_helper(helper_path, retained)
assert loaded.VALUE == "attested"
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(bootstrap),
            str(helper),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_secret_scan_ignores_binary_sources_but_scans_controls() -> None:
    files = [
        ("content/_sources/a/source.bin", 30, b"ghp_" + b"x" * 30),
        ("control/initial/postfetch.log", 5, b"safe\n"),
    ]
    assert pilot._secret_scan(files) == []
    files.append(("control/execution.txt", 30, b"ghp_" + b"x" * 30))
    assert pilot._secret_scan(files) == ["control/execution.txt"]


def test_execution_evidence_hash_and_non_admission() -> None:
    records = [_record(1)]
    statuses = {records[0].article_id: "downloaded"}
    attempts = {
        attempt: _attempt(attempt, records, statuses) for attempt in pilot.ATTEMPTS
    }
    attestation = pilot.AdapterAttestation(
        implementation_commit="a" * 40,
        attestation_commit="b" * 40,
        head_commit="b" * 40,
        manifest_sha256="c" * 64,
        path_sha256={"adapter": "d" * 64},
        path_payloads={"adapter": b"fixed"},
        dependency_manifest_sha256="e" * 64,
        dependency_binding={"runtime_inventory_sha256": "f" * 64},
        dependency_payloads={},
        externally_pinned_bootstrap_sha256="1" * 64,
        externally_pinned_python_sha256="2" * 64,
        externally_pinned_python_library_sha256="3" * 64,
    )
    downloader = pilot.DownloaderBinding(
        worktree=Path("/tmp/worktree"),
        tree="e" * 40,
        script=Path("/tmp/worktree/downloader.py"),
        script_blob="f" * 40,
        runtime_files={"frozen_downloader.py": b"print('fixed')\n"},
    )

    evidence = pilot._execution_evidence(
        attestation=attestation,
        downloader=downloader,
        binding={"summary_hash": "1" * 64},
        attempts=attempts,
        retained_active_pids=(),
    )

    assert evidence["status"] == "PASS"
    assert evidence["network_guard"]["trust_environment"] is False
    assert evidence["concurrency"]["retained_active_pids"] == []
    assert evidence["evidence_hash"] == pilot._sha256_json(
        {key: value for key, value in evidence.items() if key != "evidence_hash"}
    )


def test_finalizer_rejects_impossible_m5_budget_and_hint_claims() -> None:
    harvest = {
        "article-1": {
            "source_assets_observed": 1,
            "figure_assets_observed": 0,
            "caption_assets_observed": 0,
            "assets": [
                {
                    "asset_id": "source",
                    "relative_path": "content/_sources/article-1/source.csv",
                    "sha256": "c" * 64,
                    "bytes": 8,
                    "doi": "10.1038/article-1",
                    "declared_asset_kind": "source_data",
                    "declared_format_tuple": ["NONE", "CSV_V1"],
                    "candidate_hints": [],
                    "first_attempt": "initial",
                }
            ],
        }
    }
    receipt = {
        "schema_version": "c2-v2-raw-attempt-receipt-v2",
        "source_harvest": harvest,
    }
    with pytest.raises(
        finalizer.C2RemediationError,
        match="cannot explain",
    ):
        finalizer._validate_m5_network_harvest_binding(
            attempt="initial",
            receipt=receipt,
            processed_ids={"article-1"},
            expected_dois={"article-1": "10.1038/article-1"},
            expected_attempted_records=200,
            network_budget={"request_count": 0, "response_bytes": 0},
        )

    execution_evidence = {
        "evidence_hash": "a" * 64,
        "adapter": {},
        "network_guard": {
            "status": "ENFORCED",
            "trust_environment": False,
            "https_only": True,
            "public_ip_only": True,
            "redirects_validated": True,
            "per_response_byte_cap": 256 * 1024 * 1024,
            "per_attempt_request_cap": 10_000,
            "per_attempt_response_byte_cap": 8 * 1024 * 1024 * 1024,
            "per_attempt_wall_timeout_seconds": 3 * 60 * 60,
        },
    }
    active_attestation = SimpleNamespace(
        implementation_commit="1" * 40,
        attestation_commit="2" * 40,
        manifest_sha256="3" * 64,
        externally_pinned_bootstrap_sha256="4" * 64,
        externally_pinned_python_sha256="5" * 64,
        externally_pinned_python_library_sha256="6" * 64,
        path_sha256={"adapter": "7" * 64},
        dependency_binding={"runtime_inventory_sha256": "8" * 64},
    )
    with pytest.raises(
        finalizer.C2RemediationError,
        match="active runtime trust binding",
    ):
        finalizer._validate_m5_execution_trust(
            execution_evidence,
            binding={"summary_hash": "9" * 64},
            attestation=active_attestation,
        )

    source_evidence = finalizer._SourceEvidence(
        descriptor_path="content/_source_evidence/article-1.json",
        descriptor_sha256="b" * 64,
        descriptor_bytes=1,
        schema_version="c2-source-evidence-v2",
        descriptor={
            "assets": [
                {
                    "asset_id": "source",
                    "relative_path": "content/_sources/article-1/source.csv",
                    "sha256": "d" * 64,
                    "bytes": 8,
                    "doi": "10.1038/article-1",
                    "declared_asset_kind": "source_data",
                    "declared_format_tuple": ["NONE", "CSV_V1"],
                    "candidate_hints": [],
                }
            ]
        },
        source_paths=(),
    )
    with pytest.raises(
        finalizer.C2RemediationError,
        match="differs from its acquisition receipt",
    ):
        finalizer._validate_m5_snapshot_source_qualification(
            execution_evidence,
            {
                "article-1": {
                    "source_evidence": source_evidence,
                    "value": {"retained_assets": []},
                }
            },
            {
                ("article-1", "source"): {
                    "asset_id": "source",
                    "relative_path": "content/_sources/article-1/source.csv",
                    "sha256": "c" * 64,
                    "bytes": 8,
                    "doi": "10.1038/article-1",
                    "declared_asset_kind": "source_data",
                    "declared_format_tuple": ["NONE", "CSV_V1"],
                    "candidate_hints": [],
                    "first_attempt": "initial",
                }
            },
        )


def test_public_execution_and_validation_have_no_injection_parameters() -> None:
    for function in (pilot.execute_source_pilot, pilot.validate_source_pilot):
        assert all(
            not parameter.startswith("_")
            for parameter in inspect.signature(function).parameters
        )


def test_network_guard_loads_and_rejects_non_allowlisted_urls(
    tmp_path: Path,
) -> None:
    guard = (
        Path(__file__).resolve().parents[1]
        / "c2_m5_sitecustomize"
    )
    _, dependency_payloads, _, _ = _dependency_closure()
    dependency_workspace = tmp_path / "dependencies"
    dependency_workspace.mkdir()
    pilot._materialize_dependencies(tmp_path, dependency_payloads)
    budget_path = tmp_path / pilot.NETWORK_BUDGET_NAME
    budget_path.write_bytes(struct.pack(">QQ", 0, 0))
    budget_path.chmod(0o600)
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "C2_M5_NETWORK_GUARD_REQUIRED": "1",
        "C2_M5_NETWORK_BUDGET_PATH": str(budget_path),
    }
    code = """
import sitecustomize
import urllib.request
assert sitecustomize._allowed_host("www.nature.com")
assert sitecustomize._allowed_host("media.springernature.com")
assert not sitecustomize._allowed_host("evilnature.com")
for url in ("http://www.nature.com/a", "https://127.0.0.1/a", "https://evil.test/a"):
    try:
        sitecustomize._validate_url(url)
    except RuntimeError:
        pass
    else:
        raise AssertionError(url)
for kwargs in (
    {"auth": ("user", "password")},
    {"headers": {"Authorization": "Bearer secret"}},
    {"headers": {"Host": "example.com"}},
    {"verify": False},
    {"verify": 0},
):
    try:
        sitecustomize._safe_get("https://www.nature.com/", **kwargs)
    except RuntimeError:
        pass
    else:
        raise AssertionError(kwargs)
for operation in (
    lambda: sitecustomize.requests.Session().get("https://example.com"),
    lambda: sitecustomize.requests.post("https://www.nature.com/"),
    lambda: urllib.request.urlopen("https://www.nature.com/"),
):
    try:
        operation()
    except RuntimeError:
        pass
    else:
        raise AssertionError("unguarded requests entry point passed")
sitecustomize._ORIGINAL_GETADDRINFO = lambda *args, **kwargs: [
    (2, 1, 6, "", ("93.184.216.34", 443))
]
try:
    sitecustomize._guarded_getaddrinfo("www.nature.com", 80)
except RuntimeError:
    pass
else:
    raise AssertionError("non-TLS destination port passed")
for family, address in ((2, "127.0.0.1"), (30, "fec0::1")):
    sitecustomize._ORIGINAL_GETADDRINFO = lambda *args, _family=family, _address=address, **kwargs: [
        (_family, 1, 6, "", (_address, 443))
    ]
    for validator in (
        sitecustomize._public_addresses,
        sitecustomize._guarded_getaddrinfo,
    ):
        try:
            validator("www.nature.com", 443)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"non-public destination passed: {address}")
sitecustomize._ORIGINAL_GETADDRINFO = lambda *args, **kwargs: [
    (2, 1, 6, "", ("93.184.216.34", 443))
]

def response(status, headers=None, content=b""):
    value = sitecustomize.requests.Response()
    value.status_code = status
    value.headers.update(headers or {})
    value._content = content
    value._content_consumed = True
    return value

for fake_response, requested_stream, expected in (
    (
        response(302, {"Location": "https://evil.test/redirect"}),
        False,
        "redirect",
    ),
    (
        response(
            200,
            {"Content-Length": str(sitecustomize._MAX_RESPONSE_BYTES + 1)},
        ),
        False,
        "Content-Length",
    ),
):
    sitecustomize._ORIGINAL_SESSION_REQUEST = (
        lambda *_args, _response=fake_response, **_kwargs: _response
    )
    try:
        sitecustomize._safe_get(
            "https://www.nature.com/source",
            stream=requested_stream,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"{expected} guard did not fail closed")

sitecustomize._MAX_RESPONSE_BYTES = 3
sitecustomize._ORIGINAL_SESSION_REQUEST = (
    lambda *_args, **_kwargs: response(200, content=b"four")
)
streamed = sitecustomize._safe_get(
    "https://www.nature.com/source",
    stream=True,
)
try:
    list(streamed.iter_content(chunk_size=2))
except RuntimeError:
    pass
else:
    raise AssertionError("streamed response overflow passed")
with open(sitecustomize._BUDGET_PATH, "rb") as handle:
    assert sitecustomize.struct.unpack(">QQ", handle.read()) == (3, 4)
with open(sitecustomize._BUDGET_PATH, "r+b") as handle:
    handle.write(
        sitecustomize.struct.pack(
            ">QQ",
            sitecustomize._MAX_TOTAL_REQUESTS,
            4,
        )
    )
try:
    sitecustomize._safe_get("https://www.nature.com/request-cap")
except RuntimeError:
    pass
else:
    raise AssertionError("cumulative request cap passed")
with open(sitecustomize._BUDGET_PATH, "rb") as handle:
    assert sitecustomize.struct.unpack(">QQ", handle.read()) == (
        sitecustomize._MAX_TOTAL_REQUESTS + 1,
        4,
    )
with open(sitecustomize._BUDGET_PATH, "r+b") as handle:
    handle.write(
        sitecustomize.struct.pack(
            ">QQ",
            0,
            sitecustomize._MAX_TOTAL_RESPONSE_BYTES - 2,
        )
    )
response = sitecustomize._safe_get(
    "https://www.nature.com/byte-cap",
    stream=True,
)
try:
    list(response.iter_content())
except RuntimeError:
    pass
else:
    raise AssertionError("cumulative response-byte cap passed")
with open(sitecustomize._BUDGET_PATH, "rb") as handle:
    assert sitecustomize.struct.unpack(">QQ", handle.read()) == (
        1,
        sitecustomize._MAX_TOTAL_RESPONSE_BYTES + 1,
    )
sitecustomize._ORIGINAL_GETADDRINFO = lambda *args, **kwargs: [
    (2, 1, 6, "", ("100.64.0.1", 443))
]
try:
    sitecustomize._guarded_getaddrinfo("www.nature.com", 443)
except RuntimeError:
    pass
else:
    raise AssertionError("non-global shared address passed")
"""
    isolated_code = (
        "import sys\n"
        f"sys.path[:0] = [{str(guard)!r}, {str(dependency_workspace)!r}]\n"
        + code
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", isolated_code],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

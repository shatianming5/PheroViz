from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import shlex
import shutil
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

    overflow_exit = pilot._run_bounded_process(
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
    assert (overflow_workspace / "postfetch.log").stat().st_size == 32

    timeout_workspace = tmp_path / "timeout"
    timeout_workspace.mkdir()
    started = time.monotonic()
    timeout_exit = pilot._run_bounded_process(
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
    assert time.monotonic() - started < 3
    assert (timeout_workspace / "postfetch.log").stat().st_size <= 32

    closed_output_workspace = tmp_path / "closed-output"
    closed_output_workspace.mkdir()
    started = time.monotonic()
    closed_output_exit = pilot._run_bounded_process(
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
    assert time.monotonic() - started < 3

    descendant_workspace = tmp_path / "descendant"
    descendant_workspace.mkdir()
    descendant_exit = pilot._run_bounded_process(
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
    child_pid = int((descendant_workspace / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


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
        shutil.copyfile(
            source_repository / "agent" / name,
            repository / "agent" / name,
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

    exit_code = pilot._run_attempt_subprocess(
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
@pytest.mark.parametrize("flags", [(), ("-S", "-B"), ("-I", "-B")])
def test_bootstraps_require_all_isolation_flags(
    relative: str,
    flags: tuple[str, ...],
) -> None:
    bootstrap = Path(__file__).resolve().parents[1] / relative
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONDONTWRITEBYTECODE", "PYTHONPATH"}
    }

    completed = subprocess.run(
        [sys.executable, *flags, str(bootstrap)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=environment,
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
    )

    assert evidence["status"] == "PASS"
    assert evidence["network_guard"]["trust_environment"] is False
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
                    "bytes": 8,
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
                    "sha256": "c" * 64,
                    "bytes": 8,
                    "declared_asset_kind": "source_data",
                    "candidate_hints": [{"panel_id": "forged"}],
                }
            ]
        },
        source_paths=(),
    )
    with pytest.raises(
        finalizer.C2RemediationError,
        match="candidate hints",
    ):
        finalizer._validate_m5_snapshot_source_qualification(
            execution_evidence,
            {"article-1": {"source_evidence": source_evidence}},
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
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.pathsep.join(
            [str(guard), str(dependency_workspace)]
        ),
        "PYTHONNOUSERSITE": "1",
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
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

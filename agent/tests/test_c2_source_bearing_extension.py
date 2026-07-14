from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from experiments import c2_remediation_root_finalizer as finalizer
from experiments.c2_source_bearing_extension import (
    SourceBearingExtensionError,
    build_source_bearing_extension,
    validate_source_bearing_extension,
    verify_source_extension_code_attestation,
)
from tests.test_c2_remediation_root_finalizer import _make_fixture
from tests.test_experiment_support import experiment_workspace


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _MemoryRoot:
    """A write-once test double for the finalizer's descriptor-rooted staging API."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def write_bytes(self, relative: str, payload: bytes) -> str:
        if relative in self.payloads:
            raise AssertionError(f"attempted overwrite: {relative}")
        self.payloads[relative] = payload
        return relative

    def read_bytes(self, relative: str) -> bytes:
        return self.payloads[relative]

    def sha256(self, relative: str) -> str:
        return _sha256(self.read_bytes(relative))


def _zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return output.getvalue()


def _xlsx() -> bytes:
    return _zip(
        {
            "[Content_Types].xml": (
                b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
                b'content-types"><Default Extension="rels" ContentType="application/'
                b'vnd.openxmlformats-package.relationships+xml"/><Default '
                b'Extension="xml" ContentType="application/xml"/><Override '
                b'PartName="/xl/workbook.xml" ContentType="application/vnd.'
                b'openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                b'<Override PartName="/xl/worksheets/sheet1.xml" ContentType='
                b'"application/vnd.openxmlformats-officedocument.spreadsheetml.'
                b'worksheet+xml"/></Types>'
            ),
            "_rels/.rels": (
                b'<Relationships xmlns="http://schemas.openxmlformats.org/package/'
                b'2006/relationships"><Relationship Id="rId1" Type="http://'
                b'schemas.openxmlformats.org/officeDocument/2006/relationships/'
                b'officeDocument" Target="xl/workbook.xml"/></Relationships>'
            ),
            "xl/workbook.xml": (
                b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
                b'2006/main" xmlns:r="http://schemas.openxmlformats.org/'
                b'officeDocument/2006/relationships"><sheets><sheet name="Sheet1" '
                b'sheetId="1" r:id="rId1"/></sheets></workbook>'
            ),
            "xl/_rels/workbook.xml.rels": (
                b'<Relationships xmlns="http://schemas.openxmlformats.org/package/'
                b'2006/relationships"><Relationship Id="rId1" Type="http://'
                b'schemas.openxmlformats.org/officeDocument/2006/relationships/'
                b'worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
            ),
            "xl/worksheets/sheet1.xml": (
                b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
                b'2006/main"><sheetData/></worksheet>'
            ),
        }
    )


def _asset(
    *,
    article_id: str,
    doi_id: str,
    asset_id: str,
    payload: bytes,
    kind: str,
    detected: list[str],
    hints: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str, bytes]:
    relative = f"content/_sources/{article_id}/{asset_id}.bin"
    return (
        {
            "asset_id": asset_id,
            "relative_path": relative,
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "doi": doi_id,
            "declared_asset_kind": kind,
            "declared_format_tuple": detected,
            "candidate_hints": hints or [],
        },
        relative,
        payload,
    )


def _source_root(
    *,
    source_assets: list[tuple[dict[str, Any], str, bytes]],
    terminal_status: str = "downloaded",
) -> tuple[
    _MemoryRoot,
    Any,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    root = _MemoryRoot()
    article_id = "article-1"
    doi_id = "10.9999/source-1"
    assets = [item[0] for item in source_assets]
    for _, relative, payload in source_assets:
        root.payloads[relative] = payload
    descriptor: dict[str, Any] = {
        "schema_version": "c2-source-evidence-v2",
        "doi": doi_id,
        "article_id": article_id,
        "provenance_relative_path": f"content/_provenance/{article_id}.json",
        "assets": assets,
    }
    descriptor["descriptor_hash"] = _sha256(_canonical(descriptor))
    descriptor_relative = f"content/_source_evidence/{article_id}.json"
    root.payloads[descriptor_relative] = _canonical(descriptor)
    root.payloads[f"content/_provenance/{article_id}.json"] = _canonical(
        {
            "doi": doi_id,
            "source_evidence": {
                "descriptor_path": descriptor_relative,
                "descriptor_sha256": _sha256(_canonical(descriptor)),
                "descriptor_bytes": len(_canonical(descriptor)),
            },
        }
    )
    reader = SimpleNamespace(
        reads={
            relative: SimpleNamespace(payload=payload, sha256=_sha256(payload))
            for _, relative, payload in source_assets
        }
    )
    evidence = SimpleNamespace(
        descriptor=descriptor,
        descriptor_path=descriptor_relative,
        descriptor_sha256=_sha256(_canonical(descriptor)),
        descriptor_bytes=len(_canonical(descriptor)),
        source_paths=(),
    )
    records = [
        {
            "article_url": "https://example.test/article-1",
            "doi": doi_id,
        }
    ]
    provenance = {
        article_id: {
            "source_evidence": evidence,
            "relative_path": f"content/_provenance/{article_id}.json",
        }
    }
    terminal_rows = [
        {
            "article_id": article_id,
            "doi": doi_id,
            "input_index_1based": 1,
            "terminal_status": terminal_status,
            "rounds": {
                "initial": "downloaded",
                "retry1": "downloaded",
                "retry2": terminal_status,
            },
        }
    ]
    root.payloads["control/terminal_outcomes.jsonl"] = _canonical(terminal_rows[0]) + b"\n"
    return root, reader, records, provenance, terminal_rows


def _bound_assets(
    *,
    table_payload: bytes = b"panel,value\na,1\n",
    table_kind: str = "source_data",
    table_format: list[str] | None = None,
    table_hints: list[dict[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], str, bytes]]:
    article_id = "article-1"
    doi_id = "10.9999/source-1"
    return [
        _asset(
            article_id=article_id,
            doi_id=doi_id,
            asset_id="table",
            payload=table_payload,
            kind=table_kind,
            detected=table_format or ["NONE", "CSV_V1"],
            hints=table_hints
            if table_hints is not None
            else [
                {
                    "member_selector_or_null": None,
                    "panel_id": "panel-a",
                    "case_group_or_null": None,
                    "figure_asset_id": "figure",
                    "caption_asset_id": "caption",
                }
            ],
        ),
        _asset(
            article_id=article_id,
            doi_id=doi_id,
            asset_id="figure",
            payload=b"\x89PNG\r\n\x1a\nsynthetic",
            kind="figure",
            detected=["NONE", "OTHER_REGISTERED_V1"],
        ),
        _asset(
            article_id=article_id,
            doi_id=doi_id,
            asset_id="caption",
            payload=b"synthetic caption",
            kind="caption",
            detected=["NONE", "OTHER_REGISTERED_V1"],
        ),
    ]


def _build(
    source_assets: list[tuple[dict[str, Any], str, bytes]],
) -> tuple[_MemoryRoot, Any]:
    root, reader, records, provenance, terminal_rows = _source_root(
        source_assets=source_assets
    )
    result = build_source_bearing_extension(
        root=root,
        raw_reader=reader,
        records=records,
        provenance=provenance,
        terminal_rows=terminal_rows,
        partition_records=1,
        source_chunk_sha256="a" * 64,
    )
    return root, result


def _records(root: _MemoryRoot, path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in root.read_bytes(path).decode("utf-8").splitlines()
    ]


def test_required_v2_schemas_are_closed_and_meta_schema_valid() -> None:
    schema_dir = Path(__file__).parents[1] / "experiments" / "schemas"
    names = (
        "c2_v2_fd_format_classifier_config_v1.schema.json",
        "c2_v2_detected_format_v1.schema.json",
        "c2_v2_container_accounting_index_v1.schema.json",
        "c2_v2_consumable_source_unit_v1.schema.json",
        "c2_v2_downstream_consumption_v1.schema.json",
        "c2_v2_candidate_set_input_v1.schema.json",
        "c2_v2_consumption_bijection_validation_v1.schema.json",
    )
    for name in names:
        schema = json.loads((schema_dir / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False


def test_code_attestation_rejects_wrong_commit_blob_and_runtime_path() -> None:
    repository = Path(__file__).resolve().parents[2]
    current_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    attestation = verify_source_extension_code_attestation(repository)
    assert attestation.attestation_commit_full == current_commit
    assert {
        blob.relative_path for blob in attestation.code_blobs
    } >= {
        "agent/experiments/c2_source_bearing_extension.py",
        "agent/experiments/c2_remediation_root_finalizer.py",
        "agent/experiments/schemas/c2_v2_fd_format_classifier_config_v1.schema.json",
    }

    with experiment_workspace("c2-source-extension-attestation") as workspace:
        clone = workspace / "attested-clone"
        subprocess.run(
            ["git", "clone", "--no-local", "--no-checkout", str(repository), str(clone)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--detach", current_commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        extension_path = clone / "agent/experiments/c2_source_bearing_extension.py"
        finalizer_path = clone / "agent/experiments/c2_remediation_root_finalizer.py"
        verify_source_extension_code_attestation(
            clone,
            loaded_extension_path=extension_path,
            loaded_finalizer_path=finalizer_path,
        )
        with pytest.raises(SourceBearingExtensionError, match="loaded runtime path"):
            verify_source_extension_code_attestation(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=Path(__file__),
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "checkout",
                "--detach",
                attestation.approved_implementation_commit_full,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        source_manifest = json.loads(
            (
                repository
                / "agent/experiments/c2_source_bearing_extension_code_attestation.json"
            ).read_text(encoding="utf-8")
        )
        source_manifest["attested_paths"][0]["sha256"] = "0" * 64
        (clone / "agent/experiments/c2_source_bearing_extension_code_attestation.json").write_bytes(
            _canonical(source_manifest) + b"\n"
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "config",
                "user.email",
                "attestation-test@example.test",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "config",
                "user.name",
                "Attestation Test",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "add",
                "agent/experiments/c2_source_bearing_extension_code_attestation.json",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "commit", "-m", "forge attestation"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with pytest.raises(SourceBearingExtensionError, match="attested blob mismatch"):
            verify_source_extension_code_attestation(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "checkout",
                "--detach",
                attestation.approved_implementation_commit_full,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with pytest.raises(
            SourceBearingExtensionError,
            match="attestation commit",
        ):
            verify_source_extension_code_attestation(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
            )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--detach", current_commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        extension_path.write_bytes(extension_path.read_bytes() + b"\n# forged\n")
        with pytest.raises(SourceBearingExtensionError, match="worktree is dirty"):
            verify_source_extension_code_attestation(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
            )


def test_csv_pipeline_replays_without_models_and_retains_single_case() -> None:
    root, result = _build(_bound_assets())

    assert result.status == "SOURCE_CLASSIFICATION_V2_COMPLETE"
    assert result.canonical["case_count"] == 1
    assert _records(root, "review_v2/structural_outcomes.jsonl")[0]["outcome"] == (
        "ACCEPTED_STRUCTURAL"
    )
    assert _records(root, "p_evidence_v2/source_classifications.jsonl")[0][
        "derived_public_stratum"
    ] == "P1"
    classification = _records(root, "p_evidence_v2/source_classifications.jsonl")[0]
    assert set(classification) == {
        "schema_version",
        "doi_id",
        "parent_doi_id",
        "cluster_id",
        "canonical_case_set_hash",
        "canonical_case_ids",
        "canonical_case_record_hashes",
        "verified_panel_descriptors_sha256",
        "qualified_panel_counts",
        "derived_public_stratum",
        "derived_code_label",
        "source_inventory_collection_hash",
        "candidate_collection_hash",
        "proposal_collection_hash",
        "review_collection_hash",
        "canonical_collection_hash",
        "consumption_bijection_hash",
        "panel_rule_hash",
        "record_hash",
    }
    assert classification["doi_id"] == classification["cluster_id"]
    assert classification["derived_code_label"] == "P=1"
    replay = validate_source_bearing_extension(
        root, partition_records=1, source_chunk_sha256="a" * 64
    )
    assert replay["status"] == "PASS"
    protocol = json.loads(root.read_bytes("review_v2/structural_protocol.json"))
    assert protocol["review_mode"] == "C2_V2_STRUCTURAL_REVIEW_V1"


def test_generic_zip_is_fd_accounted_and_every_member_is_consumed() -> None:
    archive = _zip(
        {
            "table.csv": b"panel,value\na,1\n",
            "notes.txt": b"source-only note",
            "nested.zip": _zip({"nested.csv": b"panel,value\nb,2\n"}),
        }
    )
    hints = [
        {
            "member_selector_or_null": "table.csv",
            "panel_id": "panel-a",
            "case_group_or_null": None,
            "figure_asset_id": "figure",
            "caption_asset_id": "caption",
        }
    ]
    root, result = _build(
        [
            _asset(
                article_id="article-1",
                doi_id="10.9999/source-1",
                asset_id="archive",
                payload=archive,
                kind="source_archive",
                detected=["ZIP_V1", "GENERIC_ZIP_V1"],
                hints=hints,
            ),
            *_bound_assets()[1:],
        ]
    )

    accounts = json.loads(
        root.read_bytes("source_inventory_v2/container_accounts/container-000001.json")
    )
    assert accounts["physical_entry_count"] == 3
    assert all(entry["entry_hash"] for entry in accounts["entries"])
    derived = _records(root, "source_inventory_v2/derived_archive_members.jsonl")
    consumption = _records(root, "cases_v2/downstream_consumption.jsonl")
    assert len(derived) == 4
    assert len(consumption) == 4
    assert result.canonical["case_count"] == 1
    assert validate_source_bearing_extension(
        root, partition_records=1, source_chunk_sha256="a" * 64
    )["status"] == "PASS"


def test_raw_xlsx_is_a_zip_accounted_outer_self_unit() -> None:
    xlsx = _xlsx()
    root, result = _build(
        _bound_assets(
            table_payload=xlsx,
            table_format=["ZIP_V1", "XLSX_V1"],
        )
    )

    units = _records(root, "cases_v2/consumable_source_units.jsonl")
    assert len(units) == 1
    assert units[0]["origin_kind"] == "RAW_XLSX_CONTAINER_SELF"
    account = json.loads(
        root.read_bytes("source_inventory_v2/container_accounts/container-000001.json")
    )
    assert account["content_profile"] == "XLSX_V1"
    assert {
        entry["source_only_exclusion_reason_or_null"] for entry in account["entries"]
    } == {"XLSX_PACKAGE_COMPONENT"}
    assert result.canonical["case_count"] == 1

    invalid_xlsx = _zip(
        {
            "[Content_Types].xml": b"<Types/>",
            "_rels/.rels": b"<Relationships/>",
            "xl/workbook.xml": b"<workbook/>",
            "xl/worksheets/sheet1.xml": b"<worksheet/>",
        }
    )
    with pytest.raises(SourceBearingExtensionError, match="DECLARED_KIND_FORMAT"):
        _build(
            _bound_assets(
                table_payload=invalid_xlsx,
                table_format=["ZIP_V1", "XLSX_V1"],
            )
        )


def test_declared_format_bypass_cross_doi_and_mixed_p_fail_closed() -> None:
    archive = _zip({"table.csv": b"panel,value\na,1\n"})
    with pytest.raises(SourceBearingExtensionError, match="DECLARED_KIND_FORMAT"):
        _build(
            _bound_assets(
                table_payload=archive,
                table_kind="source_data",
                table_format=["NONE", "CSV_V1"],
            )
        )

    assets = _bound_assets()
    assets[0][0]["doi"] = "10.9999/other"
    with pytest.raises(SourceBearingExtensionError, match="metadata invalid"):
        _build(assets)

    table2 = _asset(
        article_id="article-1",
        doi_id="10.9999/source-1",
        asset_id="table-2",
        payload=b"panel,value\nb,2\n",
        kind="source_data",
        detected=["NONE", "CSV_V1"],
        hints=[
            {
                "member_selector_or_null": None,
                "panel_id": "panel-b",
                "case_group_or_null": "multi",
                "figure_asset_id": "figure",
                "caption_asset_id": "caption",
            }
        ],
    )
    table3 = _asset(
        article_id="article-1",
        doi_id="10.9999/source-1",
        asset_id="table-3",
        payload=b"panel,value\nc,3\n",
        kind="source_data",
        detected=["NONE", "CSV_V1"],
        hints=[
            {
                "member_selector_or_null": None,
                "panel_id": "panel-c",
                "case_group_or_null": "multi",
                "figure_asset_id": "figure",
                "caption_asset_id": "caption",
            }
        ],
    )
    with pytest.raises(SourceBearingExtensionError, match="MULTI_STRATUM"):
        _build([*_bound_assets(), table2, table3])


def test_duplicate_doi_case_group_panel_membership_is_terminal() -> None:
    duplicate_group_hints = [
        {
            "member_selector_or_null": None,
            "panel_id": "panel-a",
            "case_group_or_null": "shared-case",
            "figure_asset_id": "figure",
            "caption_asset_id": "caption",
        }
    ]
    duplicate_table = _asset(
        article_id="article-1",
        doi_id="10.9999/source-1",
        asset_id="table-duplicate",
        payload=b"panel,value\na,2\n",
        kind="source_data",
        detected=["NONE", "CSV_V1"],
        hints=duplicate_group_hints,
    )
    with pytest.raises(SourceBearingExtensionError, match="DUPLICATE_PANEL_MEMBERSHIP"):
        _build(
            [
                *_bound_assets(table_hints=duplicate_group_hints),
                duplicate_table,
            ]
        )


def test_archive_directory_data_and_aggregate_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_bearing_directory = _zip({"directory/": b"must not be ignored"})
    with pytest.raises(SourceBearingExtensionError, match="DIRECTORY_ENTRY"):
        _build(
            [
                _asset(
                    article_id="article-1",
                    doi_id="10.9999/source-1",
                    asset_id="archive",
                    payload=data_bearing_directory,
                    kind="source_archive",
                    detected=["ZIP_V1", "GENERIC_ZIP_V1"],
                ),
                *_bound_assets()[1:],
            ]
        )

    import experiments.c2_source_bearing_extension as extension

    monkeypatch.setattr(extension, "MAX_ARCHIVE_CONTAINER_UNCOMPRESSED_BYTES", 20)
    aggregate_limited_archive = _zip(
        {"first.txt": b"123456789012", "second.txt": b"abcdefghijkl"}
    )
    with pytest.raises(
        SourceBearingExtensionError,
        match="CONTAINER_UNCOMPRESSED_LIMIT",
    ):
        _build(
            [
                _asset(
                    article_id="article-1",
                    doi_id="10.9999/source-1",
                    asset_id="archive",
                    payload=aggregate_limited_archive,
                    kind="source_archive",
                    detected=["ZIP_V1", "GENERIC_ZIP_V1"],
                ),
                *_bound_assets()[1:],
            ]
        )

    monkeypatch.setattr(extension, "MAX_ARCHIVE_CONTAINER_UNCOMPRESSED_BYTES", 1024)
    monkeypatch.setattr(extension, "MAX_ARCHIVE_RUN_UNCOMPRESSED_BYTES", 20)
    with pytest.raises(SourceBearingExtensionError, match="RUN_UNCOMPRESSED_LIMIT"):
        _build(
            [
                _asset(
                    article_id="article-1",
                    doi_id="10.9999/source-1",
                    asset_id="archive",
                    payload=aggregate_limited_archive,
                    kind="source_archive",
                    detected=["ZIP_V1", "GENERIC_ZIP_V1"],
                ),
                *_bound_assets()[1:],
            ]
        )


def test_archive_traversal_rejects_and_unmapped_valid_table_stays_accounted() -> None:
    traversal = _zip({"../escape.csv": b"panel,value\na,1\n"})
    with pytest.raises(SourceBearingExtensionError, match="SELECTOR"):
        _build(
            [
                _asset(
                    article_id="article-1",
                    doi_id="10.9999/source-1",
                    asset_id="archive",
                    payload=traversal,
                    kind="source_archive",
                    detected=["ZIP_V1", "GENERIC_ZIP_V1"],
                ),
                *_bound_assets()[1:],
            ]
        )

    root, result = _build(_bound_assets(table_hints=[]))
    candidate_inputs = _records(root, "cases_v2/candidate_set_inputs.jsonl")
    consumption = _records(root, "cases_v2/downstream_consumption.jsonl")
    exclusions = _records(root, "cases_v2/source_only_exclusions.jsonl")
    assert result.canonical["case_count"] == 0
    assert candidate_inputs[0]["candidate_outcome"] == "SOURCE_ONLY_EXCLUSION"
    assert consumption[0]["consumption_disposition"] == "CANDIDATE_SET_INPUT"
    assert exclusions[0]["reason_code"] == "NO_BOUND_FIGURE_CAPTION"
    case_sets = json.loads(
        root.read_bytes("canonical_v2/canonical_case_set_manifest.json")
    )["case_sets"]
    assert case_sets == [
        {
            "doi_id": "10.9999/source-1",
            "cases": [],
            "canonical_case_set_hash": _sha256(
                _canonical({"doi_id": "10.9999/source-1", "cases": []})
            ),
        }
    ]


def test_consumption_review_and_raw_byte_tampering_are_rejected() -> None:
    root, _ = _build(_bound_assets())
    consumption_path = "cases_v2/downstream_consumption.jsonl"
    root.payloads[consumption_path] = b""
    with pytest.raises(SourceBearingExtensionError, match="CONSUMPTION_BIJECTION"):
        validate_source_bearing_extension(
            root, partition_records=1, source_chunk_sha256="a" * 64
        )

    root, _ = _build(_bound_assets())
    root.payloads["review_v2/structural_outcomes.jsonl"] = b""
    with pytest.raises(SourceBearingExtensionError, match="review coverage"):
        validate_source_bearing_extension(
            root, partition_records=1, source_chunk_sha256="a" * 64
        )

    root, _ = _build(_bound_assets())
    root.payloads["content/_sources/article-1/table.bin"] = b"panel,value\nforged,9\n"
    with pytest.raises(SourceBearingExtensionError, match="raw source asset bytes changed"):
        validate_source_bearing_extension(
            root, partition_records=1, source_chunk_sha256="a" * 64
        )


def test_v2_opt_in_keeps_a_zero_source_root_on_the_empty_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-source-bearing-zero-source") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        result = finalizer.finalize_remediation_root(
            chunk_id="001",
            source_bearing_v2=True,
            **paths,
        )
        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES"
        assert (paths["target_root"] / "canonical_v1/canonical_summary.json").is_file()
        assert not (paths["target_root"] / "canonical_v2").exists()


def test_v2_processes_prior_attempt_source_without_assigning_terminal_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-source-bearing-prior-download") as workspace:
        paths = _make_fixture(
            workspace,
            monkeypatch,
            chunk_id="001",
            downloaded_mode="source",
            downloaded_attempt="initial",
        )
        _upgrade_raw_source_descriptor_v2(paths["raw_root"])
        finalizer.finalize_remediation_root(
            chunk_id="001",
            source_bearing_v2=True,
            **paths,
        )
        dispositions = [
            json.loads(line)
            for line in (
                paths["target_root"] / "p_evidence_v2/acquisition_dispositions.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert dispositions[0]["final_disposition"] == "NON_STRATIFIED_NO_SOURCE_DATA"
        assert (
            paths["target_root"] / "p_evidence_v2/source_classifications.jsonl"
        ).read_text(encoding="utf-8") == ""
        assert (paths["target_root"] / "canonical_v2/cases.jsonl").is_file()


def _upgrade_raw_source_descriptor_v2(raw_root: Path) -> None:
    provenance_path, provenance = next(
        (path, value)
        for path in sorted((raw_root / "content" / "_provenance").glob("*.json"))
        if (value := json.loads(path.read_text(encoding="utf-8"))).get("source_evidence")
    )
    article_id = provenance_path.stem
    old_source = raw_root / "content" / "_sources" / article_id / "source.json"
    old_source.unlink()
    source_dir = old_source.parent
    table = source_dir / "table.csv"
    figure = source_dir / "figure.png"
    caption = source_dir / "caption.txt"
    table.write_bytes(b"panel,value\na,1\n")
    figure.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    caption.write_bytes(b"fixture caption")
    doi_id = provenance["doi"]
    assets: list[dict[str, Any]] = []
    for asset_id, path, kind, detected, hints in (
        (
            "table",
            table,
            "source_data",
            ["NONE", "CSV_V1"],
            [
                {
                    "member_selector_or_null": None,
                    "panel_id": "panel-a",
                    "case_group_or_null": None,
                    "figure_asset_id": "figure",
                    "caption_asset_id": "caption",
                }
            ],
        ),
        ("figure", figure, "figure", ["NONE", "OTHER_REGISTERED_V1"], []),
        ("caption", caption, "caption", ["NONE", "OTHER_REGISTERED_V1"], []),
    ):
        payload = path.read_bytes()
        assets.append(
            {
                "asset_id": asset_id,
                "relative_path": str(path.relative_to(raw_root)),
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "doi": doi_id,
                "declared_asset_kind": kind,
                "declared_format_tuple": detected,
                "candidate_hints": hints,
            }
        )
    descriptor = {
        "schema_version": "c2-source-evidence-v2",
        "doi": doi_id,
        "article_id": article_id,
        "provenance_relative_path": f"content/_provenance/{article_id}.json",
        "assets": assets,
    }
    descriptor["descriptor_hash"] = _sha256(_canonical(descriptor))
    descriptor_path = raw_root / "content" / "_source_evidence" / f"{article_id}.json"
    descriptor_path.write_bytes(_canonical(descriptor))
    payload = descriptor_path.read_bytes()
    provenance["source_evidence"] = {
        "descriptor_path": str(descriptor_path.relative_to(raw_root)),
        "descriptor_sha256": _sha256(payload),
        "descriptor_bytes": len(payload),
    }
    provenance_path.write_bytes(json.dumps(provenance, sort_keys=True).encode("utf-8"))


@pytest.mark.parametrize(("chunk_id", "expected_records"), [("001", 200), ("013", 63)])
def test_finalizer_seals_v2_source_bearing_200_and_63_roots(
    monkeypatch: pytest.MonkeyPatch,
    chunk_id: str,
    expected_records: int,
) -> None:
    with experiment_workspace(f"c2-source-bearing-{chunk_id}") as workspace:
        paths = _make_fixture(
            workspace,
            monkeypatch,
            chunk_id=chunk_id,
            downloaded_mode="source",
        )
        _upgrade_raw_source_descriptor_v2(paths["raw_root"])
        result = finalizer.finalize_remediation_root(
            chunk_id=chunk_id,
            source_bearing_v2=True,
            **paths,
        )

        target = paths["target_root"]
        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
        assert len(
            (target / "p_evidence_v2/acquisition_dispositions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ) == expected_records
        assert (
            json.loads(
                (target / "control/v2/source_bearing_extension_validation.json").read_text(
                    encoding="utf-8"
                )
            )["status"]
            == "PASS"
        )
        assert not (target / "control/source_classification_blocked.json").exists()

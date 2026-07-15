from __future__ import annotations

import hashlib
import io
import json
import struct
import subprocess
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from experiments import c2_remediation_root_finalizer as finalizer
from experiments import c2_source_bearing_extension as source_extension
from experiments.c2_source_bearing_extension import (
    SourceBearingExtensionError,
    build_source_bearing_extension,
    build_source_bearing_extension_for_testing,
    validate_source_bearing_extension,
    validate_source_bearing_extension_for_testing,
    verify_source_extension_code_attestation_for_testing,
)
from tests.test_c2_remediation_root_finalizer import (
    _make_fixture,
    _refresh_raw_inventory,
)
from tests.test_experiment_support import experiment_workspace


@pytest.fixture(autouse=True)
def _exercise_guarded_remediation_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in (finalizer, source_extension):
        monkeypatch.setattr(
            module,
            "require_external_m1_trust_lock",
            lambda: None,
        )
    monkeypatch.setattr(
        source_extension,
        "require_test_only_source_extension_gate",
        lambda: None,
    )
    monkeypatch.setattr(
        finalizer,
        "require_test_only_finalizer_gate",
        lambda: None,
    )
    monkeypatch.setattr(
        finalizer,
        "_require_owner_remediation_policy_for_chunk",
        lambda _chunk_id, _partition: (
            SimpleNamespace(authorization_id_sha256="test-authorization"),
            SimpleNamespace(policy_id_sha256="test-policy"),
            SimpleNamespace(required_action="FRESH_REMEDIATION_REQUIRED"),
        ),
    )


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


def _with_deflate_option_flags(payload: bytes, option_bits: int) -> bytes:
    output = bytearray(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.compress_type == zipfile.ZIP_DEFLATED:
                struct.pack_into("<H", output, info.header_offset + 6, option_bits)
    eocd = output.rfind(b"PK\x05\x06")
    assert eocd >= 0
    central_offset = struct.unpack_from("<I", output, eocd + 16)[0]
    cursor = central_offset
    while cursor < eocd:
        assert output[cursor : cursor + 4] == b"PK\x01\x02"
        compression = struct.unpack_from("<H", output, cursor + 10)[0]
        if compression == zipfile.ZIP_DEFLATED:
            struct.pack_into("<H", output, cursor + 8, option_bits)
        name_size, extra_size, comment_size = struct.unpack_from(
            "<HHH", output, cursor + 28
        )
        cursor += 46 + name_size + extra_size + comment_size
    assert cursor == eocd
    return bytes(output)


def _dos_directory_with_data_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        directory = zipfile.ZipInfo("dos-directory/")
        directory.create_system = 0
        directory.external_attr = 0x10
        archive.writestr(directory, b"must-not-be-directory-data")
    return output.getvalue()


def _dos_directory_archive(directory_payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        directory = zipfile.ZipInfo("dos-directory/")
        directory.create_system = 0
        directory.external_attr = 0x10
        archive.writestr(directory, directory_payload)
        archive.writestr("table.csv", b"panel,value\na,1\n")
    return output.getvalue()


def _xlsx(
    *,
    workbook_extra_relationship: bytes = b"",
    worksheet_target: bytes = b"worksheets/sheet1.xml",
    worksheet_part: str = "xl/worksheets/sheet1.xml",
) -> bytes:
    worksheet_part_bytes = worksheet_part.encode("utf-8")
    style_parts = (
        {
            "xl/styles.xml": (
                b'<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
                b'2006/main"/>'
            )
        }
        if workbook_extra_relationship
        else {}
    )
    content_type_extra = (
        b'<Override PartName="/xl/styles.xml" ContentType="application/vnd.'
        b'openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        if workbook_extra_relationship
        else b""
    )
    return _zip(
        {
            "[Content_Types].xml": (
                b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
                b'content-types"><Default Extension="rels" ContentType="application/'
                b'vnd.openxmlformats-package.relationships+xml"/><Default '
                b'Extension="xml" ContentType="application/xml"/><Override '
                b'PartName="/xl/workbook.xml" ContentType="application/vnd.'
                b'openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                b'<Override PartName="/'
                + worksheet_part_bytes
                + b'" ContentType='
                b'"application/vnd.openxmlformats-officedocument.spreadsheetml.'
                b'worksheet+xml"/>'
                + content_type_extra
                + b"</Types>"
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
                b'worksheet" Target="'
                + worksheet_target
                + b'"/>'
                + workbook_extra_relationship
                + b"</Relationships>"
            ),
            worksheet_part: (
                b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
                b'2006/main"><sheetData/></worksheet>'
            ),
            **style_parts,
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
    result = build_source_bearing_extension_for_testing(
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


def _test_only_attestation_payload(repository: Path, commit: str) -> bytes:
    paths = (
        "agent/experiments/c2_m1_trust_boundary.py",
        "agent/experiments/c2_remediation_root_finalizer.py",
        "agent/experiments/c2_source_bearing_extension.py",
        "agent/experiments/c2_stageb_source_extension_code_attestation.py",
        "agent/experiments/cli.py",
        "agent/experiments/models.py",
        "agent/experiments/schemas/c2_v2_candidate_set_input_v1.schema.json",
        "agent/experiments/schemas/c2_v2_consumable_source_unit_v1.schema.json",
        "agent/experiments/schemas/c2_v2_consumption_bijection_validation_v1.schema.json",
        "agent/experiments/schemas/c2_v2_container_accounting_index_v1.schema.json",
        "agent/experiments/schemas/c2_v2_detected_format_v1.schema.json",
        "agent/experiments/schemas/c2_v2_downstream_consumption_v1.schema.json",
        "agent/experiments/schemas/c2_v2_fd_format_classifier_config_v1.schema.json",
    )
    entries = []
    for relative in paths:
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", f"{commit}:{relative}"],
            text=True,
        ).strip()
        payload = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{relative}"]
        )
        entries.append(
            {
                "relative_path": relative,
                "git_blob_object_id": blob,
                "sha256": _sha256(payload),
            }
        )
    return _canonical(
        {
            "schema_version": "c2_source_bearing_extension_test_attestation_v1",
            "approved_implementation_commit_full": commit,
            "attested_paths": entries,
        }
    ) + b"\n"


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


def test_test_only_code_attestation_rejects_wrong_commit_blob_and_runtime_path() -> None:
    repository = Path(__file__).resolve().parents[2]
    current_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    attestation = verify_source_extension_code_attestation_for_testing(repository)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            attestation.attestation_commit_full,
            current_commit,
        ],
        check=True,
    )
    assert {
        blob.relative_path for blob in attestation.code_blobs
    } >= {
        "agent/experiments/c2_m1_trust_boundary.py",
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
        m1_trust_boundary_path = (
            clone / "agent/experiments/c2_m1_trust_boundary.py"
        )
        verify_source_extension_code_attestation_for_testing(
            clone,
            loaded_extension_path=extension_path,
            loaded_finalizer_path=finalizer_path,
            loaded_m1_trust_boundary_path=m1_trust_boundary_path,
        )
        with pytest.raises(SourceBearingExtensionError, match="loaded runtime path"):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=Path(__file__),
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )
        with pytest.raises(SourceBearingExtensionError, match="loaded runtime path"):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=Path(__file__),
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
                / "agent/tests/fixtures/c2_source_bearing_extension_test_attestation.json"
            ).read_text(encoding="utf-8")
        )
        source_manifest["attested_paths"][0]["sha256"] = "0" * 64
        forged_manifest = (
            clone
            / "agent/tests/fixtures/c2_source_bearing_extension_test_attestation.json"
        )
        forged_manifest.parent.mkdir(parents=True, exist_ok=True)
        forged_manifest.write_bytes(
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
                "agent/tests/fixtures/c2_source_bearing_extension_test_attestation.json",
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
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
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
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
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
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--detach", current_commit],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        m1_trust_boundary_path.write_bytes(
            m1_trust_boundary_path.read_bytes() + b"\n# forged M1 boundary\n"
        )
        with pytest.raises(SourceBearingExtensionError, match="worktree is dirty"):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )

        malicious_clone = workspace / "malicious-test-only-clone"
        subprocess.run(
            [
                "git",
                "clone",
                "--no-local",
                "--no-checkout",
                str(repository),
                str(malicious_clone),
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
                str(malicious_clone),
                "checkout",
                "--detach",
                attestation.approved_implementation_commit_full,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for key, value in (
            ("user.email", "attestation-test@example.test"),
            ("user.name", "Attestation Test"),
        ):
            subprocess.run(
                ["git", "-C", str(malicious_clone), "config", key, value],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        malicious_extension = (
            malicious_clone / "agent/experiments/c2_source_bearing_extension.py"
        )
        malicious_extension.write_bytes(
            malicious_extension.read_bytes() + b"\n# malicious test-only implementation\n"
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(malicious_clone),
                "add",
                "agent/experiments/c2_source_bearing_extension.py",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(malicious_clone), "commit", "-m", "malicious implementation"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        malicious_implementation = subprocess.check_output(
            ["git", "-C", str(malicious_clone), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        malicious_manifest = (
            malicious_clone
            / "agent/tests/fixtures/c2_source_bearing_extension_test_attestation.json"
        )
        malicious_manifest.parent.mkdir(parents=True, exist_ok=True)
        malicious_manifest.write_bytes(
            _test_only_attestation_payload(
                malicious_clone,
                malicious_implementation,
            )
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(malicious_clone),
                "add",
                "agent/tests/fixtures/c2_source_bearing_extension_test_attestation.json",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(malicious_clone), "commit", "-m", "matching manifest"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        matching_test_only = verify_source_extension_code_attestation_for_testing(
            malicious_clone,
            loaded_extension_path=malicious_extension,
            loaded_finalizer_path=(
                malicious_clone
                / "agent/experiments/c2_remediation_root_finalizer.py"
            ),
            loaded_m1_trust_boundary_path=(
                malicious_clone / "agent/experiments/c2_m1_trust_boundary.py"
            ),
        )
        assert (
            matching_test_only.approved_implementation_commit_full
            == malicious_implementation
        )



def test_test_only_code_attestation_rejects_invalid_child_topologies_and_manifest() -> None:
    repository = Path(__file__).resolve().parents[2]
    attestation = verify_source_extension_code_attestation_for_testing(repository)
    implementation = attestation.approved_implementation_commit_full
    manifest_relative = (
        "agent/tests/fixtures/c2_source_bearing_extension_test_attestation.json"
    )
    source_manifest = json.loads(
        (repository / manifest_relative).read_text(encoding="utf-8")
    )

    with experiment_workspace("c2-source-extension-attestation-topology") as workspace:
        clone = workspace / "topology-clone"
        subprocess.run(
            ["git", "clone", "--no-local", "--no-checkout", str(repository), str(clone)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def git(*arguments: str) -> None:
            subprocess.run(
                ["git", "-C", str(clone), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        for key, value in (
            ("user.email", "attestation-topology-test@example.test"),
            ("user.name", "Attestation Topology Test"),
        ):
            git("config", key, value)

        manifest_path = clone / manifest_relative
        extension_path = clone / "agent/experiments/c2_source_bearing_extension.py"
        finalizer_path = clone / "agent/experiments/c2_remediation_root_finalizer.py"
        m1_trust_boundary_path = (
            clone / "agent/experiments/c2_m1_trust_boundary.py"
        )
        unexpected_child = clone / "unexpected-test-only-child.txt"

        def checkout_implementation() -> None:
            git("checkout", "--detach", implementation)
            if unexpected_child.exists():
                unexpected_child.unlink()

        def clone_manifest() -> dict[str, Any]:
            return json.loads(_canonical(source_manifest).decode("utf-8"))

        def commit_manifest(
            payload: bytes,
            message: str,
            *,
            with_unexpected_child: bool = False,
        ) -> None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_bytes(payload)
            paths = [manifest_relative]
            if with_unexpected_child:
                unexpected_child.write_text("unexpected child diff\n", encoding="utf-8")
                paths.append(unexpected_child.relative_to(clone).as_posix())
            git("add", *paths)
            git("commit", "-m", message)

        checkout_implementation()
        with pytest.raises(
            SourceBearingExtensionError,
            match="must add only its manifest",
        ):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )

        checkout_implementation()
        commit_manifest(
            _test_only_attestation_payload(clone, implementation),
            "test non-manifest attestation child",
            with_unexpected_child=True,
        )
        with pytest.raises(
            SourceBearingExtensionError,
            match="must add only its manifest",
        ):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )

        wrong_blob = clone_manifest()
        wrong_blob["attested_paths"][0]["git_blob_object_id"] = "0" * 40
        checkout_implementation()
        commit_manifest(
            _canonical(wrong_blob) + b"\n",
            "test wrong attestation blob",
        )
        with pytest.raises(SourceBearingExtensionError, match="attested blob mismatch"):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )

        wrong_path = clone_manifest()
        wrong_path["attested_paths"][0]["relative_path"] = (
            "agent/experiments/not_attested.py"
        )
        checkout_implementation()
        commit_manifest(
            _canonical(wrong_path) + b"\n",
            "test wrong attestation path",
        )
        with pytest.raises(
            SourceBearingExtensionError,
            match="code attestation blob is invalid",
        ):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )

        wrong_manifest = clone_manifest()
        wrong_manifest["approved_implementation_commit_full"] = "0" * 40
        checkout_implementation()
        commit_manifest(
            _canonical(wrong_manifest) + b"\n",
            "test wrong attestation manifest",
        )
        with pytest.raises(
            SourceBearingExtensionError,
            match="code attestation manifest is invalid",
        ):
            verify_source_extension_code_attestation_for_testing(
                clone,
                loaded_extension_path=extension_path,
                loaded_finalizer_path=finalizer_path,
                loaded_m1_trust_boundary_path=m1_trust_boundary_path,
            )


def test_production_extension_routes_use_fixed_runtime_attestation() -> None:
    root, reader, records, provenance, terminal_rows = _source_root(
        source_assets=_bound_assets()
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
    replay = validate_source_bearing_extension(
        root,
        partition_records=1,
        source_chunk_sha256="a" * 64,
    )

    assert result.status == "SOURCE_CLASSIFICATION_V2_COMPLETE"
    assert replay["status"] == "PASS"
    assert replay["source_classification_count"] == 1


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
    replay = validate_source_bearing_extension_for_testing(
        root, partition_records=1, source_chunk_sha256="a" * 64
    )
    assert replay["status"] == "PASS"
    protocol = json.loads(root.read_bytes("review_v2/structural_protocol.json"))
    assert protocol["review_mode"] == "C2_V2_STRUCTURAL_REVIEW_V1"


def test_zip_preflight_reuses_one_member_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _zip(
        {
            f"table-{index:04d}.csv": b"panel,value\na,1\n"
            for index in range(200)
        }
    )
    original = source_extension.zipfile.ZipFile
    calls = 0

    def counting_zipfile(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(source_extension.zipfile, "ZipFile", counting_zipfile)

    assert source_extension.validate_v2_source_asset_payload(archive) == (
        "ZIP_V1",
        "GENERIC_ZIP_V1",
    )
    assert calls == 2
    expired_budget = source_extension.new_v2_archive_run_budget(
        deadline_monotonic=time.monotonic() - 1
    )
    with pytest.raises(SourceBearingExtensionError, match="ARCHIVE_DEADLINE"):
        source_extension.validate_v2_source_asset_payload(
            archive,
            archive_budget=expired_budget,
        )


def test_zip_preflight_rejects_non_source_and_unaccounted_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for archive in (
        _zip({}),
        _zip({"__MACOSX/._metadata": b"resource-only"}),
        _zip({"notes.txt": b"not tabular source data"}),
    ):
        with pytest.raises(
            SourceBearingExtensionError,
            match="ARCHIVE_NO_SUBSTANTIVE_SOURCE",
        ):
            source_extension.validate_v2_source_asset_payload(archive)

    valid = _zip({"table.csv": b"panel,value\na,1\n"})
    eocd = valid.rfind(b"PK\x05\x06")
    assert eocd >= 0
    central_offset = struct.unpack_from("<I", valid, eocd + 16)[0]
    gap = b"hidden-gap"
    with_gap = bytearray(valid[:central_offset] + gap + valid[central_offset:])
    struct.pack_into(
        "<I",
        with_gap,
        eocd + len(gap) + 16,
        central_offset + len(gap),
    )
    with pytest.raises(
        SourceBearingExtensionError,
        match="ZIP_PHYSICAL_COVERAGE",
    ):
        source_extension.validate_v2_source_asset_payload(bytes(with_gap))

    monkeypatch.setattr(source_extension, "MAX_ARCHIVE_MEMBER_BYTES", 10)
    with pytest.raises(
        SourceBearingExtensionError,
        match="ZIP_MEMBER_TOO_LARGE",
    ):
        source_extension.validate_v2_source_asset_payload(
            _zip(
                {
                    "table.csv": b"a,b\n1,2\n",
                    "__MACOSX/._metadata": b"oversized-resource",
                }
            )
        )


def test_xlsx_rejects_contradictory_xml_encoding() -> None:
    workbook = _xlsx()
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        members = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }
    members["[Content_Types].xml"] = (
        b'<?xml version="1.0" encoding="UTF-16"?>'
        + members["[Content_Types].xml"]
    )

    with pytest.raises(SourceBearingExtensionError):
        source_extension.validate_v2_source_asset_payload(_zip(members))


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
    assert validate_source_bearing_extension_for_testing(
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

    traversal_style_xlsx = _xlsx(
        workbook_extra_relationship=(
            b'<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.'
            b'org/officeDocument/2006/relationships/styles" Target="../styles.xml"/>'
        )
    )
    with pytest.raises(SourceBearingExtensionError, match="DECLARED_KIND_FORMAT"):
        _build(
            _bound_assets(
                table_payload=traversal_style_xlsx,
                table_format=["ZIP_V1", "XLSX_V1"],
            )
        )

    xlsx_with_styles = _xlsx(
        workbook_extra_relationship=(
            b'<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.'
            b'org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        )
    )
    styled_root, styled_result = _build(
        _bound_assets(
            table_payload=xlsx_with_styles,
            table_format=["ZIP_V1", "XLSX_V1"],
        )
    )
    assert styled_result.canonical["case_count"] == 1
    assert validate_source_bearing_extension_for_testing(
        styled_root,
        partition_records=1,
        source_chunk_sha256="a" * 64,
    )["status"] == "PASS"


@pytest.mark.parametrize("option_bits", [0x2, 0x4, 0x6])
def test_valid_deflate_compression_option_flags_are_supported(
    option_bits: int,
) -> None:
    xlsx = _with_deflate_option_flags(_xlsx(), option_bits)

    detected = source_extension._detect_format(xlsx)

    assert detected.tuple == ("ZIP_V1", "XLSX_V1")


def test_opc_relationship_parent_segments_normalize_within_package() -> None:
    assert source_extension._resolve_internal_opc_target(
        "xl/worksheets/sheet1.xml",
        "../drawings/drawing1.xml",
    ) == "xl/drawings/drawing1.xml"
    assert (
        source_extension._resolve_internal_opc_target(
            "xl/workbook.xml",
            "../../outside.xml",
        )
        is None
    )
    assert source_extension._resolve_internal_opc_target(
        "xl/workbook.xml",
        "/xl/worksheets/sheet1.xml",
    ) == "xl/worksheets/sheet1.xml"
    assert source_extension._resolve_internal_opc_target(
        "xl/workbook.xml",
        "./worksheets/sheet1.xml",
    ) == "xl/worksheets/sheet1.xml"
    assert (
        source_extension._resolve_internal_opc_target(
            "xl/workbook.xml",
            "urn:x/../worksheets/sheet1.xml",
        )
        is None
    )


def test_opc_uri_scheme_is_not_an_internal_xlsx_target() -> None:
    disguised = _xlsx(worksheet_target=b"urn:x/../worksheets/sheet1.xml")
    package_root = _xlsx(worksheet_target=b"/xl/worksheets/sheet1.xml")

    assert source_extension._detect_format(disguised).tuple == (
        "ZIP_V1",
        "GENERIC_ZIP_V1",
    )
    assert source_extension._detect_format(package_root).tuple == (
        "ZIP_V1",
        "XLSX_V1",
    )


def test_ooxml_parser_rejects_utf16_and_entity_declarations() -> None:
    entity_xml = (
        '<?xml version="1.0" encoding="UTF-16"?>'
        '<!DOCTYPE workbook [<!ENTITY injected "fabricated">]>'
        '<workbook xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main">&injected;</workbook>'
    ).encode("utf-16")

    assert source_extension._parse_ooxml_xml(entity_xml) is None
    assert (
        source_extension._parse_ooxml_xml(
            b'<!doctype workbook><workbook xmlns="urn:test"/>'
        )
        is None
    )


@pytest.mark.parametrize(
    ("raw_target", "xml_target"),
    [
        (" worksheets/sheet1.xml", b" worksheets/sheet1.xml"),
        ("worksheets/sheet1.xml?", b"worksheets/sheet1.xml?"),
        ("worksheets/sheet1.xml#", b"worksheets/sheet1.xml#"),
        ("//[", b"//["),
        ("///xl/worksheets/sheet1.xml", b"///xl/worksheets/sheet1.xml"),
        ("1:x/../worksheets/sheet1.xml", b"1:x/../worksheets/sheet1.xml"),
        ("worksheets/she\net1.xml", b"worksheets/she&#10;et1.xml"),
        ("worksheets/sheet1.xml/.", b"worksheets/sheet1.xml/."),
        ("worksheets/%2e%2e/sheet1.xml", b"worksheets/%2e%2e/sheet1.xml"),
        ("worksheets/\u0080.xml", "worksheets/\u0080.xml".encode("utf-8")),
    ],
)
def test_malformed_opc_targets_fail_closed_without_aliasing(
    raw_target: str,
    xml_target: bytes,
) -> None:
    assert (
        source_extension._resolve_internal_opc_target(
            "xl/workbook.xml",
            raw_target,
        )
        is None
    )
    assert source_extension._detect_format(
        _xlsx(worksheet_target=xml_target)
    ).tuple == ("ZIP_V1", "GENERIC_ZIP_V1")


def test_illegal_opc_path_character_cannot_match_a_packaged_worksheet() -> None:
    target = b"worksheets/[sheet].xml"
    package = _xlsx(
        worksheet_target=target,
        worksheet_part="xl/worksheets/[sheet].xml",
    )

    assert source_extension._resolve_internal_opc_target(
        "xl/workbook.xml",
        target.decode("ascii"),
    ) is None
    assert source_extension._detect_format(package).tuple == (
        "ZIP_V1",
        "GENERIC_ZIP_V1",
    )


@pytest.mark.parametrize(
    "target",
    (
        "worksheets/sheet1.xml.",
        "worksheets/...",
        "worksheets/directory./sheet1.xml",
    ),
)
def test_opc_trailing_dot_alias_cannot_match_a_packaged_worksheet(
    target: str,
) -> None:
    package_target = target.encode("ascii")
    package_part = f"xl/{target}"
    package = _xlsx(
        worksheet_target=package_target,
        worksheet_part=package_part,
    )

    assert (
        source_extension._resolve_internal_opc_target(
            "xl/workbook.xml",
            target,
        )
        is None
    )
    assert source_extension._detect_format(package).tuple == (
        "ZIP_V1",
        "GENERIC_ZIP_V1",
    )


def test_zip_selector_rejects_unicode_control_characters() -> None:
    with pytest.raises(SourceBearingExtensionError, match="SELECTOR_UNSAFE"):
        source_extension._detect_format(_zip({"\u0080.csv": b"a,b\n1,2\n"}))


def test_v2_source_asset_preflight_reads_every_generic_zip_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_extension, "MAX_ARCHIVE_MEMBER_BYTES", 16)
    archive = _zip({"table.csv": b"a,b\n" + b"1,2\n" * 5})
    assert source_extension._detect_format(archive).tuple == (
        "ZIP_V1",
        "GENERIC_ZIP_V1",
    )

    with pytest.raises(SourceBearingExtensionError, match="MEMBER_TOO_LARGE"):
        source_extension.validate_v2_source_asset_payload(archive)


def test_v2_source_asset_preflight_bounds_recursive_entry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_extension, "MAX_ARCHIVE_RUN_ENTRIES", 2)
    nested = _zip(
        {
            "first.csv": b"a,b\n1,2\n",
            "second.csv": b"a,b\n3,4\n",
        }
    )
    archive = _zip({"nested.zip": nested})

    with pytest.raises(SourceBearingExtensionError, match="RUN_ENTRY_LIMIT"):
        source_extension.validate_v2_source_asset_payload(archive)


def test_tracked_openpyxl_workbook_is_xlsx_v1() -> None:
    sample = Path(__file__).resolve().parents[1] / "sample.xlsx"

    assert source_extension._detect_format(sample.read_bytes()).tuple == (
        "ZIP_V1",
        "XLSX_V1",
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
    import experiments.c2_source_bearing_extension as extension

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

    dos_directory_data = _dos_directory_with_data_zip()
    with pytest.raises(SourceBearingExtensionError, match="DIRECTORY_ENTRY"):
        extension._parse_zip_v1(dos_directory_data)
    with pytest.raises(SourceBearingExtensionError, match="DIRECTORY_ENTRY"):
        _build(
            [
                _asset(
                    article_id="article-1",
                    doi_id="10.9999/source-1",
                    asset_id="dos-archive",
                    payload=dos_directory_data,
                    kind="source_archive",
                    detected=["ZIP_V1", "GENERIC_ZIP_V1"],
                ),
                *_bound_assets()[1:],
            ]
        )

    valid_dos_archive = _dos_directory_archive(b"")
    root, _ = _build(
        [
            _asset(
                article_id="article-1",
                doi_id="10.9999/source-1",
                asset_id="archive",
                payload=valid_dos_archive,
                kind="source_archive",
                detected=["ZIP_V1", "GENERIC_ZIP_V1"],
                hints=[
                    {
                        "member_selector_or_null": "table.csv",
                        "panel_id": "panel-a",
                        "case_group_or_null": None,
                        "figure_asset_id": "figure",
                        "caption_asset_id": "caption",
                    }
                ],
            ),
            *_bound_assets()[1:],
        ]
    )
    replay_directory_data = _dos_directory_archive(b"forged-directory-data")
    raw_path = "content/_sources/article-1/archive.bin"
    account_path = "source_inventory_v2/container_accounts/container-000001.json"
    index_path = "source_inventory_v2/container_accounting_index.json"
    root.payloads[raw_path] = replay_directory_data
    account = json.loads(root.read_bytes(account_path))
    account["verified_file_sha256"] = _sha256(replay_directory_data)
    account["verified_bytes"] = len(replay_directory_data)
    account.pop("archive_accounting_hash")
    extension._seal(account, "archive_accounting_hash")
    root.payloads[account_path] = _canonical(account) + b"\n"
    index = json.loads(root.read_bytes(index_path))
    node = index["container_nodes"][0]
    node["verified_file_sha256"] = _sha256(replay_directory_data)
    node["verified_bytes"] = len(replay_directory_data)
    node["account_sha256"] = _sha256(root.read_bytes(account_path))
    node["archive_accounting_hash"] = account["archive_accounting_hash"]
    node.pop("entry_hash")
    extension._seal(node, "entry_hash")
    index.pop("index_hash")
    extension._seal(index, "index_hash")
    with pytest.raises(SourceBearingExtensionError, match="DIRECTORY_ENTRY"):
        extension._validate_archive_accounts(
            root,
            index,
            json.loads(
                root.read_bytes("source_inventory_v2/fd_format_classifier_config.json")
            ),
            extension._ArchiveRunBudget(),
        )

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
        validate_source_bearing_extension_for_testing(
            root, partition_records=1, source_chunk_sha256="a" * 64
        )

    root, _ = _build(_bound_assets())
    root.payloads["review_v2/structural_outcomes.jsonl"] = b""
    with pytest.raises(SourceBearingExtensionError, match="review coverage"):
        validate_source_bearing_extension_for_testing(
            root, partition_records=1, source_chunk_sha256="a" * 64
        )

    root, _ = _build(_bound_assets())
    root.payloads["content/_sources/article-1/table.bin"] = b"panel,value\nforged,9\n"
    with pytest.raises(SourceBearingExtensionError, match="raw source asset bytes changed"):
        validate_source_bearing_extension_for_testing(
            root, partition_records=1, source_chunk_sha256="a" * 64
        )


def test_v2_opt_in_keeps_a_zero_source_root_on_the_empty_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-source-bearing-zero-source") as workspace:
        paths = _make_fixture(workspace, monkeypatch, chunk_id="001")
        result = finalizer.finalize_remediation_root_for_testing(
            chunk_id="001",
            source_bearing_v2=True,
            **paths,
        )
        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES"
        assert (paths["target_root"] / "canonical_v1/canonical_summary.json").is_file()
        assert not (paths["target_root"] / "canonical_v2").exists()


def test_v2_opt_in_seals_prior_attempt_source_under_fixed_stage_b_policy(
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
        result = finalizer.finalize_remediation_root_for_testing(
            chunk_id="001",
            source_bearing_v2=True,
            **paths,
        )
        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
        assert (
            paths["target_root"]
            / "canonical_v2/canonical_case_set_manifest.json"
        ).is_file()
        assert (
            paths["target_root"]
            / "p_evidence_v2/source_classifications.jsonl"
        ).is_file()
        assert not (
            paths["target_root"] / "control/source_classification_blocked.json"
        ).exists()
        inventory = [
            json.loads(line)
            for line in (
                paths["target_root"]
                / "source_inventory_v2/source_inventory.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        classifications = [
            json.loads(line)
            for line in (
                paths["target_root"]
                / "p_evidence_v2/source_classifications.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        dispositions = [
            json.loads(line)
            for line in (
                paths["target_root"]
                / "p_evidence_v2/acquisition_dispositions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert inventory
        assert [item["doi_id"] for item in classifications] == ["10.9999/c2-1"]
        assert dispositions[0]["terminal_status"] == "NO_SOURCE_DATA"
        assert dispositions[0]["final_disposition"] == (
            "STRATIFIED_SOURCE_CANONICAL"
        )
        assert dispositions[0]["classification_reason"] == (
            "VERIFIED_PRIOR_ATTEMPT_SOURCE_CANONICAL_ALL_CASES"
        )
        assert dispositions[0]["source_inventory_binding_or_null"] is not None


def test_exact_63_prior_attempt_source_is_fully_accounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-source-bearing-prior-download-013") as workspace:
        paths = _make_fixture(
            workspace,
            monkeypatch,
            chunk_id="013",
            downloaded_mode="source",
            downloaded_attempt="initial",
        )
        _upgrade_raw_source_descriptor_v2(paths["raw_root"])
        result = finalizer.finalize_remediation_root_for_testing(
            chunk_id="013",
            source_bearing_v2=True,
            **paths,
        )

        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
        assert result["input_total"] == 63
        classifications = [
            json.loads(line)
            for line in (
                paths["target_root"]
                / "p_evidence_v2/source_classifications.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        dispositions = [
            json.loads(line)
            for line in (
                paths["target_root"]
                / "p_evidence_v2/acquisition_dispositions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(classifications) == 1
        assert len(dispositions) == 63
        assert dispositions[0]["source_inventory_binding_or_null"] is not None
        assert dispositions[0]["source_classification_record_hash_or_null"] == (
            classifications[0]["record_hash"]
        )


def test_stage_b_execution_keeps_raw_acquisition_binding_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with experiment_workspace("c2-stageb-raw-binding-separation") as workspace:
        paths = _make_fixture(
            workspace,
            monkeypatch,
            chunk_id="001",
            downloaded_mode="source",
        )
        _upgrade_raw_source_descriptor_v2(paths["raw_root"])
        assert (
            subprocess.check_output(
                ["git", "-C", str(paths["worktree"]), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            == finalizer.FROZEN_CODE_COMMIT
        )
        result = finalizer.finalize_remediation_root_for_testing(
            chunk_id="001",
            source_bearing_v2=True,
            **paths,
        )
        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
        sealed_root = paths["target_root"]
        raw_binding = json.loads(
            (sealed_root / "control/pre_download_binding.json").read_text(
                encoding="utf-8"
            )
        )
        execution_evidence = json.loads(
            (sealed_root / "control/execution_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        assert raw_binding["code_commit"] == finalizer.FROZEN_CODE_COMMIT
        assert execution_evidence["code_commit"] == finalizer.FROZEN_CODE_COMMIT
        assert (
            sealed_root / "control/v2/source_bearing_extension_validation.json"
        ).is_file()
        assert not (
            sealed_root / "control/source_classification_blocked.json"
        ).exists()


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
    _refresh_raw_inventory(raw_root)


@pytest.mark.parametrize("chunk_id", ["001", "013"])
def test_finalizer_seals_source_bearing_roots_with_fixed_stage_b_policy(
    monkeypatch: pytest.MonkeyPatch,
    chunk_id: str,
) -> None:
    with experiment_workspace(f"c2-source-bearing-{chunk_id}") as workspace:
        paths = _make_fixture(
            workspace,
            monkeypatch,
            chunk_id=chunk_id,
            downloaded_mode="source",
        )
        _upgrade_raw_source_descriptor_v2(paths["raw_root"])
        result = finalizer.finalize_remediation_root_for_testing(
            chunk_id=chunk_id,
            source_bearing_v2=True,
            **paths,
        )

        assert result["status"] == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
        assert result["input_total"] == finalizer.FROZEN_PARTITIONS[chunk_id].records
        if chunk_id == "013":
            assert result["input_total"] == 63
        assert (
            paths["target_root"]
            / "canonical_v2/canonical_case_set_manifest.json"
        ).is_file()
        assert (
            paths["target_root"]
            / "p_evidence_v2/source_classifications.jsonl"
        ).is_file()
        assert not (
            paths["target_root"] / "control/source_classification_blocked.json"
        ).exists()

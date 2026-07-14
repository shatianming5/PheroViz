from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

import experiments.c2_terminal_finalizer as c2_terminal_finalizer
import experiments.models as experiment_models
from experiments.c2_terminal_finalizer import (
    CHUNK_IDS,
    REPLACEMENT_CHUNK_IDS,
    TERMINAL_OUTCOME_STATUSES,
    C2AdmissionError,
    _finalize_manifest_for_testing as finalize_manifest,
    _finalize_to_path_for_testing as finalize_to_path,
    _prepare_finalization_for_testing as prepare_finalization,
    _write_final_report_for_testing as write_final_report,
    validate_final_report,
)
import experiments.cli as cli
from experiments.cli import main as cli_main
from experiments.models import sha256_file, sha256_json, write_json_atomic


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


@pytest.fixture(autouse=True)
def _enable_private_terminal_finalizer_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "require_external_m1_trust_lock", lambda: None)
    monkeypatch.setattr(
        cli,
        "finalize_c2_to_path",
        c2_terminal_finalizer._finalize_to_path_for_testing,
    )


class _FakeNativeFunction:
    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def __call__(self, *args: Any) -> Any:
        return self.implementation(*args)


class _FakeDarwinAclLibrary:
    def __init__(self, tag: int | None, permissions: int = 0) -> None:
        self.tag = tag
        self.permissions = permissions
        self.descriptor: int | None = None
        self.acl_type: int | None = None
        self.acl_get_fd_np = _FakeNativeFunction(self._get_fd)
        self.acl_get_entry = _FakeNativeFunction(self._get_entry)
        self.acl_get_tag_type = _FakeNativeFunction(self._get_tag)
        self.acl_get_permset_mask_np = _FakeNativeFunction(self._get_permissions)
        self.acl_free = _FakeNativeFunction(lambda _acl: 0)

    def _get_fd(self, descriptor: int, acl_type: int) -> int | None:
        self.descriptor = descriptor
        self.acl_type = acl_type
        if self.tag is None:
            ctypes.set_errno(errno.ENOENT)
            return None
        return 0xAC1

    def _get_entry(self, _acl: int, entry_id: int, entry: Any) -> int:
        if entry_id == 0:
            ctypes.cast(entry, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0xE17
            return 0
        ctypes.set_errno(errno.EINVAL)
        return -1

    def _get_tag(self, _entry: int, tag: Any) -> int:
        ctypes.cast(tag, ctypes.POINTER(ctypes.c_int)).contents.value = self.tag
        return 0

    def _get_permissions(self, _entry: int, permissions: Any) -> int:
        ctypes.cast(
            permissions,
            ctypes.POINTER(ctypes.c_uint64),
        ).contents.value = self.permissions
        return 0


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
            c2_terminal_finalizer._finalize_manifest_for_testing(manifest_path)


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
        final_report = c2_terminal_finalizer._finalize_manifest_for_testing(
            manifest_path
        )

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


def test_chunk_digest_and_validation_use_one_buffer_after_digest_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("chunk-digest-capture") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        original_bytes = report_path.read_bytes()
        original_report_hash = _read_json(report_path)["report_hash"]
        replacement = _read_json(report_path)
        replacement["execution"]["outcomes"][0]["terminal_status"] = "DOWNLOADED"
        _seal_report(replacement)
        replacement_report_hash = replacement["report_hash"]
        replacement_bytes = (
            json.dumps(replacement, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        assert replacement_report_hash != original_report_hash

        target_path = report_path.resolve()
        original_hash_bytes = c2_terminal_finalizer._sha256_bytes
        swapped = False

        def _swap_target_once() -> None:
            nonlocal swapped
            if not swapped:
                target_path.write_bytes(replacement_bytes)
                swapped = True

        def _digest_then_swap(payload: bytes) -> str:
            digest = original_hash_bytes(payload)
            if payload == original_bytes:
                _swap_target_once()
            return digest

        def _legacy_digest_then_swap(path: Path) -> str:
            digest = sha256_file(path)
            if path.resolve() == target_path:
                _swap_target_once()
            return digest

        monkeypatch.setattr(
            c2_terminal_finalizer,
            "_sha256_bytes",
            _digest_then_swap,
        )
        monkeypatch.setattr(
            c2_terminal_finalizer,
            "sha256_file",
            _legacy_digest_then_swap,
            raising=False,
        )
        final_report = c2_terminal_finalizer._finalize_manifest_for_testing(
            manifest_path
        )

        report_binding = next(
            item for item in final_report["chunks"] if item["chunk_id"] == "013"
        )
        assert swapped is True
        assert report_binding["report_file_sha256"] == hashlib.sha256(
            original_bytes
        ).hexdigest()
        assert report_binding["report_hash"] == original_report_hash
        assert report_path.read_bytes() == replacement_bytes


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


def test_cli_rejects_relative_manifest_output_collision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("manifest-output-collision") as workspace:
        manifest_path, manifest, _ = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        monkeypatch.chdir(workspace)

        assert (
            cli_main(
                [
                    "c2-terminal-finalize",
                    "admission-manifest.json",
                    "--out",
                    "./admission-manifest.json",
                ]
            )
            == 2
        )
        assert "aliases an admitted input path" in capsys.readouterr().err
        assert manifest_path.read_bytes() == manifest_bytes
        assert _read_json(manifest_path)["manifest_hash"] == manifest["manifest_hash"]


def test_cli_rejects_tilde_manifest_output_collision_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("tilde-manifest-output-collision") as workspace:
        manifest_path, _, _ = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        monkeypatch.setenv("HOME", str(workspace))
        monkeypatch.chdir(workspace)

        assert (
            cli_main(
                [
                    "c2-terminal-finalize",
                    str(manifest_path),
                    "--out",
                    "~/admission-manifest.json",
                ]
            )
            == 2
        )
        assert "aliases an admitted input path" in capsys.readouterr().err
        assert manifest_path.read_bytes() == manifest_bytes
        assert not (workspace / "~" / "admission-manifest.json").exists()


def test_library_passes_one_normalized_tilde_output_path_to_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("tilde-normalized-output") as workspace:
        manifest_path, _, _ = _build_admission(workspace)
        expected_output = workspace / "final-report.json"
        writes: list[tuple[Path, bool, bool]] = []
        original_open_target = c2_terminal_finalizer.open_secure_output_target
        monkeypatch.setenv("HOME", str(workspace))
        monkeypatch.chdir(workspace)

        def _record_open_target(
            path: Path,
            *,
            normalized_path: bool = False,
            require_trusted_parent: bool = False,
        ) -> experiment_models.SecureOutputTarget:
            writes.append((path, normalized_path, require_trusted_parent))
            return original_open_target(
                path,
                normalized_path=normalized_path,
                require_trusted_parent=require_trusted_parent,
            )

        monkeypatch.setattr(
            c2_terminal_finalizer,
            "open_secure_output_target",
            _record_open_target,
        )
        report, written_path = finalize_to_path(
            manifest_path,
            Path("~/final-report.json"),
        )

        assert written_path == expected_output
        assert writes == [(expected_output, True, True)]
        assert _read_json(expected_output) == report
        assert not (workspace / "~" / "final-report.json").exists()


def test_library_rejects_symlink_alias_to_chunk_report() -> None:
    with _workspace("chunk-output-collision") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        finalized = prepare_finalization(manifest_path)
        report_path = report_paths["013"]
        report_bytes = report_path.read_bytes()
        output_alias = workspace / "chunk-report-output.json"
        output_alias.symlink_to(report_path)

        with pytest.raises(C2AdmissionError, match="Cannot resolve final report output"):
            write_final_report(finalized, output_alias)

        assert report_path.read_bytes() == report_bytes


def test_library_rejects_hardlink_alias_to_chunk_report() -> None:
    with _workspace("chunk-hardlink-output-collision") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        finalized = prepare_finalization(manifest_path)
        report_path = report_paths["013"]
        report_bytes = report_path.read_bytes()
        output_alias = workspace / "chunk-report-hardlink-output.json"
        os.link(report_path, output_alias)

        with pytest.raises(C2AdmissionError, match="aliases an admitted input path"):
            write_final_report(finalized, output_alias)

        assert report_path.read_bytes() == report_bytes


def test_library_writes_a_noncolliding_output_path() -> None:
    with _workspace("noncolliding-output") as workspace:
        manifest_path, _, _ = _build_admission(workspace)
        output_path = workspace / "final" / "final-report.json"
        output_path.parent.mkdir()

        report, written_path = finalize_to_path(manifest_path, output_path)

        assert written_path == output_path
        assert _read_json(output_path) == report


@pytest.mark.parametrize(
    ("tag", "permissions", "expected"),
    (
        (1, 1 << 2, True),
        (2, 1 << 2, False),
        (1, 1 << 1, False),
        (1, 1 << 30, True),
    ),
)
def test_darwin_native_acl_metadata_classifies_mutating_allow_entries(
    monkeypatch: pytest.MonkeyPatch,
    tag: int,
    permissions: int,
    expected: bool,
) -> None:
    native_acl = _FakeDarwinAclLibrary(tag, permissions)
    monkeypatch.setattr(
        experiment_models.ctypes,
        "CDLL",
        lambda *args, **kwargs: native_acl,
    )

    assert experiment_models._darwin_acl_allows_mutation(73) is expected
    assert native_acl.descriptor == 73
    assert native_acl.acl_type == 0x00000100


def test_darwin_native_acl_metadata_allows_absent_acl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_acl = _FakeDarwinAclLibrary(None)
    monkeypatch.setattr(
        experiment_models.ctypes,
        "CDLL",
        lambda *args, **kwargs: native_acl,
    )

    assert experiment_models._darwin_acl_allows_mutation(73) is False


def test_darwin_native_acl_metadata_fails_closed_when_uninspectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load_native_acl(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic ACL metadata failure")

    monkeypatch.setattr(
        experiment_models.ctypes,
        "CDLL",
        fail_to_load_native_acl,
    )

    with pytest.raises(
        experiment_models.ProvenanceError,
        match="Cannot inspect Darwin ACL metadata",
    ):
        experiment_models._darwin_acl_allows_mutation(73)


def test_linux_posix_acl_metadata_rejects_present_or_uninspectable_acl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def present_acl(_descriptor: int, _name: str) -> bytes:
        return b"synthetic-acl"

    monkeypatch.setattr(
        experiment_models.os,
        "getxattr",
        present_acl,
        raising=False,
    )
    monkeypatch.setattr(
        experiment_models.os,
        "supports_fd",
        frozenset({present_acl}),
    )
    assert experiment_models._linux_acl_allows_mutation(73) is True

    def unreadable_acl(_descriptor: int, _name: str) -> bytes:
        raise OSError(errno.EPERM, "synthetic ACL metadata failure")

    monkeypatch.setattr(experiment_models.os, "getxattr", unreadable_acl)
    monkeypatch.setattr(
        experiment_models.os,
        "supports_fd",
        frozenset({unreadable_acl}),
    )
    with pytest.raises(
        experiment_models.ProvenanceError,
        match="Cannot inspect POSIX ACL metadata",
    ):
        experiment_models._linux_acl_allows_mutation(73)


def test_library_checks_every_trusted_output_ancestor_for_acl_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("acl-clean-trusted-chain") as workspace:
        manifest_path, _, _ = _build_admission(workspace)
        output_parent = workspace / "trusted-output"
        output_parent.mkdir()
        output_path = output_parent / "final-report.json"
        checked_ancestors: list[tuple[int, int]] = []

        def no_mutating_acl(descriptor: int) -> bool:
            metadata = os.fstat(descriptor)
            checked_ancestors.append((metadata.st_dev, metadata.st_ino))
            return False

        monkeypatch.setattr(
            experiment_models,
            "_trusted_acl_allows_foreign_mutation",
            no_mutating_acl,
        )

        _, written_path = finalize_to_path(manifest_path, output_path)

        assert written_path == output_path
        assert len(checked_ancestors) >= len(output_path.parent.parts)
        assert _read_json(output_path)["status"] == "ADMITTED"


def test_library_rejects_mocked_mutating_acl_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("mutating-acl-library") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_path = report_paths["013"]
        chunk_bytes = chunk_path.read_bytes()
        output_parent = workspace / "trusted-output"
        output_parent.mkdir()
        output_path = output_parent / "final-report.json"
        inspected_ancestors: list[int] = []

        def has_mutating_acl(descriptor: int) -> bool:
            inspected_ancestors.append(os.fstat(descriptor).st_ino)
            return len(inspected_ancestors) == 2

        def fail_if_writer_runs(*args: Any, **kwargs: Any) -> None:
            pytest.fail("mutating ACL reached the writer")

        monkeypatch.setattr(
            experiment_models,
            "_trusted_acl_allows_foreign_mutation",
            has_mutating_acl,
        )
        monkeypatch.setattr(
            c2_terminal_finalizer,
            "write_json_atomic_to_target",
            fail_if_writer_runs,
        )

        with pytest.raises(C2AdmissionError, match="Cannot secure final report output"):
            finalize_to_path(manifest_path, output_path)

        assert len(inspected_ancestors) == 2
        assert not output_path.exists()
        assert manifest_path.read_bytes() == manifest_bytes
        assert chunk_path.read_bytes() == chunk_bytes


def test_cli_rejects_mocked_mutating_acl_without_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("mutating-acl-cli") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_path = report_paths["013"]
        chunk_bytes = chunk_path.read_bytes()
        output_parent = workspace / "trusted-output"
        output_parent.mkdir()
        output_path = output_parent / "final-report.json"

        def fail_if_writer_runs(*args: Any, **kwargs: Any) -> None:
            pytest.fail("mutating ACL reached the writer")

        monkeypatch.setattr(
            experiment_models,
            "_trusted_acl_allows_foreign_mutation",
            lambda _descriptor: True,
        )
        monkeypatch.setattr(
            c2_terminal_finalizer,
            "write_json_atomic_to_target",
            fail_if_writer_runs,
        )

        assert (
            cli_main(
                [
                    "c2-terminal-finalize",
                    str(manifest_path),
                    "--out",
                    str(output_path),
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Cannot secure final report output" in captured.err
        assert not output_path.exists()
        assert manifest_path.read_bytes() == manifest_bytes
        assert chunk_path.read_bytes() == chunk_bytes


@pytest.mark.parametrize("mode", (0o775, 0o777))
def test_library_rejects_group_or_world_writable_output_parent(mode: int) -> None:
    with _workspace(f"untrusted-parent-{mode:o}") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_bytes = report_paths["013"].read_bytes()
        output_parent = workspace / "untrusted-output"
        output_parent.mkdir()
        output_parent.chmod(mode)
        output_path = output_parent / "final-report.json"

        with pytest.raises(C2AdmissionError, match="Cannot secure final report output"):
            finalize_to_path(manifest_path, output_path)

        assert not output_path.exists()
        assert manifest_path.read_bytes() == manifest_bytes
        assert report_paths["013"].read_bytes() == chunk_bytes


def test_library_rejects_symlinked_output_parent() -> None:
    with _workspace("symlinked-output-parent") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_bytes = report_paths["013"].read_bytes()
        actual_parent = workspace / "actual-output"
        actual_parent.mkdir()
        output_parent = workspace / "symlinked-output"
        output_parent.symlink_to(actual_parent, target_is_directory=True)
        output_path = output_parent / "final-report.json"

        with pytest.raises(C2AdmissionError, match="Cannot secure final report output"):
            finalize_to_path(manifest_path, output_path)

        assert not (actual_parent / output_path.name).exists()
        assert manifest_path.read_bytes() == manifest_bytes
        assert report_paths["013"].read_bytes() == chunk_bytes


def test_library_rejects_missing_trusted_output_parent() -> None:
    with _workspace("missing-output-parent") as workspace:
        manifest_path, _, _ = _build_admission(workspace)
        output_parent = workspace / "missing-output"

        with pytest.raises(C2AdmissionError, match="Cannot secure final report output"):
            finalize_to_path(manifest_path, output_parent / "final-report.json")

        assert not output_parent.exists()


def test_atomic_writer_ignores_old_predictable_staging_symlink_to_manifest() -> None:
    with _workspace("manifest-staging-symlink") as workspace:
        manifest_path, _, _ = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        output_path = workspace / "final-report.json"
        old_staging_path = output_path.with_name(f".{output_path.name}.tmp")
        old_staging_path.symlink_to(manifest_path)

        write_json_atomic(output_path, {"kind": "safe-output"})

        assert manifest_path.read_bytes() == manifest_bytes
        assert old_staging_path.is_symlink()
        assert _read_json(output_path) == {"kind": "safe-output"}


def test_atomic_writer_ignores_old_predictable_staging_hardlink_to_chunk() -> None:
    with _workspace("chunk-staging-hardlink") as workspace:
        _, _, report_paths = _build_admission(workspace)
        report_path = report_paths["013"]
        report_bytes = report_path.read_bytes()
        output_path = workspace / "final-report.json"
        old_staging_path = output_path.with_name(f".{output_path.name}.tmp")
        os.link(report_path, old_staging_path)

        write_json_atomic(output_path, {"kind": "safe-output"})

        assert report_path.read_bytes() == report_bytes
        assert old_staging_path.samefile(report_path)
        assert _read_json(output_path) == {"kind": "safe-output"}


def test_atomic_writer_rejects_leaf_symlink_without_mutating_target() -> None:
    with _workspace("leaf-output-symlink") as workspace:
        target_path = workspace / "unguarded-target.json"
        target_bytes = b'{"sealed":"target"}\n'
        target_path.write_bytes(target_bytes)
        output_alias = workspace / "unguarded-output.json"
        output_alias.symlink_to(target_path)

        with pytest.raises(
            experiment_models.ProvenanceError,
            match="Leaf output symlinks are forbidden",
        ):
            write_json_atomic(output_alias, {"kind": "unsafe-output"})

        assert target_path.read_bytes() == target_bytes
        assert output_alias.is_symlink()


def test_atomic_writer_fails_closed_without_descriptor_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("descriptor-fallback") as workspace:
        output_path = workspace / "final-report.json"
        monkeypatch.setattr(
            experiment_models,
            "_SECURE_OUTPUT_DIR_FD_SUPPORTED",
            False,
        )

        with pytest.raises(
            experiment_models.ProvenanceError,
            match="descriptor-relative output writes are unsupported",
        ):
            write_json_atomic(output_path, {"kind": "unsafe-fallback"})

        assert not output_path.exists()


def test_atomic_writer_leaves_private_staging_after_rename_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("staging-cleanup") as workspace:
        output_path = workspace / "final-report.json"

        def _fail_rename(
            source: str | Path,
            destination: str | Path,
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
        ) -> None:
            raise OSError("synthetic rename failure")

        monkeypatch.setattr(experiment_models.os, "rename", _fail_rename)
        with pytest.raises(OSError, match="synthetic rename failure"):
            write_json_atomic(output_path, {"kind": "failed-output"})

        staging_paths = list(workspace.glob(".final-report.json.*.tmp"))
        assert len(staging_paths) == 1
        assert staging_paths[0].stat().st_mode & 0o077 == 0
        assert not output_path.exists()


def test_atomic_writer_preserves_reused_staging_name_after_failed_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("failed-staging-name-reuse") as workspace:
        output_path = workspace / "final-report.json"
        reused_names: list[str] = []

        def _fail_rename_after_reuse(
            source: str,
            destination: str,
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
        ) -> None:
            os.unlink(source, dir_fd=src_dir_fd)
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"unrelated")
            reused_names.append(source)
            raise OSError("synthetic rename failure after reuse")

        monkeypatch.setattr(
            experiment_models.os,
            "rename",
            _fail_rename_after_reuse,
        )
        with pytest.raises(OSError, match="synthetic rename failure after reuse"):
            write_json_atomic(output_path, {"kind": "failed-output"})

        assert reused_names
        assert (workspace / reused_names[0]).read_bytes() == b"unrelated"
        assert not output_path.exists()


def test_atomic_writer_preserves_reused_staging_name_after_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("staging-name-reuse") as workspace:
        output_path = workspace / "final-report.json"
        original_rename = experiment_models.os.rename
        reused_names: list[str] = []

        def _rename_then_reuse(
            source: str,
            destination: str,
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
        ) -> None:
            original_rename(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"unrelated")
            reused_names.append(source)

        monkeypatch.setattr(experiment_models.os, "rename", _rename_then_reuse)
        write_json_atomic(output_path, {"kind": "safe-output"})

        assert reused_names
        assert (workspace / reused_names[0]).read_bytes() == b"unrelated"
        assert _read_json(output_path) == {"kind": "safe-output"}


def test_library_rejects_parent_swap_after_safe_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("output-parent-swap") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_path = report_paths["013"]
        chunk_bytes = chunk_path.read_bytes()
        output_parent = workspace / "safe-output-parent"
        output_parent.mkdir()
        output_path = output_parent / manifest_path.name
        parked_parent = workspace / "parked-output-parent"
        original_writer = c2_terminal_finalizer.write_json_atomic_to_target

        def _swap_parent_then_write(
            target: experiment_models.SecureOutputTarget,
            payload: dict[str, Any],
        ) -> None:
            output_parent.rename(parked_parent)
            output_parent.symlink_to(workspace, target_is_directory=True)
            original_writer(target, payload)

        monkeypatch.setattr(
            c2_terminal_finalizer,
            "write_json_atomic_to_target",
            _swap_parent_then_write,
        )

        with pytest.raises(
            C2AdmissionError,
            match="Final report output verification failed",
        ):
            finalize_to_path(manifest_path, output_path)

        assert manifest_path.read_bytes() == manifest_bytes
        assert chunk_path.read_bytes() == chunk_bytes
        assert (parked_parent / manifest_path.name).is_file()
        assert (output_parent / manifest_path.name).samefile(manifest_path)


def test_cli_does_not_report_success_after_chunk_parent_swap(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("output-parent-chunk-swap") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_path = report_paths["013"]
        chunk_bytes = chunk_path.read_bytes()
        output_parent = workspace / "safe-output-parent"
        output_parent.mkdir()
        output_path = output_parent / chunk_path.name
        parked_parent = workspace / "parked-output-parent"
        original_writer = c2_terminal_finalizer.write_json_atomic_to_target

        def _swap_parent_then_write(
            target: experiment_models.SecureOutputTarget,
            payload: dict[str, Any],
        ) -> None:
            output_parent.rename(parked_parent)
            output_parent.symlink_to(chunk_path.parent, target_is_directory=True)
            original_writer(target, payload)

        monkeypatch.setattr(
            c2_terminal_finalizer,
            "write_json_atomic_to_target",
            _swap_parent_then_write,
        )

        assert (
            cli_main(
                [
                    "c2-terminal-finalize",
                    str(manifest_path),
                    "--out",
                    str(output_path),
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Final report output verification failed" in captured.err

        assert manifest_path.read_bytes() == manifest_bytes
        assert chunk_path.read_bytes() == chunk_bytes
        assert (parked_parent / chunk_path.name).is_file()
        assert (output_parent / chunk_path.name).samefile(chunk_path)


def test_cli_rejects_untrusted_parent_swap_without_success_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("untrusted-parent-swap") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_path = report_paths["013"]
        chunk_bytes = chunk_path.read_bytes()
        output_parent = workspace / "untrusted-output"
        output_parent.mkdir()
        output_parent.chmod(0o777)
        parked_parent = workspace / "parked-untrusted-output"
        output_path = output_parent / manifest_path.name

        output_parent.rename(parked_parent)
        output_parent.symlink_to(workspace, target_is_directory=True)

        def fail_if_writer_runs(*args: Any, **kwargs: Any) -> None:
            pytest.fail("untrusted output parent reached the writer")

        monkeypatch.setattr(
            c2_terminal_finalizer,
            "write_json_atomic_to_target",
            fail_if_writer_runs,
        )
        assert (
            cli_main(
                [
                    "c2-terminal-finalize",
                    str(manifest_path),
                    "--out",
                    str(output_path),
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Cannot secure final report output" in captured.err
        assert manifest_path.read_bytes() == manifest_bytes
        assert chunk_path.read_bytes() == chunk_bytes
        assert (output_parent / manifest_path.name).samefile(manifest_path)


def test_library_rejects_leaf_replacement_after_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace("output-leaf-replacement") as workspace:
        manifest_path, _, report_paths = _build_admission(workspace)
        manifest_bytes = manifest_path.read_bytes()
        chunk_path = report_paths["013"]
        chunk_bytes = chunk_path.read_bytes()
        output_path = workspace / "final-report.json"
        original_writer = c2_terminal_finalizer.write_json_atomic_to_target

        def _write_then_replace_leaf(
            target: experiment_models.SecureOutputTarget,
            payload: dict[str, Any],
        ) -> None:
            original_writer(target, payload)
            os.unlink(target.leaf_name, dir_fd=target.parent_fd)
            os.link(
                manifest_path,
                target.leaf_name,
                dst_dir_fd=target.parent_fd,
            )

        monkeypatch.setattr(
            c2_terminal_finalizer,
            "write_json_atomic_to_target",
            _write_then_replace_leaf,
        )
        with pytest.raises(
            C2AdmissionError,
            match="Final report output verification failed",
        ):
            finalize_to_path(manifest_path, output_path)

        assert manifest_path.read_bytes() == manifest_bytes
        assert chunk_path.read_bytes() == chunk_bytes
        assert output_path.samefile(manifest_path)


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

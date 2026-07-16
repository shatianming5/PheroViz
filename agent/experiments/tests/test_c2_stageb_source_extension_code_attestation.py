from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from copy import deepcopy
from importlib import resources
from typing import Any

import pytest

import experiments.c2_stageb_source_extension_code_attestation as code_attestation
import experiments.c2_source_bearing_extension as source_extension
from experiments.c2_full_replacement_policy import C2FullReplacementPolicyError
from experiments.cli import _build_parser
from experiments.models import sha256_json
from tests.test_experiment_support import release_test_git_repository


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seal(entry: dict[str, Any]) -> None:
    entry["registry_id_sha256"] = sha256_json(
        {
            key: value
            for key, value in entry.items()
            if key != "registry_id_sha256"
        }
    )


def _synthetic_entry() -> dict[str, Any]:
    """Return a test-only opaque shape fixture, never an anchor/resource."""

    entry: dict[str, Any] = {
        "schema_version": "c2-stageb-source-extension-code-attestation-v1",
        "registry_entry_type": "C2_STAGEB_SOURCE_EXTENSION_CODE_ATTESTATION",
        "registry_id_sha256": "",
        "extension_implementation_commit_full": "a" * 40,
        "manifest_only_attestation_commit_full": "b" * 40,
        "manifest_sha256": _digest("synthetic-manifest"),
        "canonical_attested_blob_set_sha256": _digest("synthetic-blob-set"),
        "covered_runtime_paths": [
            {"runtime_path": path, "role": role}
            for path, role in code_attestation.SOURCE_EXTENSION_RUNTIME_PATH_ROLES
        ],
    }
    _seal(entry)
    return entry


def _entry_bytes(entry: dict[str, Any]) -> bytes:
    return json.dumps(
        entry,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _compile(
    entry: dict[str, Any],
) -> code_attestation.SourceExtensionCodeAttestationRegistryEntry:
    payload = _entry_bytes(entry)
    return (
        code_attestation.compile_source_extension_code_attestation_registry_entry_for_testing(
            payload,
            expected_resource_sha256=hashlib.sha256(payload).hexdigest(),
        )
    )


def test_runtime_path_contract_is_closed_ordered_and_nonclassification() -> None:
    expected = (
        (
            "agent/experiments/c2_m1_trust_boundary.py",
            "M1_TRUST_BOUNDARY_RUNTIME",
        ),
        (
            "agent/experiments/c2_remediation_root_finalizer.py",
            "REMEDIATION_FINALIZER_RUNTIME",
        ),
        (
            "agent/experiments/c2_source_bearing_extension.py",
            "SOURCE_EXTENSION_RUNTIME",
        ),
        (
            "agent/experiments/c2_stageb_source_extension_code_attestation.py",
            "CODE_ATTESTATION_LOADER_RUNTIME",
        ),
        ("agent/experiments/cli.py", "EXPERIMENTS_CLI_RUNTIME"),
        ("agent/experiments/models.py", "EXPERIMENTS_MODELS_RUNTIME"),
        (
            "agent/experiments/schemas/c2_v2_candidate_set_input_v1.schema.json",
            "SOURCE_EXTENSION_SCHEMA",
        ),
        (
            "agent/experiments/schemas/c2_v2_consumable_source_unit_v1.schema.json",
            "SOURCE_EXTENSION_SCHEMA",
        ),
        (
            "agent/experiments/schemas/"
            "c2_v2_consumption_bijection_validation_v1.schema.json",
            "SOURCE_EXTENSION_SCHEMA",
        ),
        (
            "agent/experiments/schemas/c2_v2_container_accounting_index_v1.schema.json",
            "SOURCE_EXTENSION_SCHEMA",
        ),
        (
            "agent/experiments/schemas/c2_v2_detected_format_v1.schema.json",
            "SOURCE_EXTENSION_SCHEMA",
        ),
        (
            "agent/experiments/schemas/c2_v2_downstream_consumption_v1.schema.json",
            "SOURCE_EXTENSION_SCHEMA",
        ),
        (
            "agent/experiments/schemas/"
            "c2_v2_fd_format_classifier_config_v1.schema.json",
            "SOURCE_EXTENSION_SCHEMA",
        ),
    )
    assert code_attestation.SOURCE_EXTENSION_RUNTIME_PATH_ROLES == expected
    assert frozenset(path for path, _role in expected) == (
        source_extension._REQUIRED_ATTESTED_CODE_PATHS
    )
    assert tuple(path for path, _role in expected) == tuple(
        sorted(path for path, _role in expected)
    )
    for path, role in expected:
        marker = f"{path}:{role}".casefold()
        assert not any(
            token in marker
            for token in (
                "doi",
                "p1",
                "cluster",
                "panel",
                "strata",
                "coverage",
                "admission",
                "outcome",
            )
        )


def test_typed_fixture_compiles_but_production_loader_stays_non_admissive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _synthetic_entry()
    payload = _entry_bytes(entry)
    compiled = _compile(entry)
    assert compiled.extension_implementation_commit_full == "a" * 40
    assert tuple(
        (binding.runtime_path, binding.role)
        for binding in compiled.covered_runtime_paths
    ) == code_attestation.SOURCE_EXTENSION_RUNTIME_PATH_ROLES
    assert compiled.registry_id == entry["registry_id_sha256"]
    assert "registry_entry_id" not in compiled.to_dict()
    assert compiled.to_dict() == entry

    monkeypatch.setattr(
        code_attestation,
        "_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        code_attestation,
        "_read_compile_pinned_resource_bytes",
        lambda: payload,
    )
    monkeypatch.setattr(
        code_attestation,
        "_SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED",
        False,
    )
    with pytest.raises(C2FullReplacementPolicyError, match="intentionally non-admissive"):
        code_attestation.load_compile_pinned_source_extension_code_attestation()


def test_loader_has_no_selector_and_ignores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert list(
        inspect.signature(
            code_attestation.load_compile_pinned_source_extension_code_attestation
        ).parameters
    ) == []
    monkeypatch.setenv("C2_STAGEB_CODE_ANCHOR_PATH", "/attacker/anchor.json")
    monkeypatch.setenv("C2_STAGEB_CODE_ANCHOR_SHA256", "f" * 64)
    monkeypatch.setenv("C2_STAGEB_CODE_ANCHOR_EVIDENCE", "/attacker/evidence")
    loaded = code_attestation.load_compile_pinned_source_extension_code_attestation()
    assert loaded.resource_sha256 == (
        code_attestation._SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256
    )

    parser = _build_parser()
    parsed = parser.parse_args(
        [
            "c2-full-replacement-finalize",
            "unavailable-manifest.json",
            "--out",
            "unavailable-output.json",
        ]
    )
    assert not any(
        token in key
        for key in vars(parsed)
        for token in ("anchor", "attestation", "policy", "resource", "digest")
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "c2-full-replacement-finalize",
                "unavailable-manifest.json",
                "--out",
                "unavailable-output.json",
                "--code-attestation",
                "attacker-anchor.json",
            ]
        )


def test_compile_pinned_loader_rejects_byte_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _entry_bytes(_synthetic_entry())
    monkeypatch.setattr(
        code_attestation,
        "_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256",
        "0" * 64,
    )
    monkeypatch.setattr(
        code_attestation,
        "_read_compile_pinned_resource_bytes",
        lambda: payload,
    )
    with pytest.raises(C2FullReplacementPolicyError, match="compiled SHA-256"):
        code_attestation.load_compile_pinned_source_extension_code_attestation()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head", "HEAD"),
        ("parent_commit_full", "a" * 40),
        ("policy_path", "attacker-policy.json"),
        ("manifest_path", "attacker-manifest.json"),
        ("evidence_path", "attacker-evidence.json"),
        ("environment", "C2_STAGEB_CODE_ANCHOR_PATH"),
        ("cli", "--code-attestation"),
        ("fallback", "other-anchor"),
        ("replacement", "other-anchor"),
        ("code_semantics", {"function": "select"}),
        ("source_claim", "verified source"),
        ("doi_p_cluster_map", []),
        ("outcome", "ADMITTED"),
    ],
)
def test_selector_dynamic_and_semantic_fields_are_rejected(
    field: str,
    value: Any,
) -> None:
    entry = _synthetic_entry()
    entry[field] = value
    _seal(entry)
    with pytest.raises(C2FullReplacementPolicyError, match="schema validation failed"):
        _compile(entry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extension_implementation_commit_full", "HEAD"),
        ("manifest_only_attestation_commit_full", "a" * 39),
        ("manifest_sha256", "g" * 64),
        ("canonical_attested_blob_set_sha256", "A" * 64),
    ],
)
def test_malformed_opaque_commit_and_blob_identities_are_rejected(
    field: str,
    value: str,
) -> None:
    entry = _synthetic_entry()
    entry[field] = value
    _seal(entry)
    with pytest.raises(C2FullReplacementPolicyError, match="schema validation failed"):
        _compile(entry)


def test_manifest_only_attestation_commit_must_differ_from_implementation() -> None:
    entry = _synthetic_entry()
    entry["manifest_only_attestation_commit_full"] = (
        entry["extension_implementation_commit_full"]
    )
    _seal(entry)
    with pytest.raises(C2FullReplacementPolicyError, match="must be distinct"):
        _compile(entry)


@pytest.mark.parametrize(
    "declared_id",
    [
        "outcome-admitted",
        "result-supported",
        "status-blocked-insufficient",
        "admission-report-not-run",
        "cluster-p1-sentinel",
    ],
)
def test_free_form_compound_outcome_and_status_ids_are_rejected(
    declared_id: str,
) -> None:
    entry = _synthetic_entry()
    entry["registry_entry_id"] = declared_id
    _seal(entry)
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="schema validation failed",
    ):
        _compile(entry)


def test_runtime_path_contract_rejects_missing_duplicate_unbound_and_unordered_paths() -> None:
    missing = _synthetic_entry()
    missing["covered_runtime_paths"].pop()
    _seal(missing)
    with pytest.raises(C2FullReplacementPolicyError, match="schema validation failed"):
        _compile(missing)

    duplicate = _synthetic_entry()
    duplicate["covered_runtime_paths"][-1] = dict(duplicate["covered_runtime_paths"][0])
    _seal(duplicate)
    with pytest.raises(C2FullReplacementPolicyError, match="repeats a runtime path"):
        _compile(duplicate)

    unbound = _synthetic_entry()
    unbound["covered_runtime_paths"][-1]["runtime_path"] = (
        "agent/experiments/unbound_runtime.py"
    )
    _seal(unbound)
    with pytest.raises(C2FullReplacementPolicyError, match="unbound runtime path"):
        _compile(unbound)

    extra = _synthetic_entry()
    extra["covered_runtime_paths"].append(
        {
            "runtime_path": "agent/experiments/unbound_runtime.py",
            "role": "M1_TRUST_BOUNDARY_RUNTIME",
        }
    )
    _seal(extra)
    with pytest.raises(C2FullReplacementPolicyError, match="schema validation failed"):
        _compile(extra)

    unordered = _synthetic_entry()
    unordered["covered_runtime_paths"].reverse()
    _seal(unordered)
    with pytest.raises(C2FullReplacementPolicyError, match="not exactly ordered"):
        _compile(unordered)


def test_runtime_path_role_mismatch_is_rejected() -> None:
    entry = deepcopy(_synthetic_entry())
    assert entry["covered_runtime_paths"][0] == {
        "runtime_path": "agent/experiments/c2_m1_trust_boundary.py",
        "role": "M1_TRUST_BOUNDARY_RUNTIME",
    }
    entry["covered_runtime_paths"][0]["role"] = "EXPERIMENTS_CLI_RUNTIME"
    _seal(entry)
    with pytest.raises(C2FullReplacementPolicyError, match="role is inconsistent"):
        _compile(entry)


def test_compile_pinned_resource_and_runtime_manifest_are_active() -> None:
    assert isinstance(
        code_attestation._SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256,
        str,
    )
    assert code_attestation._SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED
    resource = resources.files(
        code_attestation._SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_PACKAGE
    ).joinpath(*code_attestation._SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_PARTS)
    assert resource.is_file()
    entry = code_attestation.load_compile_pinned_source_extension_code_attestation()
    runtime = code_attestation.load_verified_source_extension_runtime_attestation()
    assert entry.registry_id == (
        "a918e021bc70e3f3c1242afc5ea22db9191a0f03349dd05ad87e0813214a9069"
    )
    assert runtime.code_blob_set_sha256 == entry.canonical_attested_blob_set_sha256
    assert len(runtime.code_blobs) == 13


def test_declared_manifest_attestation_commit_contains_only_exact_manifest() -> None:
    repository = release_test_git_repository()
    entry = code_attestation.load_compile_pinned_source_extension_code_attestation()
    relative = (
        "agent/experiments/resources/"
        "c2_source_extension_runtime_manifest_v1.json"
    )
    committed = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "show",
            f"{entry.manifest_only_attestation_commit_full}:{relative}",
        ]
    )
    bundled = (repository / relative).read_bytes()

    assert committed == bundled
    assert hashlib.sha256(committed).hexdigest() == entry.manifest_sha256
    changed_paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            entry.manifest_only_attestation_commit_full,
        ],
        text=True,
    ).splitlines()
    assert changed_paths == [relative]
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            entry.extension_implementation_commit_full,
            entry.manifest_only_attestation_commit_full,
        ],
        check=True,
    )
    manifest = json.loads(committed)
    for blob in manifest["attested_paths"]:
        relative_path = blob["relative_path"]
        observed_object_id = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "rev-parse",
                f"{entry.extension_implementation_commit_full}:{relative_path}",
            ],
            text=True,
        ).strip()
        observed_payload = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "show",
                f"{entry.extension_implementation_commit_full}:{relative_path}",
            ]
        )
        assert observed_object_id == blob["git_blob_object_id"]
        assert hashlib.sha256(observed_payload).hexdigest() == blob["sha256"]

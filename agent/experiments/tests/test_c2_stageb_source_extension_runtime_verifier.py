from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from typing import Any

import pytest

import experiments.c2_stageb_source_extension_code_attestation as code_attestation
import experiments.c2_stageb_source_extension_runtime_verifier as runtime_verifier
from experiments.c2_full_replacement_policy import C2FullReplacementPolicyError
from experiments.cli import _build_parser
from experiments.models import sha256_json


def _seal(entry: dict[str, Any]) -> None:
    entry["registry_id_sha256"] = sha256_json(
        {
            key: value
            for key, value in entry.items()
            if key != "registry_id_sha256"
        }
    )


def _entry_bytes(entry: dict[str, Any]) -> bytes:
    return json.dumps(
        entry,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _runtime_files() -> tuple[runtime_verifier.SourceExtensionRuntimeFixtureFile, ...]:
    python_bytes_by_path = {
        "agent/experiments/c2_remediation_root_finalizer.py": (
            b"from . import c2_source_bearing_extension\n"
        ),
        "agent/experiments/c2_source_bearing_extension.py": b"from . import models\n",
        "agent/experiments/cli.py": b"from . import models\n",
        "agent/experiments/models.py": b"import json\n",
    }
    files: list[runtime_verifier.SourceExtensionRuntimeFixtureFile] = []
    for path, role in code_attestation.SOURCE_EXTENSION_RUNTIME_PATH_ROLES:
        if path.endswith(".py"):
            runtime_bytes = python_bytes_by_path[path]
        else:
            runtime_bytes = json.dumps(
                {"fixture_schema_path": path},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        files.append(
            runtime_verifier.SourceExtensionRuntimeFixtureFile(
                runtime_path=path,
                role=role,
                runtime_bytes=runtime_bytes,
            )
        )
    return tuple(files)


def _replace_runtime_bytes(
    files: tuple[runtime_verifier.SourceExtensionRuntimeFixtureFile, ...],
    runtime_path: str,
    runtime_bytes: bytes,
) -> tuple[runtime_verifier.SourceExtensionRuntimeFixtureFile, ...]:
    replaced = False
    updated: list[runtime_verifier.SourceExtensionRuntimeFixtureFile] = []
    for runtime_file in files:
        if runtime_file.runtime_path == runtime_path:
            updated.append(replace(runtime_file, runtime_bytes=runtime_bytes))
            replaced = True
        else:
            updated.append(runtime_file)
    assert replaced
    return tuple(updated)


def _fixture(
    *,
    implementation_commit_full: str = "a" * 40,
    manifest_only_attestation_commit_full: str = "b" * 40,
    manifest_bytes: bytes = b"synthetic test-only manifest bytes",
    runtime_files: tuple[runtime_verifier.SourceExtensionRuntimeFixtureFile, ...]
    | None = None,
) -> runtime_verifier.SourceExtensionRuntimeTestFixture:
    return runtime_verifier.SourceExtensionRuntimeTestFixture(
        implementation_commit_full=implementation_commit_full,
        manifest_only_attestation_commit_full=manifest_only_attestation_commit_full,
        manifest_bytes=manifest_bytes,
        runtime_files=_runtime_files() if runtime_files is None else runtime_files,
    )


def _compile_registry(
    fixture: runtime_verifier.SourceExtensionRuntimeTestFixture,
) -> code_attestation.SourceExtensionCodeAttestationRegistryEntry:
    entry: dict[str, Any] = {
        "schema_version": "c2-stageb-source-extension-code-attestation-v1",
        "registry_entry_type": "C2_STAGEB_SOURCE_EXTENSION_CODE_ATTESTATION",
        "registry_id_sha256": "",
        "extension_implementation_commit_full": fixture.implementation_commit_full,
        "manifest_only_attestation_commit_full": (
            fixture.manifest_only_attestation_commit_full
        ),
        "manifest_sha256": hashlib.sha256(fixture.manifest_bytes).hexdigest(),
        "canonical_attested_blob_set_sha256": (
            runtime_verifier.canonical_attested_blob_set_sha256_for_testing(
                fixture.runtime_files
            )
        ),
        "covered_runtime_paths": [
            {"runtime_path": path, "role": role}
            for path, role in code_attestation.SOURCE_EXTENSION_RUNTIME_PATH_ROLES
        ],
    }
    _seal(entry)
    payload = _entry_bytes(entry)
    compiler = getattr(
        code_attestation,
        "compile_source_extension_code_attestation_registry_entry_for_testing",
    )
    return compiler(
        payload,
        expected_resource_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _compile_runtime_closure(
    runtime_files: tuple[runtime_verifier.SourceExtensionRuntimeFixtureFile, ...],
) -> runtime_verifier.SourceExtensionRuntimeByteImportClosure:
    compiler = getattr(
        runtime_verifier,
        "compile_source_extension_runtime_byte_import_closure_for_testing",
    )
    return compiler(runtime_files)


def _lock(
    registry: code_attestation.SourceExtensionCodeAttestationRegistryEntry,
    fixture: runtime_verifier.SourceExtensionRuntimeTestFixture,
) -> runtime_verifier.DeploymentPinnedSourceExtensionRuntimeLock:
    closure = _compile_runtime_closure(fixture.runtime_files)
    return runtime_verifier.DeploymentPinnedSourceExtensionRuntimeLock(
        expected_code_attestation_registry_id_sha256=registry.registry_id_sha256,
        expected_extension_implementation_commit_full=(
            registry.extension_implementation_commit_full
        ),
        expected_manifest_only_attestation_commit_full=(
            registry.manifest_only_attestation_commit_full
        ),
        expected_manifest_sha256=registry.manifest_sha256,
        expected_canonical_attested_blob_set_sha256=(
            registry.canonical_attested_blob_set_sha256
        ),
        expected_runtime_path_roles=registry.covered_runtime_paths,
        expected_runtime_byte_import_closure_sha256=closure.canonical_sha256,
    )


def _verified_inputs() -> tuple[
    runtime_verifier.DeploymentPinnedSourceExtensionRuntimeLock,
    code_attestation.SourceExtensionCodeAttestationRegistryEntry,
    runtime_verifier.SourceExtensionRuntimeTestFixture,
]:
    fixture = _fixture()
    registry = _compile_registry(fixture)
    return _lock(registry, fixture), registry, fixture


def test_test_only_typed_interface_binds_registry_manifest_blobset_and_closure(
) -> None:
    lock, registry, fixture = _verified_inputs()

    binding = runtime_verifier.verify_source_extension_runtime_for_testing(
        deployment_lock=lock,
        registry_entry=registry,
        fixture=fixture,
    )

    assert binding.code_attestation_registry_id_sha256 == registry.registry_id_sha256
    assert binding.extension_implementation_commit_full == "a" * 40
    assert binding.manifest_only_attestation_commit_full == "b" * 40
    assert binding.manifest_sha256 == hashlib.sha256(fixture.manifest_bytes).hexdigest()
    assert (
        binding.canonical_attested_blob_set_sha256
        == registry.canonical_attested_blob_set_sha256
    )
    assert tuple(
        (entry.runtime_path, entry.role) for entry in binding.covered_runtime_paths
    ) == code_attestation.SOURCE_EXTENSION_RUNTIME_PATH_ROLES
    closure = _compile_runtime_closure(fixture.runtime_files)
    local_dependencies = {
        item.runtime_path: item.resolved_project_local_dependencies
        for item in closure.bindings
    }
    assert local_dependencies[
        "agent/experiments/c2_remediation_root_finalizer.py"
    ] == (
        "agent/experiments/c2_source_bearing_extension.py",
        "agent/experiments/models.py",
    )
    assert local_dependencies["agent/experiments/c2_source_bearing_extension.py"] == (
        "agent/experiments/models.py",
    )
    assert "admitted" not in binding.to_dict()
    assert "evidence" not in binding.to_dict()


def test_unrostered_relative_alias_import_is_rejected_from_runtime_closure() -> None:
    fixture = _fixture()
    unrostered_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"from . import aggregate as aggregate_module\n",
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match="unrostered project-local dependency",
    ):
        _compile_runtime_closure(unrostered_files)


def test_nested_local_import_cycle_is_rejected_from_runtime_closure() -> None:
    fixture = _fixture()
    cyclic_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/c2_source_bearing_extension.py",
        b"from . import c2_remediation_root_finalizer\n",
    )

    with pytest.raises(C2FullReplacementPolicyError, match="local dependency cycle"):
        _compile_runtime_closure(cyclic_files)


def test_duplicate_local_import_and_relative_traversal_are_rejected() -> None:
    fixture = _fixture()
    duplicate_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"from . import models\nfrom .models import Model\n",
    )
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="duplicate local dependency",
    ):
        _compile_runtime_closure(duplicate_files)

    traversal_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"from .. import models\n",
    )
    with pytest.raises(C2FullReplacementPolicyError, match="traversal"):
        _compile_runtime_closure(traversal_files)


def test_dynamic_import_is_rejected_from_runtime_closure() -> None:
    fixture = _fixture()
    dynamic_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"import importlib\n"
        b"importlib.import_module('.aggregate', package=__package__)\n",
    )

    with pytest.raises(C2FullReplacementPolicyError, match="dynamic imports"):
        _compile_runtime_closure(dynamic_files)


@pytest.mark.parametrize(
    "source",
    [
        (
            b"from importlib import import_module\n"
            b"import_module('.aggregate', package=__package__)\n"
        ),
        (
            b"from importlib import import_module as imp\n"
            b"imp('.aggregate', package=__package__)\n"
        ),
        b"from builtins import __import__\n__import__('.aggregate')\n",
        (
            b"from builtins import __import__ as imp\n"
            b"imp('.aggregate')\n"
        ),
        (
            b"import importlib as importer\n"
            b"importer.import_module('.aggregate', package=__package__)\n"
        ),
        b"import builtins as builtin_module\nbuiltin_module.__import__('.aggregate')\n",
        (
            b"import importlib as importer\n"
            b"imp = importer.import_module\n"
            b"imp('.aggregate', package=__package__)\n"
        ),
        (
            b"import builtins as builtin_module\n"
            b"imp = builtin_module.__import__\n"
            b"imp('.aggregate')\n"
        ),
        (
            b"import importlib as importer\n"
            b"nested = importer\n"
            b"nested_again = nested\n"
            b"nested_again.import_module('.aggregate', package=__package__)\n"
        ),
        (
            b"import builtins as builtin_module\n"
            b"nested = builtin_module\n"
            b"nested_again = nested\n"
            b"nested_again.__import__('.aggregate')\n"
        ),
        (
            b"from importlib import import_module as imp\n"
            b"nested = imp\n"
            b"nested_again = nested\n"
            b"nested_again('.aggregate', package=__package__)\n"
        ),
        (
            b"from builtins import __import__ as imp\n"
            b"nested = imp\n"
            b"nested_again = nested\n"
            b"nested_again('.aggregate')\n"
        ),
        (
            b"import importlib as importer\n"
            b"imp = getattr(importer, 'import_module')\n"
            b"imp('.aggregate', package=__package__)\n"
        ),
        (
            b"import builtins as builtin_module\n"
            b"imp = getattr(builtin_module, '__import__')\n"
            b"imp('.aggregate')\n"
        ),
        (
            b"import importlib as importer\n"
            b"holder.imp = importer.import_module\n"
            b"holder.imp('.aggregate', package=__package__)\n"
        ),
        (
            b"import builtins as builtin_module\n"
            b"holder.imp = builtin_module.__import__\n"
            b"holder.imp('.aggregate')\n"
        ),
        (
            b"import importlib as importer\n"
            b"setattr(holder, 'imp', importer.import_module)\n"
            b"holder.imp('.aggregate', package=__package__)\n"
        ),
        (
            b"import builtins as builtin_module\n"
            b"class Holder:\n"
            b"    imp = builtin_module.__import__\n"
            b"Holder.imp('.aggregate')\n"
        ),
    ],
)
def test_dynamic_import_aliases_are_rejected_from_runtime_closure(
    source: bytes,
) -> None:
    fixture = _fixture()
    alias_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    with pytest.raises(C2FullReplacementPolicyError, match="dynamic imports"):
        _compile_runtime_closure(alias_files)


def test_unproven_alias_call_target_is_rejected_from_runtime_closure() -> None:
    fixture = _fixture()
    unknown_alias_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"import importlib as importer\n"
        b"imp = getattr(importer, method_name)\n"
        b"imp('.aggregate', package=__package__)\n",
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match="cannot be proven non-dynamic",
    ):
        _compile_runtime_closure(unknown_alias_files)


@pytest.mark.parametrize(
    "source",
    [
        (
            b"import importlib as importer\n"
            b"imp, unused = importer.import_module, None\n"
            b"imp('.aggregate', package=__package__)\n"
        ),
        (
            b"import importlib as importer\n"
            b"def invoke(imp):\n"
            b"    imp('.aggregate', package=__package__)\n"
            b"invoke(importer.import_module)\n"
        ),
        (
            b"import importlib as importer\n"
            b"for imp in (importer.import_module,):\n"
            b"    imp('.aggregate', package=__package__)\n"
        ),
    ],
)
def test_indirect_dynamic_alias_call_targets_fail_closed(
    source: bytes,
) -> None:
    fixture = _fixture()
    indirect_alias_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match="cannot be proven non-dynamic",
    ):
        _compile_runtime_closure(indirect_alias_files)


def test_absent_production_resource_fails_before_registry_or_fixture_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        code_attestation._SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256
        is None
    )

    def unexpected_fixture_access(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("production verifier accessed a candidate fixture")

    monkeypatch.setattr(
        runtime_verifier,
        "_verify_source_extension_runtime_fixture",
        unexpected_fixture_access,
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match="no reviewed compile-pinned internal source-extension",
    ):
        runtime_verifier.verify_compile_pinned_source_extension_runtime()


def test_public_production_verifier_has_no_selector_surface_and_ignores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert list(
        inspect.signature(
            runtime_verifier.verify_compile_pinned_source_extension_runtime
        ).parameters
    ) == []
    for name, value in {
        "C2_STAGEB_RUNTIME_LOCK": "/attacker/lock.json",
        "C2_STAGEB_RUNTIME_WORKTREE": "/attacker/worktree",
        "C2_STAGEB_RUNTIME_MANIFEST": "/attacker/manifest.json",
        "C2_STAGEB_RUNTIME_COMMIT": "HEAD",
        "C2_STAGEB_RUNTIME_DIGEST": "f" * 64,
        "C2_STAGEB_RUNTIME_EVIDENCE": "/attacker/evidence",
    }.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(C2FullReplacementPolicyError, match="Stage-A only"):
        runtime_verifier.verify_compile_pinned_source_extension_runtime()

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
        for token in (
            "runtime",
            "attestation",
            "policy",
            "resource",
            "digest",
            "evidence",
            "worktree",
            "commit",
        )
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "c2-full-replacement-finalize",
                "unavailable-manifest.json",
                "--out",
                "unavailable-output.json",
                "--runtime-lock",
                "attacker-selected-lock.json",
            ]
        )


@pytest.mark.parametrize(
    ("fixture_field", "value"),
    [
        ("implementation_commit_full", "HEAD"),
        ("manifest_only_attestation_commit_full", "parent"),
    ],
)
def test_dynamic_candidate_head_or_parent_is_rejected_even_if_manifest_claims_match(
    fixture_field: str,
    value: str,
) -> None:
    lock, registry, fixture = _verified_inputs()
    self_attesting_manifest = json.dumps(
        {
            "extension_implementation_commit_full": (
                lock.expected_extension_implementation_commit_full
            ),
            "manifest_only_attestation_commit_full": (
                lock.expected_manifest_only_attestation_commit_full
            ),
        },
        sort_keys=True,
    ).encode("utf-8")
    dynamic_fixture = replace(
        fixture,
        manifest_bytes=self_attesting_manifest,
        **{fixture_field: value},
    )

    with pytest.raises(C2FullReplacementPolicyError, match="full Git commit"):
        runtime_verifier.verify_source_extension_runtime_for_testing(
            deployment_lock=lock,
            registry_entry=registry,
            fixture=dynamic_fixture,
        )


def test_self_attesting_manifest_bytes_cannot_override_a_pinned_digest() -> None:
    lock, registry, fixture = _verified_inputs()
    self_attesting_fixture = replace(
        fixture,
        manifest_bytes=json.dumps(
            {
                "extension_implementation_commit_full": (
                    fixture.implementation_commit_full
                ),
                "manifest_only_attestation_commit_full": (
                    fixture.manifest_only_attestation_commit_full
                ),
                "manifest_sha256": lock.expected_manifest_sha256,
                "canonical_attested_blob_set_sha256": (
                    registry.canonical_attested_blob_set_sha256
                ),
            },
            sort_keys=True,
        ).encode("utf-8"),
    )

    with pytest.raises(C2FullReplacementPolicyError, match="candidate manifest bytes"):
        runtime_verifier.verify_source_extension_runtime_for_testing(
            deployment_lock=lock,
            registry_entry=registry,
            fixture=self_attesting_fixture,
        )


@pytest.mark.parametrize(
    ("lock_field", "value"),
    [
        ("expected_extension_implementation_commit_full", "HEAD"),
        ("expected_manifest_only_attestation_commit_full", "parent"),
    ],
)
def test_dynamic_deployment_expectations_are_rejected(
    lock_field: str,
    value: str,
) -> None:
    lock, registry, fixture = _verified_inputs()
    dynamic_lock = replace(lock, **{lock_field: value})

    with pytest.raises(C2FullReplacementPolicyError, match="full Git commit"):
        runtime_verifier.verify_source_extension_runtime_for_testing(
            deployment_lock=dynamic_lock,
            registry_entry=registry,
            fixture=fixture,
        )


def test_malformed_path_role_rosters_fail_for_lock_registry_and_fixture() -> None:
    lock, registry, fixture = _verified_inputs()

    malformed_lock = replace(
        lock,
        expected_runtime_path_roles=tuple(reversed(lock.expected_runtime_path_roles)),
    )
    with pytest.raises(C2FullReplacementPolicyError, match="fixed ordered"):
        runtime_verifier.verify_source_extension_runtime_for_testing(
            deployment_lock=malformed_lock,
            registry_entry=registry,
            fixture=fixture,
        )

    malformed_registry = replace(
        registry,
        covered_runtime_paths=tuple(reversed(registry.covered_runtime_paths)),
    )
    with pytest.raises(C2FullReplacementPolicyError, match="fixed ordered"):
        runtime_verifier.verify_source_extension_runtime_for_testing(
            deployment_lock=lock,
            registry_entry=malformed_registry,
            fixture=fixture,
        )

    malformed_file = replace(
        fixture.runtime_files[0],
        role="EXPERIMENTS_CLI_RUNTIME",
    )
    malformed_fixture = replace(
        fixture,
        runtime_files=(malformed_file, *fixture.runtime_files[1:]),
    )
    with pytest.raises(C2FullReplacementPolicyError, match="fixed ordered"):
        runtime_verifier.verify_source_extension_runtime_for_testing(
            deployment_lock=lock,
            registry_entry=registry,
            fixture=malformed_fixture,
        )


def test_runtime_byte_import_closure_is_a_separate_required_binding() -> None:
    lock, registry, fixture = _verified_inputs()
    changed_file = replace(
        fixture.runtime_files[0],
        runtime_bytes=b"import collections\nfrom . import models\n",
    )
    changed_fixture = replace(
        fixture,
        runtime_files=(changed_file, *fixture.runtime_files[1:]),
    )
    changed_registry = _compile_registry(changed_fixture)
    stale_closure_lock = replace(
        _lock(changed_registry, changed_fixture),
        expected_runtime_byte_import_closure_sha256=(
            lock.expected_runtime_byte_import_closure_sha256
        ),
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match="runtime byte/import closure",
    ):
        runtime_verifier.verify_source_extension_runtime_for_testing(
            deployment_lock=stale_closure_lock,
            registry_entry=changed_registry,
            fixture=changed_fixture,
        )
    assert (
        registry.canonical_attested_blob_set_sha256
        != changed_registry.canonical_attested_blob_set_sha256
    )


def test_test_only_fixture_cannot_be_a_production_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, registry, fixture = _verified_inputs()
    assert list(
        inspect.signature(
            runtime_verifier.verify_compile_pinned_source_extension_runtime
        ).parameters
    ) == []
    production_verifier: Any = (
        runtime_verifier.verify_compile_pinned_source_extension_runtime
    )
    with pytest.raises(TypeError):
        production_verifier(fixture)
    with pytest.raises(TypeError):
        production_verifier(fixture=fixture)

    def unexpected_fixture_access(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("production verifier accepted a test-only fixture")

    monkeypatch.setattr(
        runtime_verifier,
        "load_compile_pinned_source_extension_code_attestation",
        lambda: registry,
    )
    monkeypatch.setattr(
        runtime_verifier,
        "_verify_source_extension_runtime_fixture",
        unexpected_fixture_access,
    )
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="intentionally non-admissive",
    ):
        runtime_verifier.verify_compile_pinned_source_extension_runtime()

    assert lock.expected_manifest_sha256 == hashlib.sha256(
        fixture.manifest_bytes
    ).hexdigest()


def test_required_test_matrix_is_explicit_and_closed() -> None:
    assert runtime_verifier.SOURCE_EXTENSION_RUNTIME_VERIFIER_TEST_MATRIX == (
        "absence-fails-before-candidate-access",
        "public-selector-injection-is-not-an-input",
        "dynamic-head-or-parent-identity-is-rejected",
        "malformed-fixed-path-role-roster-is-rejected",
        "unrostered-or-unresolved-local-import-is-rejected",
        "duplicate-or-cyclic-local-import-graph-is-rejected",
        "dynamic-import-aliases-and-unknown-targets-are-rejected",
        "test-only-fixture-is-not-a-production-input",
    )

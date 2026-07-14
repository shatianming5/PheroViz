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

    with pytest.raises(
        C2FullReplacementPolicyError,
        match="dynamic imports|reflective|namespace|declarative AST",
    ):
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
        match="reflective or executable|declarative AST",
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
        match="dynamic imports|unbound|namespace|implicit runtime|declarative AST",
    ):
        _compile_runtime_closure(indirect_alias_files)


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (
            (
                b"from importlib import import_module as imp\n"
                b"imp.__call__('.aggregate', package=__package__)\n"
            ),
            "dynamic imports",
        ),
        (
            (
                b"from builtins import __import__ as imp\n"
                b"nested = imp\n"
                b"nested.__call__('.aggregate')\n"
            ),
            "dynamic imports",
        ),
        (
            (
                b"import importlib as importer\n"
                b"globals()['imp'] = importer.import_module\n"
                b"imp('.aggregate', package=__package__)\n"
            ),
            "namespace or mapping mutation",
        ),
        (
            (
                b"namespace['imp'] = callable_value\n"
                b"namespace['imp']('.aggregate')\n"
            ),
            "namespace or mapping mutation",
        ),
        (
            (
                b"holder.__dict__['imp'] = callable_value\n"
                b"holder.imp('.aggregate')\n"
            ),
            "reflective namespace access|namespace or mapping mutation",
        ),
        (
            b"__builtins__['__import__']('.aggregate')\n",
            "reflective namespace access",
        ),
        (b"getattr(target, 'callable')()\n", "reflective or executable"),
        (b"setattr(target, 'callable', value)\n", "reflective or executable"),
        (b"delattr(target, 'callable')\n", "reflective or executable"),
        (b"globals()\n", "reflective or executable"),
        (b"locals()\n", "reflective or executable"),
        (b"vars()\n", "reflective or executable"),
        (b"exec('pass')\n", "reflective or executable"),
        (b"eval('1')\n", "reflective or executable"),
        (b"compile('1', '<fixture>', 'eval')\n", "reflective or executable"),
        (b"unbound_callable()\n", "unbound or not a statically allowed"),
        (
            b"unknown_callable.__call__()\n",
            "unbound or not a statically allowed",
        ),
    ],
)
def test_deny_by_default_rejects_dynamic_reflection_and_unknown_calls(
    source: bytes,
    match: str,
) -> None:
    fixture = _fixture()
    prohibited_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match=f"{match}|declarative AST",
    ):
        _compile_runtime_closure(prohibited_files)


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (
            (
                b"import pydoc\n"
                b"class Meta(type):\n"
                b"    def __fspath__(cls):\n"
                b"        return 'agent/experiments/aggregate.py'\n"
                b"@pydoc.importfile\n"
                b"class C(metaclass=Meta):\n"
                b"    pass\n"
            ),
            "decorators|class base or metaclass",
        ),
        (
            b"import pydoc\n@pydoc.importfile\nclass C:\n    pass\n",
            "decorators",
        ),
        (
            b"import pydoc\n@pydoc.importfile\ndef decorated():\n    pass\n",
            "decorators",
        ),
        (
            (
                b"import pydoc\n"
                b"decorator = pydoc.importfile\n"
                b"@decorator\n"
                b"def decorated():\n"
                b"    pass\n"
            ),
            "decorators",
        ),
        (
            b"import json\n@json.dumps\ndef decorated():\n    pass\n",
            "decorators",
        ),
        (b"class C(metaclass=type):\n    pass\n", "class base or metaclass"),
        (b"def f(value=runtime_value):\n    pass\n", "candidate definition"),
        (b"def f(value: str):\n    pass\n", "candidate definition"),
        (b"callback = lambda: None\n", "implicit runtime"),
        (b"values = [value for value in source]\n", "implicit runtime"),
        (b"with context:\n    pass\n", "implicit runtime"),
        (b"async def f():\n    pass\n", "async runtime semantics"),
        (b"value[0]\n", "implicit runtime"),
        (b"result = left + right\n", "implicit runtime"),
        (b"message = f'{value}'\n", "implicit runtime"),
        (b"print(*values)\n", "implicit runtime"),
        (b"print(**mapping)\n", "implicit runtime"),
        (b"values = {value}\n", "implicit runtime"),
        (b"values = {'key': value}\n", "implicit runtime"),
        (b"left, right = values\n", "implicit runtime"),
    ],
)
def test_implicit_runtime_evaluation_routes_fail_closed(
    source: bytes,
    match: str,
) -> None:
    fixture = _fixture()
    implicit_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match=f"{match}|declarative AST",
    ):
        _compile_runtime_closure(implicit_files)


@pytest.mark.parametrize(
    "source",
    [
        (
            b"from pydoc import importfile\n"
            b"print(next(map(importfile, ['agent/experiments/aggregate.py'])))\n"
        ),
        (
            b"from pydoc import importfile as loader\n"
            b"print(next(filter(loader, ['agent/experiments/aggregate.py'])))\n"
        ),
        (
            b"from pydoc import importfile\n"
            b"print(sorted(['agent/experiments/aggregate.py'], key=importfile))\n"
        ),
        (
            b"from pydoc import importfile\n"
            b"loader = importfile\n"
            b"print(next(map(loader, ['agent/experiments/aggregate.py'])))\n"
        ),
        (
            b"from pydoc import importfile\n"
            b"print(min(['agent/experiments/aggregate.py'], key=importfile))\n"
        ),
    ],
)
def test_higher_order_callback_loading_paths_fail_closed(source: bytes) -> None:
    fixture = _fixture()
    callback_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match="callback|nonstatic|reflective|declarative AST",
    ):
        _compile_runtime_closure(callback_files)


def test_static_callable_accepts_only_harmless_static_arguments() -> None:
    fixture = _fixture()
    safe_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"print('static-literal')\n",
    )

    closure = _compile_runtime_closure(safe_files)
    assert any(
        item.runtime_path == "agent/experiments/cli.py"
        for item in closure.bindings
    )


@pytest.mark.parametrize(
    "class_body",
    [
        b"__init__ = importfile\n",
        b"__new__ = importfile\n",
        b"__call__ = importfile\n",
        b"__iter__ = importfile\n",
        b"__fspath__ = importfile\n",
        b"value = property(importfile)\n",
        b"__get__ = importfile\n",
    ],
)
def test_runtime_classes_and_magic_protocol_bindings_fail_closed(
    class_body: bytes,
) -> None:
    fixture = _fixture()
    class_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"from pydoc import importfile\n"
        b"class D:\n"
        + b"    " + class_body
        + b"D()\n",
    )

    with pytest.raises(C2FullReplacementPolicyError, match="declarative AST"):
        _compile_runtime_closure(class_files)


@pytest.mark.parametrize(
    "source",
    [
        (
            b"from pydoc import importfile\n"
            b"def loader():\n"
            b"    return importfile\n"
            b"loader()\n"
        ),
        (
            b"def outer():\n"
            b"    def inner():\n"
            b"        pass\n"
            b"    return None\n"
            b"outer()\n"
        ),
        (
            b"from json import dumps\n"
            b"callback = dumps\n"
            b"callback('static')\n"
        ),
        (
            b"from json import dumps\n"
            b"callbacks = [dumps]\n"
            b"print('static')\n"
        ),
    ],
)
def test_callable_capture_and_nested_function_forms_fail_closed(
    source: bytes,
) -> None:
    fixture = _fixture()
    capture_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    with pytest.raises(C2FullReplacementPolicyError, match="declarative AST"):
        _compile_runtime_closure(capture_files)


def test_fully_validated_module_level_function_can_be_called_directly() -> None:
    fixture = _fixture()
    function_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        b"def approved():\n"
        b"    print('static')\n"
        b"    return None\n"
        b"approved()\n",
    )

    closure = _compile_runtime_closure(function_files)
    assert any(
        item.runtime_path == "agent/experiments/cli.py"
        for item in closure.bindings
    )


@pytest.mark.parametrize(
    ("source", "expected_imports"),
    [
        (
            b"from . import models\nmodels.canonical_json({})\n",
            (".models",),
        ),
        (
            b"from .models import canonical_json\ncanonical_json({})\n",
            (".models",),
        ),
    ],
)
def test_explicit_static_local_module_call_is_allowed(
    source: bytes,
    expected_imports: tuple[str, ...],
) -> None:
    fixture = _fixture()
    static_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    closure = _compile_runtime_closure(static_files)
    cli_binding = next(
        item
        for item in closure.bindings
        if item.runtime_path == "agent/experiments/cli.py"
    )
    assert cli_binding.direct_imports == expected_imports


@pytest.mark.parametrize(
    ("source", "expected_imports"),
    [
        (b"import json\njson.dumps({})\n", ("json",)),
        (b"import json as payload\npayload.dumps({})\n", ("json",)),
        (b"from json import dumps\ndumps({})\n", ("json",)),
    ],
)
def test_closed_module_attribute_allowlist_permits_reviewed_calls(
    source: bytes,
    expected_imports: tuple[str, ...],
) -> None:
    fixture = _fixture()
    allowed_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    closure = _compile_runtime_closure(allowed_files)
    cli_binding = next(
        item
        for item in closure.bindings
        if item.runtime_path == "agent/experiments/cli.py"
    )
    assert cli_binding.direct_imports == expected_imports
    assert ("json", "dumps") in (
        runtime_verifier.SOURCE_EXTENSION_RUNTIME_MODULE_ATTRIBUTE_CALL_ALLOWLIST
    )


def test_every_reviewed_runtime_module_call_is_explicitly_allowlisted() -> None:
    allowed_calls = (
        runtime_verifier.SOURCE_EXTENSION_RUNTIME_MODULE_ATTRIBUTE_CALL_ALLOWLIST
    )
    source = "\n".join(
        line
        for index, (module_name, attribute_name) in enumerate(
            sorted(allowed_calls)
        )
        for line in (
            f"import {module_name} as module_{index}",
            f"module_{index}.{attribute_name}()",
        )
    ).encode("utf-8")
    fixture = _fixture()
    reviewed_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    closure = _compile_runtime_closure(reviewed_files)
    cli_binding = next(
        item
        for item in closure.bindings
        if item.runtime_path == "agent/experiments/cli.py"
    )
    assert len(cli_binding.direct_imports) == len(
        {
            module_name
            for module_name, _ in allowed_calls
        }
    )


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (
            b"import runpy\nrunpy.run_module('unrostered')\n",
            "reflective or executable",
        ),
        (
            b"import runpy\nrunpy.run_path('unrostered.py')\n",
            "reflective or executable",
        ),
        (
            (
                b"import importlib.machinery as machinery\n"
                b"machinery.SourceFileLoader('x', 'unrostered.py')\n"
            ),
            "dynamic imports|reflective or executable",
        ),
        (
            (
                b"from importlib.machinery import SourceFileLoader as loader\n"
                b"loader('x', 'unrostered.py')\n"
            ),
            "reflective or executable",
        ),
        (
            (
                b"import runpy\n"
                b"loader = runpy.run_module\n"
                b"loader('unrostered')\n"
            ),
            "reflective or executable",
        ),
        (
            (
                b"from runpy import run_path as loader\n"
                b"loader('unrostered.py')\n"
            ),
            "reflective or executable",
        ),
        (b"import json\njson.JSONEncoder()\n", "unbound or not a statically allowed"),
    ],
)
def test_unlisted_module_loader_and_attribute_calls_fail_closed(
    source: bytes,
    match: str,
) -> None:
    assert ("runpy", "run_module") not in (
        runtime_verifier.SOURCE_EXTENSION_RUNTIME_MODULE_ATTRIBUTE_CALL_ALLOWLIST
    )
    assert ("runpy", "run_path") not in (
        runtime_verifier.SOURCE_EXTENSION_RUNTIME_MODULE_ATTRIBUTE_CALL_ALLOWLIST
    )
    fixture = _fixture()
    prohibited_files = _replace_runtime_bytes(
        fixture.runtime_files,
        "agent/experiments/cli.py",
        source,
    )

    with pytest.raises(
        C2FullReplacementPolicyError,
        match=f"{match}|declarative AST",
    ):
        _compile_runtime_closure(prohibited_files)


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
        "deny-by-default-static-call-targets-are-required",
        "closed-module-attribute-call-allowlist-is-enforced",
        "implicit-runtime-evaluation-routes-are-rejected",
        "higher-order-callback-dispatch-is-rejected",
        "declarative-ast-subset-rejects-runtime-protocols",
        "test-only-fixture-is-not-a-production-input",
    )

"""Non-admissive Stage-B source-extension runtime-verifier foundation.

The public production entry point deliberately has no inputs and cannot inspect
Git, a worktree, a manifest, source evidence, or runtime artifacts while the
required production resources remain absent.  This module defines only the
typed comparison boundary a later reviewed adapter must use:

* a deployment-pinned lock supplies expected implementation/attestation
  identities, manifest and blob-set digests, the fixed path/role roster, and a
  runtime byte/import-closure digest;
* the compile-pinned code-attestation registry supplies the independently
  reviewed implementation/attestation and manifest/blob-set bindings; and
* a runtime adapter proves byte and deny-by-default static-import closure
  against both.

There is intentionally no deployment-lock resource, production adapter,
candidate selector, or admission route here.  The explicitly named
``*_for_testing`` APIs accept in-memory fixtures only.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Sequence

from .c2_stageb_source_extension_code_attestation import (
    SOURCE_EXTENSION_RUNTIME_PATH_ROLES,
    C2StageBCodeAttestationError,
    SourceExtensionCodeAttestationRegistryEntry,
    SourceExtensionRuntimePathRole,
    load_compile_pinned_source_extension_code_attestation,
)
from .models import sha256_json


class C2StageBSourceExtensionRuntimeVerifierError(C2StageBCodeAttestationError):
    """Raised when a source-extension runtime binding is unavailable or invalid."""


_SOURCE_EXTENSION_RUNTIME_VERIFIER_ROUTE_APPROVED = False

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_PACKAGE_PATH = ("agent", "experiments")
_RUNTIME_PRIMARY_PACKAGE = "experiments"
_RUNTIME_ALTERNATE_PACKAGE = "agent.experiments"
_ALIAS_STATIC_CALLABLE = "static-callable"
_ALIAS_STATIC_MODULE = "static-module"
_ALIAS_STATIC_VALUE = "static-value"
_ALIAS_UNKNOWN = "unknown"
_ALIAS_IMPORTLIB_MODULE = "importlib-module"
_ALIAS_BUILTINS_MODULE = "builtins-module"
_ALIAS_IMPORTLIB_CALLABLE = "importlib-import-module"
_ALIAS_BUILTINS_CALLABLE = "builtins-dunder-import"
_ALIAS_PROHIBITED_CALLABLE = "prohibited-callable"
_DYNAMIC_IMPORT_ALIAS_KINDS = frozenset(
    {
        _ALIAS_IMPORTLIB_MODULE,
        _ALIAS_BUILTINS_MODULE,
        _ALIAS_IMPORTLIB_CALLABLE,
        _ALIAS_BUILTINS_CALLABLE,
    }
)
_DYNAMIC_IMPORT_ATTRIBUTE_NAMES = frozenset({"__import__", "import_module"})
_REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES = frozenset(
    {
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "locals",
        "setattr",
        "vars",
    }
)
_REFLECTIVE_NAMESPACE_ATTRIBUTE_NAMES = frozenset({"__dict__", "__builtins__"})
_KNOWN_STATIC_BUILTIN_CALLABLE_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    }
)

SOURCE_EXTENSION_RUNTIME_VERIFIER_TEST_MATRIX = (
    "absence-fails-before-candidate-access",
    "public-selector-injection-is-not-an-input",
    "dynamic-head-or-parent-identity-is-rejected",
    "malformed-fixed-path-role-roster-is-rejected",
    "unrostered-or-unresolved-local-import-is-rejected",
    "duplicate-or-cyclic-local-import-graph-is-rejected",
    "dynamic-import-aliases-and-unknown-targets-are-rejected",
    "deny-by-default-static-call-targets-are-required",
    "test-only-fixture-is-not-a-production-input",
)


@dataclass(frozen=True, slots=True)
class DeploymentPinnedSourceExtensionRuntimeLock:
    """Expected runtime bindings for a future deployment-pinned lock.

    This is an in-memory typed interface only.  It is not a signed lock, does
    not load a lock from a path, and is currently usable only by test APIs.
    Its implementation and manifest-only attestation identities are I and A.
    """

    expected_code_attestation_registry_id_sha256: str
    expected_extension_implementation_commit_full: str
    expected_manifest_only_attestation_commit_full: str
    expected_manifest_sha256: str
    expected_canonical_attested_blob_set_sha256: str
    expected_runtime_path_roles: tuple[SourceExtensionRuntimePathRole, ...]
    expected_runtime_byte_import_closure_sha256: str


@dataclass(frozen=True, slots=True)
class SourceExtensionRuntimeFixtureFile:
    """One in-memory runtime file used only by test-only verification."""

    runtime_path: str
    role: str
    runtime_bytes: bytes


@dataclass(frozen=True, slots=True)
class SourceExtensionRuntimeTestFixture:
    """An in-memory candidate representation for explicitly test-only APIs."""

    implementation_commit_full: str
    manifest_only_attestation_commit_full: str
    manifest_bytes: bytes
    runtime_files: tuple[SourceExtensionRuntimeFixtureFile, ...]


@dataclass(frozen=True, slots=True)
class SourceExtensionRuntimeByteImportBinding:
    """The byte hash and recursive static imports for one runtime roster entry."""

    runtime_path: str
    role: str
    byte_sha256: str
    direct_imports: tuple[str, ...]
    resolved_project_local_dependencies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_path": self.runtime_path,
            "role": self.role,
            "byte_sha256": self.byte_sha256,
            "direct_imports": list(self.direct_imports),
            "resolved_project_local_dependencies": list(
                self.resolved_project_local_dependencies
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceExtensionRuntimeByteImportClosure:
    """Canonical byte/import closure required by a deployment-pinned lock."""

    bindings: tuple[SourceExtensionRuntimeByteImportBinding, ...]
    canonical_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_byte_import_closure": [
                binding.to_dict() for binding in self.bindings
            ]
        }


@dataclass(frozen=True, slots=True)
class _StaticImport:
    """One statically parsed import statement or direct import target."""

    module: str | None
    relative_level: int
    imported_names: tuple[str, ...]
    direct_imports: tuple[str, ...]
    is_from_import: bool


@dataclass(frozen=True, slots=True)
class SourceExtensionRuntimeVerificationBinding:
    """A structural runtime binding, never a production admission decision."""

    code_attestation_registry_id_sha256: str
    extension_implementation_commit_full: str
    manifest_only_attestation_commit_full: str
    manifest_sha256: str
    canonical_attested_blob_set_sha256: str
    covered_runtime_paths: tuple[SourceExtensionRuntimePathRole, ...]
    runtime_byte_import_closure_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_attestation_registry_id_sha256": (
                self.code_attestation_registry_id_sha256
            ),
            "extension_implementation_commit_full": (
                self.extension_implementation_commit_full
            ),
            "manifest_only_attestation_commit_full": (
                self.manifest_only_attestation_commit_full
            ),
            "manifest_sha256": self.manifest_sha256,
            "canonical_attested_blob_set_sha256": (
                self.canonical_attested_blob_set_sha256
            ),
            "covered_runtime_paths": [
                binding.to_dict() for binding in self.covered_runtime_paths
            ],
            "runtime_byte_import_closure_sha256": (
                self.runtime_byte_import_closure_sha256
            ),
        }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} must be a full SHA-256 digest"
        )
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} must be a full Git commit, never a dynamic HEAD or parent"
        )
    return value


def _require_exact_roster(
    value: Any,
    label: str,
) -> tuple[SourceExtensionRuntimePathRole, ...]:
    if not isinstance(value, tuple):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} must be an immutable ordered path/role roster"
        )
    parsed: list[SourceExtensionRuntimePathRole] = []
    for index, binding in enumerate(value):
        if not isinstance(binding, SourceExtensionRuntimePathRole):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                f"{label}[{index}] must be a SourceExtensionRuntimePathRole"
            )
        parsed.append(binding)
    roster = tuple(
        (binding.runtime_path, binding.role) for binding in parsed
    )
    if roster != SOURCE_EXTENSION_RUNTIME_PATH_ROLES:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} must equal the fixed ordered source-extension path/role roster"
        )
    return tuple(parsed)


def _require_equal(actual: str, expected: str, label: str) -> None:
    if not hmac.compare_digest(actual, expected):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} does not match the deployment-pinned expectation"
        )


def _validate_deployment_lock(
    lock: Any,
) -> DeploymentPinnedSourceExtensionRuntimeLock:
    if type(lock) is not DeploymentPinnedSourceExtensionRuntimeLock:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "Runtime verification requires a DeploymentPinnedSourceExtensionRuntimeLock"
        )
    _require_sha256(
        lock.expected_code_attestation_registry_id_sha256,
        "expected_code_attestation_registry_id_sha256",
    )
    _require_commit(
        lock.expected_extension_implementation_commit_full,
        "expected_extension_implementation_commit_full",
    )
    _require_commit(
        lock.expected_manifest_only_attestation_commit_full,
        "expected_manifest_only_attestation_commit_full",
    )
    _require_sha256(lock.expected_manifest_sha256, "expected_manifest_sha256")
    _require_sha256(
        lock.expected_canonical_attested_blob_set_sha256,
        "expected_canonical_attested_blob_set_sha256",
    )
    _require_exact_roster(
        lock.expected_runtime_path_roles,
        "expected_runtime_path_roles",
    )
    _require_sha256(
        lock.expected_runtime_byte_import_closure_sha256,
        "expected_runtime_byte_import_closure_sha256",
    )
    return lock


def _validate_registry_entry(
    registry_entry: Any,
) -> SourceExtensionCodeAttestationRegistryEntry:
    if type(registry_entry) is not SourceExtensionCodeAttestationRegistryEntry:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "Runtime verification requires a compiled code-attestation registry entry"
        )
    _require_sha256(registry_entry.registry_id_sha256, "registry_id_sha256")
    _require_commit(
        registry_entry.extension_implementation_commit_full,
        "extension_implementation_commit_full",
    )
    _require_commit(
        registry_entry.manifest_only_attestation_commit_full,
        "manifest_only_attestation_commit_full",
    )
    _require_sha256(registry_entry.manifest_sha256, "manifest_sha256")
    _require_sha256(
        registry_entry.canonical_attested_blob_set_sha256,
        "canonical_attested_blob_set_sha256",
    )
    _require_exact_roster(
        registry_entry.covered_runtime_paths,
        "registry_entry.covered_runtime_paths",
    )
    return registry_entry


def _validate_fixture(
    fixture: Any,
) -> SourceExtensionRuntimeTestFixture:
    if type(fixture) is not SourceExtensionRuntimeTestFixture:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "Only a SourceExtensionRuntimeTestFixture may reach a test-only verifier"
        )
    _require_commit(fixture.implementation_commit_full, "implementation_commit_full")
    _require_commit(
        fixture.manifest_only_attestation_commit_full,
        "manifest_only_attestation_commit_full",
    )
    if type(fixture.manifest_bytes) is not bytes:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "test-only manifest fixture must supply exact bytes"
        )
    if not isinstance(fixture.runtime_files, tuple):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "test-only runtime fixture must supply an immutable ordered file roster"
        )
    return fixture


def _validate_fixture_files(
    runtime_files: Sequence[SourceExtensionRuntimeFixtureFile],
) -> tuple[SourceExtensionRuntimeFixtureFile, ...]:
    if not isinstance(runtime_files, tuple):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture files must be an immutable ordered roster"
        )
    parsed: list[SourceExtensionRuntimeFixtureFile] = []
    for index, runtime_file in enumerate(runtime_files):
        if type(runtime_file) is not SourceExtensionRuntimeFixtureFile:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                f"runtime fixture file {index} has an invalid test-only type"
            )
        if type(runtime_file.runtime_bytes) is not bytes:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                f"runtime fixture file {index} must carry exact bytes"
            )
        parsed.append(runtime_file)
    roster = tuple((item.runtime_path, item.role) for item in parsed)
    if roster != SOURCE_EXTENSION_RUNTIME_PATH_ROLES:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture files must equal the fixed ordered path/role roster"
        )
    return tuple(parsed)


def _require_canonical_repo_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} must be a nonempty canonical repository-relative POSIX path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} must not contain traversal, aliases, or an absolute path"
        )
    return value


def _require_module_name(value: str, label: str) -> str:
    if not value or any(not part.isidentifier() for part in value.split(".")):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"{label} is not a statically resolvable Python module name"
        )
    return value


def _runtime_module_names(runtime_path: str) -> tuple[str, str]:
    path = _require_canonical_repo_relative_path(runtime_path, "runtime path")
    parts = PurePosixPath(path).parts
    if (
        len(parts) < 3
        or parts[:2] != _RUNTIME_PACKAGE_PATH
        or not path.endswith(".py")
    ):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime Python path must be under agent/experiments and end in .py"
        )
    stem = parts[-1][:-3]
    module_parts = list(parts[2:-1])
    if stem != "__init__":
        module_parts.append(stem)
    if any(not part.isidentifier() for part in module_parts):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"runtime path cannot map to a static Python module: {runtime_path}"
        )
    suffix = ".".join(module_parts)
    primary = (
        _RUNTIME_PRIMARY_PACKAGE
        if not suffix
        else f"{_RUNTIME_PRIMARY_PACKAGE}.{suffix}"
    )
    alternate = (
        _RUNTIME_ALTERNATE_PACKAGE
        if not suffix
        else f"{_RUNTIME_ALTERNATE_PACKAGE}.{suffix}"
    )
    return primary, alternate


def _build_runtime_module_index(
    runtime_files: Sequence[SourceExtensionRuntimeFixtureFile],
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    module_to_path: dict[str, str] = {}
    primary_module_by_path: dict[str, str] = {}
    roster_index = {
        runtime_file.runtime_path: index
        for index, runtime_file in enumerate(runtime_files)
    }
    for runtime_file in runtime_files:
        if not runtime_file.runtime_path.endswith(".py"):
            continue
        primary, alternate = _runtime_module_names(runtime_file.runtime_path)
        primary_module_by_path[runtime_file.runtime_path] = primary
        for module_name in (primary, alternate):
            existing_path = module_to_path.get(module_name)
            if existing_path is not None:
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime import closure has a duplicate local module mapping: "
                    f"{module_name} maps to both {existing_path} and "
                    f"{runtime_file.runtime_path}"
                )
            module_to_path[module_name] = runtime_file.runtime_path
    return module_to_path, primary_module_by_path, roster_index


def _alias_path(value: ast.expr | ast.expr_context) -> tuple[str, ...] | None:
    if isinstance(value, ast.Name):
        return (value.id,)
    if isinstance(value, ast.Attribute):
        parent = _alias_path(value.value)
        return None if parent is None else (*parent, value.attr)
    return None


def _assignment_alias_paths(
    target: ast.expr | ast.expr_context,
) -> tuple[tuple[str, ...], ...]:
    path = _alias_path(target)
    if path is not None:
        return (path,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            path
            for element in target.elts
            for path in _assignment_alias_paths(element)
        )
    return ()


def _class_assignment_alias_paths(
    class_name: str,
    target: ast.expr | ast.expr_context,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(target, ast.Name):
        return ((class_name, target.id),)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            path
            for element in target.elts
            for path in _class_assignment_alias_paths(class_name, element)
        )
    return ()


def _merge_alias_kind(current: str | None, incoming: str) -> str:
    if current is None or current == incoming:
        return incoming
    return _ALIAS_UNKNOWN


def _record_alias_kind(
    aliases: dict[tuple[str, ...], str],
    path: tuple[str, ...],
    kind: str,
) -> bool:
    merged = _merge_alias_kind(aliases.get(path), kind)
    if aliases.get(path) == merged:
        return False
    aliases[path] = merged
    return True


def _import_module_alias_kind(module_name: str, binding_name: str) -> str:
    if (
        module_name == "importlib"
        or (
            module_name.startswith("importlib.")
            and binding_name == "importlib"
        )
    ):
        return _ALIAS_IMPORTLIB_MODULE
    if (
        module_name == "builtins"
        or (
            module_name.startswith("builtins.")
            and binding_name == "builtins"
        )
    ):
        return _ALIAS_BUILTINS_MODULE
    return _ALIAS_STATIC_MODULE


def _from_import_alias_kind(
    module_name: str | None,
    relative_level: int,
    imported_name: str,
) -> str:
    if relative_level == 0 and module_name == "importlib":
        if imported_name == "import_module":
            return _ALIAS_IMPORTLIB_CALLABLE
        if imported_name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES:
            return _ALIAS_PROHIBITED_CALLABLE
    if relative_level == 0 and module_name == "builtins":
        if imported_name == "__import__":
            return _ALIAS_BUILTINS_CALLABLE
        if imported_name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES:
            return _ALIAS_PROHIBITED_CALLABLE
    if imported_name in _DYNAMIC_IMPORT_ATTRIBUTE_NAMES:
        return _ALIAS_IMPORTLIB_CALLABLE
    if imported_name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES:
        return _ALIAS_PROHIBITED_CALLABLE
    if relative_level and module_name is None:
        return _ALIAS_STATIC_MODULE
    return _ALIAS_STATIC_CALLABLE


def _alias_kind_for_path(
    path: tuple[str, ...],
    aliases: dict[tuple[str, ...], str],
) -> str | None:
    known = aliases.get(path)
    if known is not None:
        return known
    if len(path) == 1:
        return None
    parent_kind = _alias_kind_for_path(path[:-1], aliases)
    attribute_name = path[-1]
    if attribute_name in _REFLECTIVE_NAMESPACE_ATTRIBUTE_NAMES:
        return _ALIAS_PROHIBITED_CALLABLE
    if attribute_name in _DYNAMIC_IMPORT_ATTRIBUTE_NAMES:
        return _ALIAS_IMPORTLIB_CALLABLE
    if attribute_name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES:
        return _ALIAS_PROHIBITED_CALLABLE
    if parent_kind in {
        _ALIAS_IMPORTLIB_MODULE,
        _ALIAS_BUILTINS_MODULE,
        _ALIAS_IMPORTLIB_CALLABLE,
        _ALIAS_BUILTINS_CALLABLE,
    }:
        return _ALIAS_IMPORTLIB_CALLABLE
    if parent_kind == _ALIAS_PROHIBITED_CALLABLE:
        return _ALIAS_PROHIBITED_CALLABLE
    if parent_kind == _ALIAS_STATIC_MODULE:
        return _ALIAS_STATIC_CALLABLE
    if parent_kind == _ALIAS_STATIC_CALLABLE:
        return (
            _ALIAS_STATIC_CALLABLE
            if attribute_name == "__call__"
            else _ALIAS_UNKNOWN
        )
    if parent_kind in {_ALIAS_STATIC_VALUE, _ALIAS_UNKNOWN}:
        return _ALIAS_UNKNOWN
    return None


def _alias_kind_for_static_getattr(
    value: ast.Call,
    aliases: dict[tuple[str, ...], str],
) -> str | None:
    if (
        not isinstance(value.func, ast.Name)
        or value.func.id != "getattr"
        or len(value.args) < 2
        or not isinstance(value.args[1], ast.Constant)
        or not isinstance(value.args[1].value, str)
    ):
        return None
    return _ALIAS_PROHIBITED_CALLABLE


def _alias_kind_for_expression(
    value: ast.expr,
    aliases: dict[tuple[str, ...], str],
) -> str | None:
    path = _alias_path(value)
    if path is not None:
        return _alias_kind_for_path(path, aliases)
    if isinstance(value, ast.Call):
        return _alias_kind_for_static_getattr(value, aliases) or _ALIAS_UNKNOWN
    if isinstance(value, ast.Lambda):
        return _ALIAS_STATIC_CALLABLE
    if isinstance(value, ast.Constant):
        return _ALIAS_STATIC_VALUE
    if isinstance(value, (ast.Subscript, ast.List, ast.Set, ast.Tuple)):
        return _ALIAS_UNKNOWN
    return None


def _assignment_alias_kind(
    value: ast.expr,
    aliases: dict[tuple[str, ...], str],
) -> str:
    return _alias_kind_for_expression(value, aliases) or _ALIAS_UNKNOWN


def _collect_static_alias_kinds(
    tree: ast.AST,
) -> dict[tuple[str, ...], str]:
    aliases: dict[tuple[str, ...], str] = {
        (name,): _ALIAS_STATIC_CALLABLE
        for name in _KNOWN_STATIC_BUILTIN_CALLABLE_NAMES
    }
    aliases.update(
        {
            (name,): _ALIAS_PROHIBITED_CALLABLE
            for name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES
        }
    )
    aliases[("__import__",)] = _ALIAS_BUILTINS_CALLABLE
    aliases[("import_module",)] = _ALIAS_IMPORTLIB_CALLABLE
    assignments: list[tuple[tuple[str, ...], ast.expr | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                binding_name = imported.asname or imported.name.split(".", 1)[0]
                _record_alias_kind(
                    aliases,
                    (binding_name,),
                    _import_module_alias_kind(imported.name, binding_name),
                )
        elif isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.name == "*":
                    continue
                binding_name = imported.asname or imported.name
                _record_alias_kind(
                    aliases,
                    (binding_name,),
                    _from_import_alias_kind(
                        node.module,
                        node.level,
                        imported.name,
                    ),
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _record_alias_kind(aliases, (node.name,), _ALIAS_STATIC_CALLABLE)
        elif isinstance(node, ast.ClassDef):
            _record_alias_kind(aliases, (node.name,), _ALIAS_STATIC_CALLABLE)
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        assignments.extend(
                            (path, statement.value)
                            for path in _class_assignment_alias_paths(
                                node.name,
                                target,
                            )
                        )
                elif isinstance(statement, ast.AnnAssign):
                    assignments.extend(
                        (path, statement.value)
                        for path in _class_assignment_alias_paths(
                            node.name,
                            statement.target,
                        )
                    )
        elif isinstance(node, ast.arguments):
            arguments = (
                *node.posonlyargs,
                *node.args,
                *node.kwonlyargs,
            )
            if node.vararg is not None:
                arguments = (*arguments, node.vararg)
            if node.kwarg is not None:
                arguments = (*arguments, node.kwarg)
            for argument in arguments:
                _record_alias_kind(aliases, (argument.arg,), _ALIAS_UNKNOWN)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            for path in _assignment_alias_paths(node.target):
                _record_alias_kind(aliases, path, _ALIAS_UNKNOWN)
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None:
                for path in _assignment_alias_paths(node.optional_vars):
                    _record_alias_kind(aliases, path, _ALIAS_UNKNOWN)
        elif isinstance(node, ast.ExceptHandler):
            if node.name is not None:
                _record_alias_kind(aliases, (node.name,), _ALIAS_UNKNOWN)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                assignments.extend(
                    (path, node.value) for path in _assignment_alias_paths(target)
                )
        elif isinstance(node, ast.AnnAssign):
            assignments.extend(
                (path, node.value) for path in _assignment_alias_paths(node.target)
            )
        elif isinstance(node, ast.NamedExpr):
            assignments.extend(
                (path, node.value) for path in _assignment_alias_paths(node.target)
            )
        elif isinstance(node, ast.AugAssign):
            assignments.extend(
                (path, None) for path in _assignment_alias_paths(node.target)
            )
    while True:
        changed = False
        for path, value in assignments:
            kind = _ALIAS_UNKNOWN if value is None else _assignment_alias_kind(
                value,
                aliases,
            )
            changed = _record_alias_kind(aliases, path, kind) or changed
        if not changed:
            return aliases


def _contains_subscript_target(target: ast.expr | ast.expr_context) -> bool:
    if isinstance(target, ast.Subscript):
        return True
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_contains_subscript_target(element) for element in target.elts)
    return False


def _reject_reflective_namespace_syntax(tree: ast.AST, runtime_path: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids reflective namespace access: "
                f"{runtime_path}"
            )
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _REFLECTIVE_NAMESPACE_ATTRIBUTE_NAMES
        ):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids reflective namespace access: "
                f"{runtime_path}"
            )
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids namespace or attribute mutation: "
                f"{runtime_path}"
            )
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = (node.target,)
        elif isinstance(node, ast.Delete):
            targets = node.targets
        else:
            continue
        if any(_contains_subscript_target(target) for target in targets):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids namespace or mapping mutation: "
                f"{runtime_path}"
            )


def _reject_dynamic_or_reflective_alias_references(
    tree: ast.AST,
    aliases: dict[tuple[str, ...], str],
    runtime_path: str,
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        path = _alias_path(node)
        if path is None:
            continue
        kind = _alias_kind_for_path(path, aliases)
        if kind in _DYNAMIC_IMPORT_ALIAS_KINDS:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture import closure forbids dynamic imports: "
                f"{runtime_path}"
            )
        if kind == _ALIAS_PROHIBITED_CALLABLE:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids reflective or executable "
                f"callables: {runtime_path}"
            )


def _reject_nonstatic_call_targets(tree: ast.AST, runtime_path: str) -> None:
    _reject_reflective_namespace_syntax(tree, runtime_path)
    aliases = _collect_static_alias_kinds(tree)
    _reject_dynamic_or_reflective_alias_references(
        tree,
        aliases,
        runtime_path,
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target_kind = _alias_kind_for_expression(node.func, aliases)
        if target_kind in _DYNAMIC_IMPORT_ALIAS_KINDS:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture import closure forbids dynamic imports: "
                f"{runtime_path}"
            )
        if target_kind == _ALIAS_PROHIBITED_CALLABLE:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids reflective or executable "
                f"callables: {runtime_path}"
            )
        if target_kind != _ALIAS_STATIC_CALLABLE:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture call target is unbound or not a statically "
                "allowed non-dynamic callable: "
                f"{runtime_path}"
            )


def _parse_static_imports(
    runtime_path: str,
    runtime_bytes: bytes,
) -> tuple[_StaticImport, ...]:
    try:
        source = runtime_bytes.decode("utf-8")
        tree = ast.parse(source, filename=runtime_path, mode="exec")
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            f"runtime fixture Python bytes are not valid UTF-8 source: {runtime_path}"
        ) from exc
    _reject_nonstatic_call_targets(tree, runtime_path)
    imports: list[_StaticImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = _require_module_name(
                    alias.name,
                    f"absolute import in {runtime_path}",
                )
                imports.append(
                    _StaticImport(
                        module=module_name,
                        relative_level=0,
                        imported_names=(),
                        direct_imports=(module_name,),
                        is_from_import=False,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture import closure forbids wildcard imports: "
                    f"{runtime_path}"
                )
            imported_names = tuple(alias.name for alias in node.names)
            if any(not name.isidentifier() for name in imported_names):
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture import closure has an unresolved alias import: "
                    f"{runtime_path}"
                )
            module_name = (
                None
                if node.module is None
                else _require_module_name(
                    node.module,
                    f"from-import module in {runtime_path}",
                )
            )
            if node.level == 0 and module_name is None:
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    f"runtime fixture import has no static target: {runtime_path}"
                )
            prefix = "." * node.level
            direct_imports = (
                tuple(f"{prefix}{name}" for name in imported_names)
                if module_name is None
                else (f"{prefix}{module_name}",)
            )
            imports.append(
                _StaticImport(
                    module=module_name,
                    relative_level=node.level,
                    imported_names=imported_names,
                    direct_imports=direct_imports,
                    is_from_import=True,
                )
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture import closure forbids dynamic imports: "
                    f"{runtime_path}"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"__import__", "import_module"}
            ):
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture import closure forbids dynamic imports: "
                    f"{runtime_path}"
                )
    return tuple(imports)


def _relative_import_base(
    importer_module: str,
    relative_level: int,
    importer_path: str,
) -> str:
    package_parts = importer_module.split(".")[:-1]
    if relative_level < 1 or relative_level > len(package_parts):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture relative import has traversal or no static package base: "
            f"{importer_path}"
        )
    return ".".join(package_parts[: len(package_parts) - relative_level + 1])


def _is_project_local_module(module_name: str) -> bool:
    return (
        module_name == _RUNTIME_PRIMARY_PACKAGE
        or module_name.startswith(f"{_RUNTIME_PRIMARY_PACKAGE}.")
        or module_name == "agent"
        or module_name.startswith("agent.")
    )


def _resolve_local_module_path(
    module_name: str,
    module_to_path: dict[str, str],
    importer_path: str,
) -> str:
    path = module_to_path.get(module_name)
    if path is None:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture import resolves to an unrostered project-local "
            f"dependency: {module_name} from {importer_path}"
        )
    return _require_canonical_repo_relative_path(
        path,
        "resolved project-local dependency",
    )


def _resolve_static_local_dependencies(
    importer_path: str,
    importer_module: str,
    imports: Sequence[_StaticImport],
    module_to_path: dict[str, str],
    roster_index: dict[str, int],
) -> tuple[str, ...]:
    dependencies: list[str] = []
    for static_import in imports:
        if static_import.relative_level:
            base = _relative_import_base(
                importer_module,
                static_import.relative_level,
                importer_path,
            )
            if static_import.module is None:
                for imported_name in static_import.imported_names:
                    dependencies.append(
                        _resolve_local_module_path(
                            f"{base}.{imported_name}",
                            module_to_path,
                            importer_path,
                        )
                    )
            else:
                dependencies.append(
                    _resolve_local_module_path(
                        f"{base}.{static_import.module}",
                        module_to_path,
                        importer_path,
                    )
                )
            continue

        module_name = static_import.module
        if module_name is None or not _is_project_local_module(module_name):
            continue
        if (
            static_import.is_from_import
            and module_name
            in {_RUNTIME_PRIMARY_PACKAGE, _RUNTIME_ALTERNATE_PACKAGE}
        ):
            for imported_name in static_import.imported_names:
                dependencies.append(
                    _resolve_local_module_path(
                        f"{module_name}.{imported_name}",
                        module_to_path,
                        importer_path,
                    )
                )
        else:
            dependencies.append(
                _resolve_local_module_path(
                    module_name,
                    module_to_path,
                    importer_path,
                )
            )
    if len(dependencies) != len(set(dependencies)):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture import closure has a duplicate local dependency: "
            f"{importer_path}"
        )
    return tuple(sorted(dependencies, key=roster_index.__getitem__))


def _resolve_recursive_local_import_closures(
    graph: dict[str, tuple[str, ...]],
    roster_index: dict[str, int],
) -> dict[str, tuple[str, ...]]:
    completed: dict[str, tuple[str, ...]] = {}
    active: list[str] = []

    def visit(runtime_path: str) -> tuple[str, ...]:
        if runtime_path in active:
            cycle = (*active[active.index(runtime_path) :], runtime_path)
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture import closure has a local dependency cycle: "
                f"{' -> '.join(cycle)}"
            )
        if runtime_path in completed:
            return completed[runtime_path]
        active.append(runtime_path)
        transitive_dependencies: list[str] = []
        for dependency in graph[runtime_path]:
            if dependency not in graph:
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture import closure resolved a non-Python local "
                    f"dependency: {dependency}"
                )
            transitive_dependencies.append(dependency)
            transitive_dependencies.extend(visit(dependency))
        active.pop()
        closure = tuple(
            sorted(
                set(transitive_dependencies),
                key=roster_index.__getitem__,
            )
        )
        completed[runtime_path] = closure
        return closure

    for runtime_path in graph:
        visit(runtime_path)
    return completed


def _canonical_attested_blob_set_sha256(
    runtime_files: Sequence[SourceExtensionRuntimeFixtureFile],
) -> str:
    return sha256_json(
        {
            "attested_blob_set": [
                {
                    "runtime_path": runtime_file.runtime_path,
                    "role": runtime_file.role,
                    "byte_sha256": hashlib.sha256(
                        runtime_file.runtime_bytes
                    ).hexdigest(),
                }
                for runtime_file in runtime_files
            ]
        }
    )


def _compile_runtime_byte_import_closure(
    runtime_files: Sequence[SourceExtensionRuntimeFixtureFile],
) -> SourceExtensionRuntimeByteImportClosure:
    (
        module_to_path,
        primary_module_by_path,
        roster_index,
    ) = _build_runtime_module_index(runtime_files)
    parsed_imports_by_path: dict[str, tuple[_StaticImport, ...]] = {}
    local_dependencies_by_path: dict[str, tuple[str, ...]] = {}
    for runtime_file in runtime_files:
        if not runtime_file.runtime_path.endswith(".py"):
            continue
        imports = _parse_static_imports(
            runtime_file.runtime_path,
            runtime_file.runtime_bytes,
        )
        parsed_imports_by_path[runtime_file.runtime_path] = imports
        local_dependencies_by_path[runtime_file.runtime_path] = (
            _resolve_static_local_dependencies(
                runtime_file.runtime_path,
                primary_module_by_path[runtime_file.runtime_path],
                imports,
                module_to_path,
                roster_index,
            )
        )
    recursive_local_dependencies_by_path = _resolve_recursive_local_import_closures(
        local_dependencies_by_path,
        roster_index,
    )
    bindings = tuple(
        SourceExtensionRuntimeByteImportBinding(
            runtime_path=runtime_file.runtime_path,
            role=runtime_file.role,
            byte_sha256=hashlib.sha256(runtime_file.runtime_bytes).hexdigest(),
            direct_imports=tuple(
                sorted(
                    {
                        direct_import
                        for static_import in parsed_imports_by_path.get(
                            runtime_file.runtime_path,
                            (),
                        )
                        for direct_import in static_import.direct_imports
                    }
                )
            ),
            resolved_project_local_dependencies=(
                recursive_local_dependencies_by_path.get(
                    runtime_file.runtime_path,
                    (),
                )
            ),
        )
        for runtime_file in runtime_files
    )
    closure = SourceExtensionRuntimeByteImportClosure(
        bindings=bindings,
        canonical_sha256="",
    )
    return SourceExtensionRuntimeByteImportClosure(
        bindings=bindings,
        canonical_sha256=sha256_json(closure.to_dict()),
    )


def compile_source_extension_runtime_byte_import_closure_for_testing(
    runtime_files: tuple[SourceExtensionRuntimeFixtureFile, ...],
) -> SourceExtensionRuntimeByteImportClosure:
    """Compile an in-memory byte/import closure for tests, never production."""

    return _compile_runtime_byte_import_closure(_validate_fixture_files(runtime_files))


def canonical_attested_blob_set_sha256_for_testing(
    runtime_files: tuple[SourceExtensionRuntimeFixtureFile, ...],
) -> str:
    """Return the test-only canonical byte-derived attested blob-set digest."""

    return _canonical_attested_blob_set_sha256(_validate_fixture_files(runtime_files))


def _verify_source_extension_runtime_fixture(
    deployment_lock: DeploymentPinnedSourceExtensionRuntimeLock,
    registry_entry: SourceExtensionCodeAttestationRegistryEntry,
    fixture: SourceExtensionRuntimeTestFixture,
) -> SourceExtensionRuntimeVerificationBinding:
    lock = _validate_deployment_lock(deployment_lock)
    registry = _validate_registry_entry(registry_entry)
    test_fixture = _validate_fixture(fixture)
    files = _validate_fixture_files(test_fixture.runtime_files)

    _require_equal(
        registry.registry_id_sha256,
        lock.expected_code_attestation_registry_id_sha256,
        "code-attestation registry identity",
    )
    _require_equal(
        registry.extension_implementation_commit_full,
        lock.expected_extension_implementation_commit_full,
        "registry implementation identity",
    )
    _require_equal(
        registry.manifest_only_attestation_commit_full,
        lock.expected_manifest_only_attestation_commit_full,
        "registry manifest-only attestation identity",
    )
    _require_equal(
        registry.manifest_sha256,
        lock.expected_manifest_sha256,
        "registry manifest digest",
    )
    _require_equal(
        registry.canonical_attested_blob_set_sha256,
        lock.expected_canonical_attested_blob_set_sha256,
        "registry canonical attested blob-set digest",
    )
    _require_exact_roster(
        registry.covered_runtime_paths,
        "registry_entry.covered_runtime_paths",
    )
    _require_exact_roster(
        lock.expected_runtime_path_roles,
        "expected_runtime_path_roles",
    )

    _require_equal(
        test_fixture.implementation_commit_full,
        lock.expected_extension_implementation_commit_full,
        "candidate implementation identity",
    )
    _require_equal(
        test_fixture.manifest_only_attestation_commit_full,
        lock.expected_manifest_only_attestation_commit_full,
        "candidate manifest-only attestation identity",
    )
    _require_equal(
        hashlib.sha256(test_fixture.manifest_bytes).hexdigest(),
        lock.expected_manifest_sha256,
        "candidate manifest bytes",
    )
    _require_equal(
        _canonical_attested_blob_set_sha256(files),
        lock.expected_canonical_attested_blob_set_sha256,
        "candidate canonical attested blob set",
    )
    closure = _compile_runtime_byte_import_closure(files)
    _require_equal(
        closure.canonical_sha256,
        lock.expected_runtime_byte_import_closure_sha256,
        "candidate runtime byte/import closure",
    )
    return SourceExtensionRuntimeVerificationBinding(
        code_attestation_registry_id_sha256=registry.registry_id_sha256,
        extension_implementation_commit_full=(
            registry.extension_implementation_commit_full
        ),
        manifest_only_attestation_commit_full=(
            registry.manifest_only_attestation_commit_full
        ),
        manifest_sha256=registry.manifest_sha256,
        canonical_attested_blob_set_sha256=(
            registry.canonical_attested_blob_set_sha256
        ),
        covered_runtime_paths=registry.covered_runtime_paths,
        runtime_byte_import_closure_sha256=closure.canonical_sha256,
    )


def verify_source_extension_runtime_for_testing(
    *,
    deployment_lock: DeploymentPinnedSourceExtensionRuntimeLock,
    registry_entry: SourceExtensionCodeAttestationRegistryEntry,
    fixture: SourceExtensionRuntimeTestFixture,
) -> SourceExtensionRuntimeVerificationBinding:
    """Verify synthetic in-memory inputs through the test-only structural path."""

    return _verify_source_extension_runtime_fixture(
        deployment_lock,
        registry_entry,
        fixture,
    )


def verify_compile_pinned_source_extension_runtime() -> (
    SourceExtensionRuntimeVerificationBinding
):
    """Fail closed until reviewed production resources and adapter exist.

    This deliberately performs no candidate access.  In particular, it does not
    open Git, a worktree, a manifest, source evidence, or runtime artifacts.
    """

    # The current registry loader fails before reading its absent production
    # resource. A future adapter must bind only this no-selector, compile-pinned
    # registry and a separately reviewed deployment-pinned lock.
    registry_entry = load_compile_pinned_source_extension_code_attestation()
    _validate_registry_entry(registry_entry)
    if not _SOURCE_EXTENSION_RUNTIME_VERIFIER_ROUTE_APPROVED:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "Stage-A only: source-extension runtime verification remains intentionally "
            "non-admissive pending independent deployment-lock review"
        )
    raise C2StageBSourceExtensionRuntimeVerifierError(
        "Stage-A only: no reviewed source-extension runtime adapter exists"
    )

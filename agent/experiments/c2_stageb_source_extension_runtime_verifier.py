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
_SOURCE_EXTENSION_RUNTIME_IMPORT_ROOT_PATH = "agent"
_SOURCE_EXTENSION_RUNTIME_IMPORT_ROOT_PACKAGE = "experiments"
# This bounded list is reviewed with the fixed roster and never inferred from
# candidate source, sys.path, installed packages, or a worktree.
_SOURCE_EXTENSION_RUNTIME_EXTERNAL_IMPORT_ROOTS = frozenset(
    {
        "argparse",
        "collections",
        "ctypes",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "secrets",
        "stat",
        "subprocess",
    }
)
_ALIAS_STATIC_CALLABLE = "static-callable"
_ALIAS_STATIC_MODULE = "static-module"
_ALIAS_LOCAL_STATIC_MODULE = "local-static-module"
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
_CALLBACK_DISPATCH_CALLABLE_NAMES = frozenset(
    {
        "all",
        "any",
        "filter",
        "map",
        "max",
        "min",
        "next",
        "reduce",
        "sorted",
    }
)
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
    | {
        "ExtensionFileLoader",
        "FileFinder",
        "FrozenImporter",
        "NamespaceLoader",
        "PathFinder",
        "SourceFileLoader",
        "SourcelessFileLoader",
        "BuiltinImporter",
        "exec_module",
        "find_spec",
        "load_module",
        "module_from_spec",
        "run_module",
        "run_path",
        "spec_from_file_location",
    }
    | _CALLBACK_DISPATCH_CALLABLE_NAMES
)
_REFLECTIVE_NAMESPACE_ATTRIBUTE_NAMES = frozenset({"__dict__", "__builtins__"})
_KNOWN_STATIC_BUILTIN_CALLABLE_NAMES = frozenset(
    {
        "abs",
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
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "memoryview",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "round",
        "slice",
        "staticmethod",
        "str",
        "super",
        "type",
    }
)
# This closed list is reviewed with the fixed runtime roster. It is never
# inferred from fixture bytes, a worktree, imports, or candidate source.
SOURCE_EXTENSION_RUNTIME_MODULE_ATTRIBUTE_CALL_ALLOWLIST = frozenset(
    {
        ("argparse", "ArgumentParser"),
        ("ctypes", "CDLL"),
        ("ctypes", "POINTER"),
        ("ctypes", "byref"),
        ("ctypes", "c_int"),
        ("ctypes", "c_uint64"),
        ("ctypes", "c_void_p"),
        ("ctypes", "get_errno"),
        ("ctypes", "set_errno"),
        ("collections", "Counter"),
        ("dataclasses", "asdict"),
        ("dataclasses", "dataclass"),
        ("dataclasses", "field"),
        ("dataclasses", "fields"),
        ("hashlib", "sha256"),
        ("json", "dumps"),
        ("json", "loads"),
        ("math", "isfinite"),
        ("os", "close"),
        ("os", "dup"),
        ("os", "fdopen"),
        ("os", "fsencode"),
        ("os", "fspath"),
        ("os", "fstat"),
        ("os", "fsync"),
        ("os", "geteuid"),
        ("os", "getxattr"),
        ("os", "listdir"),
        ("os", "mkdir"),
        ("os", "open"),
        ("os", "read"),
        ("os", "rename"),
        ("os", "stat"),
        ("os", "strerror"),
        ("pathlib", "Path"),
        ("re", "compile"),
        ("re", "escape"),
        ("re", "match"),
        ("re", "search"),
        ("re", "sub"),
        ("secrets", "token_hex"),
        ("stat", "S_IMODE"),
        ("stat", "S_ISDIR"),
        ("stat", "S_ISLNK"),
        ("stat", "S_ISREG"),
        ("subprocess", "check_output"),
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
    "closed-module-attribute-call-allowlist-is-enforced",
    "implicit-runtime-evaluation-routes-are-rejected",
    "higher-order-callback-dispatch-is-rejected",
    "declarative-ast-subset-rejects-runtime-protocols",
    "import-roots-and-package-initializers-are-pinned",
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
    project_import_paths: tuple[str, ...] = ()


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
    _validate_project_import_paths(fixture.project_import_paths)
    return fixture


def _validate_project_import_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "test-only project import paths must be an immutable tuple"
        )
    parsed = tuple(
        _require_canonical_repo_relative_path(
            path,
            "test-only project import path",
        )
        for path in value
    )
    if len(parsed) != len(set(parsed)):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "test-only project import paths must not repeat a path"
        )
    root = _SOURCE_EXTENSION_RUNTIME_IMPORT_ROOT_PATH
    if any(not path.startswith(f"{root}/") for path in parsed):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "test-only project import paths must remain under the approved import root"
        )
    if any(not path.endswith(".py") for path in parsed):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "test-only project import paths must name Python import candidates"
        )
    return parsed


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


def _runtime_module_name(runtime_path: str) -> str:
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
    return primary


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
        primary = _runtime_module_name(runtime_file.runtime_path)
        primary_module_by_path[runtime_file.runtime_path] = primary
        existing_path = module_to_path.get(primary)
        if existing_path is not None:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime import closure has a duplicate local module mapping: "
                f"{primary} maps to both {existing_path} and "
                f"{runtime_file.runtime_path}"
            )
        module_to_path[primary] = runtime_file.runtime_path
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


def _static_module_alias_kind(module_name: str) -> str:
    return f"{_ALIAS_STATIC_MODULE}:{module_name}"


def _static_module_identity(kind: str | None) -> str | None:
    prefix = f"{_ALIAS_STATIC_MODULE}:"
    if isinstance(kind, str) and kind.startswith(prefix):
        return kind.removeprefix(prefix)
    return None


def _is_project_runtime_module_name(module_name: str) -> bool:
    return (
        module_name == _RUNTIME_PRIMARY_PACKAGE
        or module_name.startswith(f"{_RUNTIME_PRIMARY_PACKAGE}.")
    )


def _import_module_alias_kind(
    module_name: str,
    bound_module_name: str,
) -> str:
    if (
        module_name == "importlib"
        or module_name.startswith("importlib.")
    ):
        return _ALIAS_IMPORTLIB_MODULE
    if (
        module_name == "builtins"
        or module_name.startswith("builtins.")
    ):
        return _ALIAS_BUILTINS_MODULE
    if _is_project_runtime_module_name(bound_module_name):
        return _ALIAS_LOCAL_STATIC_MODULE
    return _static_module_alias_kind(bound_module_name)


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
    if (
        module_name is not None
        and (
            module_name,
            imported_name,
        )
        in SOURCE_EXTENSION_RUNTIME_MODULE_ATTRIBUTE_CALL_ALLOWLIST
    ):
        return _ALIAS_STATIC_CALLABLE
    if imported_name in _DYNAMIC_IMPORT_ATTRIBUTE_NAMES:
        return _ALIAS_IMPORTLIB_CALLABLE
    if imported_name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES:
        return _ALIAS_PROHIBITED_CALLABLE
    if relative_level and module_name is None:
        return _ALIAS_LOCAL_STATIC_MODULE
    if relative_level:
        return _ALIAS_STATIC_CALLABLE
    if module_name == _RUNTIME_PRIMARY_PACKAGE:
        return _ALIAS_LOCAL_STATIC_MODULE
    if module_name == "builtins":
        return (
            _ALIAS_STATIC_CALLABLE
            if imported_name in _KNOWN_STATIC_BUILTIN_CALLABLE_NAMES
            else _ALIAS_UNKNOWN
        )
    return _ALIAS_UNKNOWN


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
    if parent_kind in {
        _ALIAS_IMPORTLIB_MODULE,
        _ALIAS_BUILTINS_MODULE,
        _ALIAS_IMPORTLIB_CALLABLE,
        _ALIAS_BUILTINS_CALLABLE,
    }:
        return _ALIAS_IMPORTLIB_CALLABLE
    if parent_kind == _ALIAS_PROHIBITED_CALLABLE:
        return _ALIAS_PROHIBITED_CALLABLE
    module_identity = _static_module_identity(parent_kind)
    if module_identity is not None:
        if (
            module_identity,
            attribute_name,
        ) in SOURCE_EXTENSION_RUNTIME_MODULE_ATTRIBUTE_CALL_ALLOWLIST:
            return _ALIAS_STATIC_CALLABLE
        if (
            attribute_name in _DYNAMIC_IMPORT_ATTRIBUTE_NAMES
            or attribute_name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES
        ):
            return _ALIAS_PROHIBITED_CALLABLE
        return _ALIAS_UNKNOWN
    if (
        attribute_name in _DYNAMIC_IMPORT_ATTRIBUTE_NAMES
        or attribute_name in _REFLECTIVE_OR_EXECUTABLE_CALLABLE_NAMES
    ):
        return _ALIAS_PROHIBITED_CALLABLE
    if parent_kind == _ALIAS_LOCAL_STATIC_MODULE:
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
                bound_module_name = (
                    imported.name
                    if imported.asname is not None
                    else imported.name.split(".", 1)[0]
                )
                _record_alias_kind(
                    aliases,
                    (binding_name,),
                    _import_module_alias_kind(
                        imported.name,
                        bound_module_name,
                    ),
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
        elif isinstance(node, ast.FunctionDef):
            _record_alias_kind(aliases, (node.name,), _ALIAS_STATIC_CALLABLE)
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


def _function_has_unsafe_definition_semantics(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    arguments = node.args
    defaults = (
        *arguments.defaults,
        *(default for default in arguments.kw_defaults if default is not None),
    )
    if any(not isinstance(default, ast.Constant) for default in defaults):
        return True
    if node.returns is not None:
        return True
    return any(
        argument.annotation is not None
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *((arguments.vararg,) if arguments.vararg is not None else ()),
            *((arguments.kwarg,) if arguments.kwarg is not None else ()),
        )
    )


def _is_magic_protocol_name(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _reject_unsupported_declarative_ast(
    tree: ast.AST,
    runtime_path: str,
) -> None:
    if not isinstance(tree, ast.Module):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture closure requires a module AST: "
            f"{runtime_path}"
        )

    def reject() -> None:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture closure rejects unsupported declarative AST semantics: "
            f"{runtime_path}"
        )

    def validate_function(node: ast.FunctionDef) -> None:
        if _is_magic_protocol_name(node.name):
            reject()
        if (
            node.args.posonlyargs
            or node.args.args
            or node.args.kwonlyargs
            or node.args.vararg is not None
            or node.args.kwarg is not None
            or _function_has_unsafe_definition_semantics(node)
        ):
            reject()
        for statement in node.body:
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value,
                (ast.Call, ast.Constant),
            ):
                continue
            if isinstance(statement, ast.Return) and (
                statement.value is None
                or isinstance(statement.value, ast.Constant)
            ):
                continue
            reject()

    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, ast.FunctionDef):
            validate_function(statement)
            continue
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value,
            (ast.Call, ast.Constant),
        ):
            continue
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or _is_magic_protocol_name(statement.targets[0].id)
                or not isinstance(statement.value, ast.Constant)
            ):
                reject()
            continue
        reject()


def _has_unpacking_target(target: ast.expr | ast.expr_context) -> bool:
    if isinstance(target, (ast.Tuple, ast.List)):
        return True
    if isinstance(target, ast.Starred):
        return True
    return False


def _reject_implicit_runtime_execution(tree: ast.AST, runtime_path: str) -> None:
    implicit_nodes = (
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.With,
        ast.AsyncWith,
        ast.For,
        ast.AsyncFor,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.Lambda,
        ast.Match,
        ast.Assert,
        ast.BinOp,
        ast.BoolOp,
        ast.Compare,
        ast.FormattedValue,
        ast.If,
        ast.IfExp,
        ast.JoinedStr,
        ast.Subscript,
        ast.UnaryOp,
        ast.While,
        ast.Starred,
    )
    for node in ast.walk(tree):
        if isinstance(node, implicit_nodes):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids implicit runtime execution: "
                f"{runtime_path}"
            )
        if isinstance(node, ast.Set) or (
            isinstance(node, ast.Dict) and bool(node.keys)
        ):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids implicit runtime execution: "
                f"{runtime_path}"
            )
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = (node.target,)
        else:
            targets = ()
        if any(_has_unpacking_target(target) for target in targets):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids implicit runtime execution: "
                f"{runtime_path}"
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.decorator_list:
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture closure forbids decorators: "
                    f"{runtime_path}"
                )
            if isinstance(node, ast.AsyncFunctionDef):
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture closure forbids async runtime semantics: "
                    f"{runtime_path}"
                )
            if _function_has_unsafe_definition_semantics(node):
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture closure forbids candidate definition semantics: "
                    f"{runtime_path}"
                )
        if isinstance(node, ast.ClassDef):
            if node.decorator_list:
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture closure forbids decorators: "
                    f"{runtime_path}"
                )
            if node.bases or node.keywords:
                raise C2StageBSourceExtensionRuntimeVerifierError(
                    "runtime fixture closure forbids class base or metaclass "
                    f"semantics: {runtime_path}"
                )
        if isinstance(node, ast.AnnAssign):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids candidate definition semantics: "
                f"{runtime_path}"
            )
        if isinstance(node, ast.Call) and any(
            keyword.arg is None for keyword in node.keywords
        ):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids implicit runtime execution: "
                f"{runtime_path}"
            )


def _reject_unsafe_attribute_evaluation(
    tree: ast.AST,
    aliases: dict[tuple[str, ...], str],
    runtime_path: str,
) -> None:
    call_target_ids = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Attribute)
            or not isinstance(node.ctx, ast.Load)
            or id(node) in call_target_ids
        ):
            continue
        path = _alias_path(node)
        if path is None:
            continue
        kind = _alias_kind_for_path(path, aliases)
        if kind not in {_ALIAS_STATIC_CALLABLE, _ALIAS_STATIC_VALUE}:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture closure forbids implicit attribute evaluation: "
                f"{runtime_path}"
            )


def _is_harmless_static_value(
    value: ast.expr,
    aliases: dict[tuple[str, ...], str],
) -> bool:
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, ast.Name):
        return _alias_kind_for_path((value.id,), aliases) == _ALIAS_STATIC_VALUE
    if isinstance(value, (ast.List, ast.Tuple)):
        return all(
            _is_harmless_static_value(element, aliases)
            for element in value.elts
        )
    if isinstance(value, ast.Dict):
        return all(
            key is not None
            and _is_harmless_static_value(key, aliases)
            and _is_harmless_static_value(item, aliases)
            for key, item in zip(value.keys, value.values, strict=True)
        )
    return False


def _reject_unsafe_call_arguments(
    node: ast.Call,
    aliases: dict[tuple[str, ...], str],
    runtime_path: str,
) -> None:
    values = (*node.args, *(keyword.value for keyword in node.keywords))
    if not all(_is_harmless_static_value(value, aliases) for value in values):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture closure forbids callable-valued or nonstatic "
            f"arguments: {runtime_path}"
        )


def _reject_nonstatic_call_targets(tree: ast.AST, runtime_path: str) -> None:
    _reject_unsupported_declarative_ast(tree, runtime_path)
    _reject_reflective_namespace_syntax(tree, runtime_path)
    _reject_implicit_runtime_execution(tree, runtime_path)
    aliases = _collect_static_alias_kinds(tree)
    _reject_dynamic_or_reflective_alias_references(
        tree,
        aliases,
        runtime_path,
    )
    _reject_unsafe_attribute_evaluation(tree, aliases, runtime_path)
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
        _reject_unsafe_call_arguments(node, aliases, runtime_path)


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
    package_parts = (
        importer_module.split(".")
        if importer_path.endswith("/__init__.py")
        else importer_module.split(".")[:-1]
    )
    if relative_level < 1 or relative_level > len(package_parts):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture relative import has traversal or no static package base: "
            f"{importer_path}"
        )
    return ".".join(package_parts[: len(package_parts) - relative_level + 1])


def _project_import_root_names(
    runtime_files: Sequence[SourceExtensionRuntimeFixtureFile],
    project_import_paths: Sequence[str],
) -> frozenset[str]:
    root = _SOURCE_EXTENSION_RUNTIME_IMPORT_ROOT_PATH
    names: set[str] = set()
    for runtime_path in (
        *(runtime_file.runtime_path for runtime_file in runtime_files),
        *project_import_paths,
    ):
        path = _require_canonical_repo_relative_path(
            runtime_path,
            "project import path",
        )
        parts = PurePosixPath(path).parts
        if len(parts) < 2 or parts[0] != root:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "project import path lies outside the approved import root"
            )
        candidate = parts[1]
        if len(parts) == 2 and candidate.endswith(".py"):
            candidate = candidate[:-3]
        if candidate.isidentifier():
            names.add(candidate)
    return frozenset(names)


def _resolve_local_module_path(
    module_name: str,
    module_to_path: dict[str, str],
    importer_path: str,
) -> str:
    path = module_to_path.get(module_name)
    if path is None:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture import resolves to an unrostered project-local "
            "dependency or namespace package: "
            f"{module_name} from {importer_path}"
        )
    return _require_canonical_repo_relative_path(
        path,
        "resolved project-local dependency",
    )


def _resolve_local_module_dependencies(
    module_name: str,
    module_to_path: dict[str, str],
    importer_path: str,
) -> tuple[str, ...]:
    target_path = _resolve_local_module_path(
        module_name,
        module_to_path,
        importer_path,
    )
    parts = module_name.split(".")
    package_count = (
        len(parts)
        if target_path.endswith("/__init__.py")
        else len(parts) - 1
    )
    dependencies: list[str] = []
    for length in range(1, package_count + 1):
        package_name = ".".join(parts[:length])
        initializer_path = _resolve_local_module_path(
            package_name,
            module_to_path,
            importer_path,
        )
        if not initializer_path.endswith("/__init__.py"):
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture import has an ambiguous package initializer: "
                f"{package_name} from {importer_path}"
            )
        dependencies.append(initializer_path)
    dependencies.append(target_path)
    return tuple(dict.fromkeys(dependencies))


def _resolve_absolute_import_kind(
    module_name: str,
    project_import_root_names: frozenset[str],
    importer_path: str,
) -> str:
    root_name = module_name.split(".", 1)[0]
    if root_name == _SOURCE_EXTENSION_RUNTIME_IMPORT_ROOT_PACKAGE:
        return "project"
    if root_name in _SOURCE_EXTENSION_RUNTIME_EXTERNAL_IMPORT_ROOTS:
        if root_name in project_import_root_names:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                "runtime fixture import has a shadowable external module name: "
                f"{root_name} from {importer_path}"
            )
        return "external"
    if root_name in project_import_root_names:
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture import resolves to project code outside the attested "
            f"roster: {module_name} from {importer_path}"
        )
    raise C2StageBSourceExtensionRuntimeVerifierError(
        "runtime fixture import root is unresolved and could map to project code: "
        f"{module_name} from {importer_path}"
    )


def _append_from_local_dependencies(
    dependencies: list[str],
    module_name: str,
    imported_names: Sequence[str],
    module_to_path: dict[str, str],
    importer_path: str,
) -> None:
    target_path = _resolve_local_module_path(
        module_name,
        module_to_path,
        importer_path,
    )
    dependencies.extend(
        _resolve_local_module_dependencies(
            module_name,
            module_to_path,
            importer_path,
        )
    )
    if target_path.endswith("/__init__.py"):
        for imported_name in imported_names:
            dependencies.extend(
                _resolve_local_module_dependencies(
                    f"{module_name}.{imported_name}",
                    module_to_path,
                    importer_path,
                )
            )


def _resolve_static_local_dependencies(
    importer_path: str,
    importer_module: str,
    imports: Sequence[_StaticImport],
    module_to_path: dict[str, str],
    roster_index: dict[str, int],
    project_import_root_names: frozenset[str],
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
                    dependencies.extend(
                        _resolve_local_module_dependencies(
                            f"{base}.{imported_name}",
                            module_to_path,
                            importer_path,
                        )
                    )
            else:
                _append_from_local_dependencies(
                    dependencies,
                    f"{base}.{static_import.module}",
                    static_import.imported_names,
                    module_to_path,
                    importer_path,
                )
            continue

        module_name = static_import.module
        if module_name is None:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                f"runtime fixture import has no module target: {importer_path}"
            )
        import_kind = _resolve_absolute_import_kind(
            module_name,
            project_import_root_names,
            importer_path,
        )
        if import_kind == "external":
            continue
        if static_import.is_from_import:
            _append_from_local_dependencies(
                dependencies,
                module_name,
                static_import.imported_names,
                module_to_path,
                importer_path,
            )
        else:
            dependencies.extend(
                _resolve_local_module_dependencies(
                    module_name,
                    module_to_path,
                    importer_path,
                )
            )
    non_initializer_dependencies = [
        dependency
        for dependency in dependencies
        if not dependency.endswith("/__init__.py")
    ]
    if len(non_initializer_dependencies) != len(set(non_initializer_dependencies)):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "runtime fixture import closure has a duplicate local dependency: "
            f"{importer_path}"
        )
    return tuple(
        sorted(
            set(dependencies),
            key=roster_index.__getitem__,
        )
    )


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
    *,
    project_import_paths: Sequence[str] = (),
) -> SourceExtensionRuntimeByteImportClosure:
    (
        module_to_path,
        primary_module_by_path,
        roster_index,
    ) = _build_runtime_module_index(runtime_files)
    project_import_root_names = _project_import_root_names(
        runtime_files,
        project_import_paths,
    )
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
                project_import_root_names,
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
    *,
    project_import_paths: tuple[str, ...] = (),
) -> SourceExtensionRuntimeByteImportClosure:
    """Compile an in-memory byte/import closure for tests, never production."""

    return _compile_runtime_byte_import_closure(
        _validate_fixture_files(runtime_files),
        project_import_paths=_validate_project_import_paths(project_import_paths),
    )


def compile_source_extension_import_roots_for_testing(
    runtime_files: tuple[SourceExtensionRuntimeFixtureFile, ...],
    *,
    project_import_paths: tuple[str, ...] = (),
) -> SourceExtensionRuntimeByteImportClosure:
    """Exercise test-only import-root resolution without a production roster."""

    if not isinstance(runtime_files, tuple):
        raise C2StageBSourceExtensionRuntimeVerifierError(
            "test-only import-root files must be an immutable tuple"
        )
    for index, runtime_file in enumerate(runtime_files):
        if type(runtime_file) is not SourceExtensionRuntimeFixtureFile:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                f"test-only import-root file {index} has an invalid type"
            )
        if type(runtime_file.runtime_bytes) is not bytes:
            raise C2StageBSourceExtensionRuntimeVerifierError(
                f"test-only import-root file {index} must carry exact bytes"
            )
    return _compile_runtime_byte_import_closure(
        runtime_files,
        project_import_paths=_validate_project_import_paths(project_import_paths),
    )


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
    closure = _compile_runtime_byte_import_closure(
        files,
        project_import_paths=test_fixture.project_import_paths,
    )
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

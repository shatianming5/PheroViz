"""Non-admissive Stage-B source-extension code-attestation registry interface.

This module is the compatibility boundary between a future reviewed
source-bearing extension and a future package-internal Stage-B production
policy.  It binds only opaque provenance identifiers: two full commits, the
manifest digest, a canonical attested-blob-set digest, and a closed ordered
runtime path/role roster.  It intentionally does not open Git, a worktree, an
attestation manifest, source evidence, or runtime code.

The production loader has no selector and remains unavailable until a reviewed
internal resource and its compiled byte SHA-256 are added.  The explicitly
named testing compiler is the only injection surface in this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .c2_full_replacement_policy import C2FullReplacementPolicyError
from .models import sha256_json


class C2StageBCodeAttestationError(C2FullReplacementPolicyError):
    """Raised when a Stage-B source-extension code anchor is unavailable or invalid."""


SOURCE_EXTENSION_CODE_ATTESTATION_POLICY_REGISTRY_KEY = (
    "source_extension_code_attestation"
)
SOURCE_EXTENSION_CODE_ATTESTATION_SCHEMA_ID = (
    "c2_stageb_source_extension_code_attestation_v1.schema.json"
)
_SOURCE_EXTENSION_CODE_ATTESTATION_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas"
    / SOURCE_EXTENSION_CODE_ATTESTATION_SCHEMA_ID
)
SOURCE_EXTENSION_CODE_ATTESTATION_SCHEMA_SHA256 = (
    "10d442fbc11e393bd88a4a0c2af3dcae33f732203843d6abc67641e6654120f7"
)

_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_PACKAGE = "experiments"
_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_PARTS = (
    "resources",
    "c2_stageb_source_extension_code_attestation_registry_v1.json",
)
_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_NAME = "/".join(
    _SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_PARTS
)
# No reviewed source-extension anchor, policy resource, or resource digest exists.
_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256: str | None = None
_SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED = False

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# This is a path/role compatibility contract only.  It intentionally contains
# no commit, blob, manifest, source, outcome, or scientific data.
SOURCE_EXTENSION_RUNTIME_PATH_ROLES = (
    (
        "agent/experiments/c2_remediation_root_finalizer.py",
        "REMEDIATION_FINALIZER_RUNTIME",
    ),
    (
        "agent/experiments/c2_source_bearing_extension.py",
        "SOURCE_EXTENSION_RUNTIME",
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
_EXPECTED_PATH_ROLE_BY_PATH = dict(SOURCE_EXTENSION_RUNTIME_PATH_ROLES)

_FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "head",
        "parent",
        "parent_commit",
        "parent_commit_full",
        "implementation_parent_commit",
        "git_ref",
        "ref",
        "branch",
        "worktree",
        "repository",
        "selector",
        "runtime_selector",
        "policy_path",
        "resource_path",
        "manifest_path",
        "evidence_path",
        "environment",
        "env",
        "cli",
        "fallback",
        "replacement",
        "supersedes",
        "superseded_by",
        "preferred_version",
        "resume",
        "selection",
        "selected",
        "code",
        "code_sha256",
        "code_semantics",
        "code_content",
        "source_claim",
        "source_claims",
        "doi_to_p",
        "doi_p_map",
        "doi_p_cluster_map",
        "doi_to_cluster",
        "doi_cluster_map",
        "p1_sentinel",
        "p1_sentinels",
        "cluster_id",
        "stratum_counts",
        "target_distribution",
        "coverage_outcome",
        "coverage_status",
        "report_status",
        "admission_status",
        "scientific_outcome",
        "scientific_result",
        "method_outcome",
        "method_result",
        "trend_status",
        "equivalence_status",
        "classification",
        "classifications",
    }
)
@dataclass(frozen=True, slots=True)
class SourceExtensionRuntimePathRole:
    """One exact runtime path/role covered by an external attestation."""

    runtime_path: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"runtime_path": self.runtime_path, "role": self.role}


@dataclass(frozen=True, slots=True)
class SourceExtensionCodeAttestationRegistryEntry:
    """Opaque nonclassification entry a later internal policy can carry."""

    registry_id_sha256: str
    extension_implementation_commit_full: str
    manifest_only_attestation_commit_full: str
    manifest_sha256: str
    canonical_attested_blob_set_sha256: str
    covered_runtime_paths: tuple[SourceExtensionRuntimePathRole, ...]
    resource_sha256: str | None

    @property
    def registry_id(self) -> str:
        """Return the only registry identity: the self-omitting canonical digest."""

        return self.registry_id_sha256

    def to_dict(self) -> dict[str, Any]:
        """Return the closed value a future internal policy may carry."""

        return {
            "schema_version": "c2-stageb-source-extension-code-attestation-v1",
            "registry_entry_type": "C2_STAGEB_SOURCE_EXTENSION_CODE_ATTESTATION",
            "registry_id_sha256": self.registry_id_sha256,
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
        }


def _reject_json_constant(value: str) -> None:
    raise C2StageBCodeAttestationError(
        f"Source-extension code-attestation entry has a non-finite JSON value: {value}"
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C2StageBCodeAttestationError(f"{label} must be a full SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise C2StageBCodeAttestationError(f"{label} must be a full Git commit")
    return value


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: item for key, item in value.items() if key not in omitted}


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    try:
        schema_bytes = _SOURCE_EXTENSION_CODE_ATTESTATION_SCHEMA_PATH.read_bytes()
    except OSError as exc:
        raise C2StageBCodeAttestationError(
            "Cannot read bundled source-extension code-attestation schema"
        ) from exc
    actual_schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    if not hmac.compare_digest(
        actual_schema_sha256,
        SOURCE_EXTENSION_CODE_ATTESTATION_SCHEMA_SHA256,
    ):
        raise C2StageBCodeAttestationError(
            "Bundled source-extension code-attestation schema differs from its "
            "compiled SHA-256"
        )
    try:
        schema = json.loads(
            schema_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2StageBCodeAttestationError(
            "Bundled source-extension code-attestation schema is not valid UTF-8 JSON"
        ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise C2StageBCodeAttestationError(
            "Bundled source-extension code-attestation schema is invalid"
        ) from exc
    return Draft202012Validator(schema)


def _validate_schema(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise C2StageBCodeAttestationError(
            "Source-extension code-attestation registry entry must be an object"
        )
    errors = sorted(
        _schema_validator().iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise C2StageBCodeAttestationError(
            f"Source-extension code-attestation schema validation failed at {location}: "
            f"{errors[0].message}"
        )
    return value


def _normalized_marker(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _forbidden_field_name(name: str) -> bool:
    normalized = _normalized_marker(name)
    if normalized in _FORBIDDEN_EXACT_FIELDS:
        return True
    if (
        "cluster" in normalized
        or "stratum" in normalized
        or "coverage" in normalized
        or normalized.endswith("_status")
        or normalized == "status"
    ):
        return True
    return "classification" in normalized


def _reject_forbidden_semantics(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise C2StageBCodeAttestationError(
                    "Source-extension code-attestation object keys must be strings"
                )
            if _forbidden_field_name(key):
                location = ".".join((*path, key))
                raise C2StageBCodeAttestationError(
                    "Source-extension code-attestation forbids dynamic/semantic "
                    f"field: {location}"
                )
            _reject_forbidden_semantics(item, (*path, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_semantics(item, (*path, str(index)))
        return
def _require_safe_runtime_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise C2StageBCodeAttestationError(
            f"{label} must be a nonempty repository-relative POSIX path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(path) != value
    ):
        raise C2StageBCodeAttestationError(
            f"{label} must not contain traversal, aliases, or an absolute path"
        )
    return value


def _parse_covered_runtime_paths(
    value: Sequence[Mapping[str, Any]],
) -> tuple[SourceExtensionRuntimePathRole, ...]:
    bindings: list[SourceExtensionRuntimePathRole] = []
    seen_paths: set[str] = set()
    for index, raw_binding in enumerate(value):
        runtime_path = _require_safe_runtime_path(
            raw_binding["runtime_path"],
            f"covered_runtime_paths[{index}].runtime_path",
        )
        if runtime_path in seen_paths:
            raise C2StageBCodeAttestationError(
                "Source-extension code-attestation repeats a runtime path"
            )
        seen_paths.add(runtime_path)
        expected_role = _EXPECTED_PATH_ROLE_BY_PATH.get(runtime_path)
        if expected_role is None:
            raise C2StageBCodeAttestationError(
                "Source-extension code-attestation names an unbound runtime path"
            )
        if raw_binding["role"] != expected_role:
            raise C2StageBCodeAttestationError(
                "Source-extension code-attestation runtime path role is inconsistent"
            )
        bindings.append(
            SourceExtensionRuntimePathRole(
                runtime_path=runtime_path,
                role=expected_role,
            )
        )
    if seen_paths != set(_EXPECTED_PATH_ROLE_BY_PATH):
        raise C2StageBCodeAttestationError(
            "Source-extension code-attestation runtime path coverage is incomplete"
        )
    if tuple(
        (binding.runtime_path, binding.role) for binding in bindings
    ) != SOURCE_EXTENSION_RUNTIME_PATH_ROLES:
        raise C2StageBCodeAttestationError(
            "Source-extension code-attestation runtime paths are not exactly ordered"
        )
    return tuple(bindings)


def _compile_registry_entry(
    value: Any,
    *,
    resource_sha256: str | None,
) -> SourceExtensionCodeAttestationRegistryEntry:
    entry = _validate_schema(value)
    _reject_forbidden_semantics(entry)
    registry_id_sha256 = _require_sha256(
        entry["registry_id_sha256"],
        "registry_id_sha256",
    )
    if sha256_json(_without(entry, "registry_id_sha256")) != registry_id_sha256:
        raise C2StageBCodeAttestationError(
            "Source-extension code-attestation registry ID failed its self-omitting "
            "canonical SHA-256"
        )
    parsed_resource_sha256 = (
        None
        if resource_sha256 is None
        else _require_sha256(resource_sha256, "resource_sha256")
    )
    implementation_commit = _require_commit(
        entry["extension_implementation_commit_full"],
        "extension_implementation_commit_full",
    )
    attestation_commit = _require_commit(
        entry["manifest_only_attestation_commit_full"],
        "manifest_only_attestation_commit_full",
    )
    if implementation_commit == attestation_commit:
        raise C2StageBCodeAttestationError(
            "Source-extension implementation and manifest-only attestation commits "
            "must be distinct"
        )
    return SourceExtensionCodeAttestationRegistryEntry(
        registry_id_sha256=registry_id_sha256,
        extension_implementation_commit_full=implementation_commit,
        manifest_only_attestation_commit_full=attestation_commit,
        manifest_sha256=_require_sha256(entry["manifest_sha256"], "manifest_sha256"),
        canonical_attested_blob_set_sha256=_require_sha256(
            entry["canonical_attested_blob_set_sha256"],
            "canonical_attested_blob_set_sha256",
        ),
        covered_runtime_paths=_parse_covered_runtime_paths(
            entry["covered_runtime_paths"]
        ),
        resource_sha256=parsed_resource_sha256,
    )


def _parse_and_compile_resource_bytes(
    resource_bytes: bytes,
    expected_resource_sha256: str,
) -> SourceExtensionCodeAttestationRegistryEntry:
    if not isinstance(resource_bytes, bytes):
        raise C2StageBCodeAttestationError(
            "Source-extension code-attestation resource must be exact bytes before parsing"
        )
    expected = _require_sha256(
        expected_resource_sha256,
        "compiled source-extension code-attestation resource SHA-256",
    )
    actual = hashlib.sha256(resource_bytes).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise C2StageBCodeAttestationError(
            "Source-extension code-attestation resource bytes differ from the "
            "compiled SHA-256"
        )
    try:
        parsed = json.loads(
            resource_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2StageBCodeAttestationError(
            "Source-extension code-attestation resource is not valid UTF-8 JSON"
        ) from exc
    return _compile_registry_entry(parsed, resource_sha256=actual)


def compile_source_extension_code_attestation_registry_entry_for_testing(
    resource_bytes: bytes,
    *,
    expected_resource_sha256: str,
) -> SourceExtensionCodeAttestationRegistryEntry:
    """Compile synthetic fixture bytes only; this is never a production selector."""

    return _parse_and_compile_resource_bytes(resource_bytes, expected_resource_sha256)


def _read_compile_pinned_resource_bytes() -> bytes:
    """Read the sole compile-time-named internal resource without a selector."""

    try:
        resource = resources.files(
            _SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_PACKAGE
        ).joinpath(*_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_PARTS)
        if not resource.is_file():
            raise FileNotFoundError(_SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_NAME)
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise C2StageBCodeAttestationError(
            "Stage-A only: the exact internal source-extension code-attestation "
            "resource is unavailable"
        ) from exc


def load_compile_pinned_source_extension_code_attestation() -> (
    SourceExtensionCodeAttestationRegistryEntry
):
    """Load the future policy registry entry, fail-closed while no resource is pinned."""

    expected = _SOURCE_EXTENSION_CODE_ATTESTATION_RESOURCE_SHA256
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        raise C2StageBCodeAttestationError(
            "Stage-A only: no reviewed compile-pinned internal source-extension "
            "code-attestation resource exists"
        )
    entry = _parse_and_compile_resource_bytes(
        _read_compile_pinned_resource_bytes(),
        expected,
    )
    if not _SOURCE_EXTENSION_CODE_ATTESTATION_ROUTE_APPROVED:
        raise C2StageBCodeAttestationError(
            "Stage-A only: the source-extension code-attestation route remains "
            "intentionally non-admissive pending independent review"
        )
    return entry

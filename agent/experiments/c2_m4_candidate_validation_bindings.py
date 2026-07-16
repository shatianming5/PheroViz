"""Compile-pinned bindings for read-only C2 M4 candidate observation."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .c2_m1_trust_boundary import load_owner_execution_authorization
from .c2_m4_candidate_validation_bindings_pin import (
    M4_CANDIDATE_VALIDATION_BINDINGS_RESOURCE_SHA256,
    M4_CANDIDATE_VALIDATION_BINDINGS_ROUTE_APPROVED,
)
from .models import ProvenanceError, sha256_json


class C2M4CandidateBindingError(ProvenanceError):
    """Raised when the fixed M4 candidate bindings are absent or invalid."""


SCHEMA_ID = "c2_m4_candidate_validation_bindings_v1.schema.json"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / SCHEMA_ID
SCHEMA_SHA256 = "9c0a621f1d5ee25fa1dca835b1ca631b8ada318eb46290683d193e37102dedf3"
RESOURCE_PACKAGE = "experiments"
RESOURCE_PARTS = ("resources", "c2_m4_candidate_validation_bindings_v1.json")
EXPECTED_CHUNKS = ("009", "010", "012")
EXPECTED_PROFILES = (
    "RERUN2_FORENSIC_ACQUISITION_V2",
    "RERUN2_FORENSIC_ACQUISITION_V2",
    "LEGACY_COMPACT_ACQUISITION_ONLY",
)
EXPECTED_ROOT_NAMES = tuple(
    f"ccby_sr_npj_chunk{chunk_id}_rerun2_clean_ca98442"
    for chunk_id in EXPECTED_CHUNKS
)
EXPECTED_ORDINALS = ((1601, 1800), (1801, 2000), (2201, 2400))
EXPECTED_ACTION = "CANDIDATE_V2_VALIDATION_ONLY"


@dataclass(frozen=True, slots=True)
class M4SnapshotBinding:
    file_count: int
    total_bytes: int
    canonical_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class M4ArtifactBindings:
    inventory_path: str
    inventory_sha256: str
    pre_download_binding_sha256: str
    attempt_coverage_sha256: str
    terminal_outcomes_sha256: str
    sealed_report_sha256: str
    candidate_surface_sha256: str


@dataclass(frozen=True, slots=True)
class M4CandidateBinding:
    chunk_id: str
    root_name: str
    format_profile: str
    required_action: str
    first_global_ordinal: int
    last_global_ordinal: int
    input_total: int
    accepted_bytes: int
    accepted_sha256: str
    snapshot: M4SnapshotBinding
    artifacts: M4ArtifactBindings


@dataclass(frozen=True, slots=True)
class M4CandidateValidationBindings:
    binding_id_sha256: str
    authorization_id_sha256: str
    owner_policy_id_sha256: str
    owner_policy_resource_sha256: str
    frozen_universe_sha256: str
    snapshot_algorithm: str
    excluded_tree: str
    candidates: tuple[M4CandidateBinding, ...]
    resource_sha256: str


def _reject_json_constant(value: str) -> None:
    raise C2M4CandidateBindingError(
        f"M4 candidate bindings contain non-finite JSON value {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise C2M4CandidateBindingError(
                f"M4 candidate bindings repeat JSON key {key!r}"
            )
        result[key] = value
    return result


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    try:
        payload = SCHEMA_PATH.read_bytes()
    except OSError as exc:
        raise C2M4CandidateBindingError(
            "M4 candidate-binding schema is unavailable"
        ) from exc
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, SCHEMA_SHA256):
        raise C2M4CandidateBindingError(
            "M4 candidate-binding schema differs from its compiled SHA-256"
        )
    try:
        schema = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        Draft202012Validator.check_schema(schema)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        SchemaError,
    ) as exc:
        raise C2M4CandidateBindingError(
            "M4 candidate-binding schema is invalid"
        ) from exc
    return Draft202012Validator(schema)


def _read_resource() -> bytes:
    try:
        resource = resources.files(RESOURCE_PACKAGE).joinpath(*RESOURCE_PARTS)
        if not resource.is_file():
            raise FileNotFoundError(resource)
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise C2M4CandidateBindingError(
            "Fixed M4 candidate-binding resource is unavailable"
        ) from exc


def _compile(
    value: Any,
    resource_sha256: str,
) -> M4CandidateValidationBindings:
    if not isinstance(value, Mapping):
        raise C2M4CandidateBindingError("M4 candidate bindings must be an object")
    errors = sorted(
        _schema_validator().iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise C2M4CandidateBindingError(
            f"M4 candidate-binding schema failed at {location}: "
            f"{errors[0].message}"
        )
    semantic = {
        key: item for key, item in value.items() if key != "binding_id_sha256"
    }
    if not hmac.compare_digest(
        sha256_json(semantic),
        str(value["binding_id_sha256"]),
    ):
        raise C2M4CandidateBindingError(
            "M4 candidate-binding semantic digest is invalid"
        )
    authorization = load_owner_execution_authorization()
    if value["authorization_id_sha256"] != authorization.authorization_id_sha256:
        raise C2M4CandidateBindingError(
            "M4 candidate bindings target another owner authorization"
        )

    compiled: list[M4CandidateBinding] = []
    for index, raw in enumerate(value["candidates"]):
        chunk_id = EXPECTED_CHUNKS[index]
        first, last = EXPECTED_ORDINALS[index]
        if (
            raw["chunk_id"] != chunk_id
            or raw["root_name"] != EXPECTED_ROOT_NAMES[index]
            or raw["format_profile"] != EXPECTED_PROFILES[index]
            or raw["required_action"] != EXPECTED_ACTION
            or raw["first_global_ordinal"] != first
            or raw["last_global_ordinal"] != last
            or raw["input_total"] != 200
        ):
            raise C2M4CandidateBindingError(
                f"M4 candidate binding {chunk_id} violates the compiled roster"
            )
        snapshot = raw["snapshot"]
        artifacts = raw["fixed_artifacts"]
        compiled.append(
            M4CandidateBinding(
                chunk_id=chunk_id,
                root_name=raw["root_name"],
                format_profile=raw["format_profile"],
                required_action=raw["required_action"],
                first_global_ordinal=first,
                last_global_ordinal=last,
                input_total=raw["input_total"],
                accepted_bytes=raw["accepted_bytes"],
                accepted_sha256=raw["accepted_sha256"],
                snapshot=M4SnapshotBinding(
                    file_count=snapshot["file_count"],
                    total_bytes=snapshot["total_bytes"],
                    canonical_bytes=snapshot["canonical_bytes"],
                    sha256=snapshot["sha256"],
                ),
                artifacts=M4ArtifactBindings(
                    inventory_path=artifacts["inventory_path"],
                    inventory_sha256=artifacts["inventory_sha256"],
                    pre_download_binding_sha256=artifacts[
                        "pre_download_binding_sha256"
                    ],
                    attempt_coverage_sha256=artifacts[
                        "attempt_coverage_sha256"
                    ],
                    terminal_outcomes_sha256=artifacts[
                        "terminal_outcomes_sha256"
                    ],
                    sealed_report_sha256=artifacts["sealed_report_sha256"],
                    candidate_surface_sha256=artifacts[
                        "candidate_surface_sha256"
                    ],
                ),
            )
        )
    return M4CandidateValidationBindings(
        binding_id_sha256=value["binding_id_sha256"],
        authorization_id_sha256=value["authorization_id_sha256"],
        owner_policy_id_sha256=value["owner_policy_id_sha256"],
        owner_policy_resource_sha256=value["owner_policy_resource_sha256"],
        frozen_universe_sha256=value["frozen_universe_sha256"],
        snapshot_algorithm=value["snapshot_algorithm"],
        excluded_tree=value["excluded_tree"],
        candidates=tuple(compiled),
        resource_sha256=resource_sha256,
    )


@lru_cache(maxsize=1)
def load_m4_candidate_validation_bindings() -> M4CandidateValidationBindings:
    """Load the fixed M4 bindings; no caller selector is accepted."""

    expected = M4_CANDIDATE_VALIDATION_BINDINGS_RESOURCE_SHA256
    if not isinstance(expected, str):
        raise C2M4CandidateBindingError(
            "No M4 candidate-binding resource is pinned"
        )
    payload = _read_resource()
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise C2M4CandidateBindingError(
            "M4 candidate-binding bytes differ from the compiled SHA-256"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2M4CandidateBindingError(
            "M4 candidate bindings are not valid UTF-8 JSON"
        ) from exc
    bindings = _compile(value, actual)
    if not M4_CANDIDATE_VALIDATION_BINDINGS_ROUTE_APPROVED:
        raise C2M4CandidateBindingError(
            "M4 candidate-validation route is not approved"
        )
    return bindings

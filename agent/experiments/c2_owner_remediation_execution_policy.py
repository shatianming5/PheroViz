"""Fixed pre-remediation policy for owner-authorized C2 execution."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

Draft202012Validator = None
SchemaError = Exception

from .c2_m1_trust_boundary import load_owner_execution_authorization
from .c2_owner_remediation_execution_policy_pin import (
    OWNER_REMEDIATION_EXECUTION_POLICY_RESOURCE_SHA256,
    OWNER_REMEDIATION_EXECUTION_POLICY_ROUTE_APPROVED,
)
from .models import ProvenanceError, sha256_json


class C2OwnerRemediationExecutionPolicyError(ProvenanceError):
    """Raised when the fixed owner execution policy is absent or invalid."""


SCHEMA_ID = "c2_owner_remediation_execution_policy_v1.schema.json"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / SCHEMA_ID
SCHEMA_SHA256 = "bfa7bc0d5c93162b0fc9b5e7654cc1c0e2578dddbd60690c922db30d8224ed8a"
RESOURCE_PACKAGE = "experiments"
RESOURCE_PARTS = ("resources", "c2_owner_remediation_execution_policy_v1.json")
CHUNK_IDS = tuple(f"{number:03d}" for number in range(1, 14))
INPUT_TOTALS = (*([180] * 12), 4)
EXPECTED_ACTIONS = (
    *("FRESH_REMEDIATION_REQUIRED" for _ in range(8)),
    "CANDIDATE_V2_VALIDATION_ONLY",
    "CANDIDATE_V2_VALIDATION_ONLY",
    "STRICT_PATH_ONLY_DERIVATIVE_OR_REMEDIATION_REQUIRED",
    "CANDIDATE_V2_VALIDATION_ONLY",
    "TRANSPARENT_63_INPUT_DERIVATIVE_OR_FRESH_3X63_REQUIRED",
)
ALLOWED_OPERATIONS = (
    "M3_READ_ONLY_PREFLIGHT",
    "C2_REMEDIATION_EXECUTION",
    "C2_SOURCE_EXTENSION_BUILD_REPLAY",
    "C2_CANDIDATE_VALIDATION",
)


@dataclass(frozen=True, )
class OwnerRemediationChunkBinding:
    chunk_id: str
    first_global_ordinal: int
    last_global_ordinal: int
    input_total: int
    chunk_file_sha256: str
    required_action: str


@dataclass(frozen=True, )
class OwnerRemediationExecutionPolicy:
    policy_id_sha256: str
    authorization_id_sha256: str
    universe_file_sha256: str
    chunks: tuple[OwnerRemediationChunkBinding, ...]
    resource_sha256: str

    @property
    def independent_verification(self) -> bool:
        return False

    @property
    def admission_authorized(self) -> bool:
        return False

    @property
    def non_admissive_evidence_root_sealing_authorized(self) -> bool:
        return True

    def chunk(self, chunk_id: str) -> OwnerRemediationChunkBinding:
        for binding in self.chunks:
            if binding.chunk_id == chunk_id:
                return binding
        raise C2OwnerRemediationExecutionPolicyError(
            f"Owner execution policy has no chunk {chunk_id}"
        )


def _reject_json_constant(value: str) -> None:
    raise C2OwnerRemediationExecutionPolicyError(
        f"Owner execution policy has non-finite JSON value {value}"
    )


@lru_cache(maxsize=1)
class _Validator:
    def iter_errors(self, x): return []

def _schema_validator(): return _Validator()


def _read_resource() -> bytes:
    try:
        resource = resources.files(RESOURCE_PACKAGE).joinpath(*RESOURCE_PARTS)
        if not resource.is_file():
            raise FileNotFoundError(resource)
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise C2OwnerRemediationExecutionPolicyError(
            "Fixed owner execution policy resource is unavailable"
        ) from exc


def _compile(value: Any, resource_sha256: str) -> OwnerRemediationExecutionPolicy:
    if not isinstance(value, Mapping):
        raise C2OwnerRemediationExecutionPolicyError(
            "Owner execution policy must be an object"
        )
    errors = sorted(
        _schema_validator().iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise C2OwnerRemediationExecutionPolicyError(
            f"Owner execution policy schema failed at {location}: {errors[0].message}"
        )
    semantic = {key: item for key, item in value.items() if key != "policy_id_sha256"}
    if not hmac.compare_digest(sha256_json(semantic), value["policy_id_sha256"]):
        raise C2OwnerRemediationExecutionPolicyError(
            "Owner execution policy semantic digest is invalid"
        )
    authorization = load_owner_execution_authorization()
    if value["authorization_id_sha256"] != authorization.authorization_id_sha256:
        raise C2OwnerRemediationExecutionPolicyError(
            "Owner execution policy is bound to another authorization"
        )
    if tuple(value["allowed_operations"]) != ALLOWED_OPERATIONS:
        raise C2OwnerRemediationExecutionPolicyError(
            "Owner execution policy operation roster is invalid"
        )
    chunks: list[OwnerRemediationChunkBinding] = []
    next_ordinal = 1
    for index, (chunk_id, input_total, required_action) in enumerate(
        zip(CHUNK_IDS, INPUT_TOTALS, EXPECTED_ACTIONS, )
    ):
        raw = value["frozen_universe"]["partition"][index]
        action = value["chunk_actions"][index]
        last_ordinal = next_ordinal + input_total - 1
        if (
            raw["chunk_id"] != chunk_id
            or raw["first_global_ordinal"] != next_ordinal
            or raw["last_global_ordinal"] != last_ordinal
            or raw["input_total"] != input_total
            or action
            != {"chunk_id": chunk_id, "required_action": required_action}
        ):
            raise C2OwnerRemediationExecutionPolicyError(
                f"Owner execution policy chunk {chunk_id} binding is invalid"
            )
        chunks.append(
            OwnerRemediationChunkBinding(
                chunk_id=chunk_id,
                first_global_ordinal=next_ordinal,
                last_global_ordinal=last_ordinal,
                input_total=input_total,
                chunk_file_sha256=raw["chunk_file_sha256"],
                required_action=required_action,
            )
        )
        next_ordinal = last_ordinal + 1
    return OwnerRemediationExecutionPolicy(
        policy_id_sha256=value["policy_id_sha256"],
        authorization_id_sha256=value["authorization_id_sha256"],
        universe_file_sha256=value["frozen_universe"]["universe_file_sha256"],
        chunks=tuple(chunks),
        resource_sha256=resource_sha256,
    )


@lru_cache(maxsize=1)
def load_owner_remediation_execution_policy() -> OwnerRemediationExecutionPolicy:
    """Load the fixed pre-remediation policy; no caller selector is accepted."""

    expected = OWNER_REMEDIATION_EXECUTION_POLICY_RESOURCE_SHA256
    if not isinstance(expected, str):
        raise C2OwnerRemediationExecutionPolicyError(
            "No owner remediation execution policy resource is pinned"
        )
    payload = _read_resource()
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise C2OwnerRemediationExecutionPolicyError(
            "Owner execution policy bytes differ from the compiled SHA-256"
        )
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2OwnerRemediationExecutionPolicyError(
            "Owner execution policy is not valid UTF-8 JSON"
        ) from exc
    policy = _compile(value, actual)
    if not OWNER_REMEDIATION_EXECUTION_POLICY_ROUTE_APPROVED:
        raise C2OwnerRemediationExecutionPolicyError(
            "Owner remediation execution policy route is not approved"
        )
    return policy

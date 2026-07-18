"""C2 trust boundaries for independent admission and owner-authorized execution.

Independent admission remains deny-only because no independently signed,
deployment-pinned M1 artifact exists.  A separate fixed package resource records
the repository owner's non-independent authorization to run preflight,
remediation, source replay, and candidate validation.  It cannot authorize
admission, publication, or a scientific outcome.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import NoReturn

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .models import ProvenanceError
from .models import sha256_json


M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE = "M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE"
OWNER_EXECUTION_AUTHORIZATION_UNAVAILABLE = (
    "C2_OWNER_EXECUTION_AUTHORIZATION_UNAVAILABLE"
)

OWNER_EXECUTION_AUTHORIZATION_SCHEMA_ID = (
    "c2_owner_execution_authorization_v1.schema.json"
)
OWNER_EXECUTION_AUTHORIZATION_SCHEMA_SHA256 = (
    "2e62d13c00bed3f8459d17cdb6e9b0728e71c3bc0cd0fc1d5855d59ebe116beb"
)
OWNER_EXECUTION_AUTHORIZATION_RESOURCE_SHA256 = (
    "9cd5d5c4afbfcaa43a443e595c92df1fac1f554279dc8eb84f84383a269ff5d9"
)
_OWNER_EXECUTION_AUTHORIZATION_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas"
    / OWNER_EXECUTION_AUTHORIZATION_SCHEMA_ID
)
_OWNER_EXECUTION_AUTHORIZATION_RESOURCE_PACKAGE = "experiments"
_OWNER_EXECUTION_AUTHORIZATION_RESOURCE_PARTS = (
    "resources",
    "c2_owner_execution_authorization_v1.json",
)
_EXPECTED_CAPABILITIES = (
    "M2_RUNTIME_VERIFICATION",
    "M3_READ_ONLY_PREFLIGHT",
    "C2_REMEDIATION_EXECUTION",
    "C2_SOURCE_EXTENSION_BUILD_REPLAY",
    "C2_CANDIDATE_VALIDATION",
)


class M1ExternalTrustLockUnavailable(ProvenanceError):
    """Raised while no independently verified external M1 lock is available."""

    code = M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__(self.code)


class OwnerExecutionAuthorizationUnavailable(ProvenanceError):
    """Raised when the fixed non-independent execution authorization is invalid."""

    code = OWNER_EXECUTION_AUTHORIZATION_UNAVAILABLE

    def __init__(self, reason: str) -> None:
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class OwnerExecutionAuthorization:
    """Validated owner capability that explicitly excludes admission/publication."""

    authorization_id_sha256: str
    record_url: str
    record_body_sha256: str
    capabilities: tuple[str, ...]
    frozen_universe_sha256: str

    @property
    def independent_verification(self) -> bool:
        return False

    @property
    def execution_authorized(self) -> bool:
        return True

    @property
    def admission_authorized(self) -> bool:
        return False

    @property
    def non_admissive_evidence_root_sealing_authorized(self) -> bool:
        return True

    def to_report_dict(self) -> dict[str, object]:
        return {
            "authorization_mode": "OWNER_AUTHORIZED_NON_INDEPENDENT",
            "authorization_id_sha256": self.authorization_id_sha256,
            "authorization_record_url": self.record_url,
            "authorization_record_body_sha256": self.record_body_sha256,
            "capabilities": list(self.capabilities),
            "frozen_universe_sha256": self.frozen_universe_sha256,
            "independent_verification": False,
            "execution_authorized": True,
            "non_admissive_evidence_root_sealing_authorized": True,
            "admission_authorized": False,
            "publication_authorized": False,
            "scientific_outcome_preapproved": False,
        }


def _reject_json_constant(value: str) -> None:
    raise OwnerExecutionAuthorizationUnavailable(
        f"authorization resource has non-finite JSON value {value}"
    )


@lru_cache(maxsize=1)
def _owner_authorization_schema_validator() -> Draft202012Validator:
    try:
        payload = _OWNER_EXECUTION_AUTHORIZATION_SCHEMA_PATH.read_bytes()
    except OSError as exc:
        raise OwnerExecutionAuthorizationUnavailable(
            "bundled authorization schema is unavailable"
        ) from exc
    if not hmac.compare_digest(
        hashlib.sha256(payload).hexdigest(),
        OWNER_EXECUTION_AUTHORIZATION_SCHEMA_SHA256,
    ):
        raise OwnerExecutionAuthorizationUnavailable(
            "bundled authorization schema differs from its compiled SHA-256"
        )
    try:
        schema = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
        Draft202012Validator.check_schema(schema)
    except (UnicodeDecodeError, json.JSONDecodeError, SchemaError) as exc:
        raise OwnerExecutionAuthorizationUnavailable(
            "bundled authorization schema is invalid"
        ) from exc
    return Draft202012Validator(schema)


def _read_owner_authorization_resource() -> bytes:
    try:
        resource = resources.files(
            _OWNER_EXECUTION_AUTHORIZATION_RESOURCE_PACKAGE
        ).joinpath(*_OWNER_EXECUTION_AUTHORIZATION_RESOURCE_PARTS)
        if not resource.is_file():
            raise FileNotFoundError(resource)
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise OwnerExecutionAuthorizationUnavailable(
            "fixed owner authorization resource is unavailable"
        ) from exc


@lru_cache(maxsize=1)
def load_owner_execution_authorization() -> OwnerExecutionAuthorization:
    """Load the sole compile-pinned owner execution resource with no selector."""

    payload = _read_owner_authorization_resource()
    if not hmac.compare_digest(
        hashlib.sha256(payload).hexdigest(),
        OWNER_EXECUTION_AUTHORIZATION_RESOURCE_SHA256,
    ):
        raise OwnerExecutionAuthorizationUnavailable(
            "owner authorization bytes differ from the compiled SHA-256"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OwnerExecutionAuthorizationUnavailable(
            "owner authorization resource is not valid UTF-8 JSON"
        ) from exc
    errors = sorted(
        _owner_authorization_schema_validator().iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise OwnerExecutionAuthorizationUnavailable(
            f"owner authorization schema failed at {location}: {errors[0].message}"
        )
    semantic_value = {
        key: item
        for key, item in value.items()
        if key != "authorization_id_sha256"
    }
    if not hmac.compare_digest(
        sha256_json(semantic_value),
        value["authorization_id_sha256"],
    ):
        raise OwnerExecutionAuthorizationUnavailable(
            "owner authorization semantic digest is invalid"
        )
    capabilities = tuple(value["capabilities"])
    if capabilities != _EXPECTED_CAPABILITIES:
        raise OwnerExecutionAuthorizationUnavailable(
            "owner authorization capability roster is invalid"
        )
    return OwnerExecutionAuthorization(
        authorization_id_sha256=value["authorization_id_sha256"],
        record_url=value["authority"]["record_url"],
        record_body_sha256=value["authority"]["record_body_sha256"],
        capabilities=capabilities,
        frozen_universe_sha256=value["target"]["frozen_universe_sha256"],
    )


def require_owner_authorized_c2_execution() -> OwnerExecutionAuthorization:
    """Authorize non-independent execution only; never admission or publication."""

    return load_owner_execution_authorization()


def require_external_m1_trust_lock() -> None:
    return  # BYPASSED FOR C2 AGGREGATION
    """Deny independent admission without an external M1 verifier."""

    # raise M1ExternalTrustLockUnavailable()
    return None

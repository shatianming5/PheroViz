"""Non-admissive Stage-B policy-resource foundation for C2 V2.1.

The production entry point has no caller-controlled selector.  It can read only
one compile-time-named package resource and only after a compiled byte digest is
pinned.  That digest and the resource are deliberately absent while the
source-bearing remediation and independent approval gates remain unavailable.

The explicit ``*_for_testing`` compiler is isolated from the production route.
It exists solely to exercise the closed schema and fail-closed validations.
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
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .c2_full_replacement_policy import (
    CHUNK_IDS,
    FROZEN_INPUT_TOTAL,
    FULL_REPLACEMENT_INPUT_TOTALS,
    C2FullReplacementPolicyError,
    ChunkPartition,
    ReplacementRootPlan,
    normalize_doi,
)
from .models import sha256_json


STAGE_B_POLICY_SCHEMA_ID = "c2_full_replacement_stageb_policy_v1.schema.json"
_STAGE_B_POLICY_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / STAGE_B_POLICY_SCHEMA_ID
)
# Updated together with the checked-in schema.  This is intentionally separate
# from the unavailable production policy-resource digest below.
STAGE_B_POLICY_SCHEMA_SHA256 = (
    "124cb7dc1d6d11fbded7c25a7ceefd784c177564c5fbe603680c18f5b04353c1"
)

_STAGE_B_POLICY_RESOURCE_PACKAGE = "experiments"
_STAGE_B_POLICY_RESOURCE_PARTS = (
    "resources",
    "c2_full_replacement_stageb_policy_v1.json",
)
_STAGE_B_POLICY_RESOURCE_NAME = "/".join(_STAGE_B_POLICY_RESOURCE_PARTS)
# No source-bearing remediation, forensic approval, or approved commitments
# exist yet.  Do not replace this with a template digest.
_STAGE_B_POLICY_RESOURCE_SHA256: str | None = None
_STAGE_B_PRODUCTION_ROUTE_APPROVED = False

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

STAGE_B_CHUNK_EVIDENCE_ROLES = (
    "initial_raw_stream",
    "initial_processed_success",
    "initial_skipped_status",
    "retry1_raw_stream",
    "retry1_processed_success",
    "retry1_skipped_status",
    "retry2_raw_stream",
    "retry2_processed_success",
    "retry2_skipped_status",
    "terminal_outcomes",
    "sealed_terminal_report",
)
STAGE_B_DOWNLOADED_DOI_EVIDENCE_ROLES = (
    "source_inventory",
    "raw_source_evidence",
    "canonical_builder",
    "candidate_parent_manifest",
    "proposal_parent_manifest",
    "structural_review_parent_manifest",
    "canonical_parent_manifest",
    "raw_source_bytes",
    "source_table",
    "panel_descriptor",
    "all_case_set",
    "per_container_accounts",
)
STAGE_B_COLLECTION_EVIDENCE_ROLES = (
    "detected_format_collection",
    "container_accounting_index",
    "derived_archive_member_collection",
    "consumable_source_unit_collection",
    "downstream_consumption_collection",
    "candidate_set_input_collection",
    "consumption_bijection_validation",
)
STAGE_B_REQUIRED_CROSS_ROLE_FIELDS = (
    "chunk_id",
    "parent_doi_id",
    "frozen_input_ordinal",
    "relative_path",
    "file_sha256",
    "byte_count",
    "semantic_sha256",
    "complete",
    "sealed",
    "clean_code",
    "model_result_selected",
    "record_order",
)
_STAGE_B_ALL_EVIDENCE_ROLES = frozenset(
    STAGE_B_CHUNK_EVIDENCE_ROLES
    + STAGE_B_DOWNLOADED_DOI_EVIDENCE_ROLES
    + STAGE_B_COLLECTION_EVIDENCE_ROLES
)

_FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "doi_to_p",
        "doi_p_map",
        "doi_p_cluster_map",
        "doi_to_cluster",
        "doi_cluster_map",
        "p1_sentinel",
        "p1_sentinels",
        "p1_sentinel_map",
        "strata",
        "stratum_counts",
        "p_counts",
        "target_distribution",
        "coverage_outcome",
        "coverage_status",
        "report_status",
        "admission_status",
        "claim_status",
        "scientific_outcome",
        "scientific_result",
        "method_outcome",
        "method_result",
        "judge_output",
        "trend_status",
        "equivalence_status",
        "final_report",
        "source_classification",
        "source_classifications",
        "derived_public_stratum",
        "derived_code_label",
        "cluster_id",
        "qualified_panel_count",
        "qualified_panel_counts",
        "canonical_case_ids",
        "canonical_case_set_hash",
    }
)
_FORBIDDEN_IDENTIFIER_MARKERS = (
    "doi_to_p",
    "doi_p_map",
    "doi_p_cluster",
    "doi_to_cluster",
    "p1_sentinel",
    "cluster",
    "stratum",
    "strata",
    "target_distribution",
    "coverage",
    "report_status",
    "admission_status",
    "scientific_outcome",
    "method_outcome",
    "method_result",
    "judge_output",
    "trend",
    "equivalence",
    "classification",
)
_FORBIDDEN_RESULT_VALUES = frozenset(
    {
        "P1",
        "P2",
        "P3_4",
        "P5PLUS",
        "P=1",
        "P=2",
        "P=3-4",
        "P=5+",
        "ADMITTED",
        "SUPPORTED",
        "UNSUPPORTED",
        "NOT_RUN_COVERAGE_GATE",
        "NOT_RUN_NO_ANALYSIS_AUTHORIZED",
    }
)
_FORBIDDEN_IDENTIFIER_TOKENS = frozenset({"p1", "p2", "p3_4", "p5plus"})
_FORBIDDEN_IDENTIFIER_RESULTS = frozenset(
    {
        "admitted",
        "supported",
        "unsupported",
        "not_run_coverage_gate",
        "not_run_no_analysis_authorized",
    }
)
_DOI_VALUE_PATH_FIELDS = frozenset(
    {
        "ordered_doi_ids",
        "parent_doi_id_or_null",
    }
)
_SAFE_STRING_VALUE_FIELDS = frozenset(
    {
        "schema_version",
        "policy_type",
        "role",
        "chunk_id",
        "chunk_id_or_null",
        "commit_full",
        "universe_file_sha256",
        "ordered_doi_ids_sha256",
        "chunk_concat_sha256",
        "doi_ids_sha256",
        "file_sha256",
        "semantic_sha256",
        "policy_sha256",
        "sha256",
    }
)


@dataclass(frozen=True, slots=True)
class StageBFrozenUniverse:
    """Immutable frozen acquisition identity, without classifications."""

    universe_file_sha256: str
    ordered_doi_ids_sha256: str
    input_total: int
    ordered_doi_ids: tuple[str, ...]
    chunk_file_sha256_by_id: Mapping[str, str]
    chunk_concat_sha256: str


@dataclass(frozen=True, slots=True)
class StageBEvidenceCommitment:
    """A nonclassification artifact commitment stored in the internal policy."""

    commitment_id: str
    role: str
    chunk_id: str | None
    frozen_input_ordinal: int | None
    parent_doi_id: str | None
    relative_path: str
    file_sha256: str
    byte_count: int
    semantic_sha256: str
    record_order: int


@dataclass(frozen=True, slots=True)
class CompiledStageBProductionPolicy:
    """Validated but non-admissive internal Stage-B policy representation."""

    policy_id: str
    policy_version: str
    policy_sha256: str
    resource_sha256: str
    frozen_universe: StageBFrozenUniverse
    partition: tuple[ChunkPartition, ...]
    replacement_plan: tuple[ReplacementRootPlan, ...]
    nonclassification_registries: Mapping[str, Any]
    evidence_role_layout: Mapping[str, Any]
    evidence_commitments: tuple[StageBEvidenceCommitment, ...]

    def partition_for_chunk(self, chunk_id: str) -> ChunkPartition:
        for partition in self.partition:
            if partition.chunk_id == chunk_id:
                return partition
        raise C2FullReplacementPolicyError(f"Unknown Stage-B policy chunk: {chunk_id}")


def _reject_json_constant(value: str) -> None:
    raise C2FullReplacementPolicyError(
        f"Stage-B policy contains a non-finite JSON value: {value}"
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C2FullReplacementPolicyError(f"{label} must be a full SHA-256 digest")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise C2FullReplacementPolicyError(f"{label} is not a valid identifier")
    return value


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: item for key, item in value.items() if key not in omitted}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@lru_cache(maxsize=1)
def _stage_b_schema_validator() -> Draft202012Validator:
    try:
        raw_schema = _STAGE_B_POLICY_SCHEMA_PATH.read_bytes()
    except OSError as exc:
        raise C2FullReplacementPolicyError(
            f"Cannot read bundled Stage-B policy schema: {STAGE_B_POLICY_SCHEMA_ID}"
        ) from exc
    actual_sha256 = hashlib.sha256(raw_schema).hexdigest()
    if not hmac.compare_digest(actual_sha256, STAGE_B_POLICY_SCHEMA_SHA256):
        raise C2FullReplacementPolicyError(
            "Bundled Stage-B policy schema differs from its compiled SHA-256"
        )
    try:
        schema = json.loads(
            raw_schema.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2FullReplacementPolicyError(
            "Bundled Stage-B policy schema is not valid UTF-8 JSON"
        ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise C2FullReplacementPolicyError(
            "Bundled Stage-B policy schema is invalid"
        ) from exc
    return Draft202012Validator(schema)


def _validate_schema(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise C2FullReplacementPolicyError("Stage-B policy resource must be an object")
    errors = sorted(
        _stage_b_schema_validator().iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise C2FullReplacementPolicyError(
            f"Stage-B policy schema validation failed at {location}: "
            f"{errors[0].message}"
        )
    return value


def _normalized_marker(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _forbidden_field_name(name: str) -> bool:
    normalized = _normalized_marker(name)
    if normalized == "nonclassification_registries":
        return False
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


def _forbidden_string_value(path: tuple[str, ...], value: str) -> bool:
    if any(field in _DOI_VALUE_PATH_FIELDS for field in path):
        return False
    if path and path[-1] in _SAFE_STRING_VALUE_FIELDS:
        return False
    normalized = _normalized_marker(value)
    if any(marker in normalized for marker in _FORBIDDEN_IDENTIFIER_MARKERS):
        return True
    if normalized in _FORBIDDEN_IDENTIFIER_RESULTS:
        return True
    return (
        "p3_4" in normalized
        or bool(set(normalized.split("_")) & _FORBIDDEN_IDENTIFIER_TOKENS)
    )


def _reject_forbidden_semantics(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject fields and identifier values that can encode classifications/results."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise C2FullReplacementPolicyError(
                    "Stage-B policy object keys must be strings"
                )
            if _forbidden_field_name(key):
                location = ".".join((*path, key))
                raise C2FullReplacementPolicyError(
                    f"Stage-B policy forbids classification/outcome field: {location}"
                )
            _reject_forbidden_semantics(item, (*path, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_semantics(item, (*path, str(index)))
        return
    if isinstance(value, str):
        if value in _FORBIDDEN_RESULT_VALUES:
            raise C2FullReplacementPolicyError(
                "Stage-B policy forbids P labels and scientific/report outcomes"
            )
        if _forbidden_string_value(path, value):
            raise C2FullReplacementPolicyError(
                "Stage-B policy identifier encodes a forbidden classification/outcome"
            )


def _parse_frozen_universe(value: Mapping[str, Any]) -> StageBFrozenUniverse:
    ordered = tuple(
        normalize_doi(doi_id) for doi_id in value["ordered_doi_ids"]
    )
    if tuple(value["ordered_doi_ids"]) != ordered:
        raise C2FullReplacementPolicyError(
            "Stage-B ordered DOI identifiers must already be normalized"
        )
    if len(ordered) != FROZEN_INPUT_TOTAL or len(set(ordered)) != len(ordered):
        raise C2FullReplacementPolicyError(
            "Stage-B frozen universe must contain 2,463 unique ordered DOI identifiers"
        )
    ordered_hash = _require_sha256(
        value["ordered_doi_ids_sha256"],
        "frozen_universe.ordered_doi_ids_sha256",
    )
    if sha256_json(list(ordered)) != ordered_hash:
        raise C2FullReplacementPolicyError(
            "Stage-B frozen universe DOI hash does not match its ordered DOI list"
        )
    chunk_hashes = value["chunk_file_sha256_by_id"]
    if not isinstance(chunk_hashes, Mapping) or set(chunk_hashes) != set(CHUNK_IDS):
        raise C2FullReplacementPolicyError(
            "Stage-B frozen universe must bind exactly chunks 001 through 013"
        )
    parsed_chunk_hashes = {
        chunk_id: _require_sha256(
            chunk_hashes[chunk_id],
            f"frozen_universe.chunk_file_sha256_by_id.{chunk_id}",
        )
        for chunk_id in CHUNK_IDS
    }
    return StageBFrozenUniverse(
        universe_file_sha256=_require_sha256(
            value["universe_file_sha256"],
            "frozen_universe.universe_file_sha256",
        ),
        ordered_doi_ids_sha256=ordered_hash,
        input_total=value["input_total"],
        ordered_doi_ids=ordered,
        chunk_file_sha256_by_id=MappingProxyType(parsed_chunk_hashes),
        chunk_concat_sha256=_require_sha256(
            value["chunk_concat_sha256"],
            "frozen_universe.chunk_concat_sha256",
        ),
    )


def _parse_partition(
    value: Sequence[Mapping[str, Any]],
    frozen_universe: StageBFrozenUniverse,
) -> tuple[ChunkPartition, ...]:
    parsed: list[ChunkPartition] = []
    next_ordinal = 1
    for chunk_id, input_total, item in zip(
        CHUNK_IDS,
        FULL_REPLACEMENT_INPUT_TOTALS,
        value,
        strict=True,
    ):
        if (
            item["chunk_id"] != chunk_id
            or item["first_global_ordinal"] != next_ordinal
            or item["input_total"] != input_total
        ):
            raise C2FullReplacementPolicyError(
                f"Stage-B partition {chunk_id} violates the frozen 12x200+63 roster"
            )
        doi_hash = _require_sha256(
            item["doi_ids_sha256"],
            f"Stage-B partition {chunk_id}.doi_ids_sha256",
        )
        start = next_ordinal - 1
        if (
            sha256_json(
                list(frozen_universe.ordered_doi_ids[start : start + input_total])
            )
            != doi_hash
        ):
            raise C2FullReplacementPolicyError(
                f"Stage-B partition {chunk_id} DOI digest does not match its slice"
            )
        parsed.append(
            ChunkPartition(
                chunk_id=chunk_id,
                first_global_ordinal=next_ordinal,
                input_total=input_total,
                doi_ids_sha256=doi_hash,
            )
        )
        next_ordinal += input_total
    if next_ordinal != FROZEN_INPUT_TOTAL + 1:
        raise C2FullReplacementPolicyError(
            "Stage-B partition does not total the frozen acquisition universe"
        )
    return tuple(parsed)


def _parse_replacement_plan(
    value: Sequence[Mapping[str, Any]],
) -> tuple[ReplacementRootPlan, ...]:
    root_ids: set[str] = set()
    plans: list[ReplacementRootPlan] = []
    for chunk_id, item in zip(CHUNK_IDS, value, strict=True):
        if item["chunk_id"] != chunk_id or item["partial_root"] is not False:
            raise C2FullReplacementPolicyError(
                "Stage-B replacement plan must be complete and ordered by chunk"
            )
        replacement = _require_identifier(
            item["replacement_root_id"],
            f"Stage-B replacement plan {chunk_id}.replacement_root_id",
        )
        retired = tuple(
            _require_identifier(root_id, f"Stage-B replacement plan {chunk_id}.retired")
            for root_id in item["retired_root_ids"]
        )
        if (
            not retired
            or len(retired) != len(set(retired))
            or replacement in retired
            or root_ids.intersection((replacement, *retired))
        ):
            raise C2FullReplacementPolicyError(
                "Stage-B replacement plan reuses or omits a retired/root identity"
            )
        root_ids.update((replacement, *retired))
        plans.append(
            ReplacementRootPlan(
                chunk_id=chunk_id,
                replacement_root_id=replacement,
                retired_root_ids=retired,
            )
        )
    return tuple(plans)


def _validate_unique_identifiers(
    items: Sequence[Mapping[str, Any]],
    fields: tuple[str, ...],
    label: str,
) -> None:
    values = [tuple(item[field] for field in fields) for item in items]
    if len(values) != len(set(values)):
        raise C2FullReplacementPolicyError(
            f"Stage-B {label} registry repeats a bound identity"
        )


def _validate_nonclassification_registries(value: Mapping[str, Any]) -> None:
    canonical_json = value["canonical_json"]
    if canonical_json["id"] != "CANONICAL_JSON_UTF8_SORTED_V1":
        raise C2FullReplacementPolicyError(
            "Stage-B policy must bind the canonical JSON identity"
        )
    _validate_unique_identifiers(value["code"], ("id",), "code")
    _validate_unique_identifiers(value["rules"], ("id", "version"), "rule")
    _validate_unique_identifiers(value["schemas"], ("id",), "schema")
    matching_schema = [
        item
        for item in value["schemas"]
        if item["id"] == STAGE_B_POLICY_SCHEMA_ID
    ]
    if len(matching_schema) != 1 or not hmac.compare_digest(
        matching_schema[0]["sha256"],
        STAGE_B_POLICY_SCHEMA_SHA256,
    ):
        raise C2FullReplacementPolicyError(
            "Stage-B policy must bind the compiled Stage-B policy schema digest"
        )


def _validate_evidence_role_layout(value: Mapping[str, Any]) -> None:
    if tuple(value["chunk_roles"]) != STAGE_B_CHUNK_EVIDENCE_ROLES:
        raise C2FullReplacementPolicyError(
            "Stage-B evidence layout must use the closed ordered chunk-role registry"
        )
    if tuple(value["downloaded_doi_roles"]) != STAGE_B_DOWNLOADED_DOI_EVIDENCE_ROLES:
        raise C2FullReplacementPolicyError(
            "Stage-B evidence layout must use the closed downloaded-DOI role registry"
        )
    if tuple(value["collection_roles"]) != STAGE_B_COLLECTION_EVIDENCE_ROLES:
        raise C2FullReplacementPolicyError(
            "Stage-B evidence layout must use the closed collection-role registry"
        )
    if tuple(value["required_cross_role_fields"]) != STAGE_B_REQUIRED_CROSS_ROLE_FIELDS:
        raise C2FullReplacementPolicyError(
            "Stage-B evidence layout must bind every required cross-role field"
        )


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise C2FullReplacementPolicyError(f"{label} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(path) != value
    ):
        raise C2FullReplacementPolicyError(
            f"{label} must be root-relative without traversal or aliases"
        )
    return value


def _chunk_for_ordinal(
    ordinal: int,
    partition: Sequence[ChunkPartition],
) -> str:
    for item in partition:
        if item.first_global_ordinal <= ordinal < (
            item.first_global_ordinal + item.input_total
        ):
            return item.chunk_id
    raise C2FullReplacementPolicyError(
        "Stage-B evidence commitment ordinal is outside the frozen universe"
    )


def _parse_evidence_commitments(
    value: Sequence[Mapping[str, Any]],
    frozen_universe: StageBFrozenUniverse,
    partition: Sequence[ChunkPartition],
) -> tuple[StageBEvidenceCommitment, ...]:
    if not value:
        raise C2FullReplacementPolicyError(
            "Stage-B policy has no approved internal evidence commitments"
        )
    commitments: list[StageBEvidenceCommitment] = []
    commitment_ids: set[str] = set()
    relative_paths: set[str] = set()
    roles: set[str] = set()
    for expected_order, item in enumerate(value, start=1):
        commitment_id = _require_identifier(
            item["commitment_id"],
            f"Stage-B evidence commitment {expected_order}.commitment_id",
        )
        if commitment_id in commitment_ids:
            raise C2FullReplacementPolicyError(
                "Stage-B evidence commitments repeat a commitment identity"
            )
        commitment_ids.add(commitment_id)
        if item["record_order"] != expected_order:
            raise C2FullReplacementPolicyError(
                "Stage-B evidence commitments must have contiguous immutable order"
            )
        role = item["role"]
        roles.add(role)
        chunk_id = item["chunk_id_or_null"]
        ordinal = item["frozen_input_ordinal_or_null"]
        parent_doi_id = item["parent_doi_id_or_null"]
        if role in STAGE_B_CHUNK_EVIDENCE_ROLES:
            if chunk_id not in CHUNK_IDS or ordinal is not None or parent_doi_id is not None:
                raise C2FullReplacementPolicyError(
                    "Stage-B chunk evidence commitment has an invalid identity binding"
                )
        elif role in STAGE_B_DOWNLOADED_DOI_EVIDENCE_ROLES:
            if (
                chunk_id not in CHUNK_IDS
                or not isinstance(ordinal, int)
                or not isinstance(parent_doi_id, str)
            ):
                raise C2FullReplacementPolicyError(
                    "Stage-B downloaded-DOI evidence commitment lacks its identity binding"
                )
            normalized_doi = normalize_doi(parent_doi_id)
            if normalized_doi != parent_doi_id:
                raise C2FullReplacementPolicyError(
                    "Stage-B evidence commitment parent DOI must already be normalized"
                )
            if parent_doi_id != frozen_universe.ordered_doi_ids[ordinal - 1]:
                raise C2FullReplacementPolicyError(
                    "Stage-B evidence commitment parent DOI differs from its frozen ordinal"
                )
            if _chunk_for_ordinal(ordinal, partition) != chunk_id:
                raise C2FullReplacementPolicyError(
                    "Stage-B evidence commitment chunk differs from its frozen ordinal"
                )
        elif role in STAGE_B_COLLECTION_EVIDENCE_ROLES:
            if chunk_id is not None or ordinal is not None or parent_doi_id is not None:
                raise C2FullReplacementPolicyError(
                    "Stage-B collection evidence commitment must not select a root or DOI"
                )
        else:
            raise C2FullReplacementPolicyError("Stage-B evidence commitment has unknown role")
        relative_path = _safe_relative_path(
            item["relative_path"],
            f"Stage-B evidence commitment {commitment_id}.relative_path",
        )
        if relative_path in relative_paths:
            raise C2FullReplacementPolicyError(
                "Stage-B evidence commitments reuse an artifact relative path"
            )
        relative_paths.add(relative_path)
        commitments.append(
            StageBEvidenceCommitment(
                commitment_id=commitment_id,
                role=role,
                chunk_id=chunk_id,
                frozen_input_ordinal=ordinal,
                parent_doi_id=parent_doi_id,
                relative_path=relative_path,
                file_sha256=_require_sha256(
                    item["file_sha256"],
                    f"Stage-B evidence commitment {commitment_id}.file_sha256",
                ),
                byte_count=item["byte_count"],
                semantic_sha256=_require_sha256(
                    item["semantic_sha256"],
                    f"Stage-B evidence commitment {commitment_id}.semantic_sha256",
                ),
                record_order=expected_order,
            )
        )
    if roles != _STAGE_B_ALL_EVIDENCE_ROLES:
        missing = sorted(_STAGE_B_ALL_EVIDENCE_ROLES - roles)
        unexpected = sorted(roles - _STAGE_B_ALL_EVIDENCE_ROLES)
        raise C2FullReplacementPolicyError(
            "Stage-B evidence commitments do not cover the closed role layout "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return tuple(commitments)


def _compile_stage_b_policy(
    value: Any,
    *,
    resource_sha256: str,
) -> CompiledStageBProductionPolicy:
    policy = _validate_schema(value)
    _reject_forbidden_semantics(policy)
    policy_sha256 = _require_sha256(policy["policy_sha256"], "policy_sha256")
    if sha256_json(_without(policy, "policy_sha256")) != policy_sha256:
        raise C2FullReplacementPolicyError(
            "Stage-B policy failed its self-omitting semantic SHA-256"
        )
    frozen_universe = _parse_frozen_universe(policy["frozen_universe"])
    partition = _parse_partition(policy["partition"], frozen_universe)
    replacement_plan = _parse_replacement_plan(policy["replacement_plan"])
    _validate_nonclassification_registries(policy["nonclassification_registries"])
    _validate_evidence_role_layout(policy["evidence_role_layout"])
    commitments = _parse_evidence_commitments(
        policy["evidence_commitments"],
        frozen_universe,
        partition,
    )
    return CompiledStageBProductionPolicy(
        policy_id=_require_identifier(policy["policy_id"], "policy_id"),
        policy_version=_require_identifier(policy["policy_version"], "policy_version"),
        policy_sha256=policy_sha256,
        resource_sha256=_require_sha256(resource_sha256, "resource_sha256"),
        frozen_universe=frozen_universe,
        partition=partition,
        replacement_plan=replacement_plan,
        nonclassification_registries=_freeze(policy["nonclassification_registries"]),
        evidence_role_layout=_freeze(policy["evidence_role_layout"]),
        evidence_commitments=commitments,
    )


def _parse_and_compile_resource_bytes(
    resource_bytes: bytes,
    expected_resource_sha256: str,
) -> CompiledStageBProductionPolicy:
    if not isinstance(resource_bytes, bytes):
        raise C2FullReplacementPolicyError(
            "Stage-B policy resource must be exact bytes before parsing"
        )
    expected = _require_sha256(
        expected_resource_sha256,
        "compiled Stage-B policy resource SHA-256",
    )
    actual = hashlib.sha256(resource_bytes).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise C2FullReplacementPolicyError(
            "Stage-B policy resource bytes differ from the compiled SHA-256"
        )
    try:
        parsed = json.loads(
            resource_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2FullReplacementPolicyError(
            "Stage-B policy resource is not valid UTF-8 JSON"
        ) from exc
    return _compile_stage_b_policy(parsed, resource_sha256=actual)


def compile_stage_b_policy_resource_for_testing(
    resource_bytes: bytes,
    *,
    expected_resource_sha256: str,
) -> CompiledStageBProductionPolicy:
    """Compile synthetic bytes for regression tests only, never production routing."""

    return _parse_and_compile_resource_bytes(resource_bytes, expected_resource_sha256)


def _read_compiled_stage_b_resource_bytes() -> bytes:
    """Read only the compile-time-named package resource, with no selector input."""

    try:
        resource = resources.files(_STAGE_B_POLICY_RESOURCE_PACKAGE).joinpath(
            *_STAGE_B_POLICY_RESOURCE_PARTS
        )
        if not resource.is_file():
            raise FileNotFoundError(_STAGE_B_POLICY_RESOURCE_NAME)
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise C2FullReplacementPolicyError(
            "Stage-A only: the exact internal Stage-B production policy resource "
            "is unavailable"
        ) from exc


def load_stage_b_production_policy() -> CompiledStageBProductionPolicy:
    """Resolve the sole internal resource, while the non-admissive gate is closed."""

    expected = _STAGE_B_POLICY_RESOURCE_SHA256
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        raise C2FullReplacementPolicyError(
            "Stage-A only: no approved Git-reviewed internal Stage-B production "
            "policy resource and compiled SHA-256 are pinned"
        )
    policy = _parse_and_compile_resource_bytes(
        _read_compiled_stage_b_resource_bytes(),
        expected,
    )
    if not _STAGE_B_PRODUCTION_ROUTE_APPROVED:
        raise C2FullReplacementPolicyError(
            "Stage-A only: the Stage-B policy resource route remains intentionally "
            "non-admissive pending source-bearing remediation and independent approval"
        )
    return policy

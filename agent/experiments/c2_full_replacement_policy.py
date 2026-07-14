"""Fixed-policy primitives for source-derived C2 full-replacement V2.1.

Stage A deliberately has no production resource.  A test-only policy pins the
ordered acquisition universe and evidence rules, never P labels or clusters for
all acquired DOI.  P classification exists only after source/canonical evidence
is recomputed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from .models import ProvenanceError, sha256_json


C2_FULL_REPLACEMENT_V2_VERSION = "2.1-stagea"
CHUNK_IDS = tuple(f"{number:03d}" for number in range(1, 14))
ATTEMPT_IDS = ("initial", "retry1", "retry2")
P_DISPOSITIONS = ("P1", "P2", "P3_4", "P5PLUS")
P_CODE_LABELS = {
    "P1": "P=1",
    "P2": "P=2",
    "P3_4": "P=3-4",
    "P5PLUS": "P=5+",
}
FROZEN_INPUT_TOTAL = 2_463
FINAL_CHUNK_INPUT_TOTAL = 63
FULL_REPLACEMENT_INPUT_TOTALS = tuple(
    200 if chunk_id != "013" else FINAL_CHUNK_INPUT_TOTAL
    for chunk_id in CHUNK_IDS
)
RAW_STATUS_ADAPTER = {
    "downloaded": "DOWNLOADED",
    "no-source-data": "NO_SOURCE_DATA",
    "no-figures": "NO_FIGURES",
    "no-usable-content": "NO_USABLE_CONTENT",
    "policy-rejected": "POLICY_REJECTED",
    "fetch-error": "DOWNLOAD_FAILED",
    "download-failed": "DOWNLOAD_FAILED",
    "retry-exhausted": "RETRY_EXHAUSTED",
}
TERMINAL_OUTCOME_STATUSES = frozenset(RAW_STATUS_ADAPTER.values())
NON_STRATIFIED_DISPOSITIONS = {
    "NO_SOURCE_DATA": "NON_STRATIFIED_NO_SOURCE_DATA",
    "NO_FIGURES": "NON_STRATIFIED_NO_FIGURES",
    "NO_USABLE_CONTENT": "NON_STRATIFIED_NO_USABLE_CONTENT",
    "POLICY_REJECTED": "NON_STRATIFIED_POLICY_REJECTED",
    "DOWNLOAD_FAILED": "NON_STRATIFIED_DOWNLOAD_FAILED",
    "RETRY_EXHAUSTED": "NON_STRATIFIED_RETRY_EXHAUSTED",
}
DOWNLOADED_EMPTY_SOURCE_DISPOSITION = (
    "NON_STRATIFIED_DOWNLOADED_NO_VERIFIED_SOURCE"
)
DOWNLOADED_EMPTY_CASE_DISPOSITION = (
    "NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE"
)
STRATIFIED_DISPOSITION = "STRATIFIED_SOURCE_CANONICAL"
DOI_CASE_AGGREGATION_RULE_VERSION = "DOI_CASE_AGGREGATION_V1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class C2FullReplacementPolicyError(ProvenanceError):
    """Raised when V2.1 policy bindings are unavailable or malformed."""


def normalize_doi(value: Any) -> str:
    """Apply the frozen DOI normalization rule before any provenance comparison."""

    if not isinstance(value, str):
        raise C2FullReplacementPolicyError("DOI value must be a string")
    normalized = unquote(value).strip()
    normalized = re.sub(r"^doi:\s*", "", normalized, flags=re.IGNORECASE)
    parsed = urlparse(normalized)
    if parsed.scheme:
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or (parsed.hostname or "").casefold() not in {"doi.org", "dx.doi.org"}
        ):
            raise C2FullReplacementPolicyError("DOI URL must be an http(s) DOI URL")
        normalized = parsed.path.lstrip("/")
    else:
        normalized = normalized.split("?", 1)[0].split("#", 1)[0]
    normalized = normalized.strip().casefold()
    if (
        not normalized.startswith("10.")
        or "/" not in normalized
        or any(character.isspace() for character in normalized)
    ):
        raise C2FullReplacementPolicyError("DOI normalization produced an invalid DOI")
    return normalized


def _require_normalized_doi(value: Any, label: str) -> str:
    normalized = normalize_doi(value)
    if value != normalized:
        raise C2FullReplacementPolicyError(f"{label} must already be normalized")
    return normalized


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C2FullReplacementPolicyError(f"{label} must be a full SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise C2FullReplacementPolicyError(f"{label} must be a full Git commit")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise C2FullReplacementPolicyError(f"{label} is not a valid identifier")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if frozenset(value) != expected:
        raise C2FullReplacementPolicyError(
            f"{label} fields must be exactly {sorted(expected)!r}"
        )


@dataclass(frozen=True)
class FrozenUniverse:
    sha256: str
    doi_ids_sha256: str
    input_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "doi_ids_sha256": self.doi_ids_sha256,
            "input_total": self.input_total,
        }


@dataclass(frozen=True)
class ChunkPartition:
    chunk_id: str
    first_global_ordinal: int
    input_total: int
    doi_ids_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "first_global_ordinal": self.first_global_ordinal,
            "input_total": self.input_total,
            "doi_ids_sha256": self.doi_ids_sha256,
        }


@dataclass(frozen=True)
class ReplacementRootPlan:
    chunk_id: str
    replacement_root_id: str
    retired_root_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "replacement_root_id": self.replacement_root_id,
            "retired_root_ids": list(self.retired_root_ids),
        }


@dataclass(frozen=True)
class StratifiedSourceClassification:
    """A DOI-level classification derived from verified source/canonical inputs."""

    doi_id: str
    canonical_case_set_hash: str
    canonical_case_ids: tuple[str, ...]
    verified_panel_descriptors_sha256: str
    qualified_panel_counts: tuple[int, ...]
    derived_public_stratum: str
    derived_code_label: str
    source_binding_hashes: Mapping[str, str]

    @property
    def cluster_id(self) -> str:
        return self.doi_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "doi_id": self.doi_id,
            "cluster_id": self.doi_id,
            "canonical_case_set_hash": self.canonical_case_set_hash,
            "canonical_case_ids": list(self.canonical_case_ids),
            "verified_panel_descriptors_sha256": (
                self.verified_panel_descriptors_sha256
            ),
            "qualified_panel_counts": list(self.qualified_panel_counts),
            "derived_public_stratum": self.derived_public_stratum,
            "derived_code_label": self.derived_code_label,
            "source_binding_hashes": dict(self.source_binding_hashes),
        }


@dataclass(frozen=True)
class StratumAggregate:
    p_disposition: str
    source_doi_count: int
    independent_cluster_count: int
    coverage_required: bool
    coverage_met: bool
    inference_eligible: bool
    deficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_disposition": self.p_disposition,
            "source_doi_count": self.source_doi_count,
            "independent_cluster_count": self.independent_cluster_count,
            "coverage_required": self.coverage_required,
            "coverage_met": self.coverage_met,
            "inference_eligible": self.inference_eligible,
            "deficient": self.deficient,
        }


@dataclass(frozen=True)
class SourceCoverageAggregation:
    strata: tuple[StratumAggregate, ...]
    deficient_strata: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class CompiledFullReplacementPolicy:
    """An immutable ordered acquisition policy used only by Stage-A tests."""

    policy_id: str
    policy_version: str
    frozen_universe: FrozenUniverse
    frozen_bindings: Mapping[str, Any]
    partition: tuple[ChunkPartition, ...]
    replacement_plan: tuple[ReplacementRootPlan, ...]
    ordered_doi_ids: tuple[str, ...]
    test_policy_sha256: str

    @property
    def is_test_only(self) -> bool:
        return True

    def dois_for_chunk(self, chunk_id: str) -> tuple[str, ...]:
        partition = self.partition_for_chunk(chunk_id)
        start = partition.first_global_ordinal - 1
        return self.ordered_doi_ids[start : start + partition.input_total]

    def partition_for_chunk(self, chunk_id: str) -> ChunkPartition:
        for partition in self.partition:
            if partition.chunk_id == chunk_id:
                return partition
        raise C2FullReplacementPolicyError(f"Unknown policy chunk: {chunk_id}")

    def root_plan_for_chunk(self, chunk_id: str) -> ReplacementRootPlan:
        for plan in self.replacement_plan:
            if plan.chunk_id == chunk_id:
                return plan
        raise C2FullReplacementPolicyError(f"Unknown policy root plan chunk: {chunk_id}")


def derive_public_stratum(qualified_panel_count: int) -> str:
    if not isinstance(qualified_panel_count, int) or qualified_panel_count < 1:
        raise C2FullReplacementPolicyError(
            "Qualified canonical panel count must be a positive integer"
        )
    if qualified_panel_count == 1:
        return "P1"
    if qualified_panel_count == 2:
        return "P2"
    if qualified_panel_count in {3, 4}:
        return "P3_4"
    return "P5PLUS"


def adapt_terminal_status(raw_status: Any) -> str:
    if not isinstance(raw_status, str) or raw_status not in RAW_STATUS_ADAPTER:
        raise C2FullReplacementPolicyError(
            "Terminal raw status is not in the closed V2.1 adapter"
        )
    return RAW_STATUS_ADAPTER[raw_status]


def expected_non_stratified_disposition(terminal_status: str) -> str:
    try:
        return NON_STRATIFIED_DISPOSITIONS[terminal_status]
    except KeyError as exc:
        raise C2FullReplacementPolicyError(
            "Only non-DOWNLOADED statuses have direct non-stratified dispositions"
        ) from exc


def aggregate_stratified_source_classifications(
    classifications: Sequence[StratifiedSourceClassification],
) -> SourceCoverageAggregation:
    """Apply the frozen DOI-cluster source-coverage gate, never acquisition rows."""

    doi_by_stratum: dict[str, set[str]] = defaultdict(set)
    seen_dois: set[str] = set()
    required_binding_hashes = frozenset(
        {
            "source_inventory_sha256",
            "raw_source_evidence_sha256",
            "candidate_manifest_sha256",
            "proposal_manifest_sha256",
            "review_manifest_sha256",
            "canonical_manifest_sha256",
            "canonical_builder_sha256",
        }
    )
    for index, classification in enumerate(classifications):
        doi_id = _require_normalized_doi(
            classification.doi_id,
            f"source classification {index}.doi_id",
        )
        if doi_id in seen_dois:
            raise C2FullReplacementPolicyError(
                "Stratified source classifications contain a duplicate DOI cluster"
            )
        seen_dois.add(doi_id)
        if classification.cluster_id != doi_id:
            raise C2FullReplacementPolicyError(
                "Stratified source classification cluster_id must equal doi_id"
            )
        _require_sha256(
            classification.canonical_case_set_hash,
            f"source classification {index}.canonical_case_set_hash",
        )
        _require_sha256(
            classification.verified_panel_descriptors_sha256,
            f"source classification {index}.verified_panel_descriptors_sha256",
        )
        if frozenset(classification.source_binding_hashes) != required_binding_hashes:
            raise C2FullReplacementPolicyError(
                "Stratified source classification has incomplete source/canonical bindings"
            )
        for name, value in classification.source_binding_hashes.items():
            _require_sha256(value, f"source classification {index}.{name}")
        if classification.derived_public_stratum not in P_DISPOSITIONS:
            raise C2FullReplacementPolicyError(
                "Stratified source classification has an unknown P stratum"
            )
        if classification.derived_code_label != P_CODE_LABELS[
            classification.derived_public_stratum
        ]:
            raise C2FullReplacementPolicyError(
                "Stratified source classification code label is inconsistent"
            )
        if not classification.canonical_case_ids:
            raise C2FullReplacementPolicyError(
                "Stratified source classification must retain every canonical case"
            )
        if len(classification.canonical_case_ids) != len(
            classification.qualified_panel_counts
        ):
            raise C2FullReplacementPolicyError(
                "Stratified source classification has inconsistent case counts"
            )
        if len(classification.canonical_case_ids) != len(
            set(classification.canonical_case_ids)
        ):
            raise C2FullReplacementPolicyError(
                "Stratified source classification repeats a canonical case ID"
            )
        if tuple(sorted(classification.canonical_case_ids)) != (
            classification.canonical_case_ids
        ):
            raise C2FullReplacementPolicyError(
                "Stratified source classification case IDs are not Unicode ordered"
            )
        expected_strata = {
            derive_public_stratum(panel_count)
            for panel_count in classification.qualified_panel_counts
        }
        if expected_strata != {classification.derived_public_stratum}:
            raise C2FullReplacementPolicyError(
                "Stratified source classification violates the same-stratum rule"
            )
        doi_by_stratum[classification.derived_public_stratum].add(doi_id)

    strata: list[StratumAggregate] = []
    for disposition in P_DISPOSITIONS:
        count = len(doi_by_stratum[disposition])
        coverage_required = disposition != "P1"
        coverage_met = not coverage_required or count >= 2
        strata.append(
            StratumAggregate(
                p_disposition=disposition,
                source_doi_count=count,
                independent_cluster_count=count,
                coverage_required=coverage_required,
                coverage_met=coverage_met,
                inference_eligible=False,
                deficient=coverage_required and not coverage_met,
            )
        )
    deficient = tuple(item.p_disposition for item in strata if item.deficient)
    if not deficient:
        status = "ADMITTED"
    elif "P5PLUS" in deficient:
        status = "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
    elif "P2" in deficient:
        status = "BLOCKED_INSUFFICIENT_INDEPENDENT_P2"
    else:
        status = "BLOCKED_INSUFFICIENT_INDEPENDENT_P3_4"
    return SourceCoverageAggregation(
        strata=tuple(strata),
        deficient_strata=deficient,
        status=status,
    )


def _parse_frozen_universe(value: Any) -> FrozenUniverse:
    if not isinstance(value, Mapping):
        raise C2FullReplacementPolicyError("frozen_universe must be an object")
    _require_exact_keys(
        value,
        frozenset({"sha256", "doi_ids_sha256", "input_total"}),
        "frozen_universe",
    )
    if value["input_total"] != FROZEN_INPUT_TOTAL:
        raise C2FullReplacementPolicyError(
            "V2.1 frozen universe must contain exactly 2,463 DOI inputs"
        )
    return FrozenUniverse(
        sha256=_require_sha256(value["sha256"], "frozen_universe.sha256"),
        doi_ids_sha256=_require_sha256(
            value["doi_ids_sha256"],
            "frozen_universe.doi_ids_sha256",
        ),
        input_total=value["input_total"],
    )


def _parse_frozen_bindings(
    value: Any,
    frozen_universe: FrozenUniverse,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise C2FullReplacementPolicyError("frozen_bindings must be an object")
    expected = frozenset(
        {
            "universe_file_sha256",
            "universe_ordered_doi_ids_sha256",
            "universe_input_total",
            "universe_chunk_count",
            "universe_chunk_file_sha256_by_id",
            "universe_chunk_concat_sha256",
            "normalizer_code_commit_full",
            "normalizer_code_sha256",
            "canonical_builder_code_commit_full",
            "canonical_builder_code_sha256",
            "canonical_builder_rule_version",
            "canonical_builder_rule_sha256",
            "panel_recomputation_rule_version",
            "panel_recomputation_rule_sha256",
            "doi_case_aggregation_rule_version",
            "doi_case_aggregation_rule_sha256",
            "v2_schema_file_sha256",
        }
    )
    _require_exact_keys(value, expected, "frozen_bindings")
    for name in (
        "universe_file_sha256",
        "universe_ordered_doi_ids_sha256",
        "universe_chunk_concat_sha256",
        "normalizer_code_sha256",
        "canonical_builder_code_sha256",
        "canonical_builder_rule_sha256",
        "panel_recomputation_rule_sha256",
        "doi_case_aggregation_rule_sha256",
        "v2_schema_file_sha256",
    ):
        _require_sha256(value[name], f"frozen_bindings.{name}")
    for name in (
        "normalizer_code_commit_full",
        "canonical_builder_code_commit_full",
    ):
        _require_commit(value[name], f"frozen_bindings.{name}")
    if (
        value["universe_ordered_doi_ids_sha256"] != frozen_universe.doi_ids_sha256
        or value["universe_input_total"] != FROZEN_INPUT_TOTAL
        or value["universe_chunk_count"] != len(CHUNK_IDS)
        or value["doi_case_aggregation_rule_version"]
        != DOI_CASE_AGGREGATION_RULE_VERSION
    ):
        raise C2FullReplacementPolicyError(
            "frozen_bindings disagrees with the fixed V2.1 universe/rule contract"
        )
    if (
        not isinstance(value["canonical_builder_rule_version"], str)
        or not value["canonical_builder_rule_version"]
        or not isinstance(value["panel_recomputation_rule_version"], str)
        or not value["panel_recomputation_rule_version"]
    ):
        raise C2FullReplacementPolicyError(
            "frozen_bindings canonical builder/panel rule versions are required"
        )
    chunk_hashes = value["universe_chunk_file_sha256_by_id"]
    if not isinstance(chunk_hashes, Mapping) or tuple(chunk_hashes) != CHUNK_IDS:
        raise C2FullReplacementPolicyError(
            "frozen_bindings must hash the ordered roster 001..013"
        )
    for chunk_id in CHUNK_IDS:
        _require_sha256(
            chunk_hashes[chunk_id],
            f"frozen_bindings.universe_chunk_file_sha256_by_id.{chunk_id}",
        )
    return dict(value)


def _parse_partition(value: Any) -> tuple[ChunkPartition, ...]:
    if not isinstance(value, list) or len(value) != len(CHUNK_IDS):
        raise C2FullReplacementPolicyError("V2.1 partition must contain all 13 chunks")
    parsed: list[ChunkPartition] = []
    next_ordinal = 1
    for expected_chunk, expected_total, item in zip(
        CHUNK_IDS,
        FULL_REPLACEMENT_INPUT_TOTALS,
        value,
        strict=True,
    ):
        if not isinstance(item, Mapping):
            raise C2FullReplacementPolicyError("partition item must be an object")
        _require_exact_keys(
            item,
            frozenset(
                {
                    "chunk_id",
                    "first_global_ordinal",
                    "input_total",
                    "doi_ids_sha256",
                }
            ),
            f"partition {expected_chunk}",
        )
        if (
            item["chunk_id"] != expected_chunk
            or item["first_global_ordinal"] != next_ordinal
            or item["input_total"] != expected_total
        ):
            raise C2FullReplacementPolicyError(
                f"partition {expected_chunk} violates the frozen V2.1 roster"
            )
        parsed.append(
            ChunkPartition(
                chunk_id=expected_chunk,
                first_global_ordinal=next_ordinal,
                input_total=expected_total,
                doi_ids_sha256=_require_sha256(
                    item["doi_ids_sha256"],
                    f"partition {expected_chunk}.doi_ids_sha256",
                ),
            )
        )
        next_ordinal += expected_total
    if next_ordinal != FROZEN_INPUT_TOTAL + 1:
        raise C2FullReplacementPolicyError("V2.1 partition does not total 2,463")
    return tuple(parsed)


def _parse_replacement_plan(value: Any) -> tuple[ReplacementRootPlan, ...]:
    if not isinstance(value, list) or len(value) != len(CHUNK_IDS):
        raise C2FullReplacementPolicyError(
            "V2.1 replacement plan must contain all 13 replacement chunks"
        )
    plans: list[ReplacementRootPlan] = []
    all_root_ids: set[str] = set()
    for expected_chunk, item in zip(CHUNK_IDS, value, strict=True):
        if not isinstance(item, Mapping):
            raise C2FullReplacementPolicyError("replacement plan item must be an object")
        _require_exact_keys(
            item,
            frozenset({"chunk_id", "replacement_root_id", "retired_root_ids"}),
            f"replacement plan {expected_chunk}",
        )
        if item["chunk_id"] != expected_chunk:
            raise C2FullReplacementPolicyError(
                "V2.1 replacement plan must use ordered chunks 001..013"
            )
        replacement = _require_identifier(
            item["replacement_root_id"],
            f"replacement plan {expected_chunk}.replacement_root_id",
        )
        retired_value = item["retired_root_ids"]
        if not isinstance(retired_value, list) or not retired_value:
            raise C2FullReplacementPolicyError(
                f"replacement plan {expected_chunk} must retire at least one root"
            )
        retired = tuple(
            _require_identifier(root, f"replacement plan {expected_chunk}.retired")
            for root in retired_value
        )
        if len(retired) != len(set(retired)) or replacement in retired:
            raise C2FullReplacementPolicyError(
                f"replacement plan {expected_chunk} reuses a root identifier"
            )
        if all_root_ids.intersection((replacement, *retired)):
            raise C2FullReplacementPolicyError(
                "V2.1 replacement plan reuses a root identifier"
            )
        all_root_ids.update((replacement, *retired))
        plans.append(
            ReplacementRootPlan(
                chunk_id=expected_chunk,
                replacement_root_id=replacement,
                retired_root_ids=retired,
            )
        )
    return tuple(plans)


def _parse_ordered_doi_ids(
    value: Any,
    frozen_universe: FrozenUniverse,
    partition: Sequence[ChunkPartition],
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != FROZEN_INPUT_TOTAL:
        raise C2FullReplacementPolicyError(
            "V2.1 acquisition DOI list must contain exactly 2,463 rows"
        )
    ordered = tuple(
        _require_normalized_doi(doi_id, f"ordered_doi_ids[{index}]")
        for index, doi_id in enumerate(value)
    )
    if len(ordered) != len(set(ordered)):
        raise C2FullReplacementPolicyError("V2.1 acquisition DOI list has duplicates")
    if sha256_json(list(ordered)) != frozen_universe.doi_ids_sha256:
        raise C2FullReplacementPolicyError(
            "V2.1 acquisition DOI list differs from frozen universe hash"
        )
    for item in partition:
        start = item.first_global_ordinal - 1
        if sha256_json(list(ordered[start : start + item.input_total])) != (
            item.doi_ids_sha256
        ):
            raise C2FullReplacementPolicyError(
                f"V2.1 partition {item.chunk_id} DOI hash is inconsistent"
            )
    return ordered


def compile_synthetic_policy_for_testing(
    value: Mapping[str, Any],
) -> CompiledFullReplacementPolicy:
    """Compile a complete ordered acquisition policy for test code only."""

    if not isinstance(value, Mapping):
        raise C2FullReplacementPolicyError("Synthetic policy must be an object")
    _require_exact_keys(
        value,
        frozenset(
            {
                "policy_id",
                "policy_version",
                "frozen_universe",
                "frozen_bindings",
                "partition",
                "replacement_plan",
                "ordered_doi_ids",
            }
        ),
        "synthetic policy",
    )
    policy_id = _require_identifier(value["policy_id"], "synthetic policy.policy_id")
    policy_version = _require_identifier(
        value["policy_version"],
        "synthetic policy.policy_version",
    )
    frozen_universe = _parse_frozen_universe(value["frozen_universe"])
    frozen_bindings = _parse_frozen_bindings(
        value["frozen_bindings"],
        frozen_universe,
    )
    partition = _parse_partition(value["partition"])
    replacement_plan = _parse_replacement_plan(value["replacement_plan"])
    ordered_doi_ids = _parse_ordered_doi_ids(
        value["ordered_doi_ids"],
        frozen_universe,
        partition,
    )
    return CompiledFullReplacementPolicy(
        policy_id=policy_id,
        policy_version=policy_version,
        frozen_universe=frozen_universe,
        frozen_bindings=frozen_bindings,
        partition=partition,
        replacement_plan=replacement_plan,
        ordered_doi_ids=ordered_doi_ids,
        test_policy_sha256=sha256_json(value),
    )


def load_production_policy() -> CompiledFullReplacementPolicy:
    """Fail closed until Stage B commits a resource and compiled byte digest."""

    raise C2FullReplacementPolicyError(
        "C2 full-replacement V2.1 is Stage-A only: no Git-reviewed internal "
        "production policy resource and compiled SHA-256 are pinned"
    )

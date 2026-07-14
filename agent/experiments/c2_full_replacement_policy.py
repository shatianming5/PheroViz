"""Fixed-policy primitives for the isolated C2 full-replacement V2 path.

Stage A intentionally has no production policy resource.  The only loader used
by the production CLI therefore fails closed.  Synthetic policies are accepted
solely through explicitly test-named, in-process helpers.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import ProvenanceError, sha256_json


C2_FULL_REPLACEMENT_V2_VERSION = "2.0-stagea"
CHUNK_IDS = tuple(f"{number:03d}" for number in range(1, 14))
ATTEMPT_IDS = ("initial", "retry1", "retry2")
P_DISPOSITIONS = ("P1", "P2", "P3_4", "P5PLUS")
P1_NONINFERENTIAL_CLUSTER = "P1_NONINFERENTIAL"
FROZEN_INPUT_TOTAL = 2_463
FINAL_CHUNK_INPUT_TOTAL = 63
FULL_REPLACEMENT_INPUT_TOTALS = tuple(
    200 if chunk_id != "013" else FINAL_CHUNK_INPUT_TOTAL
    for chunk_id in CHUNK_IDS
)
TERMINAL_OUTCOME_STATUSES = frozenset(
    {
        "DOWNLOADED",
        "NO_SOURCE_DATA",
        "NO_FIGURES",
        "NO_USABLE_CONTENT",
        "POLICY_REJECTED",
        "DOWNLOAD_FAILED",
        "RETRY_EXHAUSTED",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOI_RE = re.compile(r"^10\.[0-9]{4,9}/\S+$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class C2FullReplacementPolicyError(ProvenanceError):
    """Raised when an internal V2 policy is unavailable or malformed."""


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
class PolicyRow:
    global_ordinal: int
    doi_id: str
    p_disposition: str
    independent_cluster_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_ordinal": self.global_ordinal,
            "doi_id": self.doi_id,
            "p_disposition": self.p_disposition,
            "independent_cluster_id": self.independent_cluster_id,
        }


@dataclass(frozen=True)
class StratumAggregate:
    p_disposition: str
    doi_count: int
    independent_cluster_count: int
    inference_eligible: bool
    deficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_disposition": self.p_disposition,
            "doi_count": self.doi_count,
            "independent_cluster_count": self.independent_cluster_count,
            "inference_eligible": self.inference_eligible,
            "deficient": self.deficient,
        }


@dataclass(frozen=True)
class MappingAggregation:
    strata: tuple[StratumAggregate, ...]
    deficient_strata: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class CompiledFullReplacementPolicy:
    """An immutable in-process policy object used only by Stage-A tests."""

    policy_id: str
    policy_version: str
    frozen_universe: FrozenUniverse
    partition: tuple[ChunkPartition, ...]
    replacement_plan: tuple[ReplacementRootPlan, ...]
    doi_p_cluster_map: tuple[PolicyRow, ...]
    raw_mapping_source_manifest: tuple[Mapping[str, Any], ...]
    raw_evidence_commitments: tuple[Mapping[str, Any], ...]
    test_policy_sha256: str

    @property
    def is_test_only(self) -> bool:
        return True

    def rows_for_chunk(self, chunk_id: str) -> tuple[PolicyRow, ...]:
        partition = next(
            (item for item in self.partition if item.chunk_id == chunk_id),
            None,
        )
        if partition is None:
            raise C2FullReplacementPolicyError(f"Unknown policy chunk: {chunk_id}")
        start = partition.first_global_ordinal - 1
        return self.doi_p_cluster_map[start : start + partition.input_total]

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


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise C2FullReplacementPolicyError(
            f"{label} fields must be exactly {sorted(expected)!r}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C2FullReplacementPolicyError(f"{label} must be a full SHA-256 digest")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise C2FullReplacementPolicyError(f"{label} is not a valid identifier")
    return value


def _require_doi(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.casefold()
        or _DOI_RE.fullmatch(value) is None
    ):
        raise C2FullReplacementPolicyError(f"{label} must be a normalized DOI")
    return value


def _validate_policy_row(row: PolicyRow, label: str) -> None:
    if not isinstance(row.global_ordinal, int) or row.global_ordinal < 1:
        raise C2FullReplacementPolicyError(f"{label}.global_ordinal must be positive")
    _require_doi(row.doi_id, f"{label}.doi_id")
    if row.p_disposition not in P_DISPOSITIONS:
        raise C2FullReplacementPolicyError(f"{label} has an unknown P disposition")
    _require_identifier(row.independent_cluster_id, f"{label}.independent_cluster_id")
    if row.p_disposition == "P1":
        if row.independent_cluster_id != P1_NONINFERENTIAL_CLUSTER:
            raise C2FullReplacementPolicyError(
                f"{label} P1 cluster must be {P1_NONINFERENTIAL_CLUSTER}"
            )
    elif row.independent_cluster_id == P1_NONINFERENTIAL_CLUSTER:
        raise C2FullReplacementPolicyError(
            f"{label} non-P1 cluster cannot use {P1_NONINFERENTIAL_CLUSTER}"
        )


def aggregate_policy_rows(rows: Sequence[PolicyRow]) -> MappingAggregation:
    """Derive every P count, gate, and status from one fixed mapping."""

    clusters: dict[str, set[str]] = defaultdict(set)
    doi_counts: dict[str, int] = defaultdict(int)
    cluster_dispositions: dict[str, str] = {}
    seen_ordinals: set[int] = set()
    seen_dois: set[str] = set()
    for index, row in enumerate(rows):
        _validate_policy_row(row, f"mapping row {index}")
        if row.global_ordinal in seen_ordinals:
            raise C2FullReplacementPolicyError("Policy mapping has duplicate ordinals")
        if row.doi_id in seen_dois:
            raise C2FullReplacementPolicyError("Policy mapping has duplicate DOI IDs")
        seen_ordinals.add(row.global_ordinal)
        seen_dois.add(row.doi_id)
        doi_counts[row.p_disposition] += 1
        if row.p_disposition == "P1":
            continue
        prior = cluster_dispositions.setdefault(
            row.independent_cluster_id,
            row.p_disposition,
        )
        if prior != row.p_disposition:
            raise C2FullReplacementPolicyError(
                "A non-P1 independent cluster cannot span P dispositions"
            )
        clusters[row.p_disposition].add(row.independent_cluster_id)

    strata: list[StratumAggregate] = []
    for disposition in P_DISPOSITIONS:
        doi_count = doi_counts[disposition]
        if disposition == "P1":
            strata.append(
                StratumAggregate(
                    p_disposition=disposition,
                    doi_count=doi_count,
                    independent_cluster_count=0,
                    inference_eligible=False,
                    deficient=doi_count > 0,
                )
            )
            continue
        cluster_count = len(clusters[disposition])
        deficient = doi_count > 0 and cluster_count < 2
        strata.append(
            StratumAggregate(
                p_disposition=disposition,
                doi_count=doi_count,
                independent_cluster_count=cluster_count,
                inference_eligible=doi_count > 0 and not deficient,
                deficient=deficient,
            )
        )
    deficient = tuple(item.p_disposition for item in strata if item.deficient)
    if not deficient:
        status = "ADMITTED"
    elif "P5PLUS" in deficient:
        status = "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS"
    else:
        status = {
            "P1": "BLOCKED_INSUFFICIENT_INDEPENDENT_P1",
            "P2": "BLOCKED_INSUFFICIENT_INDEPENDENT_P2",
            "P3_4": "BLOCKED_INSUFFICIENT_INDEPENDENT_P3_4",
        }[deficient[0]]
    return MappingAggregation(
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
    input_total = value["input_total"]
    if input_total != FROZEN_INPUT_TOTAL:
        raise C2FullReplacementPolicyError(
            "V2 frozen universe must contain exactly 2,463 DOI inputs"
        )
    return FrozenUniverse(
        sha256=_require_sha256(value["sha256"], "frozen_universe.sha256"),
        doi_ids_sha256=_require_sha256(
            value["doi_ids_sha256"],
            "frozen_universe.doi_ids_sha256",
        ),
        input_total=input_total,
    )


def _parse_partition(value: Any) -> tuple[ChunkPartition, ...]:
    if not isinstance(value, list) or len(value) != len(CHUNK_IDS):
        raise C2FullReplacementPolicyError("V2 partition must contain all 13 chunks")
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
        if item["chunk_id"] != expected_chunk:
            raise C2FullReplacementPolicyError(
                "V2 partition must use the exact ordered roster 001..013"
            )
        if item["first_global_ordinal"] != next_ordinal:
            raise C2FullReplacementPolicyError(
                f"partition {expected_chunk} has an invalid first ordinal"
            )
        if item["input_total"] != expected_total:
            raise C2FullReplacementPolicyError(
                f"partition {expected_chunk} must contain {expected_total} inputs"
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
        raise C2FullReplacementPolicyError("V2 partition does not total 2,463")
    return tuple(parsed)


def _parse_replacement_plan(value: Any) -> tuple[ReplacementRootPlan, ...]:
    if not isinstance(value, list) or len(value) != len(CHUNK_IDS):
        raise C2FullReplacementPolicyError(
            "V2 replacement plan must contain all 13 replacement chunks"
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
                "V2 replacement plan must use the exact ordered roster 001..013"
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
            _require_identifier(
                root_id,
                f"replacement plan {expected_chunk}.retired_root_ids",
            )
            for root_id in retired_value
        )
        if len(retired) != len(set(retired)) or replacement in retired:
            raise C2FullReplacementPolicyError(
                f"replacement plan {expected_chunk} reuses a root identifier"
            )
        if all_root_ids.intersection((replacement, *retired)):
            raise C2FullReplacementPolicyError(
                "V2 replacement plan reuses a replacement or retired root identifier"
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


def _parse_mapping(
    value: Any,
    partition: Sequence[ChunkPartition],
    frozen_universe: FrozenUniverse,
) -> tuple[PolicyRow, ...]:
    if not isinstance(value, list) or len(value) != FROZEN_INPUT_TOTAL:
        raise C2FullReplacementPolicyError(
            "V2 policy mapping must contain exactly 2,463 ordered rows"
        )
    parsed: list[PolicyRow] = []
    for ordinal, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise C2FullReplacementPolicyError("policy mapping row must be an object")
        _require_exact_keys(
            item,
            frozenset(
                {
                    "global_ordinal",
                    "doi_id",
                    "p_disposition",
                    "independent_cluster_id",
                }
            ),
            f"mapping row {ordinal}",
        )
        if item["global_ordinal"] != ordinal:
            raise C2FullReplacementPolicyError(
                "V2 policy mapping must be in exact global ordinal order"
            )
        row = PolicyRow(
            global_ordinal=ordinal,
            doi_id=_require_doi(item["doi_id"], f"mapping row {ordinal}.doi_id"),
            p_disposition=item["p_disposition"],
            independent_cluster_id=_require_identifier(
                item["independent_cluster_id"],
                f"mapping row {ordinal}.independent_cluster_id",
            ),
        )
        _validate_policy_row(row, f"mapping row {ordinal}")
        parsed.append(row)
    aggregation = aggregate_policy_rows(parsed)
    del aggregation
    doi_ids = [row.doi_id for row in parsed]
    if sha256_json(doi_ids) != frozen_universe.doi_ids_sha256:
        raise C2FullReplacementPolicyError(
            "V2 policy mapping DOI list does not match frozen universe hash"
        )
    for chunk in partition:
        start = chunk.first_global_ordinal - 1
        chunk_dois = [row.doi_id for row in parsed[start : start + chunk.input_total]]
        if sha256_json(chunk_dois) != chunk.doi_ids_sha256:
            raise C2FullReplacementPolicyError(
                f"V2 mapping DOI list does not match partition {chunk.chunk_id}"
            )
    return tuple(parsed)


def compile_synthetic_policy_for_testing(
    value: Mapping[str, Any],
) -> CompiledFullReplacementPolicy:
    """Compile a complete synthetic policy for test code only.

    This function deliberately accepts an in-memory mapping rather than a path,
    resource, environment variable, or digest selector.  It is not used by the
    production CLI or production resolver.
    """

    if not isinstance(value, Mapping):
        raise C2FullReplacementPolicyError("Synthetic policy must be an object")
    _require_exact_keys(
        value,
        frozenset(
            {
                "policy_id",
                "policy_version",
                "frozen_universe",
                "partition",
                "replacement_plan",
                "doi_p_cluster_map",
                "raw_mapping_source_manifest",
                "raw_evidence_commitments",
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
    partition = _parse_partition(value["partition"])
    replacement_plan = _parse_replacement_plan(value["replacement_plan"])
    mapping = _parse_mapping(value["doi_p_cluster_map"], partition, frozen_universe)
    source_manifest = value["raw_mapping_source_manifest"]
    evidence_commitments = value["raw_evidence_commitments"]
    if not isinstance(source_manifest, list) or not isinstance(evidence_commitments, list):
        raise C2FullReplacementPolicyError(
            "Synthetic raw source manifests and evidence commitments must be arrays"
        )
    if not all(isinstance(item, Mapping) for item in source_manifest):
        raise C2FullReplacementPolicyError(
            "Synthetic raw mapping source manifest entries must be objects"
        )
    if not all(isinstance(item, Mapping) for item in evidence_commitments):
        raise C2FullReplacementPolicyError(
            "Synthetic raw evidence commitments entries must be objects"
        )
    return CompiledFullReplacementPolicy(
        policy_id=policy_id,
        policy_version=policy_version,
        frozen_universe=frozen_universe,
        partition=partition,
        replacement_plan=replacement_plan,
        doi_p_cluster_map=mapping,
        raw_mapping_source_manifest=tuple(dict(item) for item in source_manifest),
        raw_evidence_commitments=tuple(dict(item) for item in evidence_commitments),
        test_policy_sha256=sha256_json(value),
    )


def load_production_policy() -> CompiledFullReplacementPolicy:
    """Fail closed until Stage B commits a resource and compiled byte digest."""

    raise C2FullReplacementPolicyError(
        "C2 full-replacement V2 is Stage-A only: no Git-reviewed internal "
        "production policy resource and compiled SHA-256 are pinned"
    )

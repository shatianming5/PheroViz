from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import experiments.c2_full_replacement_stageb_policy as stageb_policy
from experiments.c2_full_replacement_finalizer import (
    C2FullReplacementError,
    prepare_full_replacement_finalization,
)
from experiments.c2_full_replacement_policy import (
    CHUNK_IDS,
    FROZEN_INPUT_TOTAL,
    FULL_REPLACEMENT_INPUT_TOTALS,
    C2FullReplacementPolicyError,
    load_production_policy,
)
from experiments.cli import _build_parser
from experiments.models import sha256_json


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seal_policy(policy: dict[str, Any]) -> None:
    policy["policy_sha256"] = sha256_json(
        {key: value for key, value in policy.items() if key != "policy_sha256"}
    )


def _synthetic_policy() -> dict[str, Any]:
    """Return a test-only shape fixture, never a production policy resource."""

    ordered_doi_ids = [
        f"10.9000/stageb-synthetic-{ordinal:04d}"
        for ordinal in range(1, FROZEN_INPUT_TOTAL + 1)
    ]
    partition: list[dict[str, Any]] = []
    first_ordinal = 1
    for chunk_id, input_total in zip(
        CHUNK_IDS,
        FULL_REPLACEMENT_INPUT_TOTALS,
        strict=True,
    ):
        partition.append(
            {
                "chunk_id": chunk_id,
                "first_global_ordinal": first_ordinal,
                "input_total": input_total,
                "doi_ids_sha256": sha256_json(
                    ordered_doi_ids[
                        first_ordinal - 1 : first_ordinal - 1 + input_total
                    ]
                ),
            }
        )
        first_ordinal += input_total
    commitments: list[dict[str, Any]] = []

    def append_commitment(
        role: str,
        *,
        chunk_id: str | None,
        ordinal: int | None,
        parent_doi_id: str | None,
    ) -> None:
        order = len(commitments) + 1
        commitments.append(
            {
                "commitment_id": f"synthetic-commitment-{order:03d}",
                "role": role,
                "chunk_id_or_null": chunk_id,
                "frozen_input_ordinal_or_null": ordinal,
                "parent_doi_id_or_null": parent_doi_id,
                "relative_path": f"synthetic/{order:03d}-{role}.json",
                "file_sha256": _digest(f"file-{role}"),
                "byte_count": order,
                "semantic_sha256": _digest(f"semantic-{role}"),
                "complete": True,
                "sealed": True,
                "clean_code": True,
                "model_result_selected": False,
                "record_order": order,
            }
        )

    for role in stageb_policy.STAGE_B_CHUNK_EVIDENCE_ROLES:
        append_commitment(role, chunk_id="001", ordinal=None, parent_doi_id=None)
    for role in stageb_policy.STAGE_B_DOWNLOADED_DOI_EVIDENCE_ROLES:
        append_commitment(
            role,
            chunk_id="001",
            ordinal=1,
            parent_doi_id=ordered_doi_ids[0],
        )
    for role in stageb_policy.STAGE_B_COLLECTION_EVIDENCE_ROLES:
        append_commitment(role, chunk_id=None, ordinal=None, parent_doi_id=None)

    policy: dict[str, Any] = {
        "schema_version": "c2-stageb-policy-v1",
        "policy_type": "C2_FULL_REPLACEMENT_STAGEB_NONCLASSIFICATION_POLICY",
        "policy_id": "synthetic-stageb-policy",
        "policy_version": "test-v1",
        "policy_sha256": "",
        "frozen_universe": {
            "universe_file_sha256": _digest("synthetic-universe-file"),
            "ordered_doi_ids_sha256": sha256_json(ordered_doi_ids),
            "input_total": FROZEN_INPUT_TOTAL,
            "ordered_doi_ids": ordered_doi_ids,
            "chunk_file_sha256_by_id": {
                chunk_id: _digest(f"synthetic-chunk-file-{chunk_id}")
                for chunk_id in CHUNK_IDS
            },
            "chunk_concat_sha256": _digest("synthetic-chunk-concat"),
        },
        "partition": partition,
        "replacement_plan": [
            {
                "chunk_id": chunk_id,
                "replacement_root_id": f"synthetic-replacement-{chunk_id}",
                "retired_root_ids": [f"synthetic-retired-{chunk_id}"],
                "partial_root": False,
            }
            for chunk_id in CHUNK_IDS
        ],
        "nonclassification_registries": {
            "canonical_json": {
                "id": "CANONICAL_JSON_UTF8_SORTED_V1",
                "version": "v1",
                "sha256": _digest("canonical-json"),
            },
            "error_enum": {
                "id": "C2_STAGEB_ERROR_ENUM",
                "version": "v1",
                "sha256": _digest("error-enum"),
            },
            "code": [
                {
                    "id": "SYNTHETIC_SOURCE_BUILDER",
                    "commit_full": "a" * 40,
                    "sha256": _digest("source-builder"),
                }
            ],
            "rules": [
                {
                    "id": "SYNTHETIC_SOURCE_ONLY_RULE",
                    "version": "v1",
                    "sha256": _digest("source-only-rule"),
                }
            ],
            "schemas": [
                {
                    "id": stageb_policy.STAGE_B_POLICY_SCHEMA_ID,
                    "sha256": stageb_policy.STAGE_B_POLICY_SCHEMA_SHA256,
                }
            ],
        },
        "evidence_role_layout": {
            "layout_id": "C2_STAGEB_EVIDENCE_LAYOUT",
            "layout_version": "v1",
            "root_relative_only": True,
            "descriptor_read_required": True,
            "no_symlink": True,
            "chunk_roles": list(stageb_policy.STAGE_B_CHUNK_EVIDENCE_ROLES),
            "downloaded_doi_roles": list(
                stageb_policy.STAGE_B_DOWNLOADED_DOI_EVIDENCE_ROLES
            ),
            "collection_roles": list(stageb_policy.STAGE_B_COLLECTION_EVIDENCE_ROLES),
            "required_cross_role_fields": list(
                stageb_policy.STAGE_B_REQUIRED_CROSS_ROLE_FIELDS
            ),
        },
        "evidence_commitments": commitments,
    }
    _seal_policy(policy)
    return policy


def _policy_bytes(policy: dict[str, Any]) -> bytes:
    return json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _compile(policy: dict[str, Any]) -> stageb_policy.CompiledStageBProductionPolicy:
    resource_bytes = _policy_bytes(policy)
    return stageb_policy.compile_stage_b_policy_resource_for_testing(
        resource_bytes,
        expected_resource_sha256=hashlib.sha256(resource_bytes).hexdigest(),
    )


def test_nonclassification_test_fixture_compiles_but_production_route_stays_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _synthetic_policy()
    resource_bytes = _policy_bytes(policy)
    compiled = _compile(policy)
    assert compiled.frozen_universe.input_total == FROZEN_INPUT_TOTAL
    assert compiled.partition[-1].chunk_id == "013"
    assert len(compiled.evidence_commitments) == (
        len(stageb_policy.STAGE_B_CHUNK_EVIDENCE_ROLES)
        + len(stageb_policy.STAGE_B_DOWNLOADED_DOI_EVIDENCE_ROLES)
        + len(stageb_policy.STAGE_B_COLLECTION_EVIDENCE_ROLES)
    )

    monkeypatch.setattr(
        stageb_policy,
        "_STAGE_B_POLICY_RESOURCE_SHA256",
        hashlib.sha256(resource_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        stageb_policy,
        "_read_compiled_stage_b_resource_bytes",
        lambda: resource_bytes,
    )
    with pytest.raises(C2FullReplacementPolicyError, match="intentionally non-admissive"):
        stageb_policy.load_stage_b_production_policy()


def test_production_resolver_and_cli_have_no_selector_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert list(inspect.signature(load_production_policy).parameters) == []
    assert list(inspect.signature(stageb_policy.load_stage_b_production_policy).parameters) == []
    monkeypatch.setenv("C2_STAGEB_POLICY_RESOURCE", "/attacker-selected-policy.json")
    monkeypatch.setenv("C2_STAGEB_POLICY_SHA256", "f" * 64)
    monkeypatch.setenv("C2_STAGEB_EVIDENCE", "/attacker-selected-evidence")
    with pytest.raises(C2FullReplacementPolicyError, match="Stage-A only"):
        load_production_policy()
    with pytest.raises(C2FullReplacementError, match="Stage-A only"):
        prepare_full_replacement_finalization(Path("attacker-manifest.json"))

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
        for token in ("policy", "resource", "digest", "evidence", "map")
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "c2-full-replacement-finalize",
                "unavailable-manifest.json",
                "--out",
                "unavailable-output.json",
                "--policy",
                "attacker-selected-policy.json",
            ]
        )


def test_resource_digest_mismatch_rejects_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_bytes = _policy_bytes(_synthetic_policy())
    monkeypatch.setattr(stageb_policy, "_STAGE_B_POLICY_RESOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(
        stageb_policy,
        "_read_compiled_stage_b_resource_bytes",
        lambda: resource_bytes,
    )
    with pytest.raises(C2FullReplacementPolicyError, match="compiled SHA-256"):
        stageb_policy.load_stage_b_production_policy()


def test_schema_rejects_unknown_resource_fields() -> None:
    policy = _synthetic_policy()
    policy["unknown_selector"] = "not-allowed"
    _seal_policy(policy)
    with pytest.raises(C2FullReplacementPolicyError, match="schema validation failed"):
        _compile(policy)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("doi_p_cluster_map", []),
        ("doi_to_cluster", {}),
        ("p1_sentinel", "P1"),
        ("stratum_counts", {}),
        ("target_distribution", {"P2": 2}),
        ("coverage_outcome", "ADMITTED"),
        ("report_status", "ADMITTED"),
        ("scientific_outcome", "SUPPORTED"),
        ("source_classifications", []),
    ],
)
def test_forbidden_classification_and_outcome_fields_are_rejected(
    field: str,
    value: Any,
) -> None:
    policy = _synthetic_policy()
    policy[field] = value
    _seal_policy(policy)
    with pytest.raises(C2FullReplacementPolicyError, match="schema validation failed"):
        _compile(policy)


def test_forbidden_semantics_in_an_allowed_registry_field_are_rejected() -> None:
    policy = _synthetic_policy()
    policy["nonclassification_registries"]["rules"][0]["id"] = "P1_SENTINEL_RULE"
    _seal_policy(policy)
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="identifier encodes a forbidden classification/outcome",
    ):
        _compile(policy)


def test_p_label_encoded_in_a_commitment_path_is_rejected() -> None:
    policy = _synthetic_policy()
    policy["evidence_commitments"][0]["relative_path"] = "synthetic/P1-map.json"
    _seal_policy(policy)
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="identifier encodes a forbidden classification/outcome",
    ):
        _compile(policy)


@pytest.mark.parametrize("root_id", ["P1-sentinel-root", "P3_4-target-root"])
def test_p_label_encoded_in_a_root_identity_is_rejected(root_id: str) -> None:
    policy = _synthetic_policy()
    policy["replacement_plan"][0]["replacement_root_id"] = root_id
    _seal_policy(policy)
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="identifier encodes a forbidden classification/outcome",
    ):
        _compile(policy)


def test_missing_approved_commitments_and_incomplete_role_layout_are_rejected() -> None:
    no_commitments = _synthetic_policy()
    no_commitments["evidence_commitments"] = []
    _seal_policy(no_commitments)
    with pytest.raises(C2FullReplacementPolicyError, match="schema validation failed"):
        _compile(no_commitments)

    missing_role = _synthetic_policy()
    missing_role["evidence_commitments"].pop()
    _seal_policy(missing_role)
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="do not cover the closed role layout",
    ):
        _compile(missing_role)


def test_policy_must_bind_the_compiled_policy_schema_digest() -> None:
    policy = deepcopy(_synthetic_policy())
    policy["nonclassification_registries"]["schemas"][0]["sha256"] = "0" * 64
    _seal_policy(policy)
    with pytest.raises(
        C2FullReplacementPolicyError,
        match="compiled Stage-B policy schema digest",
    ):
        _compile(policy)

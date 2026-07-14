"""Isolated C2 full-replacement V2 finalization.

The production entry point resolves only the package-internal production policy
and therefore fails closed during Stage A.  Test-named helpers accept an
in-memory synthetic policy to exercise the complete raw-attestation path.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .c2_full_replacement_evidence import (
    C2FullReplacementEvidenceError,
    EvidenceArtifact,
    ValidatedRawEvidence,
    _validate_schema,
    load_and_validate_raw_evidence,
)
from .c2_full_replacement_policy import (
    ATTEMPT_IDS,
    CHUNK_IDS,
    P1_NONINFERENTIAL_CLUSTER,
    C2FullReplacementPolicyError,
    CompiledFullReplacementPolicy,
    aggregate_policy_rows,
    load_production_policy,
)
from .models import (
    ProvenanceError,
    SecureOutputTarget,
    normalize_trusted_output_path,
    open_secure_output_target,
    sha256_json,
    verify_secure_output_target,
)


class C2FullReplacementError(ProvenanceError):
    """Raised when V2 full-replacement evidence cannot be finalized safely."""


@dataclass(frozen=True)
class ValidatedFullReplacementAdmission:
    """A structural guard carrying verified inputs and a derived V2 report."""

    report: Mapping[str, Any]
    policy: CompiledFullReplacementPolicy
    evidence: ValidatedRawEvidence


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: item for key, item in value.items() if key not in omitted}


def _report_chunks(
    policy: CompiledFullReplacementPolicy,
    evidence: ValidatedRawEvidence,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for chunk_evidence in evidence.chunks:
        chunk_id = chunk_evidence.chunk_id
        plan = policy.root_plan_for_chunk(chunk_id)
        partition = policy.partition_for_chunk(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "replacement_root_id": plan.replacement_root_id,
                "retired_root_ids": list(plan.retired_root_ids),
                "input_total": partition.input_total,
                "input_doi_ids_sha256": partition.doi_ids_sha256,
                "canonical_mapping": chunk_evidence.canonical_mapping.to_report_dict(),
            }
        )
    return chunks


def _build_test_report(
    policy: CompiledFullReplacementPolicy,
    evidence: ValidatedRawEvidence,
) -> dict[str, Any]:
    aggregation = aggregate_policy_rows(policy.doi_p_cluster_map)
    manifest = evidence.manifest
    report: dict[str, Any] = {
        "schema_version": "2.0-stagea",
        "finalizer": "c2_full_replacement_admission",
        "status": aggregation.status,
        "claim_status": "UNSUPPORTED"
        if aggregation.deficient_strata
        else "SUPPORTED",
        "claim_scope": "TERMINAL_ADMISSION_ONLY",
        "trend_status": "NOT_RUN",
        "equivalence_status": "NOT_RUN",
        "policy_binding": {
            "authority": "SYNTHETIC_TEST_ONLY",
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "test_policy_sha256": policy.test_policy_sha256,
        },
        "p1_model": {
            "p_disposition": "P1",
            "independent_cluster_literal": P1_NONINFERENTIAL_CLUSTER,
            "countable_independent_cluster_count": 0,
            "inference_eligible": False,
        },
        "admission_manifest": {
            "file_sha256": evidence.manifest_artifact.sha256,
            "manifest_hash": manifest["manifest_hash"],
        },
        "frozen_universe": policy.frozen_universe.to_dict(),
        "code": dict(manifest["code"]),
        "chunks": _report_chunks(policy, evidence),
        "canonical_attempt_ledger": [
            attempt.to_report_dict() for attempt in evidence.attempt_ledger
        ],
        "source_doi_ids_sha256": policy.frozen_universe.doi_ids_sha256,
        "source_doi_count": policy.frozen_universe.input_total,
        "strata": [item.to_dict() for item in aggregation.strata],
        "deficient_strata": list(aggregation.deficient_strata),
    }
    report["final_report_hash"] = sha256_json(report)
    return report


def _validate_ledger_structure(
    ledger: Sequence[Mapping[str, Any]],
    policy: CompiledFullReplacementPolicy,
) -> None:
    expected_ledger_keys = [
        (chunk_id, attempt_id)
        for chunk_id in CHUNK_IDS
        for attempt_id in ATTEMPT_IDS
    ]
    actual_ledger_keys = [
        (entry["chunk_id"], entry["attempt_id"])
        for entry in ledger
    ]
    if actual_ledger_keys != expected_ledger_keys:
        raise C2FullReplacementError(
            "V2 canonical attempt ledger is not complete and ordered"
        )
    for entry in ledger:
        expected_rows = policy.rows_for_chunk(entry["chunk_id"])
        outcomes = entry["outcomes"]
        if len(outcomes) != len(expected_rows):
            raise C2FullReplacementError(
                f"V2 canonical ledger has an invalid outcome count for "
                f"chunk {entry['chunk_id']}"
            )
        for local_ordinal, (outcome, policy_row) in enumerate(
            zip(outcomes, expected_rows, strict=True),
            start=1,
        ):
            if (
                outcome["global_ordinal"] != policy_row.global_ordinal
                or outcome["local_ordinal"] != local_ordinal
                or outcome["doi_id"] != policy_row.doi_id
            ):
                raise C2FullReplacementError(
                    "V2 canonical ledger differs from the compiled DOI/ordinal "
                    "mapping"
                )


def validate_synthetic_final_report_for_testing(
    report: Mapping[str, Any],
    policy: CompiledFullReplacementPolicy,
) -> None:
    """Validate a test-only output by deriving all P fields from the policy."""

    if not policy.is_test_only:
        raise C2FullReplacementError(
            "Stage-A final report validation accepts only a synthetic policy"
        )
    _validate_schema(
        report,
        "c2_full_replacement_final_report_v2.schema.json",
        "V2 final report",
    )
    if sha256_json(_without(report, "final_report_hash")) != report["final_report_hash"]:
        raise C2FullReplacementError("V2 final report failed its semantic hash")
    expected_binding = {
        "authority": "SYNTHETIC_TEST_ONLY",
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "test_policy_sha256": policy.test_policy_sha256,
    }
    if report["policy_binding"] != expected_binding:
        raise C2FullReplacementError("V2 final report policy binding is inconsistent")
    if report["frozen_universe"] != policy.frozen_universe.to_dict():
        raise C2FullReplacementError("V2 final report frozen universe is inconsistent")
    if (
        report["source_doi_count"] != policy.frozen_universe.input_total
        or report["source_doi_ids_sha256"] != policy.frozen_universe.doi_ids_sha256
    ):
        raise C2FullReplacementError("V2 final report source DOI binding is inconsistent")
    if report["p1_model"] != {
        "p_disposition": "P1",
        "independent_cluster_literal": P1_NONINFERENTIAL_CLUSTER,
        "countable_independent_cluster_count": 0,
        "inference_eligible": False,
    }:
        raise C2FullReplacementError("V2 final report P1 model is inconsistent")
    if [chunk["chunk_id"] for chunk in report["chunks"]] != list(CHUNK_IDS):
        raise C2FullReplacementError("V2 final report chunk roster is incomplete")
    for chunk in report["chunks"]:
        plan = policy.root_plan_for_chunk(chunk["chunk_id"])
        partition = policy.partition_for_chunk(chunk["chunk_id"])
        if (
            chunk["replacement_root_id"] != plan.replacement_root_id
            or tuple(chunk["retired_root_ids"]) != plan.retired_root_ids
            or chunk["input_total"] != partition.input_total
            or chunk["input_doi_ids_sha256"] != partition.doi_ids_sha256
        ):
            raise C2FullReplacementError(
                f"V2 final report chunk {chunk['chunk_id']} differs from policy"
            )
    _validate_ledger_structure(report["canonical_attempt_ledger"], policy)
    aggregation = aggregate_policy_rows(policy.doi_p_cluster_map)
    expected_strata = [item.to_dict() for item in aggregation.strata]
    if report["strata"] != expected_strata:
        raise C2FullReplacementError(
            "V2 final report P-stratum results were not derived from policy"
        )
    if report["deficient_strata"] != list(aggregation.deficient_strata):
        raise C2FullReplacementError("V2 final report deficiency list is inconsistent")
    if report["status"] != aggregation.status:
        raise C2FullReplacementError("V2 final report gate status is inconsistent")
    expected_claim = "UNSUPPORTED" if aggregation.deficient_strata else "SUPPORTED"
    if report["claim_status"] != expected_claim:
        raise C2FullReplacementError("V2 final report claim status is inconsistent")
    if (
        report["trend_status"] != "NOT_RUN"
        or report["equivalence_status"] != "NOT_RUN"
        or report["claim_scope"] != "TERMINAL_ADMISSION_ONLY"
    ):
        raise C2FullReplacementError(
            "V2 finalizer must not create trend or equivalence analyses"
        )


def prepare_full_replacement_finalization_for_testing(
    manifest_path: Path,
    policy: CompiledFullReplacementPolicy,
) -> ValidatedFullReplacementAdmission:
    """Build a report with the only permitted Stage-A synthetic-policy injection."""

    if not policy.is_test_only:
        raise C2FullReplacementError(
            "Stage-A finalization accepts only a synthetic in-process policy"
        )
    evidence: ValidatedRawEvidence | None = None
    try:
        evidence = load_and_validate_raw_evidence(manifest_path, policy)
        report = _build_test_report(policy, evidence)
        validate_synthetic_final_report_for_testing(report, policy)
        return ValidatedFullReplacementAdmission(
            report=report,
            policy=policy,
            evidence=evidence,
        )
    except (C2FullReplacementEvidenceError, C2FullReplacementPolicyError) as exc:
        if evidence is not None:
            evidence.close()
        raise C2FullReplacementError(str(exc)) from exc
    except Exception:
        if evidence is not None:
            evidence.close()
        raise


def prepare_full_replacement_finalization(
    manifest_path: Path,
) -> ValidatedFullReplacementAdmission:
    """Production resolver: intentionally unavailable until Stage B."""

    del manifest_path
    try:
        load_production_policy()
    except C2FullReplacementPolicyError as exc:
        raise C2FullReplacementError(str(exc)) from exc
    raise AssertionError("A Stage-B production policy resolver must not return in Stage A")


def _normalized_v2_output_path(path: Path) -> Path:
    raw = Path(os.fspath(path))
    if any(part in {".", ".."} for part in raw.parts):
        raise C2FullReplacementError(
            "V2 final report output path must not contain dot traversal"
        )
    try:
        return normalize_trusted_output_path(path)
    except ProvenanceError as exc:
        raise C2FullReplacementError(
            f"Cannot normalize V2 final report output path: {path}"
        ) from exc


def _reject_output_evidence_collision(
    target: SecureOutputTarget,
    evidence: ValidatedRawEvidence,
) -> None:
    output_parent_path = target.final_path.parent
    try:
        target.final_path.relative_to(evidence.root.path)
    except ValueError:
        pass
    else:
        raise C2FullReplacementError(
            "V2 final report output must be outside the evidence root"
        )
    try:
        evidence.root.path.relative_to(output_parent_path)
    except ValueError:
        pass
    else:
        raise C2FullReplacementError(
            "V2 output parent and evidence root must be distinct and non-containing"
        )
    output_parent = os.fstat(target.parent_fd)
    evidence_root = os.fstat(evidence.root.descriptor)
    if (
        output_parent.st_dev == evidence_root.st_dev
        and output_parent.st_ino == evidence_root.st_ino
    ):
        raise C2FullReplacementError(
            "V2 final report output parent aliases the evidence root"
        )
    try:
        existing = os.stat(
            target.leaf_name,
            dir_fd=target.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise C2FullReplacementError(
            f"Cannot inspect V2 final report output: {target.final_path}"
        ) from exc
    for artifact in evidence.input_artifacts:
        if target.final_path == artifact.absolute_path:
            raise C2FullReplacementError(
                "V2 final report output aliases a validated evidence path"
            )
        if (
            existing is not None
            and existing.st_dev == artifact.device
            and existing.st_ino == artifact.inode
        ):
            raise C2FullReplacementError(
                "V2 final report output aliases a validated evidence inode"
            )
    if existing is not None:
        raise C2FullReplacementError(
            "V2 final report output already exists; no-replace publication refuses it"
        )


def _linkat_no_replace_supported() -> bool:
    return (
        os.link in os.supports_dir_fd
        and os.link in os.supports_follow_symlinks
        and os.unlink in os.supports_dir_fd
        and hasattr(os, "O_NOFOLLOW")
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise C2FullReplacementError("Cannot write V2 output staging file")
        remaining = remaining[written:]


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


def _publish_json_no_replace(
    target: SecureOutputTarget,
    report: Mapping[str, Any],
    input_artifacts: Sequence[EvidenceArtifact],
) -> None:
    if not _linkat_no_replace_supported():
        raise C2FullReplacementError(
            "Descriptor-relative linkat-style no-replace publication is unsupported"
        )
    encoded = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = -1
    staging_name = ""
    published = False
    try:
        for _ in range(128):
            staging_name = f".{target.leaf_name}.{secrets.token_hex(16)}.stagea"
            try:
                descriptor = os.open(
                    staging_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=target.parent_fd,
                )
                break
            except FileExistsError:
                continue
        else:
            raise C2FullReplacementError(
                "Cannot allocate V2 descriptor-relative staging file"
            )
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        staging = os.fstat(descriptor)
        if not stat.S_ISREG(staging.st_mode):
            raise C2FullReplacementError("V2 staging output is not a regular file")
        try:
            os.link(
                staging_name,
                target.leaf_name,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise C2FullReplacementError(
                "V2 no-replace publication found an existing output leaf"
            ) from exc
        except (NotImplementedError, OSError) as exc:
            raise C2FullReplacementError(
                "V2 descriptor-relative no-replace publication failed"
            ) from exc
        published = True
        try:
            final = os.stat(
                target.leaf_name,
                dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise C2FullReplacementError(
                "V2 published output leaf is unavailable after link publication"
            ) from exc
        if not stat.S_ISREG(final.st_mode) or not _same_identity(staging, final):
            raise C2FullReplacementError(
                "V2 published output leaf does not identify the staged inode"
            )
        for artifact in input_artifacts:
            if final.st_dev == artifact.device and final.st_ino == artifact.inode:
                raise C2FullReplacementError(
                    "V2 published output aliases a validated evidence inode"
                )
        target.published_device = final.st_dev
        target.published_inode = final.st_ino
        os.fsync(target.parent_fd)
        current_staging = os.stat(
            staging_name,
            dir_fd=target.parent_fd,
            follow_symlinks=False,
        )
        if not _same_identity(staging, current_staging):
            raise C2FullReplacementError(
                "V2 staging name was replaced before safe cleanup"
            )
        os.unlink(staging_name, dir_fd=target.parent_fd)
        os.fsync(target.parent_fd)
    except OSError as exc:
        raise C2FullReplacementError(
            "V2 descriptor-relative output publication failed"
        ) from exc
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if not published:
            # A failed link may allow the staging name to be reused.  Leaving this
            # 0600 file is safer than unlinking a pathname we no longer own.
            pass


def write_full_replacement_report(
    finalized: ValidatedFullReplacementAdmission,
    output_path: Path,
) -> Path:
    """Publish a Stage-A report with descriptor-FD atomic no-replace semantics."""

    if not isinstance(finalized, ValidatedFullReplacementAdmission):
        raise C2FullReplacementError(
            "V2 output requires a validated full-replacement admission"
        )
    validate_synthetic_final_report_for_testing(finalized.report, finalized.policy)
    try:
        finalized.evidence.verify_root()
    except ProvenanceError as exc:
        raise C2FullReplacementError(
            "V2 evidence root changed before output publication"
        ) from exc
    normalized = _normalized_v2_output_path(output_path)
    try:
        target = open_secure_output_target(
            normalized,
            normalized_path=True,
            require_trusted_parent=True,
        )
    except ProvenanceError as exc:
        raise C2FullReplacementError(
            f"Cannot secure V2 output parent: {normalized}"
        ) from exc
    try:
        _reject_output_evidence_collision(target, finalized.evidence)
        _publish_json_no_replace(
            target,
            finalized.report,
            finalized.evidence.input_artifacts,
        )
        verify_secure_output_target(target)
        finalized.evidence.verify_root()
        return target.final_path
    except ProvenanceError as exc:
        if isinstance(exc, C2FullReplacementError):
            raise
        raise C2FullReplacementError(
            f"V2 output verification failed: {normalized}"
        ) from exc
    finally:
        target.close()


def finalize_synthetic_to_path_for_testing(
    manifest_path: Path,
    output_path: Path,
    policy: CompiledFullReplacementPolicy,
) -> tuple[dict[str, Any], Path]:
    """Complete the synthetic-only Stage-A path and always close evidence FDs."""

    finalized = prepare_full_replacement_finalization_for_testing(manifest_path, policy)
    try:
        output = write_full_replacement_report(finalized, output_path)
        return dict(finalized.report), output
    finally:
        finalized.evidence.close()


def finalize_to_path(
    manifest_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Production V2 route, deliberately blocked until Stage B policy pinning."""

    del output_path
    finalized = prepare_full_replacement_finalization(manifest_path)
    try:
        raise AssertionError("Stage-A production finalization cannot reach publication")
    finally:
        finalized.evidence.close()

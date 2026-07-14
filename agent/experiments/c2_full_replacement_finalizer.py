"""Isolated source-derived C2 full-replacement V2.1 finalization."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .c2_m1_trust_boundary import require_external_m1_trust_lock
from .c2_full_replacement_evidence import (
    C2FullReplacementEvidenceError,
    EvidenceArtifact,
    ValidatedRawEvidence,
    _validate_schema,
    freeze_evidence_value,
    load_and_validate_raw_evidence,
    thaw_evidence_value,
)
from .c2_full_replacement_policy import (
    ATTEMPT_IDS,
    CHUNK_IDS,
    STRATIFIED_DISPOSITION,
    C2FullReplacementPolicyError,
    CompiledFullReplacementPolicy,
    aggregate_stratified_source_classifications,
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
    """Raised when V2.1 full-replacement evidence cannot finalize safely."""


@dataclass(frozen=True, slots=True)
class ValidatedFullReplacementAdmission:
    """A structural guard carrying immutable source-derived evidence only."""

    policy: CompiledFullReplacementPolicy
    evidence: ValidatedRawEvidence

    @property
    def report(self) -> Mapping[str, Any]:
        """Expose a read-only diagnostic projection that publication never uses."""

        return freeze_evidence_value(
            _build_validated_test_report(self.policy, self.evidence)
        )


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
                "terminal_outcomes": chunk_evidence.terminal_outcomes.to_report_dict(),
                "sealed_terminal_report": (
                    chunk_evidence.sealed_terminal_report.to_report_dict()
                ),
                "sealed_report_hash": chunk_evidence.sealed_report_hash,
            }
        )
    return chunks


def _counts(items: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def _build_test_report(
    policy: CompiledFullReplacementPolicy,
    evidence: ValidatedRawEvidence,
) -> dict[str, Any]:
    aggregation = aggregate_stratified_source_classifications(
        evidence.stratified_source_classifications
    )
    dispositions = [
        thaw_evidence_value(item) for item in evidence.acquisition_dispositions
    ]
    source_classifications = [
        item.to_dict() for item in evidence.stratified_source_classifications
    ]
    stratified_dois = [item["doi_id"] for item in source_classifications]
    non_stratified_dois = [
        item["doi_id"]
        for item in dispositions
        if item["final_disposition"] != STRATIFIED_DISPOSITION
    ]
    report: dict[str, Any] = {
        "schema_version": "2.1-stagea",
        "finalizer": "c2_full_replacement_source_classification",
        "status": aggregation.status,
        "claim_status": "UNSUPPORTED"
        if aggregation.deficient_strata
        else "SUPPORTED",
        "claim_scope": "STRATIFIED_SOURCE_TERMINAL_ADMISSION_ONLY",
        "decision": (
            "NOT_RUN_COVERAGE_GATE"
            if aggregation.deficient_strata
            else "NOT_RUN_NO_ANALYSIS_AUTHORIZED"
        ),
        "trend_status": "NOT_RUN",
        "equivalence_status": "NOT_RUN",
        "policy_binding": {
            "authority": "SYNTHETIC_TEST_ONLY",
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "test_policy_sha256": policy.test_policy_sha256,
        },
        "admission_manifest": {
            "file_sha256": evidence.manifest_artifact.sha256,
            "manifest_hash": evidence.manifest["manifest_hash"],
        },
        "frozen_universe": policy.frozen_universe.to_dict(),
        "frozen_bindings": thaw_evidence_value(policy.frozen_bindings),
        "code": thaw_evidence_value(evidence.manifest["code"]),
        "chunks": _report_chunks(policy, evidence),
        "canonical_attempt_ledger": [
            attempt.to_report_dict() for attempt in evidence.attempt_ledger
        ],
        "acquisition_dispositions": dispositions,
        "acquisition_dispositions_sha256": sha256_json(dispositions),
        "acquisition_disposition_doi_ids_sha256": sha256_json(
            [item["doi_id"] for item in dispositions]
        ),
        "stratified_source_classifications": source_classifications,
        "stratified_source_doi_ids_sha256": sha256_json(stratified_dois),
        "stratified_source_classifications_sha256": sha256_json(
            source_classifications
        ),
        "non_stratified_doi_ids_sha256": sha256_json(non_stratified_dois),
        "canonical_case_set_manifest_sha256": (
            evidence.canonical_case_set_manifest_sha256
        ),
        "per_doi_case_set_hashes_sha256": evidence.per_doi_case_set_hashes_sha256,
        "terminal_status_counts": _counts(
            [item["terminal_status"] for item in dispositions]
        ),
        "disposition_counts": _counts(
            [item["final_disposition"] for item in dispositions]
        ),
        "strata": [item.to_dict() for item in aggregation.strata],
        "deficient_strata": list(aggregation.deficient_strata),
    }
    report["final_report_hash"] = sha256_json(report)
    return report


def _build_validated_test_report(
    policy: CompiledFullReplacementPolicy,
    evidence: ValidatedRawEvidence,
) -> dict[str, Any]:
    report = _build_test_report(policy, evidence)
    validate_synthetic_final_report_for_testing(report, policy)
    return report


def _immutable_policy_snapshot(
    policy: CompiledFullReplacementPolicy,
) -> CompiledFullReplacementPolicy:
    return replace(
        policy,
        frozen_bindings=freeze_evidence_value(policy.frozen_bindings),
    )


def _validate_ledger_structure(
    ledger: Sequence[Mapping[str, Any]],
    policy: CompiledFullReplacementPolicy,
) -> None:
    expected_keys = [
        (chunk_id, attempt_id)
        for chunk_id in CHUNK_IDS
        for attempt_id in ATTEMPT_IDS
    ]
    if [(item["chunk_id"], item["attempt_id"]) for item in ledger] != expected_keys:
        raise C2FullReplacementError(
            "V2.1 canonical attempt ledger is not complete and ordered"
        )
    for item in ledger:
        expected_dois = policy.dois_for_chunk(item["chunk_id"])
        outcomes = item["outcomes"]
        if len(outcomes) != len(expected_dois):
            raise C2FullReplacementError(
                f"V2.1 attempt ledger has an invalid outcome count for "
                f"chunk {item['chunk_id']}"
            )
        partition = policy.partition_for_chunk(item["chunk_id"])
        for local_ordinal, (outcome, doi_id) in enumerate(
            zip(outcomes, expected_dois, strict=True),
            start=1,
        ):
            if (
                outcome["doi_id"] != doi_id
                or outcome["local_ordinal"] != local_ordinal
                or outcome["global_ordinal"]
                != partition.first_global_ordinal + local_ordinal - 1
            ):
                raise C2FullReplacementError(
                    "V2.1 canonical attempt ledger differs from frozen DOI order"
                )


def validate_synthetic_final_report_for_testing(
    report: Mapping[str, Any],
    policy: CompiledFullReplacementPolicy,
) -> None:
    """Validate test-only output without accepting report P labels as inputs."""

    if not policy.is_test_only:
        raise C2FullReplacementError(
            "Stage-A final report validation accepts only a synthetic policy"
        )
    _validate_schema(
        report,
        "c2_full_replacement_final_report_v2.schema.json",
        "V2.1 final report",
    )
    if sha256_json(_without(report, "final_report_hash")) != report["final_report_hash"]:
        raise C2FullReplacementError("V2.1 final report failed its semantic hash")
    expected_policy_binding = {
        "authority": "SYNTHETIC_TEST_ONLY",
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "test_policy_sha256": policy.test_policy_sha256,
    }
    if report["policy_binding"] != expected_policy_binding:
        raise C2FullReplacementError("V2.1 final report policy binding is inconsistent")
    if (
        report["frozen_universe"] != policy.frozen_universe.to_dict()
        or report["frozen_bindings"] != thaw_evidence_value(policy.frozen_bindings)
    ):
        raise C2FullReplacementError(
            "V2.1 final report frozen bindings are inconsistent"
        )
    if [chunk["chunk_id"] for chunk in report["chunks"]] != list(CHUNK_IDS):
        raise C2FullReplacementError("V2.1 final report chunk roster is incomplete")
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
                f"V2.1 final report chunk {chunk['chunk_id']} differs from policy"
            )
    _validate_ledger_structure(report["canonical_attempt_ledger"], policy)
    dispositions = report["acquisition_dispositions"]
    if [item["doi_id"] for item in dispositions] != list(policy.ordered_doi_ids):
        raise C2FullReplacementError(
            "V2.1 acquisition dispositions are incomplete or reordered"
        )
    if (
        report["acquisition_dispositions_sha256"] != sha256_json(dispositions)
        or report["acquisition_disposition_doi_ids_sha256"]
        != sha256_json([item["doi_id"] for item in dispositions])
    ):
        raise C2FullReplacementError("V2.1 acquisition disposition hashes mismatch")
    classifications = report["stratified_source_classifications"]
    stratified_dois = [item["doi_id"] for item in classifications]
    stratified_from_dispositions = [
        item["doi_id"]
        for item in dispositions
        if item["final_disposition"] == STRATIFIED_DISPOSITION
    ]
    if stratified_dois != stratified_from_dispositions:
        raise C2FullReplacementError(
            "V2.1 stratified source set does not match acquisition dispositions"
        )
    non_stratified_dois = [
        item["doi_id"]
        for item in dispositions
        if item["final_disposition"] != STRATIFIED_DISPOSITION
    ]
    if (
        report["stratified_source_doi_ids_sha256"] != sha256_json(stratified_dois)
        or report["stratified_source_classifications_sha256"]
        != sha256_json(classifications)
        or report["non_stratified_doi_ids_sha256"]
        != sha256_json(non_stratified_dois)
    ):
        raise C2FullReplacementError("V2.1 source/non-source set hashes mismatch")
    if any(
        item["cluster_id"] != item["doi_id"]
        for item in classifications
    ):
        raise C2FullReplacementError(
            "V2.1 source classification cluster_id must equal normalized doi_id"
        )
    from .c2_full_replacement_policy import StratifiedSourceClassification

    derived = tuple(
        StratifiedSourceClassification(
            doi_id=item["doi_id"],
            canonical_case_set_hash=item["canonical_case_set_hash"],
            canonical_case_ids=tuple(item["canonical_case_ids"]),
            verified_panel_descriptors_sha256=(
                item["verified_panel_descriptors_sha256"]
            ),
            qualified_panel_counts=tuple(item["qualified_panel_counts"]),
            derived_public_stratum=item["derived_public_stratum"],
            derived_code_label=item["derived_code_label"],
            source_binding_hashes=dict(item["source_binding_hashes"]),
        )
        for item in classifications
    )
    aggregation = aggregate_stratified_source_classifications(derived)
    if report["strata"] != [item.to_dict() for item in aggregation.strata]:
        raise C2FullReplacementError(
            "V2.1 final report source coverage was not derived from source cases"
        )
    if (
        report["deficient_strata"] != list(aggregation.deficient_strata)
        or report["status"] != aggregation.status
    ):
        raise C2FullReplacementError("V2.1 final report coverage gate is inconsistent")
    expected_claim = "UNSUPPORTED" if aggregation.deficient_strata else "SUPPORTED"
    expected_decision = (
        "NOT_RUN_COVERAGE_GATE"
        if aggregation.deficient_strata
        else "NOT_RUN_NO_ANALYSIS_AUTHORIZED"
    )
    if (
        report["claim_status"] != expected_claim
        or report["decision"] != expected_decision
        or report["claim_scope"] != "STRATIFIED_SOURCE_TERMINAL_ADMISSION_ONLY"
        or report["trend_status"] != "NOT_RUN"
        or report["equivalence_status"] != "NOT_RUN"
    ):
        raise C2FullReplacementError(
            "V2.1 final report claim/decision fields are inconsistent"
        )
    if (
        report["terminal_status_counts"]
        != _counts([item["terminal_status"] for item in dispositions])
        or report["disposition_counts"]
        != _counts([item["final_disposition"] for item in dispositions])
    ):
        raise C2FullReplacementError("V2.1 final report count summaries mismatch")


def prepare_full_replacement_finalization(
    manifest_path: Path,
) -> ValidatedFullReplacementAdmission:
    """Prepare a report only after the external M1 boundary permits it."""

    require_external_m1_trust_lock()
    try:
        policy = load_production_policy()
    except C2FullReplacementPolicyError as exc:
        raise C2FullReplacementError(str(exc)) from exc
    if not policy.is_test_only:
        raise C2FullReplacementError(
            "Stage-A finalization accepts only a synthetic in-process policy"
        )
    evidence: ValidatedRawEvidence | None = None
    try:
        evidence = load_and_validate_raw_evidence(manifest_path, policy)
        immutable_policy = _immutable_policy_snapshot(policy)
        _build_validated_test_report(immutable_policy, evidence)
        return ValidatedFullReplacementAdmission(
            policy=immutable_policy,
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


def _normalized_v2_output_path(path: Path) -> Path:
    require_external_m1_trust_lock()
    raw = Path(os.fspath(path))
    if any(part in {".", ".."} for part in raw.parts):
        raise C2FullReplacementError(
            "V2.1 final report output path must not contain dot traversal"
        )
    try:
        return normalize_trusted_output_path(path)
    except ProvenanceError as exc:
        raise C2FullReplacementError(
            f"Cannot normalize V2.1 final report output path: {path}"
        ) from exc


def _reject_output_evidence_collision(
    target: SecureOutputTarget,
    evidence: ValidatedRawEvidence,
) -> None:
    require_external_m1_trust_lock()
    output_parent_path = target.final_path.parent
    try:
        target.final_path.relative_to(evidence.root_path)
    except ValueError:
        pass
    else:
        raise C2FullReplacementError(
            "V2.1 final report output must be outside the evidence root"
        )
    try:
        evidence.root_path.relative_to(output_parent_path)
    except ValueError:
        pass
    else:
        raise C2FullReplacementError(
            "V2.1 output parent and evidence root must be distinct and non-containing"
        )
    output_parent = os.fstat(target.parent_fd)
    evidence_root = os.fstat(evidence.root.descriptor)
    if (
        output_parent.st_dev == evidence_root.st_dev
        and output_parent.st_ino == evidence_root.st_ino
    ):
        raise C2FullReplacementError(
            "V2.1 final report output parent aliases the evidence root"
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
            f"Cannot inspect V2.1 final report output: {target.final_path}"
        ) from exc
    for artifact in evidence.input_artifacts:
        if target.final_path == artifact.absolute_path:
            raise C2FullReplacementError(
                "V2.1 final report output aliases a validated evidence path"
            )
        if (
            existing is not None
            and existing.st_dev == artifact.device
            and existing.st_ino == artifact.inode
        ):
            raise C2FullReplacementError(
                "V2.1 final report output aliases a validated evidence inode"
            )
    if existing is not None:
        raise C2FullReplacementError(
            "V2.1 final report output already exists; no-replace publication refuses it"
        )


def _linkat_no_replace_supported() -> bool:
    return (
        os.link in os.supports_dir_fd
        and os.link in os.supports_follow_symlinks
        and os.unlink in os.supports_dir_fd
        and hasattr(os, "O_NOFOLLOW")
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    require_external_m1_trust_lock()
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise C2FullReplacementError("Cannot write V2.1 output staging file")
        remaining = remaining[written:]


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


def _publish_json_no_replace(
    target: SecureOutputTarget,
    report: Mapping[str, Any],
    input_artifacts: Sequence[EvidenceArtifact],
) -> None:
    require_external_m1_trust_lock()
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
                "Cannot allocate V2.1 descriptor-relative staging file"
            )
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        staging = os.fstat(descriptor)
        if not stat.S_ISREG(staging.st_mode):
            raise C2FullReplacementError("V2.1 staging output is not a regular file")
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
                "V2.1 no-replace publication found an existing output leaf"
            ) from exc
        except (NotImplementedError, OSError) as exc:
            raise C2FullReplacementError(
                "V2.1 descriptor-relative no-replace publication failed"
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
                "V2.1 published output leaf is unavailable after link publication"
            ) from exc
        if not stat.S_ISREG(final.st_mode) or not _same_identity(staging, final):
            raise C2FullReplacementError(
                "V2.1 published output leaf does not identify the staged inode"
            )
        for artifact in input_artifacts:
            if final.st_dev == artifact.device and final.st_ino == artifact.inode:
                raise C2FullReplacementError(
                    "V2.1 published output aliases a validated evidence inode"
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
                "V2.1 staging name was replaced before safe cleanup"
            )
        os.unlink(staging_name, dir_fd=target.parent_fd)
        os.fsync(target.parent_fd)
    except OSError as exc:
        raise C2FullReplacementError(
            "V2.1 descriptor-relative output publication failed"
        ) from exc
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if not published:
            # A failed link can race with staging-name reuse; preserve the 0600
            # file rather than unlinking a name that is no longer proven ours.
            pass


def write_full_replacement_report(
    finalized: ValidatedFullReplacementAdmission,
    output_path: Path,
) -> Path:
    """Publish a source-derived report only after the external M1 boundary."""

    require_external_m1_trust_lock()
    if not isinstance(finalized, ValidatedFullReplacementAdmission):
        raise C2FullReplacementError(
            "V2.1 output requires a validated full-replacement admission"
        )
    try:
        finalized.evidence.verify_artifacts()
    except ProvenanceError as exc:
        raise C2FullReplacementError(
            f"V2.1 evidence changed before output publication: {exc}"
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
            f"Cannot secure V2.1 output parent: {normalized}"
        ) from exc
    try:
        _reject_output_evidence_collision(target, finalized.evidence)
        try:
            finalized.evidence.verify_artifacts()
        except ProvenanceError as exc:
            raise C2FullReplacementError(
                "V2.1 evidence changed immediately before output publication: "
                f"{exc}"
            ) from exc
        report = _build_validated_test_report(
            finalized.policy,
            finalized.evidence,
        )
        _publish_json_no_replace(
            target,
            report,
            finalized.evidence.input_artifacts,
        )
        verify_secure_output_target(target)
        finalized.evidence.verify_artifacts()
        return target.final_path
    except ProvenanceError as exc:
        if isinstance(exc, C2FullReplacementError):
            raise
        raise C2FullReplacementError(
            f"V2.1 output verification failed: {normalized}"
        ) from exc
    finally:
        target.close()


def finalize_to_path(
    manifest_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Prepare and publish only after the external M1 boundary permits it."""

    require_external_m1_trust_lock()
    finalized = prepare_full_replacement_finalization(manifest_path)
    try:
        output = write_full_replacement_report(finalized, output_path)
        return thaw_evidence_value(finalized.report), output
    finally:
        finalized.evidence.close()

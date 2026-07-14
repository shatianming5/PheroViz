"""Descriptor-rooted raw-evidence validation for C2 full-replacement V2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .c2_full_replacement_policy import (
    ATTEMPT_IDS,
    CHUNK_IDS,
    TERMINAL_OUTCOME_STATUSES,
    ChunkPartition,
    CompiledFullReplacementPolicy,
    PolicyRow,
)
from .models import (
    ProvenanceError,
    TrustedDirectory,
    open_trusted_directory,
    sha256_json,
    verify_trusted_directory,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class C2FullReplacementEvidenceError(ProvenanceError):
    """Raised when V2 raw evidence cannot be safely attested."""


@dataclass(frozen=True)
class EvidenceArtifact:
    relative_path: str
    absolute_path: Path
    sha256: str
    byte_count: int
    device: int
    inode: int
    payload: bytes

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class CanonicalOutcome:
    global_ordinal: int
    local_ordinal: int
    doi_id: str
    terminal_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_ordinal": self.global_ordinal,
            "local_ordinal": self.local_ordinal,
            "doi_id": self.doi_id,
            "terminal_status": self.terminal_status,
        }


@dataclass(frozen=True)
class AttemptLedger:
    chunk_id: str
    attempt_id: str
    raw_stream: EvidenceArtifact
    processed_success: EvidenceArtifact
    skipped_status: EvidenceArtifact
    outcomes: tuple[CanonicalOutcome, ...]

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "attempt_id": self.attempt_id,
            "raw_stream": self.raw_stream.to_report_dict(),
            "processed_success": self.processed_success.to_report_dict(),
            "skipped_status": self.skipped_status.to_report_dict(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True)
class ChunkEvidence:
    chunk_id: str
    canonical_mapping: EvidenceArtifact
    attempts: tuple[AttemptLedger, ...]

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "canonical_mapping": self.canonical_mapping.to_report_dict(),
            "attempts": [attempt.to_report_dict() for attempt in self.attempts],
        }


@dataclass
class ValidatedRawEvidence:
    root: TrustedDirectory
    manifest: Mapping[str, Any]
    manifest_artifact: EvidenceArtifact
    chunks: tuple[ChunkEvidence, ...]
    input_artifacts: tuple[EvidenceArtifact, ...]

    def close(self) -> None:
        self.root.close()

    def verify_root(self) -> None:
        verify_trusted_directory(self.root)

    @property
    def attempt_ledger(self) -> tuple[AttemptLedger, ...]:
        return tuple(
            attempt
            for chunk in self.chunks
            for attempt in chunk.attempts
        )


def _reject_json_constant(value: str) -> None:
    raise C2FullReplacementEvidenceError(
        f"Non-finite JSON value is forbidden: {value}"
    )


@lru_cache(maxsize=None)
def _schema_validator(schema_name: str) -> Draft202012Validator:
    schema_path = Path(__file__).resolve().parent / "schemas" / schema_name
    try:
        schema = json.loads(
            schema_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2FullReplacementEvidenceError(
            f"Cannot read bundled V2 schema {schema_name}"
        ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise C2FullReplacementEvidenceError(
            f"Bundled V2 schema is invalid: {schema_name}"
        ) from exc
    return Draft202012Validator(schema)


def _validation_location(error_path: Sequence[Any]) -> str:
    return ".".join(str(part) for part in error_path) or "<root>"


def _validate_schema(value: Any, schema_name: str, label: str) -> None:
    errors = sorted(
        _schema_validator(schema_name).iter_errors(value),
        key=lambda error: _validation_location(tuple(error.absolute_path)),
    )
    if errors:
        error = errors[0]
        raise C2FullReplacementEvidenceError(
            f"{label} schema validation failed at "
            f"{_validation_location(tuple(error.absolute_path))}: {error.message}"
        )


def _safe_relative_path(value: Any, label: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise C2FullReplacementEvidenceError(f"{label} must be a nonempty path")
    if "\\" in value:
        raise C2FullReplacementEvidenceError(f"{label} must use portable separators")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {".", ".."} for part in relative.parts)
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} must be a safe manifest-relative path"
        )
    return value, tuple(relative.parts)


def _read_no_follow_artifact(
    root: TrustedDirectory,
    relative_path: str,
    label: str,
) -> EvidenceArtifact:
    if root.descriptor == -1:
        raise C2FullReplacementEvidenceError("Evidence root is already closed")
    safe_path, components = _safe_relative_path(relative_path, label)
    directory_descriptor = os.dup(root.descriptor)
    file_descriptor = -1
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
            file_flags |= os.O_CLOEXEC
        for component in components[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            components[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise C2FullReplacementEvidenceError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        while True:
            block = os.read(file_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(file_descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise C2FullReplacementEvidenceError(
                f"{label} changed while it was read"
            )
        return EvidenceArtifact(
            relative_path=safe_path,
            absolute_path=root.path.joinpath(*components),
            sha256=digest.hexdigest(),
            byte_count=len(payload),
            device=before.st_dev,
            inode=before.st_ino,
            payload=payload,
        )
    except OSError as exc:
        raise C2FullReplacementEvidenceError(
            f"Cannot descriptor-read {label}: {relative_path}"
        ) from exc
    finally:
        if file_descriptor != -1:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _parse_json_object(artifact: EvidenceArtifact, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            artifact.payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise C2FullReplacementEvidenceError(f"{label} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise C2FullReplacementEvidenceError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise C2FullReplacementEvidenceError(f"{label} must be a JSON object")
    return value


def _parse_jsonl_objects(
    artifact: EvidenceArtifact,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    try:
        text = artifact.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise C2FullReplacementEvidenceError(f"{label} is not UTF-8 JSONL") from exc
    if not text:
        raise C2FullReplacementEvidenceError(f"{label} must not be empty")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise C2FullReplacementEvidenceError(
                f"{label} has a blank JSONL row at line {line_number}"
            )
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise C2FullReplacementEvidenceError(
                f"{label} has invalid JSONL at line {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise C2FullReplacementEvidenceError(
                f"{label} JSONL row {line_number} must be an object"
            )
        rows.append(row)
    return tuple(rows)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C2FullReplacementEvidenceError(f"{label} must be a full SHA-256 digest")
    return value


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: item for key, item in value.items() if key not in omitted}


class _ArtifactCollector:
    def __init__(self, root: TrustedDirectory) -> None:
        self.root = root
        self._paths: set[str] = set()
        self._identities: set[tuple[int, int]] = set()
        self._artifacts: list[EvidenceArtifact] = []

    @property
    def artifacts(self) -> tuple[EvidenceArtifact, ...]:
        return tuple(self._artifacts)

    def read(
        self,
        relative_path: str,
        expected_sha256: str | None,
        label: str,
    ) -> EvidenceArtifact:
        safe_path, _ = _safe_relative_path(relative_path, label)
        if safe_path in self._paths:
            raise C2FullReplacementEvidenceError(
                f"V2 evidence reuses an artifact path: {safe_path}"
            )
        artifact = _read_no_follow_artifact(self.root, safe_path, label)
        if expected_sha256 is not None and artifact.sha256 != expected_sha256:
            raise C2FullReplacementEvidenceError(
                f"{label} bytes do not match the manifest SHA-256 binding"
            )
        identity = (artifact.device, artifact.inode)
        if identity in self._identities:
            raise C2FullReplacementEvidenceError(
                f"V2 evidence reuses an artifact inode: {safe_path}"
            )
        self._paths.add(safe_path)
        self._identities.add(identity)
        self._artifacts.append(artifact)
        return artifact


def _open_manifest_root(manifest_path: Path) -> tuple[TrustedDirectory, str]:
    raw_path = Path(os.fspath(manifest_path))
    if raw_path.name in {"", ".", ".."} or raw_path.suffix != ".json":
        raise C2FullReplacementEvidenceError(
            "V2 admission manifest must name a JSON file"
        )
    if any(part in {".", ".."} for part in raw_path.parts):
        raise C2FullReplacementEvidenceError(
            "V2 admission manifest path must not contain dot traversal"
        )
    try:
        absolute = Path(os.path.abspath(os.fspath(raw_path.expanduser())))
    except (OSError, RuntimeError) as exc:
        raise C2FullReplacementEvidenceError(
            f"Cannot normalize V2 admission manifest path: {manifest_path}"
        ) from exc
    try:
        root = open_trusted_directory(absolute.parent)
    except ProvenanceError as exc:
        raise C2FullReplacementEvidenceError(
            f"Cannot open trusted V2 evidence root: {absolute.parent}"
        ) from exc
    return root, absolute.name


def _validate_manifest(
    manifest: Mapping[str, Any],
    policy: CompiledFullReplacementPolicy,
) -> tuple[Mapping[str, Any], ...]:
    _validate_schema(
        manifest,
        "c2_full_replacement_admission_manifest_v2.schema.json",
        "V2 admission manifest",
    )
    declared_hash = _require_sha256(manifest["manifest_hash"], "manifest_hash")
    if sha256_json(_without(manifest, "manifest_hash")) != declared_hash:
        raise C2FullReplacementEvidenceError(
            "V2 admission manifest failed its semantic hash"
        )
    if dict(manifest["frozen_universe"]) != policy.frozen_universe.to_dict():
        raise C2FullReplacementEvidenceError(
            "V2 manifest frozen universe differs from the compiled policy"
        )
    code = manifest["code"]
    if (
        not isinstance(code["commit"], str)
        or _COMMIT_RE.fullmatch(code["commit"]) is None
        or code["dirty"] is not False
    ):
        raise C2FullReplacementEvidenceError(
            "V2 admission manifest requires a full clean code commit"
        )
    chunks = manifest["chunks"]
    if not isinstance(chunks, list) or len(chunks) != len(CHUNK_IDS):
        raise C2FullReplacementEvidenceError(
            "V2 admission manifest must contain all 13 chunks"
        )
    chunk_mappings: list[Mapping[str, Any]] = []
    for chunk_id, chunk in zip(CHUNK_IDS, chunks, strict=True):
        if not isinstance(chunk, Mapping) or chunk["chunk_id"] != chunk_id:
            raise C2FullReplacementEvidenceError(
                "V2 admission manifest must use ordered chunks 001..013"
            )
        partition = policy.partition_for_chunk(chunk_id)
        root_plan = policy.root_plan_for_chunk(chunk_id)
        if chunk["input_total"] != partition.input_total:
            raise C2FullReplacementEvidenceError(
                f"V2 chunk {chunk_id} has an invalid pinned input total"
            )
        if chunk["input_doi_ids_sha256"] != partition.doi_ids_sha256:
            raise C2FullReplacementEvidenceError(
                f"V2 chunk {chunk_id} DOI hash differs from compiled partition"
            )
        root = chunk["root"]
        if (
            root["replacement_root_id"] != root_plan.replacement_root_id
            or tuple(root["retired_root_ids"]) != root_plan.retired_root_ids
            or root["partial_root"] is not False
        ):
            raise C2FullReplacementEvidenceError(
                f"V2 chunk {chunk_id} root plan differs from compiled policy"
            )
        attempts = chunk["attempts"]
        if [attempt["attempt_id"] for attempt in attempts] != list(ATTEMPT_IDS):
            raise C2FullReplacementEvidenceError(
                f"V2 chunk {chunk_id} must contain initial, retry1, retry2 in order"
            )
        chunk_mappings.append(chunk)
    return tuple(chunk_mappings)


def _read_bound_artifact(
    collector: _ArtifactCollector,
    binding: Mapping[str, Any],
    label: str,
) -> EvidenceArtifact:
    expected_sha256 = _require_sha256(binding["sha256"], f"{label}.sha256")
    return collector.read(binding["path"], expected_sha256, label)


def _validate_mapping_artifact(
    artifact: EvidenceArtifact,
    chunk_id: str,
    partition: ChunkPartition,
    policy_rows: Sequence[PolicyRow],
) -> None:
    rows = _parse_jsonl_objects(artifact, f"chunk {chunk_id} canonical mapping")
    if len(rows) != partition.input_total:
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} canonical mapping has an invalid row count"
        )
    for local_ordinal, (record, policy_row) in enumerate(
        zip(rows, policy_rows, strict=True),
        start=1,
    ):
        _validate_schema(
            record,
            "c2_full_replacement_mapping_row_v2.schema.json",
            f"chunk {chunk_id} canonical mapping row {local_ordinal}",
        )
        expected = {
            "global_ordinal": policy_row.global_ordinal,
            "local_ordinal": local_ordinal,
            "doi_id": policy_row.doi_id,
            "p_disposition": policy_row.p_disposition,
            "independent_cluster_id": policy_row.independent_cluster_id,
        }
        if dict(record) != expected:
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} canonical mapping differs from compiled policy "
                f"at local ordinal {local_ordinal}"
            )


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        record["global_ordinal"],
        record["local_ordinal"],
        record["doi_id"],
    )


def _validate_attempt(
    *,
    collector: _ArtifactCollector,
    binding: Mapping[str, Any],
    chunk_id: str,
    partition: ChunkPartition,
    policy_rows: Sequence[PolicyRow],
) -> AttemptLedger:
    attempt_id = binding["attempt_id"]
    raw_artifact = _read_bound_artifact(
        collector,
        binding["raw_stream"],
        f"chunk {chunk_id} {attempt_id} raw stream",
    )
    processed_artifact = _read_bound_artifact(
        collector,
        binding["processed_success"],
        f"chunk {chunk_id} {attempt_id} processed-success",
    )
    skipped_artifact = _read_bound_artifact(
        collector,
        binding["skipped_status"],
        f"chunk {chunk_id} {attempt_id} skipped-status",
    )
    raw_rows = _parse_jsonl_objects(
        raw_artifact,
        f"chunk {chunk_id} {attempt_id} raw stream",
    )
    if len(raw_rows) != partition.input_total:
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} {attempt_id} raw stream has an invalid row count"
        )
    for local_ordinal, (record, policy_row) in enumerate(
        zip(raw_rows, policy_rows, strict=True),
        start=1,
    ):
        _validate_schema(
            record,
            "c2_full_replacement_raw_record_v2.schema.json",
            f"chunk {chunk_id} {attempt_id} raw row {local_ordinal}",
        )
        if (
            record["attempt_id"] != attempt_id
            or record["global_ordinal"] != policy_row.global_ordinal
            or record["local_ordinal"] != local_ordinal
            or record["doi_id"] != policy_row.doi_id
        ):
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} raw stream order differs from "
                f"compiled DOI/ordinal policy"
            )

    processed = _parse_json_object(
        processed_artifact,
        f"chunk {chunk_id} {attempt_id} processed-success",
    )
    _validate_schema(
        processed,
        "c2_full_replacement_processed_success_v2.schema.json",
        f"chunk {chunk_id} {attempt_id} processed-success",
    )
    skipped = _parse_json_object(
        skipped_artifact,
        f"chunk {chunk_id} {attempt_id} skipped-status",
    )
    _validate_schema(
        skipped,
        "c2_full_replacement_skipped_status_v2.schema.json",
        f"chunk {chunk_id} {attempt_id} skipped-status",
    )
    if (
        processed["chunk_id"] != chunk_id
        or processed["attempt_id"] != attempt_id
        or skipped["chunk_id"] != chunk_id
        or skipped["attempt_id"] != attempt_id
    ):
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} {attempt_id} ledger metadata is mismatched"
        )

    expected_processed = [
        {
            "global_ordinal": row["global_ordinal"],
            "local_ordinal": row["local_ordinal"],
            "doi_id": row["doi_id"],
        }
        for row in raw_rows
        if row["raw_disposition"] == "PROCESSED"
    ]
    expected_skipped_keys = [
        _record_key(row)
        for row in raw_rows
        if row["raw_disposition"] == "SKIPPED"
    ]
    if processed["records"] != expected_processed:
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} {attempt_id} processed-success does not exactly "
            "match the raw processed partition"
        )
    skipped_keys = [_record_key(row) for row in skipped["records"]]
    if skipped_keys != expected_skipped_keys:
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} {attempt_id} skipped-status does not exactly match "
            "the raw skipped partition"
        )
    statuses_by_key: dict[tuple[int, int, str], str] = {}
    for row in skipped["records"]:
        status = row["terminal_status"]
        if status == "DOWNLOADED" or status not in TERMINAL_OUTCOME_STATUSES:
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} skipped-status illegally encodes "
                "DOWNLOADED or a nonterminal status"
            )
        key = _record_key(row)
        if key in statuses_by_key:
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} skipped-status has a duplicate DOI"
            )
        statuses_by_key[key] = status
    outcomes = tuple(
        CanonicalOutcome(
            global_ordinal=row["global_ordinal"],
            local_ordinal=row["local_ordinal"],
            doi_id=row["doi_id"],
            terminal_status=(
                "DOWNLOADED"
                if row["raw_disposition"] == "PROCESSED"
                else statuses_by_key[_record_key(row)]
            ),
        )
        for row in raw_rows
    )
    return AttemptLedger(
        chunk_id=chunk_id,
        attempt_id=attempt_id,
        raw_stream=raw_artifact,
        processed_success=processed_artifact,
        skipped_status=skipped_artifact,
        outcomes=outcomes,
    )


def load_and_validate_raw_evidence(
    manifest_path: Path,
    policy: CompiledFullReplacementPolicy,
) -> ValidatedRawEvidence:
    """Hash, parse, and validate every raw artifact from one descriptor root."""

    if not policy.is_test_only:
        raise C2FullReplacementEvidenceError(
            "Stage-A raw validation accepts only an in-process synthetic policy"
        )
    root: TrustedDirectory | None = None
    try:
        root, manifest_name = _open_manifest_root(manifest_path)
        collector = _ArtifactCollector(root)
        manifest_artifact = collector.read(
            manifest_name,
            None,
            "V2 admission manifest",
        )
        manifest = _parse_json_object(manifest_artifact, "V2 admission manifest")
        chunks = _validate_manifest(manifest, policy)
        chunk_evidence: list[ChunkEvidence] = []
        for chunk_id, chunk in zip(CHUNK_IDS, chunks, strict=True):
            partition = policy.partition_for_chunk(chunk_id)
            policy_rows = policy.rows_for_chunk(chunk_id)
            mapping = _read_bound_artifact(
                collector,
                chunk["canonical_mapping"],
                f"chunk {chunk_id} canonical mapping",
            )
            _validate_mapping_artifact(
                mapping,
                chunk_id,
                partition,
                policy_rows,
            )
            attempts = tuple(
                _validate_attempt(
                    collector=collector,
                    binding=attempt,
                    chunk_id=chunk_id,
                    partition=partition,
                    policy_rows=policy_rows,
                )
                for attempt in chunk["attempts"]
            )
            if len(attempts) != len(ATTEMPT_IDS):
                raise C2FullReplacementEvidenceError(
                    f"chunk {chunk_id} does not have exactly three raw attempts"
                )
            if chunk_id == "013":
                expected_ordinals = tuple(range(2401, 2464))
                for attempt in attempts:
                    if (
                        tuple(
                            outcome.global_ordinal for outcome in attempt.outcomes
                        )
                        != expected_ordinals
                        or tuple(
                            outcome.local_ordinal for outcome in attempt.outcomes
                        )
                        != tuple(range(1, 64))
                    ):
                        raise C2FullReplacementEvidenceError(
                            "chunk 013 must contain exactly local ordinals 1..63 "
                            "and global ordinals 2401..2463 for every attempt"
                        )
            chunk_evidence.append(
                ChunkEvidence(
                    chunk_id=chunk_id,
                    canonical_mapping=mapping,
                    attempts=attempts,
                )
            )
        return ValidatedRawEvidence(
            root=root,
            manifest=manifest,
            manifest_artifact=manifest_artifact,
            chunks=tuple(chunk_evidence),
            input_artifacts=collector.artifacts,
        )
    except Exception:
        if root is not None:
            root.close()
        raise

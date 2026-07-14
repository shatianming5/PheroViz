"""Descriptor-rooted V2.1 acquisition and source/canonical attestation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from . import models as _models
from .c2_full_replacement_policy import (
    ATTEMPT_IDS,
    CHUNK_IDS,
    P_CODE_LABELS,
    STRATIFIED_DISPOSITION,
    TERMINAL_OUTCOME_STATUSES,
    ChunkPartition,
    CompiledFullReplacementPolicy,
    StratifiedSourceClassification,
    adapt_terminal_status,
    derive_public_stratum,
    expected_non_stratified_disposition,
    normalize_doi,
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
    """Raised when V2.1 raw/source evidence cannot be safely attested."""


def _validate_trusted_evidence_metadata(
    descriptor: int,
    metadata: os.stat_result,
    path: Path,
    label: str,
) -> None:
    if not hasattr(os, "geteuid"):
        raise ProvenanceError(f"{label} requires an effective uid")
    if metadata.st_uid not in {0, os.geteuid()}:
        raise ProvenanceError(f"{label} has an unsafe owner: {path}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ProvenanceError(f"{label} is group/world writable: {path}")
    if _models._trusted_acl_allows_foreign_mutation(descriptor):
        raise ProvenanceError(f"{label} has a mutating ACL: {path}")


def validate_trusted_directory_descriptor(
    descriptor: int,
    path: Path,
    *,
    label: str = "Trusted evidence directory",
) -> os.stat_result:
    """Validate an opened evidence directory's immutable trust boundary."""

    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ProvenanceError(f"{label} is not a directory: {path}")
    _validate_trusted_evidence_metadata(descriptor, metadata, path, label)
    return metadata


def validate_trusted_regular_file_descriptor(
    descriptor: int,
    path: Path,
    *,
    label: str = "Trusted evidence artifact",
) -> os.stat_result:
    """Validate an opened evidence leaf without following its pathname."""

    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ProvenanceError(f"{label} is not a regular file: {path}")
    _validate_trusted_evidence_metadata(descriptor, metadata, path, label)
    if metadata.st_nlink != 1:
        raise ProvenanceError(f"{label} has an unsafe hard-link count: {path}")
    return metadata


def freeze_evidence_value(value: Any) -> Any:
    """Recursively freeze parsed evidence so callers cannot alter its snapshot."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: freeze_evidence_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_evidence_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_evidence_value(item) for item in value)
    return value


def thaw_evidence_value(value: Any) -> Any:
    """Return a mutable JSON-compatible copy of an immutable evidence snapshot."""

    if isinstance(value, Mapping):
        return {key: thaw_evidence_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_evidence_value(item) for item in value]
    if isinstance(value, frozenset):
        return [thaw_evidence_value(item) for item in sorted(value)]
    return value


@dataclass(frozen=True)
class EvidencePathIdentity:
    relative_path: str
    device: int
    inode: int
    mode: int
    owner_uid: int
    link_count: int
    mtime_ns: int
    ctime_ns: int


def _path_identity(relative_path: str, metadata: os.stat_result) -> EvidencePathIdentity:
    return EvidencePathIdentity(
        relative_path=relative_path,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=stat.S_IMODE(metadata.st_mode),
        owner_uid=metadata.st_uid,
        link_count=metadata.st_nlink,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


@dataclass(frozen=True)
class EvidenceArtifact:
    relative_path: str
    absolute_path: Path
    sha256: str
    byte_count: int
    device: int
    inode: int
    mode: int
    owner_uid: int
    link_count: int
    mtime_ns: int
    ctime_ns: int
    traversed_directories: tuple[EvidencePathIdentity, ...]
    payload: bytes

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }

    def to_binding(self) -> dict[str, str]:
        return {"path": self.relative_path, "sha256": self.sha256}


@dataclass(frozen=True)
class CanonicalOutcome:
    global_ordinal: int
    local_ordinal: int
    doi_id: str
    terminal_status_raw: str
    terminal_status: str

    def to_attempt_dict(self) -> dict[str, Any]:
        return {
            "global_ordinal": self.global_ordinal,
            "local_ordinal": self.local_ordinal,
            "doi_id": self.doi_id,
            "terminal_status_raw": self.terminal_status_raw,
            "terminal_status": self.terminal_status,
        }

    def to_terminal_dict(self) -> dict[str, Any]:
        return {
            "doi_id": self.doi_id,
            "attempt_count": 3,
            "terminal": True,
            "terminal_status_raw": self.terminal_status_raw,
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
            "outcomes": [outcome.to_attempt_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True)
class ChunkEvidence:
    chunk_id: str
    terminal_outcomes: EvidenceArtifact
    sealed_terminal_report: EvidenceArtifact
    sealed_report_hash: str
    attempts: tuple[AttemptLedger, ...]
    final_outcomes: tuple[CanonicalOutcome, ...]

    def terminal_evidence_binding(
        self,
        input_doi_ids_sha256: str,
    ) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "input_doi_ids_sha256": input_doi_ids_sha256,
            "terminal_outcome_file_sha256": self.terminal_outcomes.sha256,
            "sealed_report_file_sha256": self.sealed_terminal_report.sha256,
            "sealed_report_hash": self.sealed_report_hash,
            "attempt_count": 3,
            "terminal": True,
        }

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "terminal_outcomes": self.terminal_outcomes.to_report_dict(),
            "sealed_terminal_report": self.sealed_terminal_report.to_report_dict(),
            "sealed_report_hash": self.sealed_report_hash,
            "attempts": [attempt.to_report_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class SourceCanonicalEvidence:
    doi_id: str
    source_inventory: EvidenceArtifact
    raw_source_evidence: EvidenceArtifact
    canonical_builder: EvidenceArtifact
    eligible_case_ids: tuple[str, ...]
    canonical_case_set_hash: str
    case_descriptor: Mapping[str, Any]
    final_disposition: str
    classification_reason: str
    classification: StratifiedSourceClassification | None


@dataclass(frozen=True)
class ValidatedRawEvidence:
    root: TrustedDirectory
    root_path: Path
    root_identity: EvidencePathIdentity
    manifest: Mapping[str, Any]
    manifest_artifact: EvidenceArtifact
    chunks: tuple[ChunkEvidence, ...]
    source_canonical: tuple[SourceCanonicalEvidence, ...]
    acquisition_dispositions: tuple[Mapping[str, Any], ...]
    stratified_source_classifications: tuple[StratifiedSourceClassification, ...]
    canonical_case_set_manifest_sha256: str
    per_doi_case_set_hashes_sha256: str
    input_artifacts: tuple[EvidenceArtifact, ...]

    def close(self) -> None:
        self.root.close()

    def verify_root(self) -> None:
        verify_trusted_directory(self.root)

    def verify_artifacts(self) -> None:
        """Reopen and compare every validated artifact before/after publication."""

        try:
            self.verify_root()
            root_metadata = validate_trusted_directory_descriptor(
                self.root.descriptor,
                self.root_path,
                label="V2.1 evidence root",
            )
        except ProvenanceError as exc:
            raise C2FullReplacementEvidenceError(
                "V2.1 evidence root changed before publication"
            ) from exc
        if _path_identity("", root_metadata) != self.root_identity:
            raise C2FullReplacementEvidenceError(
                "V2.1 evidence root metadata changed before publication"
            )
        for artifact in self.input_artifacts:
            current = _read_no_follow_artifact(
                self.root,
                artifact.relative_path,
                f"V2.1 revalidation of {artifact.relative_path}",
                root_path=self.root_path,
            )
            if current != artifact:
                raise C2FullReplacementEvidenceError(
                    "V2.1 validated evidence artifact changed before publication: "
                    f"{artifact.relative_path}"
                )
        try:
            self.verify_root()
            root_metadata = validate_trusted_directory_descriptor(
                self.root.descriptor,
                self.root_path,
                label="V2.1 evidence root",
            )
        except ProvenanceError as exc:
            raise C2FullReplacementEvidenceError(
                "V2.1 evidence root changed before publication"
            ) from exc
        if _path_identity("", root_metadata) != self.root_identity:
            raise C2FullReplacementEvidenceError(
                "V2.1 evidence root metadata changed before publication"
            )

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
            f"Cannot read bundled V2.1 schema {schema_name}"
        ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise C2FullReplacementEvidenceError(
            f"Bundled V2.1 schema is invalid: {schema_name}"
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
    *,
    root_path: Path | None = None,
) -> EvidenceArtifact:
    if root.descriptor == -1:
        raise C2FullReplacementEvidenceError("Evidence root is already closed")
    safe_path, components = _safe_relative_path(relative_path, label)
    immutable_root_path = root.path if root_path is None else root_path
    directory_descriptor = os.dup(root.descriptor)
    file_descriptor = -1
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            file_flags |= os.O_NONBLOCK
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
            file_flags |= os.O_CLOEXEC
        try:
            validate_trusted_directory_descriptor(
                directory_descriptor,
                immutable_root_path,
                label=f"{label} evidence root",
            )
        except ProvenanceError as exc:
            raise C2FullReplacementEvidenceError(
                f"{label} evidence root is not trusted: {exc}"
            ) from exc
        traversed_directories: list[EvidencePathIdentity] = []
        directory_components: list[str] = []
        for component in components[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            directory_components.append(component)
            relative_directory = "/".join(directory_components)
            try:
                metadata = validate_trusted_directory_descriptor(
                    next_descriptor,
                    immutable_root_path.joinpath(*directory_components),
                    label=f"{label} evidence directory",
                )
            except ProvenanceError as exc:
                os.close(next_descriptor)
                raise C2FullReplacementEvidenceError(
                    f"{label} evidence directory is not trusted: {relative_directory}: "
                    f"{exc}"
                ) from exc
            traversed_directories.append(_path_identity(relative_directory, metadata))
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            components[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
        try:
            before = validate_trusted_regular_file_descriptor(
                file_descriptor,
                immutable_root_path.joinpath(*components),
                label=f"{label} evidence artifact",
            )
        except ProvenanceError as exc:
            raise C2FullReplacementEvidenceError(
                f"{label} evidence artifact is not trusted: {exc}"
            ) from exc
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        while True:
            block = os.read(file_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            blocks.append(block)
        payload = b"".join(blocks)
        try:
            after = validate_trusted_regular_file_descriptor(
                file_descriptor,
                immutable_root_path.joinpath(*components),
                label=f"{label} evidence artifact",
            )
        except ProvenanceError as exc:
            raise C2FullReplacementEvidenceError(
                f"{label} evidence artifact is not trusted after reading: {exc}"
            ) from exc
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or before.st_uid != after.st_uid
            or stat.S_IMODE(before.st_mode) != stat.S_IMODE(after.st_mode)
            or before.st_nlink != after.st_nlink
        ):
            raise C2FullReplacementEvidenceError(
                f"{label} changed while it was read"
            )
        return EvidenceArtifact(
            relative_path=safe_path,
            absolute_path=immutable_root_path.joinpath(*components),
            sha256=digest.hexdigest(),
            byte_count=len(payload),
            device=before.st_dev,
            inode=before.st_ino,
            mode=stat.S_IMODE(before.st_mode),
            owner_uid=before.st_uid,
            link_count=before.st_nlink,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
            traversed_directories=tuple(traversed_directories),
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


def _require_normalized_doi(value: Any, label: str) -> str:
    try:
        normalized = normalize_doi(value)
    except ProvenanceError as exc:
        raise C2FullReplacementEvidenceError(f"{label} is not a valid DOI") from exc
    if value != normalized:
        raise C2FullReplacementEvidenceError(f"{label} must already be normalized")
    return normalized


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: item for key, item in value.items() if key not in omitted}


class _ArtifactCollector:
    def __init__(self, root: TrustedDirectory) -> None:
        self.root = root
        self._by_path: dict[str, EvidenceArtifact] = {}
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
        *,
        allow_reuse: bool = False,
    ) -> EvidenceArtifact:
        safe_path, _ = _safe_relative_path(relative_path, label)
        existing = self._by_path.get(safe_path)
        if existing is not None:
            if not allow_reuse:
                raise C2FullReplacementEvidenceError(
                    f"V2.1 evidence reuses an artifact path: {safe_path}"
                )
            if expected_sha256 is not None and existing.sha256 != expected_sha256:
                raise C2FullReplacementEvidenceError(
                    f"{label} reuses an artifact with a mismatched SHA-256"
                )
            return existing
        artifact = _read_no_follow_artifact(self.root, safe_path, label)
        if expected_sha256 is not None and artifact.sha256 != expected_sha256:
            raise C2FullReplacementEvidenceError(
                f"{label} bytes do not match the manifest SHA-256 binding"
            )
        identity = (artifact.device, artifact.inode)
        if identity in self._identities:
            raise C2FullReplacementEvidenceError(
                f"V2.1 evidence reuses an artifact inode: {safe_path}"
            )
        self._by_path[safe_path] = artifact
        self._identities.add(identity)
        self._artifacts.append(artifact)
        return artifact


def _open_manifest_root(manifest_path: Path) -> tuple[TrustedDirectory, str]:
    raw_path = Path(os.fspath(manifest_path))
    if raw_path.name in {"", ".", ".."} or raw_path.suffix != ".json":
        raise C2FullReplacementEvidenceError(
            "V2.1 admission manifest must name a JSON file"
        )
    if any(part in {".", ".."} for part in raw_path.parts):
        raise C2FullReplacementEvidenceError(
            "V2.1 admission manifest path must not contain dot traversal"
        )
    try:
        absolute = Path(os.path.abspath(os.fspath(raw_path.expanduser())))
    except (OSError, RuntimeError) as exc:
        raise C2FullReplacementEvidenceError(
            f"Cannot normalize V2.1 admission manifest path: {manifest_path}"
        ) from exc
    try:
        root = open_trusted_directory(absolute.parent)
    except ProvenanceError as exc:
        raise C2FullReplacementEvidenceError(
            f"Cannot open trusted V2.1 evidence root: {absolute.parent}"
        ) from exc
    return root, absolute.name


def _validate_manifest(
    manifest: Mapping[str, Any],
    policy: CompiledFullReplacementPolicy,
) -> tuple[Mapping[str, Any], ...]:
    _validate_schema(
        manifest,
        "c2_full_replacement_admission_manifest_v2.schema.json",
        "V2.1 admission manifest",
    )
    if sha256_json(_without(manifest, "manifest_hash")) != manifest["manifest_hash"]:
        raise C2FullReplacementEvidenceError(
            "V2.1 admission manifest failed its semantic hash"
        )
    if dict(manifest["frozen_universe"]) != policy.frozen_universe.to_dict():
        raise C2FullReplacementEvidenceError(
            "V2.1 manifest frozen universe differs from the compiled policy"
        )
    if dict(manifest["frozen_bindings"]) != dict(policy.frozen_bindings):
        raise C2FullReplacementEvidenceError(
            "V2.1 manifest frozen bindings differ from the compiled policy"
        )
    code = manifest["code"]
    if (
        not isinstance(code["commit"], str)
        or _COMMIT_RE.fullmatch(code["commit"]) is None
        or code["dirty"] is not False
    ):
        raise C2FullReplacementEvidenceError(
            "V2.1 admission manifest requires a full clean code commit"
        )
    chunks = manifest["chunks"]
    if [chunk["chunk_id"] for chunk in chunks] != list(CHUNK_IDS):
        raise C2FullReplacementEvidenceError(
            "V2.1 admission manifest must use ordered chunks 001..013"
        )
    for chunk_id, chunk in zip(CHUNK_IDS, chunks, strict=True):
        partition = policy.partition_for_chunk(chunk_id)
        root_plan = policy.root_plan_for_chunk(chunk_id)
        if (
            chunk["input_total"] != partition.input_total
            or chunk["input_doi_ids_sha256"] != partition.doi_ids_sha256
        ):
            raise C2FullReplacementEvidenceError(
                f"V2.1 chunk {chunk_id} has an invalid pinned partition"
            )
        root = chunk["root"]
        if (
            root["replacement_root_id"] != root_plan.replacement_root_id
            or tuple(root["retired_root_ids"]) != root_plan.retired_root_ids
            or root["partial_root"] is not False
        ):
            raise C2FullReplacementEvidenceError(
                f"V2.1 chunk {chunk_id} root plan differs from compiled policy"
            )
        if [attempt["attempt_id"] for attempt in chunk["attempts"]] != list(
            ATTEMPT_IDS
        ):
            raise C2FullReplacementEvidenceError(
                f"V2.1 chunk {chunk_id} must contain initial, retry1, retry2"
            )
    return tuple(chunks)


def _read_bound_artifact(
    collector: _ArtifactCollector,
    binding: Mapping[str, Any],
    label: str,
    *,
    allow_reuse: bool = False,
) -> EvidenceArtifact:
    return collector.read(
        binding["path"],
        _require_sha256(binding["sha256"], f"{label}.sha256"),
        label,
        allow_reuse=allow_reuse,
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
    expected_dois: Sequence[str],
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
    for local_ordinal, (record, doi_id) in enumerate(
        zip(raw_rows, expected_dois, strict=True),
        start=1,
    ):
        _validate_schema(
            record,
            "c2_full_replacement_raw_record_v2.schema.json",
            f"chunk {chunk_id} {attempt_id} raw row {local_ordinal}",
        )
        if (
            record["attempt_id"] != attempt_id
            or record["global_ordinal"]
            != partition.first_global_ordinal + local_ordinal - 1
            or record["local_ordinal"] != local_ordinal
            or record["doi_id"] != doi_id
        ):
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} raw stream differs from frozen DOI order"
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
            f"chunk {chunk_id} {attempt_id} processed-success is not the exact "
            "raw processed partition"
        )
    if [_record_key(row) for row in skipped["records"]] != expected_skipped_keys:
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} {attempt_id} skipped-status is not the exact raw "
            "skipped partition"
        )
    statuses_by_key: dict[tuple[int, int, str], tuple[str, str]] = {}
    for row in skipped["records"]:
        raw_status = row["terminal_status_raw"]
        try:
            terminal_status = adapt_terminal_status(raw_status)
        except ProvenanceError as exc:
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} has an unapproved raw status"
            ) from exc
        if terminal_status == "DOWNLOADED":
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} skipped-status encodes DOWNLOADED"
            )
        key = _record_key(row)
        if key in statuses_by_key:
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} skipped-status repeats a DOI"
            )
        statuses_by_key[key] = (raw_status, terminal_status)
    outcomes: list[CanonicalOutcome] = []
    for row in raw_rows:
        if row["raw_disposition"] == "PROCESSED":
            raw_status = "downloaded"
            terminal_status = adapt_terminal_status(raw_status)
        else:
            raw_status, terminal_status = statuses_by_key[_record_key(row)]
        if terminal_status not in TERMINAL_OUTCOME_STATUSES:
            raise C2FullReplacementEvidenceError(
                f"chunk {chunk_id} {attempt_id} has a nonterminal status"
            )
        outcomes.append(
            CanonicalOutcome(
                global_ordinal=row["global_ordinal"],
                local_ordinal=row["local_ordinal"],
                doi_id=row["doi_id"],
                terminal_status_raw=raw_status,
                terminal_status=terminal_status,
            )
        )
    return AttemptLedger(
        chunk_id=chunk_id,
        attempt_id=attempt_id,
        raw_stream=raw_artifact,
        processed_success=processed_artifact,
        skipped_status=skipped_artifact,
        outcomes=tuple(outcomes),
    )


def _validate_terminal_chunk(
    *,
    collector: _ArtifactCollector,
    chunk: Mapping[str, Any],
    chunk_id: str,
    partition: ChunkPartition,
    attempts: tuple[AttemptLedger, ...],
) -> ChunkEvidence:
    terminal_outcomes = _read_bound_artifact(
        collector,
        chunk["terminal_outcomes"],
        f"chunk {chunk_id} terminal outcomes",
    )
    terminal = _parse_json_object(terminal_outcomes, f"chunk {chunk_id} terminal outcomes")
    _validate_schema(
        terminal,
        "c2_full_replacement_terminal_outcomes_v2.schema.json",
        f"chunk {chunk_id} terminal outcomes",
    )
    expected_outcomes = [item.to_terminal_dict() for item in attempts[-1].outcomes]
    if (
        terminal["chunk_id"] != chunk_id
        or terminal["input_doi_ids_sha256"] != partition.doi_ids_sha256
        or terminal["outcomes"] != expected_outcomes
    ):
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} terminal outcome ledger differs from retry2 evidence"
        )
    sealed_artifact = _read_bound_artifact(
        collector,
        chunk["sealed_terminal_report"],
        f"chunk {chunk_id} sealed terminal report",
    )
    sealed = _parse_json_object(
        sealed_artifact,
        f"chunk {chunk_id} sealed terminal report",
    )
    _validate_schema(
        sealed,
        "c2_full_replacement_sealed_terminal_report_v2.schema.json",
        f"chunk {chunk_id} sealed terminal report",
    )
    report_hash = sealed["report_hash"]
    if sha256_json(_without(sealed, "report_hash", "seal")) != report_hash:
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} sealed terminal report hash mismatch"
        )
    seal = sealed["seal"]
    if (
        seal["status"] != "TERMINAL"
        or seal["sealed_report_hash"] != report_hash
        or sha256_json(_without(seal, "seal_hash")) != seal["seal_hash"]
    ):
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} terminal seal is invalid"
        )
    if (
        sealed["chunk_id"] != chunk_id
        or sealed["input_doi_ids_sha256"] != partition.doi_ids_sha256
        or sealed["terminal_outcomes_file_sha256"] != terminal_outcomes.sha256
    ):
        raise C2FullReplacementEvidenceError(
            f"chunk {chunk_id} sealed terminal report binding is invalid"
        )
    return ChunkEvidence(
        chunk_id=chunk_id,
        terminal_outcomes=terminal_outcomes,
        sealed_terminal_report=sealed_artifact,
        sealed_report_hash=report_hash,
        attempts=attempts,
        final_outcomes=attempts[-1].outcomes,
    )


def _require_sha_fields(value: Mapping[str, Any], names: Sequence[str], label: str) -> None:
    for name in names:
        _require_sha256(value[name], f"{label}.{name}")


def _validate_source_table(
    *,
    collector: _ArtifactCollector,
    panel: Mapping[str, Any],
    parent_doi_id: str,
    candidate_raw_sources: Mapping[str, EvidenceArtifact],
    case_bindings: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    if panel["source_table_path"] in {
        artifact.relative_path for artifact in candidate_raw_sources.values()
    }:
        raise C2FullReplacementEvidenceError(
            f"{label} source table aliases a verified raw source artifact"
        )
    source_artifact = collector.read(
        panel["source_table_path"],
        panel["source_table_sha256"],
        f"{label} source table",
        allow_reuse=True,
    )
    table = _parse_json_object(source_artifact, f"{label} source table")
    _validate_schema(
        table,
        "c2_full_replacement_source_table_v2.schema.json",
        f"{label} source table",
    )
    if (
        _require_normalized_doi(table["parent_doi_id"], f"{label} table parent")
        != parent_doi_id
        or table["source_candidate_id"] != panel["source_candidate_id"]
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} source table has a cross-DOI or candidate mismatch"
        )
    raw_source = candidate_raw_sources.get(panel["source_candidate_id"])
    if (
        raw_source is None
        or panel["raw_source_sha256"] != raw_source.sha256
        or table["raw_source_sha256"] != raw_source.sha256
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} source table does not bind its verified raw source bytes"
        )
    for name in (
        "candidate_binding_sha256",
        "proposal_binding_sha256",
        "canonical_case_binding_sha256",
        "verification_evidence_sha256",
    ):
        if table[name] != panel[name]:
            raise C2FullReplacementEvidenceError(
                f"{label} source table binding differs from panel evidence"
            )
    if (
        panel["candidate_binding_sha256"]
        != case_bindings["candidate_binding_sha256"]
        or panel["proposal_binding_sha256"]
        != case_bindings["proposal_binding_sha256"]
        or panel["canonical_case_binding_sha256"]
        != case_bindings["canonical_case_binding_sha256"]
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} panel binding differs from canonical case binding"
        )
    return table


def _case_descriptor(
    *,
    case: Mapping[str, Any],
    parent_doi_id: str,
    source_candidates: set[str],
    candidate_raw_sources: Mapping[str, EvidenceArtifact],
    collector: _ArtifactCollector,
    label: str,
    validate_case_bindings: Mapping[str, str],
) -> tuple[Mapping[str, Any], bool]:
    case_id = case["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise C2FullReplacementEvidenceError(f"{label}.case_id is required")
    if _require_normalized_doi(case["doi_id"], f"{label}.doi_id") != parent_doi_id:
        raise C2FullReplacementEvidenceError(f"{label} has an asserted cross-DOI case")
    bindings = case["bindings"]
    _require_sha_fields(
        bindings,
        (
            "raw_source_evidence_sha256",
            "candidate_binding_sha256",
            "proposal_binding_sha256",
            "review_binding_sha256",
            "canonical_case_binding_sha256",
        ),
        f"{label}.bindings",
    )
    for name, expected in validate_case_bindings.items():
        if bindings[name] != expected:
            raise C2FullReplacementEvidenceError(
                f"{label} binding differs from canonical builder evidence"
            )
    canonical_case_binding_input = {
        name: case[name]
        for name in (
            "case_id",
            "doi_id",
            "case_kind",
            "curation_status",
            "eligible_for_experiment",
            "asserted_panel_ids",
            "expected_evaluation_panel_ids",
            "asserted_panel_count",
            "asserted_public_stratum",
            "asserted_code_label",
        )
    }
    if bindings["canonical_case_binding_sha256"] != sha256_json(
        canonical_case_binding_input
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} canonical-case binding is not derived from the case record"
        )
    panels = case["verified_panels"]
    seen_panel_ids: set[str] = set()
    seen_candidates: set[str] = set()
    seen_fingerprints: set[str] = set()
    normalized_panels: list[Mapping[str, Any]] = []
    for panel_index, panel in enumerate(panels):
        panel_label = f"{label}.verified_panels[{panel_index}]"
        if not isinstance(panel["panel_id"], str) or not panel["panel_id"]:
            raise C2FullReplacementEvidenceError(f"{panel_label}.panel_id is required")
        if (
            not isinstance(panel["source_candidate_id"], str)
            or not panel["source_candidate_id"]
        ):
            raise C2FullReplacementEvidenceError(
                f"{panel_label}.source_candidate_id is required"
            )
        if (
            _require_normalized_doi(panel["parent_doi_id"], f"{panel_label}.parent")
            != parent_doi_id
        ):
            raise C2FullReplacementEvidenceError(
                f"{panel_label} has a cross-DOI parent binding"
            )
        if panel["panel_id"] in seen_panel_ids:
            raise C2FullReplacementEvidenceError(f"{label} repeats a panel_id")
        if panel["source_candidate_id"] in seen_candidates:
            raise C2FullReplacementEvidenceError(
                f"{label} repeats a source_candidate_id"
            )
        fingerprint = sha256_json(dict(panel))
        if fingerprint in seen_fingerprints:
            raise C2FullReplacementEvidenceError(f"{label} repeats a panel fingerprint")
        seen_panel_ids.add(panel["panel_id"])
        seen_candidates.add(panel["source_candidate_id"])
        seen_fingerprints.add(fingerprint)
        if panel["source_candidate_id"] not in source_candidates:
            raise C2FullReplacementEvidenceError(
                f"{panel_label} does not resolve to a verified same-DOI source case"
            )
        _validate_source_table(
            collector=collector,
            panel=panel,
            parent_doi_id=parent_doi_id,
            candidate_raw_sources=candidate_raw_sources,
            case_bindings=bindings,
            label=panel_label,
        )
        normalized_panels.append(dict(panel))
    ordered_panels = tuple(
        sorted(
            normalized_panels,
            key=lambda item: (item["panel_id"], item["source_candidate_id"]),
        )
    )
    panel_ids = [panel["panel_id"] for panel in ordered_panels]
    if (
        case["asserted_panel_ids"] != panel_ids
        or case["expected_evaluation_panel_ids"] != panel_ids
        or case["asserted_panel_count"] != len(panel_ids)
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} asserted panel membership/count does not match recomputation"
        )
    if case["case_kind"] == "single" and len(panel_ids) != 1:
        raise C2FullReplacementEvidenceError(
            f"{label} single case must have exactly one verified panel"
        )
    if case["case_kind"] == "multi" and len(panel_ids) < 2:
        raise C2FullReplacementEvidenceError(
            f"{label} multi case must have at least two verified panels"
        )
    derived_stratum = derive_public_stratum(len(panel_ids))
    if (
        case["asserted_public_stratum"] != derived_stratum
        or case["asserted_code_label"] != P_CODE_LABELS[derived_stratum]
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} asserted P label differs from recomputed panels"
        )
    descriptor = {
        "case_id": case_id,
        "parent_doi_id": parent_doi_id,
        "verified_panels": list(ordered_panels),
        "qualified_panel_count": len(panel_ids),
        "derived_public_stratum": derived_stratum,
        "derived_code_label": P_CODE_LABELS[derived_stratum],
        "bindings": dict(bindings),
    }
    eligible = (
        case["curation_status"] == "verified"
        and case["eligible_for_experiment"] is True
    )
    return descriptor, eligible


def _read_parent_binding(
    *,
    collector: _ArtifactCollector,
    binding: Mapping[str, Any],
    expected_artifact_type: str,
    parent_doi_id: str,
    label: str,
) -> EvidenceArtifact:
    artifact = _read_bound_artifact(collector, binding, label)
    value = _parse_json_object(artifact, label)
    _validate_schema(
        value,
        "c2_full_replacement_parent_binding_v2.schema.json",
        label,
    )
    if (
        value["artifact_type"] != expected_artifact_type
        or _require_normalized_doi(value["parent_doi_id"], f"{label}.parent_doi_id")
        != parent_doi_id
        or value["complete"] is not True
        or value["model_result_selected"] is not False
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} has an invalid parent DOI/provenance binding"
        )
    return artifact


def _read_raw_source_evidence(
    *,
    collector: _ArtifactCollector,
    inventory: Mapping[str, Any],
    parent_doi_id: str,
    candidate_ids: tuple[str, ...],
    label: str,
) -> tuple[EvidenceArtifact, dict[str, EvidenceArtifact]]:
    artifact = _read_bound_artifact(
        collector,
        inventory["raw_source_evidence"],
        f"{label} raw source evidence",
    )
    value = _parse_json_object(artifact, f"{label} raw source evidence")
    _validate_schema(
        value,
        "c2_full_replacement_raw_source_evidence_v2.schema.json",
        f"{label} raw source evidence",
    )
    if (
        _require_normalized_doi(
            value["parent_doi_id"],
            f"{label} raw source evidence parent",
        )
        != parent_doi_id
        or value["complete"] is not True
        or value["model_result_selected"] is not False
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} raw source evidence has an invalid parent/provenance binding"
        )
    records = value["source_candidates"]
    raw_candidate_ids = tuple(record["source_candidate_id"] for record in records)
    if (
        raw_candidate_ids != candidate_ids
        or raw_candidate_ids != tuple(sorted(raw_candidate_ids))
        or len(raw_candidate_ids) != len(set(raw_candidate_ids))
    ):
        raise C2FullReplacementEvidenceError(
            f"{label} raw source evidence candidate coverage is incomplete"
        )
    raw_sources: dict[str, EvidenceArtifact] = {}
    for record in records:
        candidate_id = record["source_candidate_id"]
        if (
            _require_normalized_doi(
                record["parent_doi_id"],
                f"{label} raw source candidate {candidate_id}",
            )
            != parent_doi_id
        ):
            raise C2FullReplacementEvidenceError(
                f"{label} raw source candidate has a cross-DOI parent binding"
            )
        raw_sources[candidate_id] = collector.read(
            record["raw_source_path"],
            _require_sha256(
                record["raw_source_sha256"],
                f"{label} raw source candidate {candidate_id}.sha256",
            ),
            f"{label} raw source candidate {candidate_id}",
        )
    return artifact, raw_sources


def _validate_source_canonical(
    *,
    collector: _ArtifactCollector,
    binding: Mapping[str, Any],
    policy: CompiledFullReplacementPolicy,
    global_case_ids: set[str],
) -> SourceCanonicalEvidence:
    doi_id = _require_normalized_doi(binding["doi_id"], "source_canonical.doi_id")
    source_inventory = _read_bound_artifact(
        collector,
        binding["source_inventory"],
        f"DOI {doi_id} source inventory",
    )
    inventory = _parse_json_object(source_inventory, f"DOI {doi_id} source inventory")
    _validate_schema(
        inventory,
        "c2_full_replacement_source_inventory_v2.schema.json",
        f"DOI {doi_id} source inventory",
    )
    if _require_normalized_doi(inventory["parent_doi_id"], "source inventory parent") != doi_id:
        raise C2FullReplacementEvidenceError(
            "Source inventory has a cross-DOI parent binding"
        )
    candidate_ids = tuple(inventory["source_candidate_ids"])
    if candidate_ids != tuple(sorted(candidate_ids)) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise C2FullReplacementEvidenceError(
            f"DOI {doi_id} source inventory candidate IDs must be unique and ordered"
        )
    raw_source_evidence, candidate_raw_sources = _read_raw_source_evidence(
        collector=collector,
        inventory=inventory,
        parent_doi_id=doi_id,
        candidate_ids=candidate_ids,
        label=f"DOI {doi_id}",
    )
    builder_artifact = _read_bound_artifact(
        collector,
        binding["canonical_builder"],
        f"DOI {doi_id} canonical builder",
    )
    builder = _parse_json_object(builder_artifact, f"DOI {doi_id} canonical builder")
    _validate_schema(
        builder,
        "c2_full_replacement_canonical_builder_v2.schema.json",
        f"DOI {doi_id} canonical builder",
    )
    if (
        _require_normalized_doi(builder["parent_doi_id"], "canonical builder parent")
        != doi_id
        or builder["source_inventory_sha256"] != source_inventory.sha256
        or builder["raw_source_evidence_sha256"] != raw_source_evidence.sha256
        or builder["parent_doi_ids_sha256"] != policy.frozen_universe.doi_ids_sha256
    ):
        raise C2FullReplacementEvidenceError(
            f"DOI {doi_id} canonical builder has an invalid source/parent binding"
        )
    builder_code = builder["code"]
    frozen = policy.frozen_bindings
    if (
        builder_code["commit"] != frozen["canonical_builder_code_commit_full"]
        or builder_code["sha256"] != frozen["canonical_builder_code_sha256"]
        or builder_code["dirty"] is not False
        or builder["builder_rule"]["version"]
        != frozen["canonical_builder_rule_version"]
        or builder["builder_rule"]["sha256"]
        != frozen["canonical_builder_rule_sha256"]
    ):
        raise C2FullReplacementEvidenceError(
            f"DOI {doi_id} canonical builder code/rule is not frozen"
        )
    candidate_manifest = _read_parent_binding(
        collector=collector,
        binding=builder["candidate_manifest"],
        expected_artifact_type="c2_v21_candidate_manifest",
        parent_doi_id=doi_id,
        label=f"DOI {doi_id} candidate manifest",
    )
    proposal_manifest = _read_parent_binding(
        collector=collector,
        binding=builder["proposal_manifest"],
        expected_artifact_type="c2_v21_proposal_manifest",
        parent_doi_id=doi_id,
        label=f"DOI {doi_id} proposal manifest",
    )
    review_manifest = _read_parent_binding(
        collector=collector,
        binding=builder["review_manifest"],
        expected_artifact_type="c2_v21_review_manifest",
        parent_doi_id=doi_id,
        label=f"DOI {doi_id} review manifest",
    )
    canonical_manifest = _read_parent_binding(
        collector=collector,
        binding=builder["canonical_manifest"],
        expected_artifact_type="c2_v21_canonical_manifest",
        parent_doi_id=doi_id,
        label=f"DOI {doi_id} canonical manifest",
    )
    source_cases = builder["source_cases"]
    source_case_ids = tuple(item["source_candidate_id"] for item in source_cases)
    if (
        source_case_ids != candidate_ids
        or len(source_case_ids) != len(set(source_case_ids))
    ):
        raise C2FullReplacementEvidenceError(
            f"DOI {doi_id} canonical builder source-case coverage is incomplete"
        )
    for source_case in source_cases:
        if (
            _require_normalized_doi(
                source_case["parent_doi_id"],
                "source case parent",
            )
            != doi_id
            or source_case["verified"] is not True
            or source_case["eligible_single_source"] is not True
            or source_case["raw_source_sha256"]
            != candidate_raw_sources[source_case["source_candidate_id"]].sha256
        ):
            raise C2FullReplacementEvidenceError(
                f"DOI {doi_id} source candidate is not a verified same-DOI source case"
            )
    if not candidate_ids and (builder["source_cases"] or builder["cases"]):
        raise C2FullReplacementEvidenceError(
            f"DOI {doi_id} empty source inventory cannot emit canonical cases"
        )
    case_bindings = {
        "raw_source_evidence_sha256": raw_source_evidence.sha256,
        "candidate_binding_sha256": candidate_manifest.sha256,
        "proposal_binding_sha256": proposal_manifest.sha256,
        "review_binding_sha256": review_manifest.sha256,
    }
    descriptors: list[Mapping[str, Any]] = []
    for case_index, case in enumerate(builder["cases"]):
        descriptor, eligible = _case_descriptor(
            case=case,
            parent_doi_id=doi_id,
            source_candidates=set(candidate_ids),
            candidate_raw_sources=candidate_raw_sources,
            collector=collector,
            label=f"DOI {doi_id} canonical case {case_index}",
            validate_case_bindings=case_bindings,
        )
        if descriptor["case_id"] in global_case_ids:
            raise C2FullReplacementEvidenceError(
                "Canonical builder output repeats a globally unique case_id"
            )
        global_case_ids.add(descriptor["case_id"])
        if eligible:
            descriptors.append(descriptor)
    descriptors.sort(key=lambda item: item["case_id"])
    case_ids = tuple(item["case_id"] for item in descriptors)
    case_set_hash = sha256_json({"doi_id": doi_id, "cases": descriptors})
    empty_inventory = not candidate_ids
    if empty_inventory:
        final_disposition = "NON_STRATIFIED_DOWNLOADED_NO_VERIFIED_SOURCE"
        reason = "DOWNLOADED_EMPTY_VERIFIED_SOURCE_INVENTORY"
        classification = None
    elif not descriptors:
        final_disposition = "NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE"
        reason = "DOWNLOADED_NO_QUALIFYING_CANONICAL_CASE"
        classification = None
    else:
        strata = {item["derived_public_stratum"] for item in descriptors}
        if len(strata) != 1:
            raise C2FullReplacementEvidenceError(
                f"DOI {doi_id} is MULTI_STRATUM_CANONICAL_DOI"
            )
        disposition = next(iter(strata))
        counts = tuple(item["qualified_panel_count"] for item in descriptors)
        panel_descriptors = [
            item["verified_panels"]
            for item in descriptors
        ]
        classification = StratifiedSourceClassification(
            doi_id=doi_id,
            canonical_case_set_hash=case_set_hash,
            canonical_case_ids=case_ids,
            verified_panel_descriptors_sha256=sha256_json(panel_descriptors),
            qualified_panel_counts=counts,
            derived_public_stratum=disposition,
            derived_code_label=P_CODE_LABELS[disposition],
            source_binding_hashes=freeze_evidence_value(
                {
                    "source_inventory_sha256": source_inventory.sha256,
                    "raw_source_evidence_sha256": raw_source_evidence.sha256,
                    "candidate_manifest_sha256": candidate_manifest.sha256,
                    "proposal_manifest_sha256": proposal_manifest.sha256,
                    "review_manifest_sha256": review_manifest.sha256,
                    "canonical_manifest_sha256": canonical_manifest.sha256,
                    "canonical_builder_sha256": builder_artifact.sha256,
                }
            ),
        )
        final_disposition = STRATIFIED_DISPOSITION
        reason = "VERIFIED_SOURCE_CANONICAL_CASES"
    return SourceCanonicalEvidence(
        doi_id=doi_id,
        source_inventory=source_inventory,
        raw_source_evidence=raw_source_evidence,
        canonical_builder=builder_artifact,
        eligible_case_ids=case_ids,
        canonical_case_set_hash=case_set_hash,
        case_descriptor=freeze_evidence_value({"doi_id": doi_id, "cases": descriptors}),
        final_disposition=final_disposition,
        classification_reason=reason,
        classification=classification,
    )


def _expected_non_downloaded_disposition(
    *,
    outcome: CanonicalOutcome,
    terminal_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    final_disposition = expected_non_stratified_disposition(outcome.terminal_status)
    reason = {
        "NO_SOURCE_DATA": "TERMINAL_NO_SOURCE_DATA",
        "NO_FIGURES": "TERMINAL_NO_FIGURES",
        "NO_USABLE_CONTENT": "TERMINAL_NO_USABLE_CONTENT",
        "POLICY_REJECTED": "TERMINAL_POLICY_REJECTED",
        "DOWNLOAD_FAILED": "TERMINAL_DOWNLOAD_FAILED",
        "RETRY_EXHAUSTED": "TERMINAL_RETRY_EXHAUSTED",
    }[outcome.terminal_status]
    return {
        "doi_id": outcome.doi_id,
        "terminal_status_raw": outcome.terminal_status_raw,
        "terminal_status": outcome.terminal_status,
        "terminal_evidence_binding": dict(terminal_evidence),
        "final_disposition": final_disposition,
        "source_inventory_binding_or_null": None,
        "canonical_builder_binding_or_null": None,
        "all_eligible_case_ids": [],
        "classification_reason": reason,
    }


def _build_acquisition_dispositions(
    *,
    policy: CompiledFullReplacementPolicy,
    chunks: Sequence[ChunkEvidence],
    source_evidence: Mapping[str, SourceCanonicalEvidence],
) -> tuple[Mapping[str, Any], ...]:
    expected: list[Mapping[str, Any]] = []
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    for chunk_id in CHUNK_IDS:
        partition = policy.partition_for_chunk(chunk_id)
        chunk = by_chunk[chunk_id]
        terminal_binding = chunk.terminal_evidence_binding(partition.doi_ids_sha256)
        for outcome in chunk.final_outcomes:
            if outcome.terminal_status != "DOWNLOADED":
                expected.append(
                    _expected_non_downloaded_disposition(
                        outcome=outcome,
                        terminal_evidence=terminal_binding,
                    )
                )
                continue
            source = source_evidence.get(outcome.doi_id)
            if source is None:
                raise C2FullReplacementEvidenceError(
                    f"DOWNLOADED DOI {outcome.doi_id} lacks source/canonical coverage"
                )
            expected.append(
                {
                    "doi_id": outcome.doi_id,
                    "terminal_status_raw": outcome.terminal_status_raw,
                    "terminal_status": outcome.terminal_status,
                    "terminal_evidence_binding": dict(terminal_binding),
                    "final_disposition": source.final_disposition,
                    "source_inventory_binding_or_null": source.source_inventory.to_binding(),
                    "canonical_builder_binding_or_null": source.canonical_builder.to_binding(),
                    "all_eligible_case_ids": list(source.eligible_case_ids),
                    "classification_reason": source.classification_reason,
                }
            )
    if tuple(item["doi_id"] for item in expected) != policy.ordered_doi_ids:
        raise C2FullReplacementEvidenceError(
            "V2.1 acquisition dispositions do not cover ordered frozen DOI inputs"
        )
    return tuple(expected)


def load_and_validate_raw_evidence(
    manifest_path: Path,
    policy: CompiledFullReplacementPolicy,
) -> ValidatedRawEvidence:
    """Read every dynamic artifact once and derive V2.1 source classifications."""

    if not policy.is_test_only:
        raise C2FullReplacementEvidenceError(
            "Stage-A evidence validation accepts only a synthetic policy"
        )
    root: TrustedDirectory | None = None
    try:
        root, manifest_name = _open_manifest_root(manifest_path)
        collector = _ArtifactCollector(root)
        manifest_artifact = collector.read(manifest_name, None, "V2.1 admission manifest")
        manifest = _parse_json_object(manifest_artifact, "V2.1 admission manifest")
        manifest_chunks = _validate_manifest(manifest, policy)
        chunks: list[ChunkEvidence] = []
        for chunk_id, chunk in zip(CHUNK_IDS, manifest_chunks, strict=True):
            partition = policy.partition_for_chunk(chunk_id)
            attempts = tuple(
                _validate_attempt(
                    collector=collector,
                    binding=attempt,
                    chunk_id=chunk_id,
                    partition=partition,
                    expected_dois=policy.dois_for_chunk(chunk_id),
                )
                for attempt in chunk["attempts"]
            )
            if chunk_id == "013":
                expected_global = tuple(range(2401, 2464))
                for attempt in attempts:
                    if (
                        tuple(item.global_ordinal for item in attempt.outcomes)
                        != expected_global
                        or tuple(item.local_ordinal for item in attempt.outcomes)
                        != tuple(range(1, 64))
                    ):
                        raise C2FullReplacementEvidenceError(
                            "chunk 013 must contain exactly ordinals 2401..2463 "
                            "and local positions 1..63 for every attempt"
                        )
            chunks.append(
                _validate_terminal_chunk(
                    collector=collector,
                    chunk=chunk,
                    chunk_id=chunk_id,
                    partition=partition,
                    attempts=attempts,
                )
            )
        downloaded_dois = tuple(
            outcome.doi_id
            for chunk in chunks
            for outcome in chunk.final_outcomes
            if outcome.terminal_status == "DOWNLOADED"
        )
        source_bindings = manifest["source_canonical"]
        if tuple(item["doi_id"] for item in source_bindings) != downloaded_dois:
            raise C2FullReplacementEvidenceError(
                "V2.1 source/canonical entries must exactly cover ordered DOWNLOADED DOI"
            )
        source_records: list[SourceCanonicalEvidence] = []
        global_case_ids: set[str] = set()
        for source_binding in source_bindings:
            source_records.append(
                _validate_source_canonical(
                    collector=collector,
                    binding=source_binding,
                    policy=policy,
                    global_case_ids=global_case_ids,
                )
            )
        source_by_doi = {item.doi_id: item for item in source_records}
        expected_dispositions = _build_acquisition_dispositions(
            policy=policy,
            chunks=chunks,
            source_evidence=source_by_doi,
        )
        if tuple(manifest["acquisition_dispositions"]) != expected_dispositions:
            raise C2FullReplacementEvidenceError(
                "V2.1 acquisition dispositions are incomplete, selective, or not "
                "derived from terminal/source/canonical evidence"
            )
        classifications = tuple(
            item.classification
            for item in source_records
            if item.classification is not None
        )
        case_manifest = [
            {
                "doi_id": doi_id,
                "cases": (
                    thaw_evidence_value(source_by_doi[doi_id].case_descriptor)["cases"]
                    if doi_id in source_by_doi
                    else []
                ),
            }
            for doi_id in policy.ordered_doi_ids
        ]
        per_doi_case_hashes = [
            sha256_json(item) for item in case_manifest
        ]
        try:
            root_metadata = validate_trusted_directory_descriptor(
                root.descriptor,
                root.path,
                label="V2.1 evidence root",
            )
        except ProvenanceError as exc:
            raise C2FullReplacementEvidenceError(
                "V2.1 evidence root is not trusted after validation"
            ) from exc
        return ValidatedRawEvidence(
            root=root,
            root_path=root.path,
            root_identity=_path_identity("", root_metadata),
            manifest=freeze_evidence_value(manifest),
            manifest_artifact=manifest_artifact,
            chunks=tuple(chunks),
            source_canonical=tuple(source_records),
            acquisition_dispositions=tuple(
                freeze_evidence_value(item) for item in expected_dispositions
            ),
            stratified_source_classifications=classifications,
            canonical_case_set_manifest_sha256=sha256_json(case_manifest),
            per_doi_case_set_hashes_sha256=sha256_json(per_doi_case_hashes),
            input_artifacts=collector.artifacts,
        )
    except Exception:
        if root is not None:
            root.close()
        raise

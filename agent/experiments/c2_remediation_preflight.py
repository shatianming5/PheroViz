"""Read-only, fail-closed C2 remediation evidence preflight.

This module inventories explicitly named inputs only.  It neither runs
acquisition nor invokes either C2 finalizer, and it never treats a legacy or
candidate root as a final C2 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import models as _models
from .c2_full_replacement_policy import (
    C2FullReplacementPolicyError,
    load_production_policy,
    normalize_doi,
)
from .c2_remediation_root_finalizer import (
    FROZEN_PARTITIONS as _FINALIZER_FROZEN_PARTITIONS,
)
from .c2_remediation_root_finalizer import (
    FROZEN_UNIVERSE_SHA256 as _FINALIZER_FROZEN_UNIVERSE_SHA256,
)
from .models import ProvenanceError, canonical_json, open_trusted_directory
from .models import verify_trusted_directory


PREFLIGHT_SCHEMA_VERSION = "c2-remediation-preflight-v1"
FROZEN_UNIVERSE_SHA256 = (
    "51848466c6bf6bf400b58faf539953830349abab21438e527eaefd74103450df"
)
CHUNK_IDS = tuple(f"{number:03d}" for number in range(1, 14))
_ROOT_INVENTORY_EXCLUSIONS = frozenset({".pipeline_worktree/"})
_ALLOWED_ROOT_ROLES = frozenset(
    {
        "legacy",
        "candidate_v2",
        "fresh_remediation",
        "strict_path_derivative",
        "transparent_63_derivative",
        "final",
    }
)
_ROOT_ROLES_BY_CHUNK: Mapping[str, frozenset[str]] = {
    **{
        chunk_id: frozenset({"legacy", "fresh_remediation"})
        for chunk_id in CHUNK_IDS[:8]
    },
    "009": frozenset({"candidate_v2"}),
    "010": frozenset({"candidate_v2"}),
    "011": frozenset(
        {"legacy", "strict_path_derivative", "fresh_remediation"}
    ),
    "012": frozenset({"candidate_v2"}),
    "013": frozenset(
        {"legacy", "transparent_63_derivative", "fresh_remediation"}
    ),
}
_ACTION_BY_CHUNK: Mapping[str, str] = {
    **{
        chunk_id: "FRESH_REMEDIATION_REQUIRED"
        for chunk_id in CHUNK_IDS[:8]
    },
    "009": "CANDIDATE_V2_VALIDATION_ONLY",
    "010": "CANDIDATE_V2_VALIDATION_ONLY",
    "011": "STRICT_PATH_ONLY_DERIVATIVE_OR_REMEDIATION_REQUIRED",
    "012": "CANDIDATE_V2_VALIDATION_ONLY",
    "013": "TRANSPARENT_63_INPUT_DERIVATIVE_OR_FRESH_3X63_REQUIRED",
}


class C2RemediationPreflightError(ProvenanceError):
    """Raised when an input plan cannot enumerate the frozen C2 universe."""


@dataclass(frozen=True)
class _FrozenChunkBinding:
    """The internal compile-time binding for one exact frozen universe slice."""

    chunk_id: str
    input_total: int
    sha256: str
    first_global_ordinal: int
    last_global_ordinal: int


@dataclass(frozen=True)
class _FrozenBindings:
    """Internal fixed C2 universe bindings used by the production API."""

    frozen_universe_sha256: str
    chunks: tuple[_FrozenChunkBinding, ...]

    def chunk(self, chunk_id: str) -> _FrozenChunkBinding:
        for binding in self.chunks:
            if binding.chunk_id == chunk_id:
                return binding
        raise C2RemediationPreflightError(
            f"No frozen binding is available for chunk {chunk_id}"
        )


@dataclass(frozen=True)
class TestOnlyChunkBinding:
    """Synthetic chunk data accepted only by ``run_preflight_for_testing``."""

    chunk_id: str
    input_total: int
    sha256: str
    first_global_ordinal: int
    last_global_ordinal: int


@dataclass(frozen=True)
class TestOnlyPreflightBindings:
    """Synthetic bindings for tests; never accepted by ``run_preflight``."""

    frozen_universe_sha256: str
    chunks: tuple[TestOnlyChunkBinding, ...]

    def _to_internal_bindings(self) -> _FrozenBindings:
        return _FrozenBindings(
            frozen_universe_sha256=self.frozen_universe_sha256,
            chunks=tuple(
                _FrozenChunkBinding(
                    chunk_id=chunk.chunk_id,
                    input_total=chunk.input_total,
                    sha256=chunk.sha256,
                    first_global_ordinal=chunk.first_global_ordinal,
                    last_global_ordinal=chunk.last_global_ordinal,
                )
                for chunk in self.chunks
            ),
        )


@dataclass(frozen=True)
class _ArtifactSnapshot:
    sha256: str
    byte_count: int
    device: int
    inode: int
    mode: int
    owner_uid: int
    link_count: int
    mtime_ns: int
    ctime_ns: int

    def inventory_entry(self, path: str) -> dict[str, Any]:
        return {
            "path": path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }

    def identity(self) -> tuple[int, int, int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.mode,
            self.owner_uid,
            self.link_count,
            self.mtime_ns,
            self.ctime_ns,
        )


@dataclass(frozen=True)
class _RootInventory:
    root_device: int
    root_inode: int
    artifact_count: int
    total_bytes: int
    inventory_sha256: str
    artifacts: Mapping[str, _ArtifactSnapshot]

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "artifact_count": self.artifact_count,
            "total_bytes": self.total_bytes,
            "inventory_sha256": self.inventory_sha256,
            "excluded_non_evidence_paths": sorted(_ROOT_INVENTORY_EXCLUSIONS),
        }


def _default_bindings() -> _FrozenBindings:
    if _FINALIZER_FROZEN_UNIVERSE_SHA256 != FROZEN_UNIVERSE_SHA256:
        raise RuntimeError("C2 frozen universe bindings disagree across modules")
    bindings = tuple(
        _FrozenChunkBinding(
            chunk_id=chunk_id,
            input_total=partition.records,
            sha256=partition.source_sha256,
            first_global_ordinal=partition.start_index_1based,
            last_global_ordinal=partition.end_index_1based,
        )
        for chunk_id, partition in _FINALIZER_FROZEN_PARTITIONS.items()
    )
    if tuple(binding.chunk_id for binding in bindings) != CHUNK_IDS:
        raise RuntimeError("C2 frozen chunk binding table is incomplete or unordered")
    if (
        sum(binding.input_total for binding in bindings) != 2_463
        or any(binding.input_total != 200 for binding in bindings[:12])
        or bindings[-1].input_total != 63
    ):
        raise RuntimeError("C2 frozen chunk binding table has an invalid partition")
    return _FrozenBindings(
        frozen_universe_sha256=FROZEN_UNIVERSE_SHA256,
        chunks=bindings,
    )


DEFAULT_BINDINGS = _default_bindings()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_safe_metadata(metadata: os.stat_result, *, directory: bool) -> bool:
    if directory:
        return stat.S_ISDIR(metadata.st_mode)
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _validate_metadata(
    descriptor: int,
    metadata: os.stat_result,
    *,
    label: str,
    directory: bool,
) -> None:
    if not _is_safe_metadata(metadata, directory=directory):
        kind = "directory" if directory else "regular single-link file"
        raise C2RemediationPreflightError(f"{label} is not a {kind}")
    if not hasattr(os, "geteuid") or metadata.st_uid not in {0, os.geteuid()}:
        raise C2RemediationPreflightError(f"{label} has an unsafe owner")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise C2RemediationPreflightError(f"{label} is group/world writable")
    if _models._trusted_acl_allows_foreign_mutation(descriptor):
        raise C2RemediationPreflightError(f"{label} has a mutating ACL")


def _snapshot(metadata: os.stat_result, payload: bytes) -> _ArtifactSnapshot:
    return _ArtifactSnapshot(
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=stat.S_IMODE(metadata.st_mode),
        owner_uid=metadata.st_uid,
        link_count=metadata.st_nlink,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IMODE(left.st_mode),
        left.st_uid,
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IMODE(right.st_mode),
        right.st_uid,
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_flags() -> int:
    if not hasattr(os, "O_NONBLOCK"):
        raise C2RemediationPreflightError(
            "Safe nonblocking evidence-file reads are unsupported on this platform"
        )
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )


def _read_regular_file_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[bytes, _ArtifactSnapshot]:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise C2RemediationPreflightError(f"{label} is a symlink")
        if not _is_safe_metadata(before, directory=False):
            raise C2RemediationPreflightError(
                f"{label} is not a regular single-link file before open"
            )
        descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno is not None:
            raise C2RemediationPreflightError(f"Cannot safely open {label}") from exc
        raise
    try:
        opened = os.fstat(descriptor)
        if not _same_identity(before, opened):
            raise C2RemediationPreflightError(f"{label} changed while opening")
        _validate_metadata(
            descriptor,
            opened,
            label=label,
            directory=False,
        )
        payload = b"".join(
            iter(lambda: os.read(descriptor, 1024 * 1024), b"")
        )
        after = os.fstat(descriptor)
        if not _same_identity(opened, after):
            raise C2RemediationPreflightError(f"{label} changed while reading")
        return payload, _snapshot(after, payload)
    finally:
        os.close(descriptor)


def _read_external_regular_file(path: Path, *, label: str) -> tuple[bytes, _ArtifactSnapshot]:
    if not path.is_absolute():
        raise C2RemediationPreflightError(f"{label} must be an absolute path")
    if path.name in {"", ".", ".."}:
        raise C2RemediationPreflightError(f"{label} must name a file")
    try:
        parent = open_trusted_directory(path.parent)
    except ProvenanceError as exc:
        raise C2RemediationPreflightError(f"Cannot open {label} parent safely") from exc
    try:
        verify_trusted_directory(parent)
        payload, artifact = _read_regular_file_at(
            parent.descriptor,
            path.name,
            label=label,
        )
        verify_trusted_directory(parent)
        return payload, artifact
    except C2RemediationPreflightError:
        raise
    except ProvenanceError as exc:
        raise C2RemediationPreflightError(
            f"{label} changed while being read"
        ) from exc
    finally:
        parent.close()


def _inventory_root(
    root_path: Path,
) -> tuple[_RootInventory, bytes]:
    """Inventory an evidence root twice while retaining its trusted root FD."""

    if not root_path.is_absolute():
        raise C2RemediationPreflightError("Evidence root must be an absolute path")
    try:
        root = open_trusted_directory(root_path)
    except ProvenanceError as exc:
        raise C2RemediationPreflightError("Cannot safely open evidence root") from exc

    def scan_once() -> _RootInventory:
        verify_trusted_directory(root)
        root_metadata = os.fstat(root.descriptor)
        _validate_metadata(
            root.descriptor,
            root_metadata,
            label="Evidence root",
            directory=True,
        )
        artifacts: dict[str, _ArtifactSnapshot] = {}

        def descend(directory_fd: int, prefix: tuple[str, ...]) -> None:
            for name in sorted(os.listdir(directory_fd)):
                if not name or "/" in name or name in {".", ".."}:
                    raise C2RemediationPreflightError("Evidence root has an invalid name")
                relative = "/".join((*prefix, name))
                if not prefix and f"{name}/" in _ROOT_INVENTORY_EXCLUSIONS:
                    continue
                try:
                    before = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise C2RemediationPreflightError(
                        f"Cannot stat evidence artifact {relative}"
                    ) from exc
                if stat.S_ISLNK(before.st_mode):
                    raise C2RemediationPreflightError(
                        f"Evidence artifact {relative} is a symlink"
                    )
                if stat.S_ISDIR(before.st_mode):
                    try:
                        child_fd = os.open(
                            name,
                            _directory_flags(),
                            dir_fd=directory_fd,
                        )
                    except OSError as exc:
                        raise C2RemediationPreflightError(
                            f"Cannot open evidence directory {relative}"
                        ) from exc
                    try:
                        opened = os.fstat(child_fd)
                        if not _same_identity(before, opened):
                            raise C2RemediationPreflightError(
                                f"Evidence directory {relative} changed while opening"
                            )
                        _validate_metadata(
                            child_fd,
                            opened,
                            label=f"Evidence directory {relative}",
                            directory=True,
                        )
                        descend(child_fd, (*prefix, name))
                        after = os.fstat(child_fd)
                        if not _same_identity(opened, after):
                            raise C2RemediationPreflightError(
                                f"Evidence directory {relative} changed while reading"
                            )
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(before.st_mode):
                    raise C2RemediationPreflightError(
                        f"Evidence artifact {relative} has an unsupported type"
                    )
                _, artifact = _read_regular_file_at(
                    directory_fd,
                    name,
                    label=f"Evidence artifact {relative}",
                )
                artifacts[relative] = artifact

        root_fd = os.dup(root.descriptor)
        try:
            descend(root_fd, ())
        finally:
            os.close(root_fd)
        verify_trusted_directory(root)
        entries = [
            artifact.inventory_entry(path)
            for path, artifact in sorted(artifacts.items())
        ]
        return _RootInventory(
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            artifact_count=len(entries),
            total_bytes=sum(int(entry["byte_count"]) for entry in entries),
            inventory_sha256=_sha256_bytes(
                canonical_json(
                    {
                        "schema_version": "c2-remediation-root-inventory-v1",
                        "excluded_non_evidence_paths": sorted(
                            _ROOT_INVENTORY_EXCLUSIONS
                        ),
                        "artifacts": entries,
                    }
                ).encode("utf-8")
            ),
            artifacts=artifacts,
        )

    try:
        first = scan_once()
        accepted_payload, accepted_artifact = _read_regular_file_at(
            root.descriptor,
            "accepted.jsonl",
            label="Evidence root accepted.jsonl",
        )
        if first.artifacts.get("accepted.jsonl") != accepted_artifact:
            raise C2RemediationPreflightError(
                "accepted.jsonl changed outside the immutable root inventory"
            )
        second = scan_once()
        if (
            first.root_device != second.root_device
            or first.root_inode != second.root_inode
            or first.inventory_sha256 != second.inventory_sha256
            or first.artifacts != second.artifacts
        ):
            raise C2RemediationPreflightError(
                "Evidence root changed during immutable inventory"
            )
        return second, accepted_payload
    except C2RemediationPreflightError:
        raise
    except ProvenanceError as exc:
        raise C2RemediationPreflightError(
            "Evidence root changed during immutable inventory"
        ) from exc
    finally:
        root.close()


def _parse_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise C2RemediationPreflightError(f"{label} is not UTF-8") from exc
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise C2RemediationPreflightError(f"{label} has blank or padded rows")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise C2RemediationPreflightError(
                f"{label} row {index} is not JSON"
            ) from exc
        if not isinstance(value, dict):
            raise C2RemediationPreflightError(
                f"{label} row {index} is not an object"
            )
        records.append(value)
    return records


def _validate_ordered_dois(records: Sequence[Mapping[str, Any]], *, label: str) -> None:
    dois: list[str] = []
    for index, record in enumerate(records, start=1):
        try:
            doi = normalize_doi(record.get("doi"))
        except C2FullReplacementPolicyError as exc:
            raise C2RemediationPreflightError(
                f"{label} row {index} has an invalid DOI"
            ) from exc
        dois.append(doi)
    if len(dois) != len(set(dois)):
        raise C2RemediationPreflightError(f"{label} has duplicate DOI values")


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise C2RemediationPreflightError(f"{label} must be a nonempty path string")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise C2RemediationPreflightError(f"{label} must be absolute")
    return Path(os.path.abspath(os.fspath(raw)))


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    if frozenset(value) != expected:
        raise C2RemediationPreflightError(
            f"{label} fields must be exactly {sorted(expected)!r}"
        )


def _parse_plan(plan: Mapping[str, Any]) -> tuple[Path, dict[str, Mapping[str, Any]]]:
    if not isinstance(plan, Mapping):
        raise C2RemediationPreflightError("Preflight plan must be an object")
    _require_exact_keys(
        plan,
        frozenset({"schema_version", "frozen_universe_path", "chunks"}),
        label="Preflight plan",
    )
    if plan["schema_version"] != PREFLIGHT_SCHEMA_VERSION:
        raise C2RemediationPreflightError("Preflight plan schema version is invalid")
    frozen_universe = _absolute_path(
        plan["frozen_universe_path"],
        label="frozen_universe_path",
    )
    chunks_value = plan["chunks"]
    if not isinstance(chunks_value, list) or len(chunks_value) != len(CHUNK_IDS):
        raise C2RemediationPreflightError(
            "Preflight plan must name exactly the 13 frozen chunks"
        )
    chunks: dict[str, Mapping[str, Any]] = {}
    chunk_paths: set[Path] = set()
    root_paths: set[Path] = set()
    for index, item in enumerate(chunks_value, start=1):
        if not isinstance(item, Mapping):
            raise C2RemediationPreflightError(f"Plan chunk {index} is not an object")
        _require_exact_keys(
            item,
            frozenset({"chunk_id", "frozen_chunk_path", "root"}),
            label=f"Plan chunk {index}",
        )
        chunk_id = item["chunk_id"]
        if not isinstance(chunk_id, str) or chunk_id not in CHUNK_IDS:
            raise C2RemediationPreflightError(f"Plan chunk {index} has an invalid ID")
        if chunk_id in chunks:
            raise C2RemediationPreflightError(f"Plan duplicates chunk {chunk_id}")
        chunk_path = _absolute_path(
            item["frozen_chunk_path"],
            label=f"Plan chunk {chunk_id} frozen_chunk_path",
        )
        if chunk_path in chunk_paths:
            raise C2RemediationPreflightError(
                "Plan reuses a frozen chunk path; merging/repartitioning is forbidden"
            )
        root = item["root"]
        if not isinstance(root, Mapping):
            raise C2RemediationPreflightError(f"Plan chunk {chunk_id} root is invalid")
        _require_exact_keys(
            root,
            frozenset({"path", "role", "source_classification_claim"}),
            label=f"Plan chunk {chunk_id} root",
        )
        root_path = _absolute_path(
            root["path"],
            label=f"Plan chunk {chunk_id} root.path",
        )
        if root_path in root_paths:
            raise C2RemediationPreflightError(
                "Plan reuses an evidence root; merging chunks is forbidden"
            )
        if not isinstance(root["role"], str) or root["role"] not in _ALLOWED_ROOT_ROLES:
            raise C2RemediationPreflightError(
                f"Plan chunk {chunk_id} root role is invalid"
            )
        chunk_paths.add(chunk_path)
        root_paths.add(root_path)
        chunks[chunk_id] = item
    if tuple(sorted(chunks)) != CHUNK_IDS:
        raise C2RemediationPreflightError(
            "Preflight plan chunk IDs must be exactly 001 through 013"
        )
    return frozen_universe, chunks


def _stage_gates() -> dict[str, dict[str, str]]:
    """Check only integrated code gates; caller input cannot mark either gate passed."""

    source_extension = {
        "status": "BLOCKED",
        "reason": (
            "No approved integrated source-bearing strict-evidence validator is "
            "available; source classification claims are rejected."
        ),
    }
    try:
        policy = load_production_policy()
    except C2FullReplacementPolicyError as exc:
        stage_b = {
            "status": "BLOCKED",
            "reason": str(exc),
        }
    else:
        stage_b = (
            {
                "status": "BLOCKED",
                "reason": "Stage-B resolver returned a test-only policy",
            }
            if policy.is_test_only
            else {
                "status": "PASS",
                "reason": "A non-test Stage-B production policy is installed.",
            }
        )
    return {
        "source_extension": source_extension,
        "stage_b_policy_resource": stage_b,
    }


def _universe_result(
    frozen_universe_path: Path,
    *,
    bindings: _FrozenBindings,
) -> tuple[dict[str, Any], bytes | None, list[dict[str, Any]] | None, str | None]:
    expected = bindings.frozen_universe_sha256
    result: dict[str, Any] = {
        "path": str(frozen_universe_path),
        "expected_sha256": expected,
    }
    try:
        payload, artifact = _read_external_regular_file(
            frozen_universe_path,
            label="Frozen universe",
        )
        result["observed_sha256"] = artifact.sha256
        if artifact.sha256 != expected:
            raise C2RemediationPreflightError(
                "Frozen universe SHA-256 does not match the pinned value"
            )
        records = _parse_jsonl(payload, label="Frozen universe")
        if len(records) != 2_463:
            raise C2RemediationPreflightError(
                "Frozen universe does not contain exactly 2,463 inputs"
            )
        _validate_ordered_dois(records, label="Frozen universe")
    except C2RemediationPreflightError as exc:
        result["status"] = "REJECTED"
        result["reason"] = str(exc)
        return result, None, None, str(exc)
    result["status"] = "PASS"
    result["records"] = len(records)
    return result, payload, records, None


def _inspect_chunk(
    item: Mapping[str, Any],
    *,
    binding: _FrozenChunkBinding,
    universe_records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    chunk_id = binding.chunk_id
    root = item["root"]
    root_path = _absolute_path(root["path"], label=f"Plan chunk {chunk_id} root.path")
    frozen_chunk_path = _absolute_path(
        item["frozen_chunk_path"],
        label=f"Plan chunk {chunk_id} frozen_chunk_path",
    )
    violations: list[dict[str, str]] = []
    frozen_result: dict[str, Any] = {
        "path": str(frozen_chunk_path),
        "expected_sha256": binding.sha256,
        "expected_input_total": binding.input_total,
        "first_global_ordinal": binding.first_global_ordinal,
        "last_global_ordinal": binding.last_global_ordinal,
    }
    frozen_payload: bytes | None = None
    frozen_records: list[dict[str, Any]] | None = None
    try:
        frozen_payload, artifact = _read_external_regular_file(
            frozen_chunk_path,
            label=f"Frozen chunk {chunk_id}",
        )
        frozen_result["observed_sha256"] = artifact.sha256
        if artifact.sha256 != binding.sha256:
            raise C2RemediationPreflightError(
                "Frozen chunk SHA-256 does not match its compiled binding"
            )
        frozen_records = _parse_jsonl(
            frozen_payload,
            label=f"Frozen chunk {chunk_id}",
        )
        if len(frozen_records) != binding.input_total:
            raise C2RemediationPreflightError(
                f"Frozen chunk {chunk_id} does not contain exactly "
                f"{binding.input_total} inputs"
            )
        _validate_ordered_dois(frozen_records, label=f"Frozen chunk {chunk_id}")
        if universe_records is not None:
            start = binding.first_global_ordinal - 1
            end = binding.last_global_ordinal
            if frozen_records != list(universe_records[start:end]):
                raise C2RemediationPreflightError(
                    "Frozen chunk is not the exact ordered frozen-universe slice"
                )
    except C2RemediationPreflightError as exc:
        frozen_result["status"] = "REJECTED"
        frozen_result["reason"] = str(exc)
        violations.append(
            {
                "code": "FROZEN_CHUNK_INVALID",
                "reason": str(exc),
            }
        )
    else:
        frozen_result["status"] = "PASS"
        frozen_result["records"] = len(frozen_records)

    root_result: dict[str, Any] = {
        "path": str(root_path),
        "role": root["role"],
    }
    try:
        inventory, accepted_payload = _inventory_root(root_path)
        root_result["inventory"] = inventory.to_report_dict()
        if frozen_payload is None:
            raise C2RemediationPreflightError(
                "Cannot bind accepted.jsonl until the frozen chunk is valid"
            )
        if accepted_payload != frozen_payload:
            raise C2RemediationPreflightError(
                "accepted.jsonl is not the exact frozen chunk bytes; padding, "
                "repartitioning, and merging are forbidden"
            )
        accepted_records = _parse_jsonl(
            accepted_payload,
            label=f"Evidence root {chunk_id} accepted.jsonl",
        )
        if len(accepted_records) != binding.input_total:
            raise C2RemediationPreflightError(
                f"accepted.jsonl does not contain exactly {binding.input_total} inputs"
            )
        _validate_ordered_dois(
            accepted_records,
            label=f"Evidence root {chunk_id} accepted.jsonl",
        )
    except C2RemediationPreflightError as exc:
        root_result["status"] = "REJECTED"
        root_result["reason"] = str(exc)
        violations.append({"code": "ROOT_INPUT_UNSAFE_OR_UNBOUND", "reason": str(exc)})
    else:
        root_result["status"] = "PASS"
        root_result["accepted_input_sha256"] = _sha256_bytes(accepted_payload)

    role = root["role"]
    if role == "final":
        violations.append(
            {
                "code": "LEGACY_OR_CANDIDATE_ROOT_PASSED_AS_FINAL",
                "reason": (
                    "This preflight never accepts a root as final evidence; "
                    "publication and admission remain prohibited."
                ),
            }
        )
    elif role not in _ROOT_ROLES_BY_CHUNK[chunk_id]:
        violations.append(
            {
                "code": "ROOT_ROLE_NOT_ALLOWED_FOR_CHUNK",
                "reason": (
                    f"Chunk {chunk_id} cannot use root role {role!r}; expected one "
                    f"of {sorted(_ROOT_ROLES_BY_CHUNK[chunk_id])!r}."
                ),
            }
        )
    if root["source_classification_claim"] is not None:
        violations.append(
            {
                "code": "SOURCE_CLASSIFICATION_CLAIM_WITHOUT_STRICT_EVIDENCE",
                "reason": (
                    "No approved integrated source-bearing strict-evidence validator "
                    "exists, so this preflight rejects every source-classification "
                    "claim rather than accepting asserted P/case/panel evidence."
                ),
            }
        )

    return {
        "chunk_id": chunk_id,
        "expected_input_total": binding.input_total,
        "required_action": _ACTION_BY_CHUNK[chunk_id],
        "frozen_chunk": frozen_result,
        "root": root_result,
        "input_inventory_status": "PASS" if not violations else "REJECTED",
        "evidence_status": (
            _ACTION_BY_CHUNK[chunk_id] if not violations else "REJECTED"
        ),
        "violations": violations,
    }


def _run_preflight(
    plan: Mapping[str, Any],
    *,
    bindings: _FrozenBindings,
) -> dict[str, Any]:
    """Build a deterministic report from an internal frozen binding."""

    frozen_universe_path, chunks = _parse_plan(plan)
    universe, _, universe_records, _ = _universe_result(
        frozen_universe_path,
        bindings=bindings,
    )
    chunk_reports = [
        _inspect_chunk(
            chunks[chunk_id],
            binding=bindings.chunk(chunk_id),
            universe_records=universe_records,
        )
        for chunk_id in CHUNK_IDS
    ]
    gates = _stage_gates()
    inputs_pass = universe["status"] == "PASS" and all(
        item["input_inventory_status"] == "PASS" for item in chunk_reports
    )
    gates_pass = all(gate["status"] == "PASS" for gate in gates.values())
    report: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "mode": "READ_ONLY_NO_FINALIZER_NO_PUBLICATION_NO_RERUN",
        "frozen_universe": universe,
        "chunks": chunk_reports,
        "gates_before_real_execution": gates,
        "inputs_pass": inputs_pass,
        "execution_authorized": False,
        "overall_status": (
            "PREFLIGHT_INPUTS_READY_SEPARATE_AUTHORIZATION_REQUIRED"
            if inputs_pass and gates_pass
            else "BLOCKED"
        ),
        "prohibitions": [
            "No remediation root is published by this preflight.",
            "No source-bearing finalizer is invoked by this preflight.",
            "No legacy or candidate root is treated as final evidence.",
            "No source classification, P label, stratum, admission, or rerun is created.",
        ],
    }
    report["report_sha256"] = _sha256_bytes(
        canonical_json(report).encode("utf-8")
    )
    return report


def _build_production_runner(
    canonical_runner: Callable[..., dict[str, Any]],
    compiled_bindings: _FrozenBindings,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Capture canonical code and bindings outside the mutable module namespace."""

    def run_preflight(plan: Mapping[str, Any]) -> dict[str, Any]:
        """Run production preflight with only the compiled frozen C2 bindings.

        The public API deliberately has no binding argument.  Callers cannot
        replace the 2,463 DOI universe, the 12×200+63 partition, or any chunk
        SHA-256.
        """

        return canonical_runner(plan, bindings=compiled_bindings)

    return run_preflight


run_preflight = _build_production_runner(_run_preflight, DEFAULT_BINDINGS)


def run_preflight_for_testing(
    plan: Mapping[str, Any],
    *,
    test_bindings: TestOnlyPreflightBindings,
) -> dict[str, Any]:
    """Exercise synthetic fixtures; this test-only helper is not a production API."""

    return _run_preflight(
        plan,
        bindings=test_bindings._to_internal_bindings(),
    )


def _load_plan_from_path(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2RemediationPreflightError("Cannot parse preflight plan JSON") from exc
    if not isinstance(value, Mapping):
        raise C2RemediationPreflightError("Preflight plan JSON must be an object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.c2_remediation_preflight",
        description=(
            "Read-only C2 remediation inventory preflight; it cannot publish, "
            "finalize, classify, admit, or rerun evidence."
        ),
    )
    parser.add_argument(
        "plan",
        type=Path,
        help=(
            "Exact JSON plan with one explicit frozen chunk and one explicit root "
            "for every ID 001 through 013; globs are not supported."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_preflight(_load_plan_from_path(args.plan))
    except C2RemediationPreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["overall_status"] != "BLOCKED" else 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed finalization of a fresh, separately acquired C2 remediation root.

This module deliberately consumes an immutable raw-evidence root and publishes a
new, previously absent remediation root.  It never runs acquisition, calls a
model, or mutates the raw-evidence root.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .c2_m1_trust_boundary import (
    require_owner_authorized_c2_execution as require_external_m1_trust_lock,
)
from .models import (
    ProvenanceError,
    SecureOutputTarget,
    canonical_json,
    normalize_trusted_output_path,
    open_secure_output_target,
    sha256_file,
)
from .c2_source_bearing_extension import (
    SourceBearingExtensionError,
    build_source_bearing_extension,
    validate_source_bearing_extension,
    validate_source_evidence_descriptor_v2,
)


class C2RemediationError(ProvenanceError):
    """Raised when fresh remediation evidence cannot be sealed safely."""


ATTEMPTS = ("initial", "retry1", "retry2")
ATTEMPT_INDEX = {attempt: index for index, attempt in enumerate(ATTEMPTS)}
FROZEN_UNIVERSE_SHA256 = (
    "51848466c6bf6bf400b58faf539953830349abab21438e527eaefd74103450df"
)
FROZEN_FREEZE_SUMMARY_SHA256 = (
    "64684e64b6a4e54c508685fb745e303dfca84e6d499e96568f40c80c5dff368b"
)
FROZEN_FREEZE_SUMMARY_HASH = (
    "6cdb5f8398cc70960d108c6319c93c2121327ce757af405bc44bca687a71056d"
)
FROZEN_CODE_COMMIT = "ca98442b9e805110089b03083cb240e19b58d4a2"
STATUS_VALUES = frozenset(
    {"downloaded", "no-source-data", "no-figures", "fetch-error"}
)
SKIPPED_STATUS_VALUES = STATUS_VALUES - {"downloaded"}
SECRET_PATTERNS = (
    (
        "credential-assignment",
        re.compile(
            rb"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)"
            rb"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}"
        ),
    ),
    ("aws-access-key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("github-token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")),
)


@dataclass(frozen=True)
class FrozenPartition:
    """One statically bound frozen chunk."""

    chunk_id: str
    records: int
    source_sha256: str
    start_index_1based: int
    end_index_1based: int


FROZEN_PARTITIONS: Mapping[str, FrozenPartition] = {
    "001": FrozenPartition(
        "001",
        200,
        "e6b5d229d94fab7768a95482bbe79b895808ab925e950a0dd4352b206f32faa0",
        1,
        200,
    ),
    "002": FrozenPartition(
        "002",
        200,
        "fa4c8a566bc31c685f18ae41bd4f1706da7fed8fa22c7bb003737a2235fe3cc2",
        201,
        400,
    ),
    "003": FrozenPartition(
        "003",
        200,
        "e9551e7f78de013a76836aaaa9d689f37158292f2ca23328b73a2c1a0af9c71f",
        401,
        600,
    ),
    "004": FrozenPartition(
        "004",
        200,
        "fc6c116c3ce6b1d9f7b84a34eb48879a1295d18414977e386743810834fbff25",
        601,
        800,
    ),
    "005": FrozenPartition(
        "005",
        200,
        "527e059cf0542cec972186d1d116683b37034019237035b2ec666f7b6f48d2c0",
        801,
        1000,
    ),
    "006": FrozenPartition(
        "006",
        200,
        "8d655f46993dffcce32d56b44562f29a1589ce9242c8e2390543699e97125c95",
        1001,
        1200,
    ),
    "007": FrozenPartition(
        "007",
        200,
        "6ce5ad09e06bcef088dd4ccfaff2f23a7f3ebe607046b816e52e2821e6518517",
        1201,
        1400,
    ),
    "008": FrozenPartition(
        "008",
        200,
        "19504a671479ebaee75d0743a5961f656668a487038ab41a4d348168b52c8d3d",
        1401,
        1600,
    ),
    "009": FrozenPartition(
        "009",
        200,
        "d56a075c9323a3154c511674c3828a607ac86b808707aadd01f273b9faff174d",
        1601,
        1800,
    ),
    "010": FrozenPartition(
        "010",
        200,
        "ccdc184da2b9476e21d2d9c40a46b315af8d5e3f757fcece8409bfeb913bc9d0",
        1801,
        2000,
    ),
    "011": FrozenPartition(
        "011",
        200,
        "b1f8fa222278391decd6fec78c9284da1f6ee1660fd082d664e68eeef7593bc7",
        2001,
        2200,
    ),
    "012": FrozenPartition(
        "012",
        200,
        "41ba910500ef58eebba453ffeb5d4203549846694d9a985ce64ff273468f00d2",
        2201,
        2400,
    ),
    "013": FrozenPartition(
        "013",
        63,
        "7454ec45409115cf81b085d8cb6821d6d05d811e4ed97365a53238464ab83d03",
        2401,
        2463,
    ),
}

OLD_ROOT_PRESERVATION: Mapping[str, Mapping[str, Any]] = {
    "001": {
        "root_name": "ccby_sr_npj_chunk001_clean_ca98442",
        "artifact_count": 262,
        "total_bytes": 6544263,
        "inventory_hash": "ba9e1facda6f77ec881d72c76e8db9f97b96ed7690434b6513dfd086ab1ab770",
        "sealed_report_sha256": "c6511ccd76fad8f41cac2d297ddbd6710fc861e7b0615aa46dfd8b7e9fdaa5cc",
    },
    "002": {
        "root_name": "ccby_sr_npj_chunk002_clean_ca98442",
        "artifact_count": 245,
        "total_bytes": 2408846,
        "inventory_hash": "3af61958b6c4752d104589f3b72f0ca72907e6296612f2e596ae47d95a9995fe",
        "sealed_report_sha256": "96aacc7dab4341a289b02ca244001980d221d82cd252f47c6cb3ea6860dc93cf",
    },
    "003": {
        "root_name": "ccby_sr_npj_chunk003_clean_ca98442",
        "artifact_count": 245,
        "total_bytes": 2204574,
        "inventory_hash": "b72b2fae93f6efdb3f3145b03fa81e719eee6ed45587b823ab26ef76080ddbdf",
        "sealed_report_sha256": "7e4b03a97127ae4fe03c518c6fee5581e89a19162b51de9d8b5aa0e19cc8036c",
    },
    "004": {
        "root_name": "ccby_sr_npj_chunk004_clean_ca98442",
        "artifact_count": 245,
        "total_bytes": 2105793,
        "inventory_hash": "8dd172624357e494838e3d1f4ae57caf2862b2429fdb5282df1a8cf001c5a4fa",
        "sealed_report_sha256": "91418323cb0f5e6e43f49ab9acb258d5056068621b1ec46dec71c656702a733c",
    },
    "005": {
        "root_name": "ccby_sr_npj_chunk005_clean_ca98442",
        "artifact_count": 244,
        "total_bytes": 2359578,
        "inventory_hash": "1dae5c33652499a50322f45566c588619125c29767a12f9eea818fc51d963c35",
        "sealed_report_sha256": "d2b4859b539849a11d4e723c50e028fc4f83920755f555a144fed88e3171a68e",
    },
    "006": {
        "root_name": "ccby_sr_npj_chunk006_clean_ca98442",
        "artifact_count": 244,
        "total_bytes": 2333002,
        "inventory_hash": "d312e8ee66977c7638ba0e1f5bc03c36ffb550fd7a36afe04505962ac2fc3f30",
        "sealed_report_sha256": "64bb31ba387fba2f694ef98cc2fae106f375e511c57a16024efd650b14622479",
    },
    "007": {
        "root_name": "ccby_sr_npj_chunk007_clean_ca98442",
        "artifact_count": 244,
        "total_bytes": 2393370,
        "inventory_hash": "b8892e422afe72e82ee5e6d6d6dce3e86c566d387db1deda7349f15f0e50f1ea",
        "sealed_report_sha256": "55bc3734bd8428c1898d69a648378d52d7dd87823d3da8b12c8688d47d6ca1e7",
    },
    "008": {
        "root_name": "ccby_sr_npj_chunk008_clean_ca98442",
        "artifact_count": 244,
        "total_bytes": 2432786,
        "inventory_hash": "0a8d873e8ca4d8bbeff73e3a6b03a34a1c9f7159b2747196ad1be8042da769b4",
        "sealed_report_sha256": "2267280e6ac6887267a696e541f61a516435f78c29f636b283156999bdf7d1b5",
    },
    "011": {
        "root_name": "ccby_sr_npj_chunk011_rerun_clean_ca98442",
        "artifact_count": 254,
        "total_bytes": 2833857,
        "inventory_hash": "4487ca1eca5d702bf4894d09a4f703e90591e3cd92a58555ba12af52976f1c3d",
        "sealed_report_sha256": "091b2f646d0194a6751824b9d54756f75dab92034d1a55f580a26d447084591b",
    },
    "013": {
        "root_name": "ccby_sr_npj_chunk013_clean_ca98442",
        "artifact_count": 107,
        "total_bytes": 797890,
        "inventory_hash": "ee31eef8834fb6187d1cbfaabb07b553bff0e1a65af632f2cf9507d4496d7254",
        "sealed_report_sha256": "c93d456d5f62a601c4a030f167983b44f78b0cf4a73e44bc13e12e30d8ff7ec6",
    },
}

RETAINED_CANDIDATE_EXCLUSIONS = (
    "ccby_sr_npj_chunk009_rerun2_clean_ca98442",
    "ccby_sr_npj_chunk010_rerun2_clean_ca98442",
    "ccby_sr_npj_chunk012_rerun2_clean_ca98442",
)
SUPPORTED_REMEDIATION_CHUNKS = frozenset(OLD_ROOT_PRESERVATION)


def expected_target_root_name(chunk_id: str) -> str:
    """Return the single non-colliding rerun3 name bound to a frozen chunk."""

    _require(
        chunk_id in SUPPORTED_REMEDIATION_CHUNKS,
        "chunk is not authorized for fresh remediation",
    )
    return f"ccby_sr_npj_chunk{chunk_id}_rerun3_clean_ca98442"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2RemediationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _closed_status_counts(statuses: Iterable[str]) -> dict[str, int]:
    observed = Counter(statuses)
    return {status: int(observed[status]) for status in sorted(STATUS_VALUES)}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    ignored = set(keys)
    return {key: item for key, item in value.items() if key not in ignored}


def _read_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2RemediationError(f"{label} is not UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _read_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise C2RemediationError(f"{label} is not UTF-8") from exc
    _require(all(line.strip() for line in lines), f"{label} contains a blank line")
    try:
        records = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise C2RemediationError(f"{label} is not JSONL") from exc
    _require(
        all(isinstance(record, dict) for record in records),
        f"{label} must contain JSON objects",
    )
    return records


def _ensure_relative(relative: str | Path, label: str) -> Path:
    path = Path(relative)
    _require(not path.is_absolute(), f"{label} must be relative")
    _require(".." not in path.parts and path.parts, f"{label} escapes its root")
    return path


def _verify_trusted_existing_root(root: Path, label: str) -> Path:
    require_external_m1_trust_lock()
    _require(root.is_absolute(), f"{label} must be absolute")
    normalized = normalize_trusted_output_path(root)
    _require(normalized.is_dir() and not normalized.is_symlink(), f"{label} is unsafe")
    probe = open_secure_output_target(
        normalized / ".c2_remediation_probe",
        normalized_path=True,
        require_trusted_parent=True,
    )
    probe.close()
    return normalized


@dataclass(frozen=True)
class _RawRead:
    payload: bytes
    sha256: str
    metadata: Mapping[str, int]


@dataclass(frozen=True)
class _SourceEvidence:
    """Validated source-bearing evidence linked to one terminal provenance file."""

    descriptor_path: str
    descriptor_sha256: str
    descriptor_bytes: int
    schema_version: str
    descriptor: Mapping[str, Any]
    source_paths: tuple[Mapping[str, Any], ...]


class _RawRootReader:
    """Single-read, descriptor-relative reader for immutable raw evidence."""

    def __init__(self, root: Path) -> None:
        self._root_fd = -1
        require_external_m1_trust_lock()
        self.root = _verify_trusted_existing_root(root, "raw root")
        probe = open_secure_output_target(
            self.root / ".c2_remediation_probe",
            normalized_path=True,
            require_trusted_parent=True,
        )
        try:
            self._root_fd = os.dup(probe.parent_fd)
        finally:
            probe.close()
        self._seen: set[str] = set()
        self.reads: dict[str, _RawRead] = {}

    def close(self) -> None:
        if self._root_fd != -1:
            os.close(self._root_fd)
            self._root_fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    def __enter__(self) -> "_RawRootReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _metadata(file_stat: os.stat_result) -> dict[str, int]:
        return {
            "device": file_stat.st_dev,
            "inode": file_stat.st_ino,
            "mode": stat.S_IMODE(file_stat.st_mode),
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
            "ctime_ns": file_stat.st_ctime_ns,
        }

    def read(self, relative: str | Path, label: str) -> bytes:
        require_external_m1_trust_lock()
        path = _ensure_relative(relative, label)
        relative_text = path.as_posix()
        _require(relative_text not in self._seen, f"{label} was opened more than once")
        current_fd = os.dup(self._root_fd)
        leaf_fd = -1
        close_on_exec = getattr(os, "O_CLOEXEC", 0)
        try:
            for component in path.parts[:-1]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec,
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
                _require(
                    stat.S_ISDIR(os.fstat(current_fd).st_mode),
                    f"{label} parent is not a directory",
                )
            leaf_fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | close_on_exec,
                dir_fd=current_fd,
            )
            before = os.fstat(leaf_fd)
            _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
            chunks: list[bytes] = []
            while True:
                block = os.read(leaf_fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            payload = b"".join(chunks)
            after = os.fstat(leaf_fd)
            _require(
                self._metadata(before) == self._metadata(after),
                f"{label} changed while being read",
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise C2RemediationError(f"{label} is a symlink") from exc
            raise C2RemediationError(f"cannot securely read {label}") from exc
        finally:
            if leaf_fd != -1:
                os.close(leaf_fd)
            os.close(current_fd)
        self._seen.add(relative_text)
        self.reads[relative_text] = _RawRead(
            payload=payload,
            sha256=_sha256_bytes(payload),
            metadata=self._metadata(before),
        )
        return payload

    def files_under(self, relative: str | Path, label: str) -> tuple[str, ...]:
        """Return a no-follow, descriptor-relative recursive regular-file listing."""

        require_external_m1_trust_lock()
        path = _ensure_relative(relative, label)
        current_fd = os.dup(self._root_fd)
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )

        def descend(directory_fd: int, prefix: tuple[str, ...]) -> Iterable[str]:
            for name in sorted(os.listdir(directory_fd)):
                file_stat = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                relative_name = "/".join((*prefix, name))
                _require(
                    not stat.S_ISLNK(file_stat.st_mode),
                    f"{label} contains a symlink: {relative_name}",
                )
                if stat.S_ISDIR(file_stat.st_mode):
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                    try:
                        yield from descend(child_fd, (*prefix, name))
                    finally:
                        os.close(child_fd)
                    continue
                _require(
                    stat.S_ISREG(file_stat.st_mode),
                    f"{label} contains a non-regular artifact: {relative_name}",
                )
                yield relative_name

        try:
            for component in path.parts:
                next_fd = os.open(component, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return tuple(descend(current_fd, tuple(path.parts)))
        except C2RemediationError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise C2RemediationError(f"{label} contains a symlink") from exc
            raise C2RemediationError(f"cannot securely list {label}") from exc
        finally:
            os.close(current_fd)


_DARWIN_RENAME_EXCL = 0x00000004
_LINUX_RENAME_NOREPLACE = 0x00000001
_STAGING_CREATE_ATTEMPTS = 128


def _publication_platform() -> str:
    return sys.platform


def _open_private_staging_parent(target_root: Path) -> SecureOutputTarget:
    """Open the pre-existing, descriptor-validated parent used for staging."""

    require_external_m1_trust_lock()
    normalized_target = normalize_trusted_output_path(target_root)
    target = open_secure_output_target(
        normalized_target,
        normalized_path=True,
        require_trusted_parent=True,
    )
    try:
        _require(
            target.trusted_parent and target.parent_fd != -1,
            "private staging parent was not descriptor-validated",
        )
        _require(
            hasattr(os, "geteuid"),
            "private staging parent requires an effective uid",
        )
        metadata = os.fstat(target.parent_fd)
        _require(
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid in {0, os.geteuid()}
            and not (metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)),
            "private staging parent is unsafe",
        )
        # require_trusted_parent=True also verifies no-follow ancestry and ACL safety.
        return target
    except Exception:
        target.close()
        raise


def _publish_staging_directory(
    *,
    parent_fd: int,
    staging_name: str,
    target_name: str,
) -> None:
    """Atomically publish a staging directory without replacing any target."""

    require_external_m1_trust_lock()
    _require(
        staging_name
        and target_name
        and "/" not in staging_name
        and "/" not in target_name
        and "\x00" not in staging_name
        and "\x00" not in target_name,
        "atomic publication names are invalid",
    )
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        platform = _publication_platform()
        if platform == "darwin":
            rename_no_replace = libc.renameatx_np
            rename_no_replace.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            rename_no_replace.restype = ctypes.c_int
            ctypes.set_errno(0)
            result = rename_no_replace(
                parent_fd,
                os.fsencode(staging_name),
                parent_fd,
                os.fsencode(target_name),
                _DARWIN_RENAME_EXCL,
            )
        elif platform.startswith("linux"):
            rename_no_replace = libc.renameat2
            rename_no_replace.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            rename_no_replace.restype = ctypes.c_int
            ctypes.set_errno(0)
            result = rename_no_replace(
                parent_fd,
                os.fsencode(staging_name),
                parent_fd,
                os.fsencode(target_name),
                _LINUX_RENAME_NOREPLACE,
            )
        else:
            raise C2RemediationError(
                "atomic no-replace directory publication is unsupported on this platform"
            )
    except AttributeError as exc:
        raise C2RemediationError(
            "atomic no-replace directory publication is unavailable on this platform"
        ) from exc

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise C2RemediationError(
            "canonical target appeared before atomic staging publication"
        )
    raise C2RemediationError(
        "atomic no-replace directory publication failed"
    ) from OSError(error_number, os.strerror(error_number))


class _SecureTargetRoot:
    """Descriptor-anchored staging root with an atomic no-replace publication."""

    def __init__(
        self,
        *,
        canonical_path: Path,
        parent_fd: int,
        target_name: str,
        staging_name: str,
        root_fd: int,
        staging_identity: tuple[int, int],
    ) -> None:
        require_external_m1_trust_lock()
        self.path = canonical_path
        self._parent_fd = parent_fd
        self._target_name = target_name
        self._staging_name = staging_name
        self._root_fd = root_fd
        self._parent_identity = os.fstat(parent_fd)
        self._root_identity = os.fstat(root_fd)
        self._staging_identity = staging_identity
        self._published = False
        _require(
            stat.S_ISDIR(self._root_identity.st_mode)
            and (self._root_identity.st_dev, self._root_identity.st_ino)
            == self._staging_identity,
            "created private staging root identity is invalid",
        )

    @property
    def name(self) -> str:
        return self._target_name

    @property
    def staging_name(self) -> str:
        return self._staging_name

    @property
    def published(self) -> bool:
        return self._published

    def close(self) -> None:
        for attribute in ("_root_fd", "_parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor != -1:
                os.close(descriptor)
                setattr(self, attribute, -1)

    def _open_parent(
        self,
        relative: str | Path,
        *,
        create: bool,
    ) -> tuple[int, str]:
        require_external_m1_trust_lock()
        if create:
            _require(
                not self._published,
                "refusing to create or modify artifacts after publication",
            )
        path = _ensure_relative(relative, f"target {relative}")
        current_fd = os.dup(self._root_fd)
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            for component in path.parts[:-1]:
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    except FileExistsError:
                        existing = os.stat(
                            component,
                            dir_fd=current_fd,
                            follow_symlinks=False,
                        )
                        _require(
                            not stat.S_ISLNK(existing.st_mode),
                            f"target path contains a symlink: {relative}",
                        )
                next_fd = os.open(component, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
                _require(
                    stat.S_ISDIR(os.fstat(current_fd).st_mode),
                    f"target parent is not a directory: {component}",
                )
            return current_fd, path.name
        except C2RemediationError:
            os.close(current_fd)
            raise
        except OSError as exc:
            os.close(current_fd)
            if exc.errno == errno.ELOOP:
                raise C2RemediationError(
                    f"target path contains a symlink: {relative}"
                ) from exc
            raise C2RemediationError(
                f"cannot securely open target parent: {relative}"
            ) from exc

    @staticmethod
    def _read_regular_at(parent_fd: int, name: str, label: str) -> bytes:
        require_external_m1_trust_lock()
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            before = os.fstat(descriptor)
            _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
            _require(
                (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                == (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ),
                f"{label} changed while being read",
            )
            return b"".join(chunks)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise C2RemediationError(f"{label} is a symlink") from exc
            raise C2RemediationError(f"cannot securely read {label}") from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)

    def write_bytes(self, relative: str, payload: bytes) -> str:
        require_external_m1_trust_lock()
        _require(not self._published, "refusing to write artifacts after publication")
        parent_fd, leaf_name = self._open_parent(relative, create=True)
        descriptor = -1
        try:
            descriptor = os.open(
                leaf_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return _ensure_relative(relative, f"target {relative}").as_posix()
        except FileExistsError as exc:
            raise C2RemediationError(
                f"refusing to overwrite generated artifact {relative}"
            ) from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise C2RemediationError(
                    f"target path contains a symlink: {relative}"
                ) from exc
            raise C2RemediationError(
                f"cannot securely write generated artifact {relative}"
            ) from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)
            os.close(parent_fd)

    def read_bytes(self, relative: str) -> bytes:
        require_external_m1_trust_lock()
        parent_fd, leaf_name = self._open_parent(relative, create=False)
        try:
            return self._read_regular_at(parent_fd, leaf_name, f"target {relative}")
        finally:
            os.close(parent_fd)

    def sha256(self, relative: str) -> str:
        require_external_m1_trust_lock()
        return _sha256_bytes(self.read_bytes(relative))

    def mkdir(self, relative: str) -> None:
        require_external_m1_trust_lock()
        _require(not self._published, "refusing to create artifacts after publication")
        parent_fd, leaf_name = self._open_parent(relative, create=True)
        try:
            try:
                os.mkdir(leaf_name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise C2RemediationError(
                    f"refusing to overwrite generated directory {relative}"
                ) from exc
            descriptor = os.open(
                leaf_name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            try:
                _require(
                    stat.S_ISDIR(os.fstat(descriptor).st_mode),
                    f"generated target directory is invalid: {relative}",
                )
            finally:
                os.close(descriptor)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise C2RemediationError(
                    f"target path contains a symlink: {relative}"
                ) from exc
            raise C2RemediationError(
                f"cannot securely create target directory {relative}"
            ) from exc
        finally:
            os.close(parent_fd)

    def files(self, excludes: set[str] | None = None) -> Iterable[tuple[str, int, bytes]]:
        require_external_m1_trust_lock()
        excluded = excludes or set()

        def is_excluded(relative: str) -> bool:
            return any(
                relative == item.rstrip("/") or relative.startswith(item)
                for item in excluded
            )

        def descend(directory_fd: int, prefix: tuple[str, ...]) -> Iterable[tuple[str, int, bytes]]:
            for name in sorted(os.listdir(directory_fd)):
                relative = "/".join((*prefix, name))
                file_stat = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                _require(
                    not stat.S_ISLNK(file_stat.st_mode),
                    f"target artifact symlink {relative}",
                )
                if stat.S_ISDIR(file_stat.st_mode):
                    if is_excluded(f"{relative}/"):
                        continue
                    child_fd = os.open(
                        name,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        yield from descend(child_fd, (*prefix, name))
                    finally:
                        os.close(child_fd)
                    continue
                _require(
                    stat.S_ISREG(file_stat.st_mode),
                    f"unsupported target artifact {relative}",
                )
                if is_excluded(relative):
                    continue
                yield (
                    relative,
                    file_stat.st_size,
                    self._read_regular_at(
                        directory_fd,
                        name,
                        f"target artifact {relative}",
                    ),
                )

        root_fd = os.dup(self._root_fd)
        try:
            yield from descend(root_fd, ())
        finally:
            os.close(root_fd)

    def _verify_anchored_entry(self, name: str, label: str) -> None:
        require_external_m1_trust_lock()
        _require(self._root_fd != -1 and self._parent_fd != -1, "target is closed")
        anchored_root = os.fstat(self._root_fd)
        try:
            anchored_leaf = os.stat(
                name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise C2RemediationError(f"{label} directory identity changed") from exc
        _require(
            stat.S_ISDIR(anchored_leaf.st_mode)
            and (anchored_leaf.st_dev, anchored_leaf.st_ino)
            == (self._root_identity.st_dev, self._root_identity.st_ino)
            and (anchored_root.st_dev, anchored_root.st_ino)
            == (self._root_identity.st_dev, self._root_identity.st_ino),
            f"anchored {label} directory identity changed",
        )

    def _verify_visible_parent(self) -> None:
        require_external_m1_trust_lock()
        probe = open_secure_output_target(
            self.path,
            normalized_path=True,
            require_trusted_parent=True,
        )
        try:
            visible_parent = os.fstat(probe.parent_fd)
            _require(
                (visible_parent.st_dev, visible_parent.st_ino)
                == (self._parent_identity.st_dev, self._parent_identity.st_ino),
                "visible target parent directory identity changed",
            )
        finally:
            probe.close()

    def verify_staging_identity(self) -> None:
        require_external_m1_trust_lock()
        _require(not self._published, "private staging root is already published")
        self._verify_anchored_entry(self._staging_name, "private staging")
        self._verify_visible_parent()

    def verify_published_identity(self) -> None:
        require_external_m1_trust_lock()
        _require(self._published, "private staging root has not been published")
        self._verify_anchored_entry(self._target_name, "canonical target")
        self._verify_visible_parent()
        probe = open_secure_output_target(
            self.path,
            normalized_path=True,
            require_trusted_parent=True,
        )
        try:
            visible_leaf = os.stat(
                probe.leaf_name,
                dir_fd=probe.parent_fd,
                follow_symlinks=False,
            )
            _require(
                stat.S_ISDIR(visible_leaf.st_mode)
                and (visible_leaf.st_dev, visible_leaf.st_ino)
                == (self._root_identity.st_dev, self._root_identity.st_ino),
                "visible canonical target directory identity changed",
            )
        finally:
            probe.close()

    def publish(self) -> None:
        require_external_m1_trust_lock()
        self.verify_staging_identity()
        _publish_staging_directory(
            parent_fd=self._parent_fd,
            staging_name=self._staging_name,
            target_name=self._target_name,
        )
        self._published = True
        self.verify_published_identity()


def _read_external_file_once(path: Path, label: str) -> bytes:
    require_external_m1_trust_lock()
    _require(path.is_absolute(), f"{label} must be absolute")
    _require(path.is_file() and not path.is_symlink(), f"{label} is unsafe")
    with _RawRootReader(path.parent) as reader:
        return reader.read(path.name, label)


def _article_id(record: Mapping[str, Any]) -> str:
    article_url = str(record.get("article_url") or "").rstrip("/")
    article_id = article_url.rsplit("/", 1)[-1]
    _require(article_id and "/" not in article_id, "source record article_url is invalid")
    return article_id


def _normalized_doi(record: Mapping[str, Any]) -> str:
    doi = str(record.get("doi") or "").strip().lower()
    _require(doi.startswith("10.") and "/" in doi, "source record DOI is invalid")
    return doi


def _validate_frozen_record_policy(record: Mapping[str, Any]) -> None:
    _require(record.get("policy_accepted") is True, "source record is not policy accepted")
    _require(
        record.get("download_eligible") is True,
        "source record is not download eligible",
    )
    _require(record.get("journal_allowed") is True, "source record journal is not allowed")
    _require(record.get("require_cc_by") is True, "source record does not require CC-BY")
    _require(record.get("reject_reasons") == [], "source record has rejection reasons")
    license_record = record.get("license")
    _require(isinstance(license_record, dict), "source record license is invalid")
    _require(
        license_record.get("license_id") == "CC-BY-4.0",
        "source record is not CC-BY-4.0",
    )
    _require(
        license_record.get("content_version") == "vor",
        "source record is not version of record",
    )
    normalized_url = license_record.get("normalized_url")
    _require(
        isinstance(normalized_url, str)
        and normalized_url == "https://creativecommons.org/licenses/by/4.0/",
        "source record license URL is invalid",
    )


def _parse_processed(
    payload: bytes,
    expected_ids: set[str],
    label: str,
) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise C2RemediationError(f"{label} is not UTF-8") from exc
    _require(
        all(line and line == line.strip() for line in lines),
        f"{label} has blank or padded rows",
    )
    _require(len(lines) == len(set(lines)), f"{label} has duplicate article IDs")
    _require(set(lines).issubset(expected_ids), f"{label} has unknown article IDs")
    return {line: _sha256_bytes(line.encode("utf-8")) for line in lines}


def _parse_skipped(
    payload: bytes,
    expected_ids: set[str],
    label: str,
) -> dict[str, tuple[str, str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise C2RemediationError(f"{label} is not UTF-8") from exc
    statuses: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(lines, start=1):
        _require(line, f"{label} has a blank row")
        _require(line.count("\t") == 1, f"{label}:{line_number} must have one tab")
        article_id, status = line.split("\t")
        _require(article_id in expected_ids, f"{label}:{line_number} has unknown article ID")
        _require(article_id not in statuses, f"{label}:{line_number} duplicates article ID")
        _require(
            status in SKIPPED_STATUS_VALUES,
            f"{label}:{line_number} has invalid skipped status",
        )
        statuses[article_id] = (status, _sha256_bytes(line.encode("utf-8")))
    return statuses


def _verify_worktree(worktree: Path) -> dict[str, Any]:
    require_external_m1_trust_lock()
    worktree = _verify_trusted_existing_root(worktree, "worktree")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD^{tree}"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C2RemediationError("cannot establish clean worktree binding") from exc
    _require(commit == FROZEN_CODE_COMMIT, "worktree commit is not the frozen acquisition commit")
    _require(not status.strip(), "worktree is dirty")
    return {"commit": commit, "tree": tree, "dirty": False}


def _verify_frozen_inputs(
    partition: FrozenPartition,
    source_chunk: Path,
    frozen_universe: Path,
    freeze_summary: Path,
) -> tuple[bytes, list[dict[str, Any]]]:
    require_external_m1_trust_lock()
    for path, label in (
        (source_chunk, "source chunk"),
        (frozen_universe, "frozen universe"),
        (freeze_summary, "freeze summary"),
    ):
        _require(path.is_absolute(), f"{label} must be absolute")
        _verify_trusted_existing_root(path.parent, f"{label} parent")
        _require(path.is_file() and not path.is_symlink(), f"{label} is unsafe")
    source_bytes = _read_external_file_once(source_chunk, "source chunk")
    universe_bytes = _read_external_file_once(frozen_universe, "frozen universe")
    freeze_summary_bytes = _read_external_file_once(
        freeze_summary,
        "freeze summary",
    )
    _require(
        _sha256_bytes(source_bytes) == partition.source_sha256,
        "source chunk hash does not match static frozen partition",
    )
    _require(
        _sha256_bytes(universe_bytes) == FROZEN_UNIVERSE_SHA256,
        "frozen universe hash mismatch",
    )
    _require(
        _sha256_bytes(freeze_summary_bytes) == FROZEN_FREEZE_SUMMARY_SHA256,
        "freeze summary file hash mismatch",
    )
    summary = _read_json_bytes(freeze_summary_bytes, "freeze summary")
    _require(
        summary.get("summary_hash") == FROZEN_FREEZE_SUMMARY_HASH,
        "freeze summary semantic hash binding mismatch",
    )
    _require(
        _sha256_json(_json_without(summary, "summary_hash"))
        == FROZEN_FREEZE_SUMMARY_HASH,
        "freeze summary semantic hash is invalid",
    )
    _require(
        summary.get("universe_sha256") == FROZEN_UNIVERSE_SHA256,
        "freeze summary universe binding mismatch",
    )
    _require(
        summary.get("universe_records") == 2463,
        "freeze summary universe record count mismatch",
    )
    chunks = summary.get("chunks")
    _require(isinstance(chunks, list) and len(chunks) == 13, "freeze summary chunks invalid")
    summary_by_id: dict[str, Mapping[str, Any]] = {}
    for item in chunks:
        _require(isinstance(item, dict), "freeze summary chunk entry is invalid")
        summary_chunk = item.get("chunk")
        _require(
            isinstance(summary_chunk, int) and 1 <= summary_chunk <= 13,
            "freeze summary chunk ID is invalid",
        )
        chunk_id = f"{summary_chunk:03d}"
        _require(chunk_id not in summary_by_id, "freeze summary has duplicate chunk")
        summary_by_id[chunk_id] = item
    _require(
        set(summary_by_id) == set(FROZEN_PARTITIONS),
        "freeze summary chunk IDs do not match static partition",
    )
    for expected_chunk_id, expected_partition in FROZEN_PARTITIONS.items():
        chunk_summary = summary_by_id[expected_chunk_id]
        _require(
            chunk_summary.get("records") == expected_partition.records
            and chunk_summary.get("sha256") == expected_partition.source_sha256
            and chunk_summary.get("start_index_1based")
            == expected_partition.start_index_1based
            and chunk_summary.get("end_index_1based")
            == expected_partition.end_index_1based,
            "freeze summary partition binding mismatch",
        )
    records = _read_jsonl_bytes(source_bytes, "source chunk")
    _require(len(records) == partition.records, "source chunk count mismatch")
    universe_records = _read_jsonl_bytes(
        universe_bytes,
        "frozen universe",
    )
    _require(len(universe_records) == 2463, "frozen universe record count mismatch")
    expected_slice = universe_records[
        partition.start_index_1based - 1 : partition.end_index_1based
    ]
    _require(
        records == expected_slice,
        "source chunk is not the statically bound frozen-universe slice",
    )
    for record in records:
        _validate_frozen_record_policy(record)
    dois = [_normalized_doi(record) for record in records]
    article_ids = [_article_id(record) for record in records]
    _require(len(dois) == len(set(dois)), "source chunk has duplicate DOI")
    _require(len(article_ids) == len(set(article_ids)), "source chunk has duplicate article ID")
    return source_bytes, records


def _parse_receipt(
    payload: bytes,
    *,
    attempt: str,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    receipt = _read_json_bytes(payload, f"{attempt} raw attempt receipt")
    _require(
        receipt.get("schema_version") == "c2-v2-raw-attempt-receipt-v1",
        f"{attempt} receipt schema mismatch",
    )
    _require(receipt.get("attempt") == attempt, f"{attempt} receipt attempt mismatch")
    _require(
        receipt.get("attempt_index") == ATTEMPT_INDEX[attempt],
        f"{attempt} receipt attempt index mismatch",
    )
    for key, expected in expected_hashes.items():
        _require(receipt.get(key) == expected, f"{attempt} receipt {key} mismatch")
    for key in ("start_monotonic_ns", "end_monotonic_ns"):
        _require(
            isinstance(receipt.get(key), int) and not isinstance(receipt[key], bool),
            f"{attempt} receipt {key} is invalid",
        )
    _require(
        receipt["end_monotonic_ns"] >= receipt["start_monotonic_ns"],
        f"{attempt} receipt timestamps are invalid",
    )
    declared = receipt.get("receipt_hash")
    _require(isinstance(declared, str), f"{attempt} receipt hash missing")
    _require(
        _sha256_json(_json_without(receipt, "receipt_hash")) == declared,
        f"{attempt} receipt semantic hash mismatch",
    )
    return receipt


def _validate_events(
    payload: bytes,
    *,
    attempt: str,
    records: list[dict[str, Any]],
    statuses: Mapping[str, str],
    start_monotonic_ns: int,
    end_monotonic_ns: int,
) -> list[dict[str, Any]]:
    events = _read_jsonl_bytes(payload, f"{attempt} raw attempt events")
    _require(len(events) == len(records), f"{attempt} event count mismatch")
    previous_event_time: int | None = None
    for ordinal, (event, record) in enumerate(zip(events, records, strict=True), start=1):
        article_id = _article_id(record)
        expected = {
            "attempt": attempt,
            "attempt_index": ATTEMPT_INDEX[attempt],
            "article_id": article_id,
            "doi": _normalized_doi(record),
            "input_ordinal": ordinal,
            "processed": statuses[article_id] == "downloaded",
            "status": statuses[article_id],
        }
        _require(
            set(event) == {*expected, "event_monotonic_ns"},
            f"{attempt} event {ordinal} fields are invalid",
        )
        _require(
            all(event[key] == value for key, value in expected.items()),
            f"{attempt} event {ordinal} differs from raw status derivation",
        )
        event_time = event.get("event_monotonic_ns")
        _require(
            isinstance(event_time, int) and not isinstance(event_time, bool),
            f"{attempt} event {ordinal} timestamp is invalid",
        )
        _require(
            previous_event_time is None or event_time >= previous_event_time,
            f"{attempt} event timing is out of order",
        )
        _require(
            start_monotonic_ns <= event_time <= end_monotonic_ns,
            f"{attempt} event timestamp is outside its receipt interval",
        )
        previous_event_time = event_time
    return events


def _reject_absolute_path_values(
    value: object,
    *,
    raw_root: Path,
    label: str,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_path_values(
                item,
                raw_root=raw_root,
                label=f"{label}.{key}",
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_path_values(
                item,
                raw_root=raw_root,
                label=f"{label}[{index}]",
            )
        return
    if not isinstance(value, str):
        return
    normalized = value.replace("\\", "/")
    _require(
        not (
            normalized.startswith("/")
            or normalized.startswith("file:")
            or normalized.startswith("~/")
            or re.match(r"^[A-Za-z]:/", normalized)
            or str(raw_root) in value
        ),
        f"{label} contains an absolute filesystem reference",
    )


def _raw_relative_files() -> tuple[str, ...]:
    files = [
        "accepted.jsonl",
        "control/acquisition_config.json",
        "control/pre_download_binding.json",
        "control/pre_download_binding.sha256",
        "control/execution_evidence.json",
    ]
    for attempt in ATTEMPTS:
        prefix = f"control/{attempt}"
        files.extend(
            (
                f"{prefix}/accepted.jsonl",
                f"{prefix}/processed.txt",
                f"{prefix}/_skipped.txt",
                f"{prefix}/postfetch.log",
                f"{prefix}/postfetch.exit",
                f"{prefix}/raw_attempt_receipt.json",
                f"{prefix}/raw_attempt_events.jsonl",
            )
        )
    return tuple(files)


def _validate_pre_download_controls(
    raw_bytes: Mapping[str, bytes],
    *,
    partition: FrozenPartition,
    code: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json_bytes(
        raw_bytes["control/acquisition_config.json"],
        "raw acquisition config",
    )
    _require(
        isinstance(config.get("config_hash"), str)
        and _sha256_json(_json_without(config, "config_hash")) == config["config_hash"],
        "raw acquisition config hash mismatch",
    )
    _require(config.get("attempts") == list(ATTEMPTS), "raw acquisition config attempts mismatch")
    _require(config.get("no_model_calls") is True, "raw acquisition config permits model calls")
    _require(config.get("no_early_stop") is True, "raw acquisition config permits early stop")
    _require(
        config.get("outcome_independent") is True,
        "raw acquisition config is outcome-dependent",
    )
    arguments = config.get("arguments")
    _require(isinstance(arguments, dict), "raw acquisition config arguments missing")
    expected_arguments = {
        "require_cc_by": True,
        "sort": "input",
        "max_figs": 12,
        "max_empty_figs": 2,
        "sleep_seconds": 1.0,
        "timeout_seconds": 300,
        "max_retries": 3,
    }
    for key, expected in expected_arguments.items():
        _require(
            arguments.get(key) == expected,
            f"raw acquisition config {key} mismatch",
        )
    _require(
        isinstance(arguments.get("workers"), int)
        and not isinstance(arguments["workers"], bool)
        and arguments["workers"] > 0,
        "raw acquisition config workers is invalid",
    )
    binding = _read_json_bytes(
        raw_bytes["control/pre_download_binding.json"],
        "raw pre-download binding",
    )
    _require(
        isinstance(binding.get("summary_hash"), str)
        and _sha256_json(_json_without(binding, "summary_hash")) == binding["summary_hash"],
        "raw pre-download binding hash mismatch",
    )
    sidecar = raw_bytes["control/pre_download_binding.sha256"].decode("utf-8")
    expected_sidecar = (
        f"{_sha256_bytes(raw_bytes['control/pre_download_binding.json'])}"
        "  pre_download_binding.json\n"
    )
    _require(sidecar == expected_sidecar, "raw pre-download binding sidecar mismatch")
    expected_fields = {
        "chunk": int(partition.chunk_id),
        "chunk_records": partition.records,
        "chunk_start_index_1based": partition.start_index_1based,
        "chunk_end_index_1based": partition.end_index_1based,
        "attempts": list(ATTEMPTS),
        "attempt_coverage_required_each": partition.records,
        "code_commit": code["commit"],
        "code_dirty": False,
        "exact_byte_copy": True,
        "execution_accepted_sha256": partition.source_sha256,
        "outcome_independent": True,
        "review_or_experiment_outcomes_used": False,
        "source_chunk_sha256": partition.source_sha256,
        "source_universe_sha256": FROZEN_UNIVERSE_SHA256,
        "source_freeze_summary_sha256": FROZEN_FREEZE_SUMMARY_SHA256,
        "source_freeze_summary_hash": FROZEN_FREEZE_SUMMARY_HASH,
    }
    for key, expected in expected_fields.items():
        _require(binding.get(key) == expected, f"raw pre-download binding {key} mismatch")
    _require(
        binding.get("acquisition_config_hash") == config["config_hash"],
        "raw pre-download binding acquisition config hash mismatch",
    )
    _require(
        binding.get("selection_rule")
        == (
            "execute every frozen chunk record in initial and both fixed retries "
            "regardless of every preceding outcome"
        ),
        "raw pre-download binding selection rule mismatch",
    )
    return config, binding


def _validate_execution_evidence(
    raw_bytes: Mapping[str, bytes],
    code: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _read_json_bytes(
        raw_bytes["control/execution_evidence.json"],
        "raw execution evidence",
    )
    _require(
        evidence.get("schema_version") == "c2-remediation-execution-evidence-v1",
        "raw execution evidence schema mismatch",
    )
    _require(evidence.get("status") == "PASS", "raw execution evidence is not PASS")
    _require(
        evidence.get("code_commit") == code["commit"] and evidence.get("code_dirty") is False,
        "raw execution evidence code binding mismatch",
    )
    _require(
        evidence.get("secret_scan_status") == "CLEAN",
        "raw execution evidence secret scan is not clean",
    )
    concurrency = evidence.get("concurrency")
    _require(
        isinstance(concurrency, dict),
        "raw execution evidence concurrency record is invalid",
    )
    _require(
        concurrency.get("status") == "PASS"
        and concurrency.get("max_fresh_roots") == 1
        and concurrency.get("fresh_roots_active") == 1
        and concurrency.get("retained_active_pids") == [],
        "raw execution evidence concurrency gate failed",
    )
    tests = evidence.get("tests")
    _require(isinstance(tests, list) and tests, "raw execution evidence has no tests")
    for index, test in enumerate(tests, start=1):
        _require(isinstance(test, dict), f"raw execution test {index} is invalid")
        _require(
            isinstance(test.get("command"), str) and test["command"].strip(),
            f"raw execution test {index} command is invalid",
        )
        _require(
            test.get("exit_code") == 0,
            f"raw execution test {index} failed",
        )
        _require(
            _is_sha256(test.get("output_sha256")),
            f"raw execution test {index} output hash is invalid",
        )
    _require(
        isinstance(evidence.get("evidence_hash"), str)
        and _sha256_json(_json_without(evidence, "evidence_hash"))
        == evidence["evidence_hash"],
        "raw execution evidence hash mismatch",
    )
    return evidence


def _validate_raw_root_from_reader(
    reader: _RawRootReader,
    source_bytes: bytes,
    records: list[dict[str, Any]],
    *,
    partition: FrozenPartition,
    code: Mapping[str, Any],
) -> tuple[
    dict[str, bytes],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    _RawRootReader,
]:
    require_external_m1_trust_lock()
    expected_ids = {_article_id(record) for record in records}
    expected_dois = {_normalized_doi(record) for record in records}
    input_line_hashes = [
        _sha256_bytes(line)
        for line in source_bytes.splitlines()
    ]
    _require(
        len(input_line_hashes) == len(records),
        "source chunk line count does not match parsed records",
    )
    raw_bytes: dict[str, bytes] = {}
    for relative in _raw_relative_files():
        raw_bytes[relative] = reader.read(relative, f"raw {relative}")
    _require(raw_bytes["accepted.jsonl"] == source_bytes, "raw accepted bytes differ from source")
    config, binding = _validate_pre_download_controls(
        raw_bytes,
        partition=partition,
        code=code,
    )
    execution_evidence = _validate_execution_evidence(raw_bytes, code)
    for label, value in (
        ("raw acquisition config", config),
        ("raw pre-download binding", binding),
        ("raw execution evidence", execution_evidence),
    ):
        _reject_absolute_path_values(
            value,
            raw_root=reader.root,
            label=label,
        )

    events_by_attempt: dict[str, list[dict[str, Any]]] = {}
    statuses_by_attempt: dict[str, dict[str, str]] = {}
    attestation_entries: list[dict[str, Any]] = []
    previous_attempt_end: int | None = None
    for attempt in ATTEMPTS:
        prefix = f"control/{attempt}"
        accepted_key = f"{prefix}/accepted.jsonl"
        processed_key = f"{prefix}/processed.txt"
        skipped_key = f"{prefix}/_skipped.txt"
        log_key = f"{prefix}/postfetch.log"
        exit_key = f"{prefix}/postfetch.exit"
        receipt_key = f"{prefix}/raw_attempt_receipt.json"
        events_key = f"{prefix}/raw_attempt_events.jsonl"
        _require(
            raw_bytes[accepted_key] == source_bytes,
            f"{attempt} accepted bytes differ from source",
        )
        processed = _parse_processed(raw_bytes[processed_key], expected_ids, processed_key)
        skipped = _parse_skipped(raw_bytes[skipped_key], expected_ids, skipped_key)
        processed_set = set(processed)
        skipped_set = set(skipped)
        _require(processed_set.isdisjoint(skipped_set), f"{attempt} processed/skipped overlap")
        _require(
            processed_set | skipped_set == expected_ids,
            f"{attempt} processed/skipped partition is incomplete",
        )
        _require(raw_bytes[log_key], f"{attempt} postfetch log is empty")
        _require(
            raw_bytes[exit_key].decode("utf-8").strip() == "0",
            f"{attempt} postfetch exit is nonzero",
        )
        hashes = {
            "input_sha256": _sha256_bytes(raw_bytes[accepted_key]),
            "config_hash": str(config["config_hash"]),
            "processed_sha256": _sha256_bytes(raw_bytes[processed_key]),
            "skipped_sha256": _sha256_bytes(raw_bytes[skipped_key]),
            "postfetch_log_sha256": _sha256_bytes(raw_bytes[log_key]),
            "postfetch_exit_sha256": _sha256_bytes(raw_bytes[exit_key]),
            "source_chunk_sha256": partition.source_sha256,
            "source_universe_sha256": FROZEN_UNIVERSE_SHA256,
            "source_freeze_summary_sha256": FROZEN_FREEZE_SUMMARY_SHA256,
            "source_freeze_summary_hash": FROZEN_FREEZE_SUMMARY_HASH,
        }
        receipt = _parse_receipt(
            raw_bytes[receipt_key],
            attempt=attempt,
            expected_hashes=hashes,
        )
        _reject_absolute_path_values(
            receipt,
            raw_root=reader.root,
            label=f"{attempt} raw attempt receipt",
        )
        _require(
            previous_attempt_end is None
            or receipt["start_monotonic_ns"] >= previous_attempt_end,
            f"{attempt} receipt is out of fixed attempt order",
        )
        previous_attempt_end = receipt["end_monotonic_ns"]
        statuses = {
            article_id: (
                "downloaded"
                if article_id in processed_set
                else skipped[article_id][0]
            )
            for article_id in expected_ids
        }
        events_by_attempt[attempt] = _validate_events(
            raw_bytes[events_key],
            attempt=attempt,
            records=records,
            statuses=statuses,
            start_monotonic_ns=receipt["start_monotonic_ns"],
            end_monotonic_ns=receipt["end_monotonic_ns"],
        )
        for event in events_by_attempt[attempt]:
            _reject_absolute_path_values(
                event,
                raw_root=reader.root,
                label=f"{attempt} raw attempt event",
            )
        statuses_by_attempt[attempt] = statuses
        bundle = {
            "schema": "c2-v2-attempt-bundle-v1",
            "attempt": attempt,
            "attempt_index": ATTEMPT_INDEX[attempt],
            "accepted_sha256": hashes["input_sha256"],
            "processed_sha256": hashes["processed_sha256"],
            "skipped_sha256": hashes["skipped_sha256"],
            "postfetch_log_sha256": hashes["postfetch_log_sha256"],
            "postfetch_exit_sha256": hashes["postfetch_exit_sha256"],
        }
        bundle_digest = _sha256_json(bundle)
        for ordinal, record in enumerate(records, start=1):
            article_id = _article_id(record)
            attestation_entries.append(
                {
                    "doi": _normalized_doi(record),
                    "input_ordinal": ordinal,
                    "attempt_index": ATTEMPT_INDEX[attempt],
                    "processed": article_id in processed_set,
                    "status": statuses[article_id],
                    "raw_artifact_digest": _sha256_json(
                        {
                            "schema": "c2-v2-raw-entry-v1",
                            "doi": _normalized_doi(record),
                            "article_id": article_id,
                            "input_ordinal": ordinal,
                            "attempt": attempt,
                            "attempt_index": ATTEMPT_INDEX[attempt],
                            "input_line_sha256": input_line_hashes[ordinal - 1],
                            "processed_line_sha256_or_null": processed.get(article_id),
                            "skipped_line_sha256_or_null": (
                                None
                                if article_id in processed_set
                                else skipped[article_id][1]
                            ),
                            "attempt_bundle_digest": bundle_digest,
                            "processed": article_id in processed_set,
                            "status": statuses[article_id],
                        }
                    ),
                }
            )
    _require(len(expected_dois) == len(records), "raw root DOI validation failed")
    return (
        raw_bytes,
        events_by_attempt,
        statuses_by_attempt,
        attestation_entries,
        binding,
        execution_evidence,
        reader,
    )


def _validate_raw_root(
    raw_root: Path,
    source_bytes: bytes,
    records: list[dict[str, Any]],
    *,
    partition: FrozenPartition,
    code: Mapping[str, Any],
) -> tuple[
    dict[str, bytes],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    _RawRootReader,
]:
    require_external_m1_trust_lock()
    reader = _RawRootReader(raw_root)
    try:
        return _validate_raw_root_from_reader(
            reader,
            source_bytes,
            records,
            partition=partition,
            code=code,
        )
    except Exception:
        reader.close()
        raise


def _validate_download_source_evidence(
    reader: _RawRootReader,
    *,
    article_id: str,
    doi: str,
    provenance_path: str,
    provenance: Mapping[str, Any],
) -> _SourceEvidence:
    require_external_m1_trust_lock()
    _require(
        provenance.get("download_status") == "downloaded",
        f"downloaded provenance status is invalid: {article_id}",
    )
    evidence = provenance.get("source_evidence")
    _require(
        isinstance(evidence, dict),
        f"downloaded provenance lacks source evidence: {article_id}",
    )
    expected_descriptor_path = f"content/_source_evidence/{article_id}.json"
    _require(
        evidence.get("descriptor_path") == expected_descriptor_path,
        f"source evidence descriptor path is invalid: {article_id}",
    )
    _require(
        _is_sha256(evidence.get("descriptor_sha256"))
        and isinstance(evidence.get("descriptor_bytes"), int)
        and not isinstance(evidence["descriptor_bytes"], bool)
        and evidence["descriptor_bytes"] > 0,
        f"source evidence descriptor metadata is invalid: {article_id}",
    )
    descriptor_payload = reader.read(
        expected_descriptor_path,
        f"source evidence descriptor {article_id}",
    )
    _require(
        len(descriptor_payload) == evidence["descriptor_bytes"]
        and _sha256_bytes(descriptor_payload) == evidence["descriptor_sha256"],
        f"source evidence descriptor digest mismatch: {article_id}",
    )
    descriptor = _read_json_bytes(
        descriptor_payload,
        f"source evidence descriptor {article_id}",
    )
    _reject_absolute_path_values(
        descriptor,
        raw_root=reader.root,
        label=f"source evidence descriptor {article_id}",
    )
    schema_version = descriptor.get("schema_version")
    _require(
        schema_version in {"c2-source-evidence-v1", "c2-source-evidence-v2"},
        f"source evidence descriptor schema mismatch: {article_id}",
    )
    _require(
        isinstance(descriptor.get("descriptor_hash"), str)
        and _sha256_json(_json_without(descriptor, "descriptor_hash"))
        == descriptor["descriptor_hash"],
        f"source evidence descriptor semantic hash mismatch: {article_id}",
    )
    if schema_version == "c2-source-evidence-v2":
        try:
            sources = list(
                validate_source_evidence_descriptor_v2(
                    descriptor,
                    article_id=article_id,
                    doi_id=doi,
                    provenance_relative_path=provenance_path,
                )
            )
        except SourceBearingExtensionError as exc:
            raise C2RemediationError(
                f"source evidence V2 descriptor is invalid: {article_id}"
            ) from exc
    else:
        _require(
            descriptor.get("doi") == doi
            and descriptor.get("article_id") == article_id
            and descriptor.get("provenance_path") == provenance_path,
            f"source evidence descriptor provenance linkage mismatch: {article_id}",
        )
        sources = descriptor.get("sources")
        _require(
            isinstance(sources, list) and sources,
            f"source evidence descriptor has no source files: {article_id}",
        )
    expected_prefix = f"content/_sources/{article_id}/"
    source_paths: list[Mapping[str, Any]] = []
    seen_paths: set[str] = set()
    for index, source in enumerate(sources, start=1):
        _require(
            isinstance(source, dict),
            f"source evidence source entry {index} is invalid: {article_id}",
        )
        relative_path = source.get("relative_path")
        _require(
            isinstance(relative_path, str)
            and relative_path.startswith(expected_prefix)
            and relative_path not in seen_paths,
            f"source evidence source path is invalid: {article_id}",
        )
        _ensure_relative(relative_path, f"source evidence source path {article_id}")
        _require(
            source.get("doi") == doi
            and _is_sha256(source.get("sha256"))
            and isinstance(source.get("bytes"), int)
            and not isinstance(source["bytes"], bool)
            and source["bytes"] > 0,
            f"source evidence source metadata is invalid: {article_id}",
        )
        source_payload = reader.read(
            relative_path,
            f"source evidence source {article_id}:{index}",
        )
        _require(
            len(source_payload) == source["bytes"]
            and _sha256_bytes(source_payload) == source["sha256"],
            f"source evidence source digest mismatch: {article_id}",
        )
        seen_paths.add(relative_path)
        source_paths.append(
            {
                "relative_path": relative_path,
                "sha256": source["sha256"],
                "bytes": source["bytes"],
                "doi": doi,
            }
        )
    return _SourceEvidence(
        descriptor_path=expected_descriptor_path,
        descriptor_sha256=evidence["descriptor_sha256"],
        descriptor_bytes=evidence["descriptor_bytes"],
        schema_version=str(schema_version),
        descriptor=descriptor,
        source_paths=tuple(source_paths),
    )


def _read_provenance(
    reader: _RawRootReader,
    records: list[dict[str, Any]],
    statuses_by_attempt: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    require_external_m1_trust_lock()
    provenance: dict[str, dict[str, Any]] = {}
    for record in records:
        article_id = _article_id(record)
        relative = f"content/_provenance/{article_id}.json"
        payload = reader.read(relative, f"raw provenance {article_id}")
        value = _read_json_bytes(payload, f"raw provenance {article_id}")
        _reject_absolute_path_values(
            value,
            raw_root=reader.root,
            label=f"raw provenance {article_id}",
        )
        _require(
            value.get("doi") == _normalized_doi(record),
            f"provenance DOI mismatch: {article_id}",
        )
        reasons = value.get("rejection_reasons")
        _require(isinstance(reasons, list), f"provenance reasons invalid: {article_id}")
        terminal_status = statuses_by_attempt["retry2"][article_id]
        source_bearing = any(
            statuses_by_attempt[attempt][article_id] == "downloaded"
            for attempt in ATTEMPTS
        )
        if source_bearing:
            _require(
                reasons == ([] if terminal_status == "downloaded" else [terminal_status]),
                f"source-bearing provenance rejection reason mismatch: {article_id}",
            )
            source_evidence = _validate_download_source_evidence(
                reader,
                article_id=article_id,
                doi=_normalized_doi(record),
                provenance_path=relative,
                provenance=value,
            )
            acquisition_disposition = (
                "SOURCE_BEARING_DOWNLOAD"
                if terminal_status == "downloaded"
                else (
                    "SOURCE_BEARING_PRIOR_ATTEMPT_DOWNLOAD_"
                    f"RETRY2_{terminal_status.upper().replace('-', '_')}"
                )
            )
            source_classification_state = (
                "BLOCKED_CANONICAL_P_BUILDER_REQUIRED"
            )
        else:
            _require(
                reasons == [terminal_status],
                f"provenance rejection reason mismatch: {article_id}",
            )
            _require(
                "download_status" not in value and "source_evidence" not in value,
                f"source-less provenance contains source claim: {article_id}",
            )
            source_evidence = None
            acquisition_disposition = (
                f"SOURCELESS_{terminal_status.upper().replace('-', '_')}"
            )
            source_classification_state = "EMPTY_CHAIN_ELIGIBLE_NO_SOURCE"
        provenance[article_id] = {
            "relative_path": relative,
            "sha256": _sha256_bytes(payload),
            "payload": payload,
            "value": value,
            "source_evidence": source_evidence,
            "acquisition_disposition": acquisition_disposition,
            "source_classification_state": source_classification_state,
        }
    return provenance


def _validate_content_closure(
    reader: _RawRootReader,
    provenance: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Reject raw content not bound to terminal provenance/source descriptors."""

    require_external_m1_trust_lock()
    expected_paths = {
        str(entry["relative_path"])
        for entry in provenance.values()
    }
    for article_id, entry in provenance.items():
        evidence = entry["source_evidence"]
        if evidence is None:
            continue
        expected_paths.add(evidence.descriptor_path)
        expected_paths.update(
            str(source["relative_path"])
            for source in evidence.source_paths
        )
    observed_paths = set(reader.files_under("content", "raw content"))
    _require(
        observed_paths == expected_paths,
        "raw content contains unreferenced, missing, or source-less source artifacts",
    )
    return tuple(sorted(observed_paths))


def _write_bytes(root: _SecureTargetRoot, relative: str, payload: bytes) -> str:
    require_external_m1_trust_lock()
    return root.write_bytes(relative, payload)


def _copy_raw_tree(
    reader: _RawRootReader,
    destination_root: _SecureTargetRoot,
    content_files: Iterable[str],
) -> None:
    require_external_m1_trust_lock()
    destination_root.mkdir("content")
    for source_relative in content_files:
        if source_relative in reader.reads:
            payload = reader.reads[source_relative].payload
        else:
            payload = reader.read(source_relative, f"raw {source_relative}")
        destination_root.write_bytes(source_relative, payload)


def _write_json(
    root: _SecureTargetRoot,
    relative: str,
    value: Mapping[str, Any],
) -> str:
    return _write_bytes(
        root,
        relative,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(
    root: _SecureTargetRoot,
    relative: str,
    values: Iterable[Mapping[str, Any]],
) -> str:
    payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for value in values
    ).encode("utf-8")
    return _write_bytes(root, relative, payload)


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = _sha256_json(_json_without(value, field))
    return value


def _artifact_entries(root: Path, excludes: set[str]) -> list[dict[str, Any]]:
    require_external_m1_trust_lock()
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(
            relative == excluded.rstrip("/") or relative.startswith(excluded)
            for excluded in excludes
        ):
            continue
        _require(not path.is_symlink(), f"target artifact symlink {relative}")
        if path.is_file():
            entries.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def _target_artifact_entries(
    root: _SecureTargetRoot,
    excludes: set[str],
) -> list[dict[str, Any]]:
    require_external_m1_trust_lock()
    return [
        {
            "path": relative,
            "bytes": byte_count,
            "sha256": _sha256_bytes(payload),
        }
        for relative, byte_count, payload in root.files(excludes)
    ]


def _write_secret_scan(root: _SecureTargetRoot) -> tuple[str, dict[str, Any]]:
    require_external_m1_trust_lock()
    hits: list[dict[str, str]] = []
    files_scanned = 0
    for relative, _byte_count, payload in root.files():
        files_scanned += 1
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                hits.append({"path": relative, "pattern": label})
    scan = {
        "schema_version": "c2-remediation-secret-scan-v1",
        "status": "CLEAN" if not hits else "BLOCKED_SECRET_HIT",
        "files_scanned": files_scanned,
        "patterns": [label for label, _ in SECRET_PATTERNS],
        "hits": hits,
    }
    _seal(scan, "scan_hash")
    path = _write_json(root, "control/secret_scan.json", scan)
    _require(not hits, "secret scan found credential-like material")
    return path, scan


def _protected_root_inventory(root: Path) -> dict[str, Any]:
    require_external_m1_trust_lock()
    _require(root.is_dir() and not root.is_symlink(), "protected old root is unsafe")
    entries = _artifact_entries(root, {".pipeline_worktree/"})
    tuples = [
        (entry["path"], entry["bytes"], entry["sha256"])
        for entry in entries
    ]
    return {
        "artifact_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "inventory_hash": _sha256_bytes(canonical_json(tuples).encode("utf-8")),
    }


def _verify_protected_old_root(
    target_parent: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    require_external_m1_trust_lock()
    root_name = contract.get("root_name")
    _require(isinstance(root_name, str) and root_name, "protected old root name invalid")
    old_root = target_parent / root_name
    observed = _protected_root_inventory(old_root)
    for key in ("artifact_count", "total_bytes", "inventory_hash"):
        _require(
            observed[key] == contract.get(key),
            f"protected old root {root_name} {key} mismatch",
        )
    report = old_root / "sealed_report_v1" / "sealed_report.json"
    _require(
        report.is_file()
        and not report.is_symlink()
        and sha256_file(report) == contract.get("sealed_report_sha256"),
        f"protected old root {root_name} sealed report mismatch",
    )
    return {
        "root_name": root_name,
        **observed,
        "sealed_report_sha256": sha256_file(report),
    }


def _create_target_root(target_root: Path) -> _SecureTargetRoot:
    """Create private staging beneath an existing trusted parent, never the target."""

    require_external_m1_trust_lock()
    _require(target_root.is_absolute(), "target root must be absolute")
    target = _open_private_staging_parent(target_root)
    staging_parent_fd = target.parent_fd
    target.parent_fd = -1
    root_fd = -1
    try:
        # This is only an early rejection; publication independently uses NO-REPLACE.
        try:
            os.stat(
                target.leaf_name,
                dir_fd=staging_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise C2RemediationError("target root already exists")
        # The retained parent FD is pre-existing, no-follow, owner/ACL validated, and
        # not writable by competing principals. A malicious process at this EUID is
        # outside this local-user threat boundary.
        for _ in range(_STAGING_CREATE_ATTEMPTS):
            staging_name = (
                f".{target.leaf_name}.c2-remediation-staging-{secrets.token_hex(16)}"
            )
            try:
                os.mkdir(staging_name, mode=0o700, dir_fd=staging_parent_fd)
            except FileExistsError:
                continue
            except OSError as exc:
                raise C2RemediationError(
                    "cannot create private remediation staging root"
                ) from exc
            try:
                staged_entry = os.stat(
                    staging_name,
                    dir_fd=staging_parent_fd,
                    follow_symlinks=False,
                )
                _require(
                    stat.S_ISDIR(staged_entry.st_mode),
                    "created private staging root is not a directory",
                )
                root_fd = os.open(
                    staging_name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=staging_parent_fd,
                )
                opened_entry = os.fstat(root_fd)
                _require(
                    (opened_entry.st_dev, opened_entry.st_ino)
                    == (staged_entry.st_dev, staged_entry.st_ino),
                    "private staging root changed while opening",
                )
                secure_root = _SecureTargetRoot(
                    canonical_path=target.final_path,
                    parent_fd=staging_parent_fd,
                    target_name=target.leaf_name,
                    staging_name=staging_name,
                    root_fd=root_fd,
                    staging_identity=(staged_entry.st_dev, staged_entry.st_ino),
                )
                root_fd = -1
                return secure_root
            except Exception:
                if root_fd != -1:
                    os.close(root_fd)
                    root_fd = -1
                raise
        raise C2RemediationError(
            "cannot allocate a collision-free private remediation staging root"
        )
    except Exception:
        if root_fd != -1:
            os.close(root_fd)
        os.close(staging_parent_fd)
        raise
    finally:
        target.close()


def _source_manifest(reads: Mapping[str, _RawRead]) -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "bytes": len(read.payload),
            "sha256": read.sha256,
            "fd_metadata": dict(read.metadata),
        }
        for relative, read in sorted(reads.items())
    ]
    manifest = {
        "schema_version": "c2-v2-raw-source-manifest-v1",
        "raw_root_locator": "external-immutable-evidence-root",
        "files": files,
    }
    return _seal(manifest, "source_manifest_hash")


def _preservation_ledger(
    chunk_id: str,
    *,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    old_root = OLD_ROOT_PRESERVATION.get(chunk_id)
    old_roots = [] if old_root is None else [dict(old_root)]
    _require(
        (before is None and after is None) or (before is not None and after is not None),
        "protected old-root preservation evidence is incomplete",
    )
    _require(
        before is None or before == after,
        "protected old root changed during finalization",
    )
    ledger = {
        "schema_version": "c2-remediation-preservation-ledger-v1",
        "fingerprint_algorithm": (
            "SHA-256 compact JSON of sorted "
            "(relative_posix_path, byte_size, content_sha256) tuples "
            "excluding .pipeline_worktree"
        ),
        "old_root_contracts": old_roots,
        "before": before,
        "after": after,
        "preservation_status": (
            "NOT_APPLICABLE_NO_PROTECTED_ROOT"
            if before is None
            else "PASS_UNCHANGED"
        ),
        "retained_candidates": [
            {"root_name": name, "disposition": "EXCLUDED_NOT_TARGET"}
            for name in RETAINED_CANDIDATE_EXCLUSIONS
        ],
        "verification_requirement": (
            "This ledger records checks before target creation and before report "
            "publication; sealed_report_v1/postseal_preservation.json records the "
            "post-publication check. Any mismatch blocks completion."
        ),
    }
    return _seal(ledger, "ledger_hash")


def finalize_remediation_root(
    *,
    chunk_id: str,
    raw_root: Path,
    target_root: Path,
    source_chunk: Path,
    frozen_universe: Path,
    freeze_summary: Path,
    worktree: Path,
    source_bearing_v2: bool = False,
) -> dict[str, Any]:
    """Seal one fresh remediation root from independently acquired raw evidence."""

    require_external_m1_trust_lock()
    _require(
        chunk_id in SUPPORTED_REMEDIATION_CHUNKS,
        "chunk is not authorized for fresh remediation",
    )
    partition = FROZEN_PARTITIONS[chunk_id]
    _require(raw_root.is_absolute(), "raw root must be absolute")
    _require(target_root.is_absolute(), "target root must be absolute")
    _require(source_chunk.is_absolute(), "source chunk must be absolute")
    _require(frozen_universe.is_absolute(), "frozen universe must be absolute")
    _require(freeze_summary.is_absolute(), "freeze summary must be absolute")
    _require(worktree.is_absolute(), "worktree must be absolute")
    normalized_raw = normalize_trusted_output_path(raw_root)
    normalized_target = normalize_trusted_output_path(target_root)
    _require(
        normalized_target.name == expected_target_root_name(chunk_id),
        "target root name does not match the frozen remediation naming contract",
    )
    _require(normalized_raw != normalized_target, "raw root and target root must differ")
    _require(
        not normalized_target.is_relative_to(normalized_raw)
        and not normalized_raw.is_relative_to(normalized_target),
        "raw root and target root must be non-overlapping",
    )
    for evidence_path, label in (
        (source_chunk, "source chunk"),
        (frozen_universe, "frozen universe"),
        (freeze_summary, "freeze summary"),
    ):
        evidence_parent = normalize_trusted_output_path(evidence_path).parent
        _require(
            not normalized_target.is_relative_to(evidence_parent),
            f"target root may not be created inside the {label} root",
        )
    _require(
        not normalized_target.is_relative_to(normalize_trusted_output_path(worktree)),
        "target root may not be created inside the worktree root",
    )
    source_bytes, records = _verify_frozen_inputs(
        partition,
        source_chunk,
        frozen_universe,
        freeze_summary,
    )
    code = _verify_worktree(worktree)
    (
        raw_bytes,
        _events,
        statuses_by_attempt,
        attestation_entries,
        binding,
        execution_evidence,
        raw_reader,
    ) = _validate_raw_root(
        raw_root,
        source_bytes,
        records,
        partition=partition,
        code=code,
    )
    try:
        provenance = _read_provenance(
            raw_reader,
            records,
            statuses_by_attempt,
        )
        content_files = _validate_content_closure(raw_reader, provenance)
    except Exception:
        raw_reader.close()
        raise

    protected_contract = OLD_ROOT_PRESERVATION.get(chunk_id)
    try:
        preservation_before = (
            None
            if protected_contract is None
            else _verify_protected_old_root(
                normalized_target.parent,
                protected_contract,
            )
        )
    except Exception:
        raw_reader.close()
        raise

    target: _SecureTargetRoot | None = None
    try:
        target = _create_target_root(target_root)
    except Exception:
        raw_reader.close()
        raise
    try:
        for relative, payload in raw_bytes.items():
            _write_bytes(target, relative, payload)
        _copy_raw_tree(raw_reader, target, content_files)

        source_manifest = _source_manifest(raw_reader.reads)
        _write_json(target, "control/v2/raw_source_manifest.json", source_manifest)
        raw_attestation = {
            "schema": "c2-v2-raw-attestation-v1",
            "source_chunk_sha256": partition.source_sha256,
            "N": partition.records,
            "entries": attestation_entries,
        }
        raw_attestation["raw_attempt_attestation_hash"] = _sha256_json(raw_attestation)
        _write_json(target, "control/v2/raw_attempt_attestation.json", raw_attestation)
        provenance_relocation = {
            "schema_version": "c2-v2-provenance-relocation-v1",
            "source_root_locator": "external-immutable-evidence-root",
            "target_root_locator": "content/_provenance",
            "entries": [
                {
                    "article_id": _article_id(record),
                    "doi": _normalized_doi(record),
                    "raw_relative_path": provenance[_article_id(record)][
                        "relative_path"
                    ],
                    "target_relative_path": provenance[_article_id(record)][
                        "relative_path"
                    ],
                    "sha256": provenance[_article_id(record)]["sha256"],
                    "fd_metadata": dict(
                        raw_reader.reads[
                            provenance[_article_id(record)]["relative_path"]
                        ].metadata
                    ),
                }
                for record in records
            ],
        }
        _seal(provenance_relocation, "relocation_hash")
        provenance_relocation_path = _write_json(
            target,
            "control/v2/provenance_relocation.json",
            provenance_relocation,
        )

        evidence: list[dict[str, Any]] = []
        ledgers: dict[str, list[dict[str, Any]]] = {}
        for attempt in ATTEMPTS:
            statuses = statuses_by_attempt[attempt]
            ledger = [
                {
                    "article_id": _article_id(record),
                    "attempt": attempt,
                    "doi": _normalized_doi(record),
                    "input_index_1based": ordinal,
                    "processed": statuses[_article_id(record)] == "downloaded",
                    "status": statuses[_article_id(record)],
                }
                for ordinal, record in enumerate(records, start=1)
            ]
            ledger_path = _write_jsonl(
                target,
                f"control/{attempt}/attempt_ledger.jsonl",
                ledger,
            )
            summary = {
                "schema_version": "1.0",
                "attempt": attempt,
                "coverage_exact": True,
                "coverage_records": partition.records,
                "input_records": partition.records,
                "input_sha256": partition.source_sha256,
                "ledger_sha256": target.sha256(ledger_path),
                "postfetch_exit": 0,
                "postfetch_log_sha256": _sha256_bytes(
                    raw_bytes[f"control/{attempt}/postfetch.log"]
                ),
                "processed_records": sum(
                    item["processed"] for item in ledger
                ),
                "processed_sha256": _sha256_bytes(
                    raw_bytes[f"control/{attempt}/processed.txt"]
                ),
                "processed_unique": sum(item["processed"] for item in ledger),
                "skipped_records": sum(
                    not item["processed"] for item in ledger
                ),
                "skipped_sha256": _sha256_bytes(
                    raw_bytes[f"control/{attempt}/_skipped.txt"]
                ),
                "statuses": _closed_status_counts(
                    item["status"] for item in ledger
                ),
            }
            _seal(summary, "summary_hash")
            summary_path = _write_json(
                target,
                f"control/{attempt}/attempt_summary.json",
                summary,
            )
            evidence.append(
                {
                    **summary,
                    "summary_file_sha256": target.sha256(summary_path),
                }
            )
            ledgers[attempt] = ledger

        coverage = {
            "schema_version": "2.0",
            "all_coverage_exact": True,
            "all_inputs_exact_byte_equal": True,
            "attempt_count": 3,
            "records_per_attempt": partition.records,
            "total_ledger_records": 3 * partition.records,
            "attempts": evidence,
        }
        _seal(coverage, "summary_hash")
        coverage_path = _write_json(target, "control/attempt_coverage.json", coverage)

        terminal_rows: list[dict[str, Any]] = []
        for ordinal, record in enumerate(records, start=1):
            article_id = _article_id(record)
            final_provenance = provenance[article_id]
            terminal_rows.append(
                {
                    "article_id": article_id,
                    "doi": _normalized_doi(record),
                    "input_index_1based": ordinal,
                    "provenance_path": final_provenance["relative_path"],
                    "provenance_sha256": final_provenance["sha256"],
                    "acquisition_disposition": final_provenance[
                        "acquisition_disposition"
                    ],
                    "source_classification_state": final_provenance[
                        "source_classification_state"
                    ],
                    "rounds": {
                        attempt: statuses_by_attempt[attempt][article_id]
                        for attempt in ATTEMPTS
                    },
                    "terminal_status": statuses_by_attempt["retry2"][article_id],
                }
            )
        terminal_path = _write_jsonl(
            target,
            "control/terminal_outcomes.jsonl",
            terminal_rows,
        )
        terminal_counts = _closed_status_counts(
            row["terminal_status"] for row in terminal_rows
        )
        source_bearing_terminal_records = sum(
            provenance[row["article_id"]]["source_evidence"] is not None
            for row in terminal_rows
        )
        cleanup = {"schema_version": "1.0", "records": []}
        cleanup_path = _write_json(target, "control/terminal_cleanup.json", cleanup)
        retry_policy = {
            "schema_version": "3.0",
            "fixed_terminal_retry_rounds": 2,
            "frozen_accepted_sha256": partition.source_sha256,
            "outcome_independent": True,
            "selection_rule": binding["selection_rule"],
            "attempt_coverage_sha256": target.sha256(coverage_path),
            "attempt_coverage_summary_hash": coverage["summary_hash"],
            "terminal_outcomes_sha256": target.sha256(terminal_path),
            "terminal_counts": terminal_counts,
        }
        _seal(retry_policy, "policy_hash")
        retry_path = _write_json(target, "control/retry_policy.json", retry_policy)
        terminal_summary = {
            "schema_version": "3.0",
            "accepted_count": partition.records,
            "accepted_sha256": partition.source_sha256,
            "attempt_count": 3,
            "attempt_coverage_exact": True,
            "attempt_coverage_sha256": target.sha256(coverage_path),
            "provenance_records": partition.records,
            "terminal": terminal_counts,
            "source_classification": {
                "source_bearing_terminal_records": source_bearing_terminal_records,
                "source_less_terminal_records": (
                    partition.records - source_bearing_terminal_records
                ),
                "empty_chain_authorized": source_bearing_terminal_records == 0,
            },
            "terminal_outcomes_sha256": target.sha256(terminal_path),
            "terminal_cleanup_sha256": target.sha256(cleanup_path),
            "retry_policy_sha256": target.sha256(retry_path),
        }
        _seal(terminal_summary, "summary_hash")
        terminal_summary_path = _write_json(
            target,
            "control/postfetch_terminal_summary.json",
            terminal_summary,
        )

        source_bearing_rows = [
            {
                "article_id": row["article_id"],
                "doi": row["doi"],
                "provenance_path": row["provenance_path"],
                "provenance_sha256": row["provenance_sha256"],
                "source_evidence": {
                    "descriptor_path": provenance[row["article_id"]][
                        "source_evidence"
                    ].descriptor_path,
                    "descriptor_sha256": provenance[row["article_id"]][
                        "source_evidence"
                    ].descriptor_sha256,
                    "descriptor_bytes": provenance[row["article_id"]][
                        "source_evidence"
                    ].descriptor_bytes,
                    "source_paths": list(
                        provenance[row["article_id"]]["source_evidence"].source_paths
                    ),
                },
            }
            for row in terminal_rows
            if provenance[row["article_id"]]["source_evidence"] is not None
        ]
        source_extension: Any | None = None
        if source_bearing_rows and not source_bearing_v2:
            blocked = {
                "schema_version": "c2-source-classification-block-v1",
                "status": "NOT_SEALABLE_SOURCE_CLASSIFICATION_BUILDER_REQUIRED",
                "reason": (
                    "source-bearing terminal records require a separately approved "
                    "deterministic canonical/P builder; empty canonical/P evidence "
                    "is forbidden"
                ),
                "terminal_outcomes_sha256": target.sha256(terminal_path),
                "terminal_summary_sha256": target.sha256(terminal_summary_path),
                "source_bearing_rows": source_bearing_rows,
            }
            _seal(blocked, "block_hash")
            _write_json(target, "control/source_classification_blocked.json", blocked)
            raise C2RemediationError(
                "source-bearing terminal rows require an approved canonical/P builder"
            )
        if source_bearing_rows:
            try:
                source_extension = build_source_bearing_extension(
                    root=target,
                    raw_reader=raw_reader,
                    records=records,
                    provenance=provenance,
                    terminal_rows=terminal_rows,
                    partition_records=partition.records,
                    source_chunk_sha256=partition.source_sha256,
                )
            except SourceBearingExtensionError as exc:
                blocked = {
                    "schema_version": "c2-source-classification-block-v1",
                    "status": "NOT_SEALABLE_SOURCE_EXTENSION_STAGEB_POLICY_REQUIRED",
                    "reason": (
                        "source-bearing V2 execution requires a compile-pinned, "
                        "package-internal Stage-B production policy/code-registry "
                        "commitment; candidate worktree and manifest identities "
                        "cannot supply one"
                    ),
                    "terminal_outcomes_sha256": target.sha256(terminal_path),
                    "terminal_summary_sha256": target.sha256(terminal_summary_path),
                    "source_bearing_rows": source_bearing_rows,
                }
                _seal(blocked, "block_hash")
                _write_json(target, "control/source_classification_blocked.json", blocked)
                raise C2RemediationError(
                    "source-bearing V2 canonical/P construction is not sealable: "
                    f"{exc}"
                ) from exc

        manifest_rows = [
            {
                "doi": _normalized_doi(record),
                "article_id": _article_id(record),
                "terminal_status": statuses_by_attempt["retry2"][_article_id(record)],
                "provenance_path": provenance[_article_id(record)]["relative_path"],
                "provenance_sha256": provenance[_article_id(record)]["sha256"],
                "policy_accepted": record["policy_accepted"],
                "download_eligible": record["download_eligible"],
                "journal_allowed": record["journal_allowed"],
                "require_cc_by": record["require_cc_by"],
                "license_id": record["license"]["license_id"],
                "content_version": record["license"]["content_version"],
                "license_normalized_url": record["license"]["normalized_url"],
            }
            for record in records
        ]
        manifest_path = _write_jsonl(
            target,
            "manifest_v1/corpus_manifest.jsonl",
            manifest_rows,
        )
        manifest_summary = {
            "schema_version": "2.0",
            "articles": partition.records,
            "valid": partition.records,
            "invalid": 0,
            "unique_dois": partition.records,
            "strict_vor_ccby": True,
            "cc_by_vor_valid": partition.records,
            "manifest_sha256": target.sha256(manifest_path),
        }
        _write_json(target, "manifest_v1/manifest_summary.json", manifest_summary)
        _write_jsonl(
            target,
            "manifest_v1/manifest_validation.jsonl",
            (
                {
                    "doi": row["doi"],
                    "valid": True,
                    "errors": [],
                    "checks": {
                        "cc_by_4_0": True,
                        "version_of_record": True,
                        "policy_accepted": True,
                    },
                }
                for row in manifest_rows
            ),
        )

        if source_extension is None:
            empty_sha = _sha256_bytes(b"")
            _write_bytes(target, "cases_v1/candidates.jsonl", b"")
            _write_bytes(target, "cases_v1/ambiguous.jsonl", b"")
            cases = {
                "schema_version": "2.0",
                "articles_total": partition.records,
                "candidates": 0,
                "ambiguous": 0,
                "eligible_for_experiment": 0,
                "llm_calls": 0,
                "candidates_sha256": empty_sha,
                "ambiguous_sha256": empty_sha,
                "reason": "fresh remediation empty-case policy; no model-generated case selection",
            }
            _seal(cases, "summary_hash")
            cases_path = _write_json(target, "cases_v1/summary.json", cases)
            _write_bytes(target, "proposals_v1/proposed.jsonl", b"")
            _write_bytes(target, "proposals_v1/rejected.jsonl", b"")
            proposals = {
                "schema_version": "2.0",
                "input_candidates_sha256": empty_sha,
                "input_count": 0,
                "proposals_total": 0,
                "single_proposals": 0,
                "multi_panel_proposals": 0,
                "rejected": 0,
                "eligible_for_experiment": 0,
                "llm_calls": 0,
            }
            _seal(proposals, "summary_hash")
            proposals_path = _write_json(target, "proposals_v1/summary.json", proposals)
            review = {
                "schema_version": "2.0",
                "status": "SKIPPED_NO_PROPOSALS",
                "proposals": 0,
                "reviews": 0,
                "models_invoked": [],
                "secret_hits": [],
                "api_or_infra_failures": [],
            }
            _seal(review, "summary_hash")
            review_path = _write_json(target, "control/review_validation.json", review)
            canonical = {
                "schema_version": "2.0",
                "status": "CANONICAL_EMPTY_NO_PROPOSALS",
                "input_proposals_sha256": empty_sha,
                "input_candidates_sha256": empty_sha,
                "input_count": 0,
                "accepted_single": 0,
                "accepted_multi": 0,
                "derivation_run": False,
                "reason": "zero deterministic proposals",
            }
            _seal(canonical, "summary_hash")
            canonical_path = _write_json(
                target,
                "canonical_v1/canonical_summary.json",
                canonical,
            )
            p_summary = {
                "schema_version": "2.0",
                "status": "CANONICAL_EMPTY_NO_PROPOSALS",
                "chunk": int(chunk_id),
                "input_proposals": 0,
                "canonical_summary_sha256": target.sha256(canonical_path),
                "canonical_summary_hash": canonical["summary_hash"],
                "reason": "no canonical accepted cases",
            }
            _seal(p_summary, "summary_hash")
            p_path = _write_json(target, "p_evidence_v1/p_summary.json", p_summary)
            benchmark = {
                "schema_version": "2.0",
                "status": "NOT_EMITTED_NO_REVIEWED_CASES",
                "assembly_run": False,
                "eligible_cases": 0,
                "canonical_summary_sha256": target.sha256(canonical_path),
                "p_summary_sha256": target.sha256(p_path),
            }
            _write_json(target, "control/sealed_benchmark_status.json", benchmark)
        else:
            _write_json(
                target,
                "control/sealed_benchmark_status.json",
                {
                    "schema_version": "c2-source-bearing-benchmark-status-v2",
                    "status": "NOT_EMITTED_SOURCE_CLASSIFICATION_ONLY",
                    "assembly_run": False,
                    "eligible_cases": source_extension.canonical["case_count"],
                    "source_bearing_extension_validation_sha256": (
                        source_extension.extension_validation["sha256"]
                    ),
                },
            )

        preservation_after = (
            None
            if protected_contract is None
            else _verify_protected_old_root(
                normalized_target.parent,
                protected_contract,
            )
        )
        preservation_ledger = _preservation_ledger(
            chunk_id,
            before=preservation_before,
            after=preservation_after,
        )
        preservation_ledger_path = _write_json(
            target,
            "control/preservation_ledger.json",
            preservation_ledger,
        )
        secret_scan_path, secret_scan = _write_secret_scan(target)
        postfinal_code = _verify_worktree(worktree)
        _require(
            postfinal_code == code,
            "clean frozen code binding changed during finalization",
        )
        source_extension_replay: Mapping[str, Any] | None = None
        if source_extension is not None:
            try:
                source_extension_replay = validate_source_bearing_extension(
                    target,
                    partition_records=partition.records,
                    source_chunk_sha256=partition.source_sha256,
                )
                stored_extension_validation = _read_json_bytes(
                    target.read_bytes(
                        "control/v2/source_bearing_extension_validation.json"
                    ),
                    "source-bearing extension validation",
                )
                _require(
                    stored_extension_validation == source_extension_replay,
                    "source-bearing extension replay differs from staged validation",
                )
            except SourceBearingExtensionError as exc:
                raise C2RemediationError(
                    "source-bearing extension failed prepublication replay: "
                    f"{exc}"
                ) from exc
        preseal = {
            "schema_version": "2.0",
            "status": "PASS",
            "gates": {
                "all_attempts_exact": True,
                "all_terminal_outcomes_exact": True,
                "clean_frozen_code_binding": True,
                "corpus_manifest_valid": True,
                "no_model_calls": True,
                "secret_scan_clean": True,
                "tests_passed": True,
                "private_trusted_staging_parent": True,
            },
            "execution_evidence_sha256": _sha256_bytes(
                raw_bytes["control/execution_evidence.json"]
            ),
            "execution_evidence_hash": execution_evidence["evidence_hash"],
            "test_results": execution_evidence["tests"],
            "secret_scan_status": execution_evidence["secret_scan_status"],
            "generated_secret_scan_sha256": target.sha256(secret_scan_path),
            "generated_secret_scan_hash": secret_scan["scan_hash"],
            "code_before": code,
            "code_after": postfinal_code,
        }
        if source_extension_replay is not None:
            preseal["gates"]["source_bearing_extension_replayed"] = True
            preseal["source_bearing_extension_validation_hash"] = (
                source_extension_replay["validation_hash"]
            )
        _seal(preseal, "validation_hash")
        preseal_path = _write_json(target, "control/preseal_validation.json", preseal)
        target.verify_staging_identity()

        inventory_excludes = {
            ".pipeline_worktree/",
            "control/root_inventory.json",
            "sealed_report_v1/",
        }
        inventory_entries = _target_artifact_entries(target, inventory_excludes)
        inventory = {
            "schema_version": "1.0",
            "root_name": target.name,
            "artifact_count": len(inventory_entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in inventory_entries),
            "excludes": sorted(inventory_excludes),
            "files": inventory_entries,
        }
        _seal(inventory, "inventory_hash")
        inventory_path = _write_json(target, "control/root_inventory.json", inventory)

        manifest_excludes = {".pipeline_worktree/", "sealed_report_v1/"}
        artifact_entries = _target_artifact_entries(target, manifest_excludes)
        artifact_manifest = {
            "schema_version": "1.0",
            "root_name": target.name,
            "artifact_count": len(artifact_entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in artifact_entries),
            "excludes": sorted(manifest_excludes),
            "files": artifact_entries,
        }
        _seal(artifact_manifest, "manifest_hash")
        artifact_manifest_path = _write_json(
            target,
            "sealed_report_v1/artifact_manifest.json",
            artifact_manifest,
        )
        _write_bytes(
            target,
            "sealed_report_v1/artifact_manifest.sha256",
            f"{target.sha256(artifact_manifest_path)}  artifact_manifest.json\n".encode(
                "utf-8"
            ),
        )
        target.verify_staging_identity()

        if source_extension is None:
            report_status = "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES"
            report_cases: Mapping[str, Any] = {
                "summary_file_sha256": target.sha256(cases_path),
                "summary_hash": cases["summary_hash"],
                "candidates": 0,
                "ambiguous": 0,
            }
            report_proposals: Mapping[str, Any] = {
                "summary_file_sha256": target.sha256(proposals_path),
                "summary_hash": proposals["summary_hash"],
                "proposals": 0,
                "rejected": 0,
            }
            report_canonical: Mapping[str, Any] = {
                "summary_file_sha256": target.sha256(canonical_path),
                "summary_hash": canonical["summary_hash"],
                "status": canonical["status"],
            }
            report_p: Mapping[str, Any] = {
                "summary_file_sha256": target.sha256(p_path),
                "summary_hash": p_summary["summary_hash"],
                "status": p_summary["status"],
            }
            report_review: Mapping[str, Any] = {
                "validation_file_sha256": target.sha256(review_path),
                "summary_hash": review["summary_hash"],
                "status": review["status"],
            }
            source_extension_trust: Mapping[str, Any] = {}
        else:
            report_status = "SEALED_COMPLETE_ATTEMPT_EVIDENCE_SOURCE_V2"
            report_cases = dict(source_extension.cases)
            report_proposals = dict(source_extension.proposals)
            report_canonical = dict(source_extension.canonical)
            report_p = dict(source_extension.p_evidence)
            report_review = dict(source_extension.review)
            source_extension_trust = {
                "source_bearing_extension_validation": dict(
                    source_extension.extension_validation
                )
            }

        report = {
            "sealed_report_version": "3.0",
            "status": report_status,
            "root": str(target.path),
            "publication": {
                "method": (
                    "descriptor-relative atomic no-replace directory rename "
                    "(renameatx_np RENAME_EXCL or renameat2 RENAME_NOREPLACE)"
                ),
                "canonical_target_identity_postpublication_check": "REQUIRED",
                "staging_parent": (
                    "pre-existing descriptor-validated owner/ACL-safe parent"
                ),
                "threat_boundary": (
                    "protects against racing local principals other than the "
                    "staging-parent owner; malicious same-EUID filesystem control "
                    "is out of scope"
                ),
            },
            "code": code,
            "code_after_finalization": postfinal_code,
            "frozen_input": {
                "chunk": int(chunk_id),
                "records": partition.records,
                "sha256": partition.source_sha256,
                "universe_sha256": FROZEN_UNIVERSE_SHA256,
                "freeze_summary_sha256": FROZEN_FREEZE_SUMMARY_SHA256,
                "freeze_summary_hash": FROZEN_FREEZE_SUMMARY_HASH,
            },
            "policy": {
                "attempts": list(ATTEMPTS),
                "records_per_attempt": partition.records,
                "total_attempt_records": 3 * partition.records,
                "no_early_stop": True,
                "no_outcome_selection": True,
                "outcome_independent": True,
                "no_model_calls": True,
            },
            "preservation": {
                "ledger_file_sha256": target.sha256(preservation_ledger_path),
                "ledger_hash": preservation_ledger["ledger_hash"],
                "protected_old_root_contracts": len(
                    preservation_ledger["old_root_contracts"]
                ),
                "retained_candidate_exclusions": list(
                    RETAINED_CANDIDATE_EXCLUSIONS
                ),
                "verification_requirement": preservation_ledger[
                    "verification_requirement"
                ],
            },
            "attempt_evidence": {
                "coverage_exact": True,
                "coverage_file_sha256": target.sha256(coverage_path),
                "coverage_summary_hash": coverage["summary_hash"],
            },
            "postfetch": {
                "terminal_outcomes_sha256": target.sha256(terminal_path),
                "summary_file_sha256": target.sha256(terminal_summary_path),
                "summary_hash": terminal_summary["summary_hash"],
                "terminal_counts": terminal_counts,
                "provenance_records": partition.records,
                "provenance_relocation_sha256": target.sha256(
                    provenance_relocation_path
                ),
                "provenance_relocation_hash": provenance_relocation[
                    "relocation_hash"
                ],
            },
            "corpus_manifest": manifest_summary,
            "cases": report_cases,
            "proposals": report_proposals,
            "canonical": report_canonical,
            "p_strata": report_p,
            "review": report_review,
            "trust_chain": {
                "raw_source_manifest_sha256": target.sha256(
                    "control/v2/raw_source_manifest.json"
                ),
                "raw_attempt_attestation_sha256": target.sha256(
                    "control/v2/raw_attempt_attestation.json"
                ),
                "provenance_relocation_sha256": target.sha256(
                    provenance_relocation_path
                ),
                "execution_evidence_sha256": _sha256_bytes(
                    raw_bytes["control/execution_evidence.json"]
                ),
                "execution_evidence_hash": execution_evidence["evidence_hash"],
                "secret_scan_sha256": target.sha256(secret_scan_path),
                "secret_scan_hash": secret_scan["scan_hash"],
                "preseal_validation_sha256": target.sha256(preseal_path),
                "preseal_validation_hash": preseal["validation_hash"],
                "root_inventory_path": "control/root_inventory.json",
                "root_inventory_sha256": target.sha256(inventory_path),
                "root_inventory_hash": inventory["inventory_hash"],
                "artifact_manifest_path": "sealed_report_v1/artifact_manifest.json",
                "artifact_manifest_file_sha256": target.sha256(
                    artifact_manifest_path
                ),
                "artifact_manifest_hash": artifact_manifest["manifest_hash"],
                "artifact_hashes": {
                    entry["path"]: entry["sha256"] for entry in artifact_entries
                },
                **source_extension_trust,
            },
        }
        _seal(report, "report_hash")
        report_path = _write_json(target, "sealed_report_v1/sealed_report.json", report)
        _write_bytes(
            target,
            "sealed_report_v1/sealed_report.sha256",
            f"{target.sha256(report_path)}  sealed_report.json\n".encode("utf-8"),
        )
        postseal_code = _verify_worktree(worktree)
        _require(
            postseal_code == code,
            "clean frozen code binding changed after report publication",
        )
        postseal_after = (
            None
            if protected_contract is None
            else _verify_protected_old_root(
                normalized_target.parent,
                protected_contract,
            )
        )
        _require(
            preservation_before is None or preservation_before == postseal_after,
            "protected old root changed after sealed-report publication",
        )
        postseal_preservation = {
            "schema_version": "c2-remediation-postseal-preservation-v1",
            "status": (
                "NOT_APPLICABLE_NO_PROTECTED_ROOT"
                if preservation_before is None
                else "PASS_UNCHANGED"
            ),
            "before": preservation_before,
            "after": postseal_after,
        }
        _seal(postseal_preservation, "verification_hash")
        postseal_preservation_path = _write_json(
            target,
            "sealed_report_v1/postseal_preservation.json",
            postseal_preservation,
        )
        validation = {
            "schema_version": "2.0",
            "status": "PASS",
            "all_gates_pass": True,
            "sealed_report_file_sha256": target.sha256(report_path),
            "sealed_report_hash": report["report_hash"],
            "code_after_sealed_report": postseal_code,
            "postseal_preservation_sha256": target.sha256(
                postseal_preservation_path
            ),
            "postseal_preservation_hash": postseal_preservation[
                "verification_hash"
            ],
            "gates": {
                "source_bytes": True,
                "clean_code_binding": True,
                "all_attempts_exact": True,
                "all_terminal_outcomes_exact": True,
                "relative_provenance_paths": True,
                "descriptor_bound_provenance_relocation": True,
                "no_early_or_partial_selection": True,
                "root_inventory": True,
                "artifact_manifest": True,
                "cross_links": True,
                "execution_evidence": True,
                "generated_secret_scan": True,
                "postseal_old_root_preservation": True,
                "atomic_no_replace_publication": True,
                "canonical_target_identity": True,
                "private_trusted_staging_parent": True,
            },
        }
        if source_extension is None:
            validation["gates"]["empty_case_chain"] = True
        else:
            validation["gates"]["source_bearing_v2_chain"] = True
            validation["source_bearing_extension_validation_sha256"] = (
                source_extension.extension_validation["sha256"]
            )
            validation["source_bearing_extension_validation_hash"] = (
                source_extension.extension_validation["validation_hash"]
            )
        _seal(validation, "validation_hash")
        validation_path = _write_json(
            target,
            "sealed_report_v1/validation.json",
            validation,
        )

        _require(
            target.sha256(report_path) == validation["sealed_report_file_sha256"],
            "generated report checksum mismatch",
        )
        _require(target.read_bytes(validation_path), "generated validation is missing")
        # All extension and report artifacts are complete at this point.  The
        # only remaining state transition is the native no-replace publication.
        # Replay the source-bearing chain again after every final staging write;
        # all subsequent operations are descriptor-rooted reads/checks.
        if source_extension is not None:
            try:
                prepublish_extension_replay = validate_source_bearing_extension(
                    target,
                    partition_records=partition.records,
                    source_chunk_sha256=partition.source_sha256,
                )
                _require(
                    prepublish_extension_replay == source_extension_replay
                    and prepublish_extension_replay
                    == _read_json_bytes(
                        target.read_bytes(
                            "control/v2/source_bearing_extension_validation.json"
                        ),
                        "source-bearing extension validation",
                    ),
                    "source-bearing extension changed before publication",
                )
            except SourceBearingExtensionError as exc:
                raise C2RemediationError(
                    "source-bearing extension failed immediate prepublication replay: "
                    f"{exc}"
                ) from exc
        target.verify_staging_identity()
        target.publish()
        target.verify_published_identity()
        return {
            "chunk_id": chunk_id,
            "target_root": str(target.path),
            "input_total": partition.records,
            "sealed_report_sha256": target.sha256(report_path),
            "report_hash": report["report_hash"],
            "status": report["status"],
        }
    except Exception:
        # Never clean up an incomplete staging/published root by pathname.
        # A later invocation rejects any canonical target rather than overwriting it.
        raise
    finally:
        if target is not None:
            target.close()
        raw_reader.close()

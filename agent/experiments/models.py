from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


SCHEMA_VERSION = "2.0"
RECORD_FILENAME = "run_record.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_STATUSES = {"running", "completed", "failed"}
_BUDGET_TYPES = {"renders", "wall_clock_seconds"}
_SECURE_OUTPUT_DIR_FD_SUPPORTED = all(
    operation in os.supports_dir_fd
    for operation in (os.open, os.mkdir, os.rename, os.stat, os.unlink)
)
_DARWIN_ACL_TYPE_EXTENDED = 0x00000100
_DARWIN_ACL_FIRST_ENTRY = 0
_DARWIN_ACL_NEXT_ENTRY = -1
_DARWIN_ACL_EXTENDED_ALLOW = 1
_DARWIN_ACL_EXTENDED_DENY = 2
_DARWIN_MUTATING_ACL_PERMISSIONS = (
    (1 << 2)  # ACL_WRITE_DATA / ACL_ADD_FILE
    | (1 << 4)  # ACL_DELETE
    | (1 << 5)  # ACL_APPEND_DATA / ACL_ADD_SUBDIRECTORY
    | (1 << 6)  # ACL_DELETE_CHILD
    | (1 << 8)  # ACL_WRITE_ATTRIBUTES
    | (1 << 10)  # ACL_WRITE_EXTATTRIBUTES
    | (1 << 12)  # ACL_WRITE_SECURITY
    | (1 << 13)  # ACL_CHANGE_OWNER
)
_DARWIN_NON_MUTATING_ACL_PERMISSIONS = (
    (1 << 1)  # ACL_READ_DATA / ACL_LIST_DIRECTORY
    | (1 << 3)  # ACL_EXECUTE / ACL_SEARCH
    | (1 << 7)  # ACL_READ_ATTRIBUTES
    | (1 << 9)  # ACL_READ_EXTATTRIBUTES
    | (1 << 11)  # ACL_READ_SECURITY
    | (1 << 20)  # ACL_SYNCHRONIZE
)
_ACL_ABSENT_ERRNOS = frozenset(
    code
    for code in (
        errno.ENOENT,
        getattr(errno, "ENODATA", None),
        getattr(errno, "ENOATTR", None),
    )
    if code is not None
)


class ProvenanceError(ValueError):
    """Raised when a run cannot establish or verify its provenance."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProvenanceError(f"{name} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ProvenanceError(f"{name} must include a timezone")
    return parsed


def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def normalize_output_path(path: Path) -> Path:
    try:
        expanded = path.expanduser()
        absolute = Path(os.path.abspath(os.fspath(expanded)))
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(f"Cannot resolve output path: {path}") from exc
    if absolute.name in {"", ".", ".."}:
        raise ProvenanceError(f"Output path must name a file: {path}")
    if absolute.is_symlink():
        raise ProvenanceError(f"Leaf output symlinks are forbidden: {path}")
    try:
        return absolute.parent.resolve(strict=False) / absolute.name
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(f"Cannot resolve output parent: {path}") from exc


def normalize_trusted_output_path(path: Path) -> Path:
    """Normalize lexically without following a parent symlink."""

    try:
        expanded = path.expanduser()
        normalized = Path(os.path.abspath(os.fspath(expanded)))
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(f"Cannot resolve output path: {path}") from exc
    if normalized.name in {"", ".", ".."}:
        raise ProvenanceError(f"Output path must name a file: {path}")
    if normalized.is_symlink():
        raise ProvenanceError(f"Leaf output symlinks are forbidden: {path}")
    return normalized


@dataclass
class SecureOutputTarget:
    """Descriptor-anchored output location that never reopens its parent path."""

    final_path: Path
    leaf_name: str
    parent_fd: int
    published_device: Optional[int] = None
    published_inode: Optional[int] = None
    trusted_parent: bool = False

    def close(self) -> None:
        if self.parent_fd != -1:
            try:
                os.close(self.parent_fd)
            finally:
                self.parent_fd = -1


@dataclass
class TrustedDirectory:
    """A trusted directory held by descriptor rather than a reopenable pathname."""

    path: Path
    descriptor: int

    def close(self) -> None:
        if self.descriptor != -1:
            try:
                os.close(self.descriptor)
            finally:
                self.descriptor = -1


def _require_secure_output_primitives() -> None:
    if (
        not _SECURE_OUTPUT_DIR_FD_SUPPORTED
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise ProvenanceError(
            "Secure descriptor-relative output writes are unsupported on this platform"
        )


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _darwin_acl_entries(descriptor: int) -> tuple[tuple[int, int], ...]:
    """Read Darwin extended ACL entries from an already-open directory."""

    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        acl_get_fd_np = library.acl_get_fd_np
        acl_get_entry = library.acl_get_entry
        acl_get_tag_type = library.acl_get_tag_type
        acl_get_permset_mask_np = library.acl_get_permset_mask_np
        acl_free = library.acl_free
    except (AttributeError, OSError) as exc:
        raise ProvenanceError("Cannot inspect Darwin ACL metadata") from exc

    acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    acl_get_fd_np.restype = ctypes.c_void_p
    acl_get_entry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    acl_get_entry.restype = ctypes.c_int
    acl_get_tag_type.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    acl_get_tag_type.restype = ctypes.c_int
    acl_get_permset_mask_np.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    acl_get_permset_mask_np.restype = ctypes.c_int
    acl_free.argtypes = [ctypes.c_void_p]
    acl_free.restype = ctypes.c_int

    ctypes.set_errno(0)
    acl = acl_get_fd_np(descriptor, _DARWIN_ACL_TYPE_EXTENDED)
    if not acl:
        error = ctypes.get_errno()
        if error in _ACL_ABSENT_ERRNOS:
            return ()
        raise ProvenanceError("Cannot inspect Darwin ACL metadata")

    try:
        entries: list[tuple[int, int]] = []
        entry_id = _DARWIN_ACL_FIRST_ENTRY
        while True:
            entry = ctypes.c_void_p()
            ctypes.set_errno(0)
            result = acl_get_entry(acl, entry_id, ctypes.byref(entry))
            error = ctypes.get_errno()
            if result in {0, 1} and entry.value:
                tag = ctypes.c_int()
                permissions = ctypes.c_uint64()
                if acl_get_tag_type(entry, ctypes.byref(tag)) != 0:
                    raise ProvenanceError("Cannot inspect Darwin ACL metadata")
                if (
                    acl_get_permset_mask_np(entry, ctypes.byref(permissions))
                    != 0
                ):
                    raise ProvenanceError("Cannot inspect Darwin ACL metadata")
                entries.append((tag.value, permissions.value))
                entry_id = _DARWIN_ACL_NEXT_ENTRY
                continue
            if result == -1 and error == errno.EINVAL and entries:
                return tuple(entries)
            raise ProvenanceError("Cannot inspect Darwin ACL metadata")
    finally:
        acl_free(acl)


def _darwin_acl_allows_mutation(descriptor: int) -> bool:
    for tag, permissions in _darwin_acl_entries(descriptor):
        if tag == _DARWIN_ACL_EXTENDED_ALLOW:
            if permissions & _DARWIN_MUTATING_ACL_PERMISSIONS:
                return True
            if permissions & ~_DARWIN_NON_MUTATING_ACL_PERMISSIONS:
                return True
        elif tag != _DARWIN_ACL_EXTENDED_DENY:
            raise ProvenanceError("Cannot classify Darwin ACL metadata")
    return False


def _linux_acl_allows_mutation(descriptor: int) -> bool:
    if not hasattr(os, "getxattr") or os.getxattr not in os.supports_fd:
        raise ProvenanceError("Cannot inspect POSIX ACL metadata")
    for name in ("system.posix_acl_access", "system.posix_acl_default"):
        try:
            os.getxattr(descriptor, name)
        except OSError as exc:
            if exc.errno in _ACL_ABSENT_ERRNOS:
                continue
            raise ProvenanceError("Cannot inspect POSIX ACL metadata") from exc
        return True
    return False


def _trusted_acl_allows_foreign_mutation(descriptor: int) -> bool:
    if sys.platform == "darwin":
        return _darwin_acl_allows_mutation(descriptor)
    if sys.platform.startswith("linux"):
        return _linux_acl_allows_mutation(descriptor)
    raise ProvenanceError("Trusted output ACL inspection is unsupported")


def _validate_trusted_directory(descriptor: int, path: Path) -> None:
    if not hasattr(os, "geteuid"):
        raise ProvenanceError("Trusted output parents require an effective uid")
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ProvenanceError(f"Trusted output parent is not a directory: {path}")
    if metadata.st_uid not in {0, os.geteuid()}:
        raise ProvenanceError(f"Trusted output parent has an unsafe owner: {path}")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ProvenanceError(
            f"Trusted output parent is group/world writable: {path}"
        )
    if _trusted_acl_allows_foreign_mutation(descriptor):
        raise ProvenanceError(
            f"Trusted output parent has a mutating ACL: {path}"
        )


def _open_secure_parent(
    parent_path: Path,
    *,
    create: bool = True,
    require_trusted_chain: bool = False,
) -> int:
    _require_secure_output_primitives()
    if not parent_path.is_absolute():
        raise ProvenanceError("Secure output parent must be absolute")

    descriptor = os.open("/", _directory_open_flags())
    current_path = Path("/")
    try:
        if require_trusted_chain:
            _validate_trusted_directory(descriptor, current_path)
        for component in parent_path.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise ProvenanceError(
                        f"Trusted output parent is missing: {parent_path}"
                    )
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
            current_path = current_path / component
            if require_trusted_chain:
                _validate_trusted_directory(descriptor, current_path)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def open_trusted_directory(path: Path) -> TrustedDirectory:
    """Open an existing absolute directory through a trusted no-follow chain."""

    try:
        absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(f"Cannot resolve trusted directory: {path}") from exc
    if not absolute.is_absolute():
        raise ProvenanceError("Trusted directory path must be absolute")
    try:
        descriptor = _open_secure_parent(
            absolute,
            create=False,
            require_trusted_chain=True,
        )
    except OSError as exc:
        raise ProvenanceError(f"Cannot securely open trusted directory: {absolute}") from exc
    return TrustedDirectory(path=absolute, descriptor=descriptor)


def verify_trusted_directory(directory: TrustedDirectory) -> None:
    """Reject if a visible trusted directory name no longer denotes its descriptor."""

    if directory.descriptor == -1:
        raise ProvenanceError("Trusted directory is already closed")
    anchored = os.fstat(directory.descriptor)
    try:
        visible_descriptor = _open_secure_parent(
            directory.path,
            create=False,
            require_trusted_chain=True,
        )
    except OSError as exc:
        raise ProvenanceError(
            f"Trusted directory changed after opening: {directory.path}"
        ) from exc
    try:
        visible = os.fstat(visible_descriptor)
        if (
            visible.st_dev != anchored.st_dev
            or visible.st_ino != anchored.st_ino
        ):
            raise ProvenanceError(
                f"Trusted directory changed after opening: {directory.path}"
            )
    finally:
        os.close(visible_descriptor)


def open_secure_output_target(
    path: Path,
    *,
    normalized_path: bool = False,
    require_trusted_parent: bool = False,
) -> SecureOutputTarget:
    """Open the output parent without following any directory or leaf symlink."""

    final_path = path if normalized_path else normalize_output_path(path)
    if not final_path.is_absolute():
        raise ProvenanceError("Normalized output path must be absolute")
    try:
        parent_fd = _open_secure_parent(
            final_path.parent,
            create=not require_trusted_parent,
            require_trusted_chain=require_trusted_parent,
        )
    except OSError as exc:
        raise ProvenanceError(
            f"Cannot securely open output parent: {final_path.parent}"
        ) from exc
    try:
        try:
            leaf = os.stat(
                final_path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            leaf = None
        if leaf is not None and stat.S_ISLNK(leaf.st_mode):
            raise ProvenanceError(
                f"Leaf output symlinks are forbidden: {final_path}"
            )
        return SecureOutputTarget(
            final_path=final_path,
            leaf_name=final_path.name,
            parent_fd=parent_fd,
            trusted_parent=require_trusted_parent,
        )
    except Exception:
        os.close(parent_fd)
        raise


def verify_secure_output_target(target: SecureOutputTarget) -> None:
    """Verify that the user-visible output still names the anchored publication."""

    if target.parent_fd == -1:
        raise ProvenanceError("Secure output target is already closed")
    if target.published_device is None or target.published_inode is None:
        raise ProvenanceError("Secure output has not been published")
    anchored_parent = os.fstat(target.parent_fd)
    try:
        anchored_leaf = os.stat(
            target.leaf_name,
            dir_fd=target.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise ProvenanceError(
            f"Final output leaf changed after publication: {target.final_path}"
        ) from exc
    if (
        stat.S_ISLNK(anchored_leaf.st_mode)
        or anchored_leaf.st_dev != target.published_device
        or anchored_leaf.st_ino != target.published_inode
    ):
        raise ProvenanceError(
            f"Final output leaf changed after publication: {target.final_path}"
        )
    try:
        visible_parent_fd = _open_secure_parent(
            target.final_path.parent,
            create=False,
            require_trusted_chain=target.trusted_parent,
        )
    except OSError as exc:
        raise ProvenanceError(
            f"Final output parent changed after publication: {target.final_path}"
        ) from exc
    try:
        visible_parent = os.fstat(visible_parent_fd)
        if (
            visible_parent.st_dev != anchored_parent.st_dev
            or visible_parent.st_ino != anchored_parent.st_ino
        ):
            raise ProvenanceError(
                f"Final output parent changed after publication: {target.final_path}"
            )
        try:
            visible_leaf = os.stat(
                target.leaf_name,
                dir_fd=visible_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ProvenanceError(
                f"Final output leaf changed after publication: {target.final_path}"
            ) from exc
        if (
            stat.S_ISLNK(visible_leaf.st_mode)
            or visible_leaf.st_dev != target.published_device
            or visible_leaf.st_ino != target.published_inode
        ):
            raise ProvenanceError(
                f"Final output leaf changed after publication: {target.final_path}"
            )
    finally:
        os.close(visible_parent_fd)


def slug_identifier(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip()).strip("-")
    if not slug:
        raise ProvenanceError(f"Cannot construct a slug from {value!r}")
    return slug


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_symlink():
        raise ProvenanceError(f"Symlink artifacts are not accepted: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise ProvenanceError(f"Artifact is neither a file nor directory: {path}")

    digest = hashlib.sha256()
    for child in sorted(resolved.rglob("*")):
        if child.is_symlink():
            raise ProvenanceError(f"Symlink artifacts are not accepted: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def verify_artifacts(record: "RunRecord", run_dir: Path) -> None:
    resolved_root = run_dir.resolve(strict=True)
    for name, relative_text in record.artifact_paths.items():
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProvenanceError(
                f"Artifact {name!r} has a non-portable path: {relative_text}"
            )
        candidate = resolved_root / relative
        if candidate.is_symlink():
            raise ProvenanceError(f"Artifact {name!r} is a symlink")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise ProvenanceError(
                f"Artifact {name!r} is missing or escaped the run directory"
            ) from exc
        actual = sha256_path(resolved)
        expected = record.artifact_hashes.get(name)
        if actual != expected:
            raise ProvenanceError(
                f"Artifact {name!r} failed its SHA-256 integrity check"
            )


def write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    normalized_path: bool = False,
) -> None:
    target = open_secure_output_target(path, normalized_path=normalized_path)
    try:
        write_json_atomic_to_target(target, payload)
    finally:
        target.close()


def write_json_atomic_to_target(
    target: SecureOutputTarget,
    payload: Mapping[str, Any],
) -> None:
    if target.parent_fd == -1:
        raise ProvenanceError("Secure output target is already closed")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    descriptor = -1
    temporary_name = ""
    try:
        for _ in range(128):
            temporary_name = (
                f".{target.leaf_name}.{secrets.token_hex(16)}.tmp"
            )
            try:
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=target.parent_fd,
                )
                break
            except FileExistsError:
                continue
        else:
            raise ProvenanceError("Cannot allocate secure output staging file")

        handle = os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            closefd=False,
        )
        with handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        published_leaf = os.fstat(descriptor)
        os.rename(
            temporary_name,
            target.leaf_name,
            src_dir_fd=target.parent_fd,
            dst_dir_fd=target.parent_fd,
        )
        target.published_device = published_leaf.st_dev
        target.published_inode = published_leaf.st_ino
        os.close(descriptor)
        descriptor = -1
        try:
            os.fsync(target.parent_fd)
        except OSError:
            pass
        verify_secure_output_target(target)
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        # A failed rename may have allowed another writer to reuse this name.
        # Never unlink by staging pathname after a failed publication attempt.


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read JSON provenance at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProvenanceError(f"Expected a JSON object at {path}")
    return data


def _strict_dataclass_kwargs(cls: type[Any], data: Mapping[str, Any]) -> Dict[str, Any]:
    known = {item.name for item in fields(cls)}
    extra = set(data) - known
    if extra:
        raise ProvenanceError(
            f"Unknown {cls.__name__} fields: {', '.join(sorted(extra))}"
        )
    return dict(data)


def _validate_json_object(name: str, value: Mapping[str, Any]) -> None:
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"{name} must be JSON serializable: {exc}") from exc


@dataclass(frozen=True)
class ExperimentSpec:
    """One fully expanded, immutable experiment run."""

    run_name: str
    method: str
    schedule: str
    backbone: str
    case_id: str
    panel_count: Optional[int]
    split: Optional[str]
    seed: int
    budget_type: str
    budget_value: float
    dataset_manifest_path: str
    dataset_manifest_hash: str
    git_commit: str
    git_dirty: bool
    provider: str
    artifact_root: str
    repo_root: str
    metric_config: Dict[str, Any]
    metric_config_hash: str
    metric_version: str
    method_config: Dict[str, Any] = field(default_factory=dict)
    provider_options: Dict[str, Any] = field(default_factory=dict)
    dataset_mode: str = "legacy"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_name",
            "method",
            "schedule",
            "backbone",
            "case_id",
            "dataset_manifest_path",
            "git_commit",
            "artifact_root",
            "repo_root",
            "metric_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ProvenanceError(f"ExperimentSpec.{name} must be non-empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ProvenanceError(
                f"Unsupported ExperimentSpec schema_version: {self.schema_version}"
            )
        case_token = f"__case-{slug_identifier(self.case_id)}"
        case_pattern = rf"(?:^|__)case-{re.escape(slug_identifier(self.case_id))}(?=__|$)"
        if re.search(case_pattern, self.run_name) is None:
            raise ProvenanceError(
                f"run_name must include the case slug {case_token!r}"
            )
        if self.panel_count is not None:
            if (
                isinstance(self.panel_count, bool)
                or not isinstance(self.panel_count, int)
                or self.panel_count < 1
            ):
                raise ProvenanceError(
                    "panel_count must be a positive integer when provided"
                )
        if self.split is not None and (
            not isinstance(self.split, str) or not self.split.strip()
        ):
            raise ProvenanceError("split must be a non-empty string when provided")
        if self.budget_type not in _BUDGET_TYPES:
            raise ProvenanceError(
                f"budget_type must be one of {sorted(_BUDGET_TYPES)}"
            )
        if self.dataset_mode not in {"legacy", "sealed_benchmark"}:
            raise ProvenanceError(
                "dataset_mode must be legacy or sealed_benchmark"
            )
        if not math.isfinite(float(self.budget_value)) or self.budget_value <= 0:
            raise ProvenanceError("budget_value must be a positive finite number")
        if self.budget_type == "renders" and not float(self.budget_value).is_integer():
            raise ProvenanceError("A renders budget must be an integer")
        if not _SHA256_RE.fullmatch(self.dataset_manifest_hash):
            raise ProvenanceError("dataset_manifest_hash must be a SHA-256 digest")
        if not _SHA256_RE.fullmatch(self.metric_config_hash):
            raise ProvenanceError("metric_config_hash must be a SHA-256 digest")
        if not _GIT_COMMIT_RE.fullmatch(self.git_commit):
            raise ProvenanceError("git_commit must be a hexadecimal commit id")
        _validate_json_object("metric_config", self.metric_config)
        _validate_json_object("method_config", self.method_config)
        _validate_json_object("provider_options", self.provider_options)
        if sha256_json(self.metric_config) != self.metric_config_hash:
            raise ProvenanceError("metric_config_hash does not match metric_config")

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        return cls(**_strict_dataclass_kwargs(cls, data))


@dataclass
class RunRecord:
    """Tamper-evident record for one attempted ExperimentSpec."""

    experiment_spec: Dict[str, Any]
    spec_hash: str
    run_name: str
    method: str
    backbone: str
    case_id: str
    panel_count: Optional[int]
    split: Optional[str]
    seed: int
    budget_type: str
    budget_value: float
    dataset_manifest_hash: str
    git_commit: str
    started_at: str
    finished_at: Optional[str]
    status: str
    render_count: int
    wall_clock_seconds: float
    artifact_paths: Dict[str, str]
    artifact_hashes: Dict[str, str]
    metric_config: Dict[str, Any]
    metric_config_hash: str
    metric_version: str
    provider: str
    provider_name: Optional[str]
    attempt: int
    test_only: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    candidates: list[Dict[str, Any]] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    best_history: list[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, str]] = None
    record_hash: str = ""
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def start(cls, spec: ExperimentSpec, attempt: int) -> "RunRecord":
        return cls(
            experiment_spec=spec.to_dict(),
            spec_hash=spec.spec_hash,
            run_name=spec.run_name,
            method=spec.method,
            backbone=spec.backbone,
            case_id=spec.case_id,
            panel_count=spec.panel_count,
            split=spec.split,
            seed=spec.seed,
            budget_type=spec.budget_type,
            budget_value=spec.budget_value,
            dataset_manifest_hash=spec.dataset_manifest_hash,
            git_commit=spec.git_commit,
            started_at=utc_now(),
            finished_at=None,
            status="running",
            render_count=0,
            wall_clock_seconds=0.0,
            artifact_paths={},
            artifact_hashes={},
            metric_config=dict(spec.metric_config),
            metric_config_hash=spec.metric_config_hash,
            metric_version=spec.metric_version,
            provider=spec.provider,
            provider_name=None,
            attempt=attempt,
            test_only=False,
        )

    def to_dict(self, *, include_record_hash: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_record_hash:
            data.pop("record_hash", None)
        return data

    def seal(self) -> None:
        self.record_hash = sha256_json(self.to_dict(include_record_hash=False))

    def write(self, path: Path) -> None:
        self.seal()
        write_json_atomic(path, self.to_dict())

    def verify_record_hash(self) -> None:
        if not _SHA256_RE.fullmatch(self.record_hash):
            raise ProvenanceError("record_hash is missing or malformed")
        expected = sha256_json(self.to_dict(include_record_hash=False))
        if expected != self.record_hash:
            raise ProvenanceError("run_record.json failed its integrity check")

    def validate_provenance(self, *, require_completed: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProvenanceError(
                f"Unsupported RunRecord schema_version: {self.schema_version}"
            )
        if self.status not in _STATUSES:
            raise ProvenanceError(f"Invalid run status: {self.status}")
        if require_completed and self.status != "completed":
            raise ProvenanceError(f"Run is not completed: {self.run_name}")
        if not self.started_at:
            raise ProvenanceError("started_at is required")
        started = _parse_timestamp(self.started_at, "started_at")
        if self.status != "running" and not self.finished_at:
            raise ProvenanceError("finished_at is required for terminal runs")
        if self.finished_at is not None:
            finished = _parse_timestamp(self.finished_at, "finished_at")
            if finished < started:
                raise ProvenanceError("finished_at cannot precede started_at")
        if self.render_count < 0:
            raise ProvenanceError("render_count cannot be negative")
        if not math.isfinite(self.wall_clock_seconds) or self.wall_clock_seconds < 0:
            raise ProvenanceError("wall_clock_seconds must be finite and non-negative")
        if self.attempt < 1:
            raise ProvenanceError("attempt must be at least one")
        if set(self.artifact_paths) != set(self.artifact_hashes):
            raise ProvenanceError("Each artifact path must have exactly one hash")
        for digest in self.artifact_hashes.values():
            if not _SHA256_RE.fullmatch(digest):
                raise ProvenanceError("Artifact hashes must be SHA-256 digests")

        spec = ExperimentSpec.from_dict(self.experiment_spec)
        if spec.spec_hash != self.spec_hash:
            raise ProvenanceError("spec_hash does not match experiment_spec")
        mirrored = {
            "run_name": self.run_name,
            "method": self.method,
            "backbone": self.backbone,
            "case_id": self.case_id,
            "panel_count": self.panel_count,
            "split": self.split,
            "seed": self.seed,
            "budget_type": self.budget_type,
            "budget_value": self.budget_value,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "git_commit": self.git_commit,
            "metric_config": self.metric_config,
            "metric_config_hash": self.metric_config_hash,
            "metric_version": self.metric_version,
            "provider": self.provider,
        }
        for name, actual in mirrored.items():
            if getattr(spec, name) != actual:
                raise ProvenanceError(f"RunRecord.{name} disagrees with ExperimentSpec")
        self.verify_record_hash()

        if self.status == "failed" and not self.error:
            raise ProvenanceError("Failed runs must include an explicit error")
        if self.status != "completed":
            return
        if self.render_count < 1:
            raise ProvenanceError("Completed runs must include at least one render")
        if not self.provider_name:
            raise ProvenanceError("Completed runs must identify the resolved provider")
        if not self.candidates:
            raise ProvenanceError("Completed runs must include candidate provenance")
        candidate_ids = {str(item.get("candidate_id", "")) for item in self.candidates}
        if not self.best_candidate_id or self.best_candidate_id not in candidate_ids:
            raise ProvenanceError("best_candidate_id must refer to a recorded candidate")
        if not self.best_history:
            raise ProvenanceError("Completed runs must include best-so-far history")
        if not self.metrics:
            raise ProvenanceError("Completed runs must include metrics")
        for name, value in self.metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ProvenanceError(f"Metric {name} is not a finite number")
        if not self.artifact_paths:
            raise ProvenanceError("Completed runs must include hashed artifacts")
        if "dataset_manifest" not in self.artifact_paths:
            raise ProvenanceError(
                "Completed runs must include a frozen dataset manifest"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        return cls(**_strict_dataclass_kwargs(cls, data))

    @classmethod
    def read(cls, path: Path) -> "RunRecord":
        return cls.from_dict(read_json(path))

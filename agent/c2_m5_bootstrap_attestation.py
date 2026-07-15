"""Stdlib-only manifest verifier used before importing the experiments package."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import sysconfig
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class C2M5BootstrapAttestationError(RuntimeError):
    """Raised when the M5 bootstrap cannot prove its fixed runtime closure."""


ATTESTATION_RELATIVE = (
    "agent/experiments/resources/c2_m5_source_pilot_attestation_v1.json"
)
DEPENDENCY_MANIFEST_RELATIVE = (
    "agent/experiments/resources/c2_m5_python_dependency_manifest_v2.json"
)
DEPENDENCY_ARCHIVE_RELATIVE = (
    "agent/experiments/resources/c2_m5_python_dependencies_py39_v1.zip"
)
REQUIRED_ATTESTED_PATHS = frozenset(
    {
        "agent/c2_m5_bootstrap_attestation.py",
        "agent/c2_m5_downloader_bootstrap.py",
        "agent/c2_m5_release_test_runner.py",
        "agent/c2_m5_sitecustomize/sitecustomize.py",
        "agent/c2_m5_source_pilot_bootstrap.py",
        "agent/experiments/__init__.py",
        "agent/experiments/c2_full_replacement_policy.py",
        "agent/experiments/cli.py",
        "agent/experiments/c2_m1_trust_boundary.py",
        "agent/experiments/c2_m5_source_pilot.py",
        "agent/experiments/c2_owner_remediation_execution_policy.py",
        "agent/experiments/c2_owner_remediation_execution_policy_pin.py",
        "agent/experiments/c2_remediation_root_finalizer.py",
        "agent/experiments/c2_source_bearing_extension.py",
        "agent/experiments/c2_stageb_source_extension_code_attestation.py",
        "agent/experiments/c2_stageb_source_extension_code_attestation_pin.py",
        "agent/experiments/models.py",
        "agent/experiments/manifest.py",
        "agent/experiments/providers.py",
        "agent/experiments/resources/c2_owner_execution_authorization_v1.json",
        "agent/experiments/resources/c2_owner_remediation_execution_policy_v1.json",
        DEPENDENCY_MANIFEST_RELATIVE,
        DEPENDENCY_ARCHIVE_RELATIVE,
        "agent/experiments/resources/c2_m5_py39_test_dependencies_v1.zip",
        "agent/experiments/resources/c2_m5_py39_test_dependency_manifest_v1.json",
        "agent/experiments/resources/c2_source_extension_runtime_manifest_v1.json",
        "agent/experiments/resources/c2_stageb_source_extension_code_attestation_registry_v1.json",
        "agent/experiments/schemas/c2_owner_execution_authorization_v1.schema.json",
        "agent/experiments/schemas/c2_owner_remediation_execution_policy_v1.schema.json",
        "agent/experiments/schemas/c2_stageb_source_extension_code_attestation_v1.schema.json",
        "agent/experiments/schemas/c2_v2_candidate_set_input_v1.schema.json",
        "agent/experiments/schemas/c2_v2_consumable_source_unit_v1.schema.json",
        "agent/experiments/schemas/c2_v2_consumption_bijection_validation_v1.schema.json",
        "agent/experiments/schemas/c2_v2_container_accounting_index_v1.schema.json",
        "agent/experiments/schemas/c2_v2_detected_format_v1.schema.json",
        "agent/experiments/schemas/c2_v2_downstream_consumption_v1.schema.json",
        "agent/experiments/schemas/c2_v2_fd_format_classifier_config_v1.schema.json",
        "agent/tests/test_c2_m5_source_pilot.py",
        "agent/tests/test_c2_remediation_root_finalizer.py",
        "agent/tests/test_c2_source_bearing_extension.py",
        "agent/tests/test_experiment_support.py",
        "agent/experiments/tests/test_c2_stageb_source_extension_code_attestation.py",
    }
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_WHEEL_FILENAMES = frozenset(
    {
        "attrs-26.1.0-py3-none-any.whl",
        "beautifulsoup4-4.14.3-py3-none-any.whl",
        "certifi-2026.4.22-py3-none-any.whl",
        "charset_normalizer-3.4.7-cp39-cp39-macosx_10_9_universal2.whl",
        "idna-3.13-py3-none-any.whl",
        "jsonschema-4.25.1-py3-none-any.whl",
        "jsonschema_specifications-2025.9.1-py3-none-any.whl",
        "referencing-0.36.2-py3-none-any.whl",
        "requests-2.32.5-py3-none-any.whl",
        "rpds_py-0.27.1-cp39-cp39-macosx_11_0_arm64.whl",
        "soupsieve-2.8.4-py3-none-any.whl",
        "typing_extensions-4.15.0-py3-none-any.whl",
        "urllib3-1.26.20-py2.py3-none-any.whl",
    }
)
_GIT = "/usr/bin/git"
_GIT_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/var/empty",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_ALLOW_PROTOCOL": "",
}
_GIT_CONFIG = (
    "--no-replace-objects",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.pager=cat",
    "-c",
    "log.showSignature=false",
    "-c",
    "diff.external=",
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.ext.allow=never",
    "-c",
    "protocol.file.allow=never",
)
_ACTIVE_ATTESTATION: AdapterAttestation | None = None


@dataclass(frozen=True)
class AdapterAttestation:
    implementation_commit: str
    attestation_commit: str
    head_commit: str
    manifest_sha256: str
    path_sha256: Mapping[str, str]
    path_payloads: Mapping[str, bytes]
    dependency_manifest_sha256: str
    dependency_binding: Mapping[str, object]
    dependency_payloads: Mapping[str, bytes]
    externally_pinned_bootstrap_sha256: str
    externally_pinned_python_sha256: str
    externally_pinned_python_library_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2M5BootstrapAttestationError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_regular_bytes(path: Path, label: str) -> bytes:
    before = path.lstat()
    _require(
        stat.S_ISREG(before.st_mode)
        and not path.is_symlink()
        and before.st_uid in {0, os.geteuid()}
        and before.st_mode & 0o022 == 0,
        f"{label} is not a private regular file",
    )
    payload = path.read_bytes()
    after = path.lstat()
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
    return payload


def _require_root_owned_ancestry(path: Path, label: str) -> None:
    resolved = path.resolve()
    chain = [Path("/"), *reversed(list(resolved.parents)[:-1]), resolved]
    for index, component in enumerate(chain):
        metadata = component.lstat()
        is_leaf = index == len(chain) - 1
        _require(
            metadata.st_uid == 0
            and metadata.st_mode & 0o022 == 0
            and not stat.S_ISLNK(metadata.st_mode)
            and (
                stat.S_ISREG(metadata.st_mode)
                if is_leaf and resolved.is_file()
                else stat.S_ISDIR(metadata.st_mode)
            ),
            f"{label} has mutable or non-root-owned ancestry: {component}",
        )


def _python_runtime_binding() -> dict[str, object]:
    launcher = Path("/usr/bin/python3")
    executable = Path(sys.executable).resolve()
    runtime_library = Path(
        "/Library/Developer/CommandLineTools/Library/Frameworks/"
        "Python3.framework/Versions/3.9/Python3"
    )
    _require(
        sys.version_info[:3] == (3, 9, 6),
        "M5 requires the fixed Apple system CPython 3.9.6",
    )
    _require(
        platform.machine() == "arm64" and os.uname().machine == "arm64",
        "M5 requires a native arm64 Python process",
    )
    _require_root_owned_ancestry(launcher, "M5 CPython launcher")
    _require_root_owned_ancestry(executable, "M5 CPython executable")
    _require_root_owned_ancestry(
        runtime_library,
        "M5 CPython runtime library",
    )
    launcher_payload = _stable_regular_bytes(
        launcher,
        "M5 CPython launcher",
    )
    executable_payload = _stable_regular_bytes(
        executable,
        "M5 CPython executable",
    )
    runtime_library_payload = _stable_regular_bytes(
        runtime_library,
        "M5 CPython runtime library",
    )
    stdlib_value = sysconfig.get_paths().get("stdlib")
    _require(isinstance(stdlib_value, str), "M5 stdlib path is unavailable")
    stdlib = Path(stdlib_value).resolve()
    _require_root_owned_ancestry(stdlib, "M5 stdlib")
    inventory: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(
        stdlib,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(stdlib)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            if (
                relative_directory == Path(".")
                and name in {"site-packages", "config-3.9-darwin"}
            ):
                continue
            child = directory_path / name
            metadata = child.lstat()
            _require(
                stat.S_ISDIR(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_uid == 0
                and metadata.st_mode & 0o022 == 0,
                f"M5 stdlib contains an unsafe directory: {child}",
            )
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            path = directory_path / name
            payload = _stable_regular_bytes(path, f"M5 stdlib file {path}")
            relative = path.relative_to(stdlib).as_posix()
            inventory.append(
                {
                    "relative_path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return {
        "implementation": platform.python_implementation(),
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "micro": sys.version_info.micro,
        "cache_tag": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
        "platform": sysconfig.get_platform(),
        "process_machine": platform.machine(),
        "launcher_path": str(launcher),
        "launcher_sha256": hashlib.sha256(launcher_payload).hexdigest(),
        "executable_path": str(executable),
        "executable_sha256": hashlib.sha256(executable_payload).hexdigest(),
        "runtime_library_path": str(runtime_library),
        "runtime_library_sha256": hashlib.sha256(
            runtime_library_payload
        ).hexdigest(),
        "stdlib_path": str(stdlib),
        "stdlib_file_count": len(inventory),
        "stdlib_bytes": sum(int(item["bytes"]) for item in inventory),
        "stdlib_inventory_sha256": hashlib.sha256(
            _canonical_bytes(inventory)
        ).hexdigest(),
        "trust_policy": (
            "ROOT_OWNED_NO_GROUP_OR_WORLD_WRITE_WITH_VALIDATED_ANCESTRY"
        ),
    }


def _python_evidence_binding(
    runtime_binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "runtime_identity": "APPLE_CLT_ROOT_OWNED_CPYTHON_3_9_6",
        "launcher_sha256": runtime_binding["launcher_sha256"],
        "executable_sha256": runtime_binding["executable_sha256"],
        "implementation": runtime_binding["implementation"],
        "major": runtime_binding["major"],
        "minor": runtime_binding["minor"],
        "micro": runtime_binding["micro"],
        "cache_tag": runtime_binding["cache_tag"],
        "soabi": runtime_binding["soabi"],
        "platform": runtime_binding["platform"],
        "process_machine": runtime_binding["process_machine"],
        "runtime_library_sha256": runtime_binding[
            "runtime_library_sha256"
        ],
        "stdlib_file_count": runtime_binding["stdlib_file_count"],
        "stdlib_bytes": runtime_binding["stdlib_bytes"],
        "stdlib_inventory_sha256": runtime_binding[
            "stdlib_inventory_sha256"
        ],
        "trust_policy": runtime_binding["trust_policy"],
    }


def _verified_dependency_payloads(
    manifest_payload: bytes,
    archive_payload: bytes,
) -> tuple[dict[str, object], dict[str, bytes]]:
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2M5BootstrapAttestationError(
            "M5 dependency manifest is invalid JSON"
        ) from exc
    _require(
        isinstance(manifest, dict)
        and set(manifest)
        == {
            "schema_version",
            "python",
            "archive",
            "source_wheels",
            "runtime_inventory",
        }
        and manifest.get("schema_version")
        == "c2_m5_python_dependency_manifest_v2"
        and manifest_payload == _canonical_bytes(manifest) + b"\n",
        "M5 dependency manifest is invalid",
    )
    python_binding = manifest["python"]
    _require(
        isinstance(python_binding, dict)
        and python_binding == _python_runtime_binding(),
        "M5 Python runtime differs from the dependency manifest",
    )
    archive = manifest["archive"]
    _require(
        isinstance(archive, dict)
        and set(archive) == {"relative_path", "bytes", "sha256"}
        and archive.get("relative_path") == DEPENDENCY_ARCHIVE_RELATIVE
        and archive.get("bytes") == len(archive_payload)
        and archive.get("sha256")
        == hashlib.sha256(archive_payload).hexdigest(),
        "M5 dependency archive binding is invalid",
    )
    source_wheels = manifest["source_wheels"]
    _require(
        isinstance(source_wheels, list)
        and len(source_wheels) == len(_REQUIRED_WHEEL_FILENAMES),
        "M5 dependency wheel roster is invalid",
    )
    observed_wheels: set[str] = set()
    for entry in source_wheels:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"filename", "bytes", "sha256"}
            and entry.get("filename") in _REQUIRED_WHEEL_FILENAMES
            and entry["filename"] not in observed_wheels
            and isinstance(entry.get("bytes"), int)
            and entry["bytes"] > 0
            and isinstance(entry.get("sha256"), str)
            and _SHA256_RE.fullmatch(entry["sha256"]) is not None,
            "M5 dependency wheel entry is invalid",
        )
        observed_wheels.add(entry["filename"])
    _require(
        observed_wheels == _REQUIRED_WHEEL_FILENAMES
        and [entry["filename"] for entry in source_wheels]
        == sorted(observed_wheels),
        "M5 dependency wheel closure is incomplete",
    )
    raw_inventory = manifest["runtime_inventory"]
    _require(
        isinstance(raw_inventory, list)
        and len(raw_inventory) <= 1_000,
        "M5 dependency runtime inventory is invalid",
    )
    declared_inventory: dict[str, tuple[int, str]] = {}
    for entry in raw_inventory:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"relative_path", "bytes", "sha256"}
            and isinstance(entry.get("relative_path"), str)
            and entry["relative_path"]
            and "\\" not in entry["relative_path"]
            and not Path(entry["relative_path"]).is_absolute()
            and ".." not in Path(entry["relative_path"]).parts
            and entry["relative_path"] not in declared_inventory
            and isinstance(entry.get("bytes"), int)
            and 0 <= entry["bytes"] <= 16 * 1024 * 1024
            and isinstance(entry.get("sha256"), str)
            and _SHA256_RE.fullmatch(entry["sha256"]) is not None,
            "M5 dependency runtime inventory entry is invalid",
        )
        declared_inventory[entry["relative_path"]] = (
            entry["bytes"],
            entry["sha256"],
        )
    _require(
        list(declared_inventory) == sorted(declared_inventory)
        and sum(size for size, _ in declared_inventory.values())
        <= 16 * 1024 * 1024,
        "M5 dependency runtime inventory is not sorted",
    )
    payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as dependency_archive:
            infos = dependency_archive.infolist()
            _require(
                not dependency_archive.comment and len(infos) == len(declared_inventory),
                "M5 dependency archive entry coverage is invalid",
            )
            for info in infos:
                relative = info.filename
                relative_path = Path(relative)
                mode = (info.external_attr >> 16) & 0xFFFF
                _require(
                    relative in declared_inventory
                    and relative not in payloads
                    and not info.is_dir()
                    and "\\" not in relative
                    and not relative_path.is_absolute()
                    and ".." not in relative_path.parts
                    and info.compress_type
                    in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    and info.flag_bits & ~0x800 == 0
                    and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                    and info.file_size == declared_inventory[relative][0],
                    "M5 dependency archive entry is unsafe",
                )
                payload = dependency_archive.read(info)
                _require(
                    len(payload) == declared_inventory[relative][0]
                    and hashlib.sha256(payload).hexdigest()
                    == declared_inventory[relative][1],
                    f"M5 dependency archive payload mismatch: {relative}",
                )
                payloads[relative] = payload
    except (OSError, zipfile.BadZipFile) as exc:
        raise C2M5BootstrapAttestationError(
            "M5 dependency archive is invalid"
        ) from exc
    _require(
        set(payloads) == set(declared_inventory)
        and {
            "attrs/__init__.py",
            "bs4/__init__.py",
            "certifi/__init__.py",
            "charset_normalizer/__init__.py",
            "idna/__init__.py",
            "jsonschema/__init__.py",
            "jsonschema_specifications/__init__.py",
            "referencing/__init__.py",
            "requests/__init__.py",
            "rpds/__init__.py",
            "soupsieve/__init__.py",
            "typing_extensions.py",
            "urllib3/__init__.py",
        }.issubset(payloads)
        and any(
            relative.startswith("rpds/")
            and relative.endswith("cpython-39-darwin.so")
            for relative in payloads
        ),
        "M5 dependency runtime import closure is incomplete",
    )
    inventory = [
        {
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in sorted(payloads.items())
    ]
    _require(
        inventory == raw_inventory,
        "M5 dependency runtime inventory differs from its archive",
    )
    binding: dict[str, object] = {
        "schema_version": "c2_m5_python_dependency_binding_v2",
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
        "python": _python_evidence_binding(python_binding),
        "source_wheels": source_wheels,
        "runtime_file_count": len(inventory),
        "runtime_bytes": sum(item["bytes"] for item in inventory),
        "runtime_inventory_sha256": hashlib.sha256(
            _canonical_bytes(inventory)
        ).hexdigest(),
    }
    return binding, payloads


def _git_text(repository: Path, arguments: Sequence[str], label: str) -> str:
    try:
        completed = subprocess.run(
            [_GIT, *_GIT_CONFIG, "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_GIT_ENV,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C2M5BootstrapAttestationError(f"cannot establish {label}") from exc
    return completed.stdout.strip()


def _git_bytes(repository: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        completed = subprocess.run(
            [_GIT, *_GIT_CONFIG, "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_GIT_ENV,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C2M5BootstrapAttestationError(f"cannot establish {label}") from exc
    return completed.stdout


def _parse_index_entries(payload: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in payload.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        pieces = metadata.split()
        _require(
            separator == b"\t" and len(pieces) == 3 and pieces[2] == b"0",
            "M5 repository index contains an unsupported entry",
        )
        try:
            relative = raw_path.decode("utf-8")
            mode = pieces[0].decode("ascii")
            object_id = pieces[1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise C2M5BootstrapAttestationError(
                "M5 repository index path is not UTF-8"
            ) from exc
        path = Path(relative)
        _require(
            mode == "100644"
            and _COMMIT_RE.fullmatch(object_id) is not None
            and relative
            and "\\" not in relative
            and not path.is_absolute()
            and ".." not in path.parts
            and relative not in entries,
            "M5 repository index entry is unsafe",
        )
        entries[relative] = (mode, object_id)
    _require(entries, "M5 repository index is empty")
    return entries


def _parse_tree_entries(payload: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in payload.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        pieces = metadata.split()
        _require(
            separator == b"\t" and len(pieces) == 3 and pieces[1] == b"blob",
            "M5 repository HEAD contains an unsupported entry",
        )
        try:
            relative = raw_path.decode("utf-8")
            mode = pieces[0].decode("ascii")
            object_id = pieces[2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise C2M5BootstrapAttestationError(
                "M5 repository HEAD path is not UTF-8"
            ) from exc
        _require(relative not in entries, "M5 repository HEAD repeats a path")
        entries[relative] = (mode, object_id)
    return entries


def _git_blob_object_id(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _verify_filter_free_clean_tree(repository: Path, head_commit: str) -> None:
    """Compare filesystem bytes, index, and HEAD without invoking Git filters."""

    _require(
        _git_text(
            repository,
            ("rev-parse", "--show-object-format"),
            "repository object format",
        )
        == "sha1",
        "M5 repository object format is unsupported",
    )
    index_entries = _parse_index_entries(
        _git_bytes(
            repository,
            ("ls-files", "--stage", "-z"),
            "repository index",
        )
    )
    head_entries = _parse_tree_entries(
        _git_bytes(
            repository,
            ("ls-tree", "-r", "-z", "--full-tree", head_commit),
            "repository HEAD tree",
        )
    )
    _require(
        index_entries == head_entries,
        "M5 repository index differs from HEAD",
    )
    expected_directories = {
        parent.as_posix()
        for relative in index_entries
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    observed_files: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        repository,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        if directory_path == repository:
            directory_names[:] = sorted(
                name for name in directory_names if name != ".git"
            )
            file_names = [name for name in file_names if name != ".git"]
        else:
            directory_names.sort()
        relative_directory = directory_path.relative_to(repository)
        for name in directory_names:
            path = directory_path / name
            relative = (relative_directory / name).as_posix()
            metadata = path.lstat()
            _require(
                stat.S_ISDIR(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
                and relative in expected_directories,
                f"M5 repository contains an unsafe or untracked directory: {relative}",
            )
        for name in sorted(file_names):
            path = directory_path / name
            relative = (relative_directory / name).as_posix()
            metadata = path.lstat()
            _require(
                metadata.st_nlink == 1,
                f"M5 repository file is hard-linked: {relative}",
            )
            payload = _stable_regular_bytes(path, f"M5 repository file {relative}")
            expected = index_entries.get(relative)
            _require(
                expected is not None
                and _git_blob_object_id(payload) == expected[1],
                f"M5 repository file differs from HEAD: {relative}",
            )
            observed_files.add(relative)
    _require(
        observed_files == set(index_entries),
        "M5 repository worktree coverage differs from HEAD",
    )


def verify_adapter_attestation(
    *,
    activate: bool,
    expected_attestation_commit: str,
    expected_manifest_sha256: str,
    expected_bootstrap_sha256: str,
    expected_python_sha256: str,
    expected_python_library_sha256: str,
) -> AdapterAttestation:
    """Verify the manifest-only commit and every runtime/import resource byte."""

    repository = Path(__file__).resolve().parents[1]
    _require(
        Path(_git_text(repository, ("rev-parse", "--show-toplevel"), "repository root"))
        .resolve()
        == repository,
        "M5 adapter is not running from its repository root",
    )
    _require(
        _COMMIT_RE.fullmatch(expected_attestation_commit) is not None
        and _SHA256_RE.fullmatch(expected_manifest_sha256) is not None
        and _SHA256_RE.fullmatch(expected_bootstrap_sha256) is not None
        and _SHA256_RE.fullmatch(expected_python_sha256) is not None
        and _SHA256_RE.fullmatch(expected_python_library_sha256) is not None,
        "M5 external launch binding is invalid",
    )
    head_commit = _git_text(repository, ("rev-parse", "HEAD"), "repository HEAD")
    observed_attestation_commit = _git_text(
        repository,
        ("log", "-1", "--format=%H", "--", ATTESTATION_RELATIVE),
        "M5 attestation commit",
    )
    _require(
        observed_attestation_commit == expected_attestation_commit,
        "M5 attestation commit differs from the external launch binding",
    )
    attestation_commit = expected_attestation_commit
    implementation_commit = _git_text(
        repository,
        ("rev-parse", f"{attestation_commit}^"),
        "M5 implementation commit",
    )
    _git_text(
        repository,
        ("merge-base", "--is-ancestor", attestation_commit, head_commit),
        "M5 attestation ancestry",
    )
    changed = _git_text(
        repository,
        (
            "diff",
            "--no-ext-diff",
            "--name-status",
            "--no-renames",
            implementation_commit,
            attestation_commit,
        ),
        "M5 attestation commit contents",
    )
    _require(
        changed == f"A\t{ATTESTATION_RELATIVE}",
        "M5 attestation commit must add only its manifest",
    )
    manifest_path = repository / ATTESTATION_RELATIVE
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "M5 attestation manifest is unavailable",
    )
    manifest_payload = _stable_regular_bytes(
        manifest_path,
        "M5 attestation manifest",
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _require(
        manifest_sha256 == expected_manifest_sha256,
        "M5 manifest differs from the external launch binding",
    )
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2M5BootstrapAttestationError(
            "M5 attestation manifest is invalid JSON"
        ) from exc
    _require(
        isinstance(manifest, dict)
        and set(manifest)
        == {
            "schema_version",
            "approved_implementation_commit_full",
            "attested_paths",
        }
        and manifest["schema_version"] == "c2_m5_source_pilot_attestation_v1"
        and manifest["approved_implementation_commit_full"] == implementation_commit
        and manifest_payload == _canonical_bytes(manifest) + b"\n",
        "M5 attestation manifest is invalid",
    )
    raw_paths = manifest["attested_paths"]
    _require(isinstance(raw_paths, list), "M5 attestation path list is invalid")
    observed: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for raw in raw_paths:
        _require(
            isinstance(raw, dict)
            and set(raw) == {"relative_path", "git_blob_object_id", "sha256"},
            "M5 attestation path entry is invalid",
        )
        relative = raw["relative_path"]
        blob = raw["git_blob_object_id"]
        digest = raw["sha256"]
        _require(
            isinstance(relative, str)
            and relative in REQUIRED_ATTESTED_PATHS
            and relative not in observed
            and isinstance(blob, str)
            and _COMMIT_RE.fullmatch(blob) is not None
            and isinstance(digest, str)
            and _SHA256_RE.fullmatch(digest) is not None,
            "M5 attestation path binding is invalid",
        )
        implementation_blob = _git_text(
            repository,
            ("rev-parse", f"{implementation_commit}:{relative}"),
            f"M5 implementation blob {relative}",
        )
        implementation_payload = _git_bytes(
            repository,
            ("show", f"{implementation_commit}:{relative}"),
            f"M5 implementation bytes {relative}",
        )
        attestation_payload = _git_bytes(
            repository,
            ("show", f"{attestation_commit}:{relative}"),
            f"M5 attestation bytes {relative}",
        )
        head_payload = _git_bytes(
            repository,
            ("show", f"{head_commit}:{relative}"),
            f"M5 HEAD bytes {relative}",
        )
        runtime_path = repository / relative
        runtime_payload = runtime_path.read_bytes()
        _require(
            runtime_path.is_file()
            and not runtime_path.is_symlink()
            and implementation_blob == blob
            and implementation_payload == attestation_payload == head_payload
            and runtime_payload == implementation_payload
            and hashlib.sha256(implementation_payload).hexdigest() == digest,
            f"M5 attested runtime mismatch: {relative}",
        )
        observed[relative] = digest
        payloads[relative] = runtime_payload
    _require(
        set(observed) == REQUIRED_ATTESTED_PATHS,
        "M5 attestation path closure is incomplete",
    )
    _require(
        observed["agent/c2_m5_source_pilot_bootstrap.py"]
        == expected_bootstrap_sha256,
        "M5 bootstrap differs from the external launch binding",
    )
    _verify_filter_free_clean_tree(repository, head_commit)
    dependency_manifest = payloads[DEPENDENCY_MANIFEST_RELATIVE]
    dependency_archive = payloads[DEPENDENCY_ARCHIVE_RELATIVE]
    dependency_binding, dependency_payloads = _verified_dependency_payloads(
        dependency_manifest,
        dependency_archive,
    )
    verified_python = dependency_binding.get("python")
    _require(
        isinstance(verified_python, dict)
        and verified_python.get("executable_sha256")
        == expected_python_sha256
        and verified_python.get("runtime_library_sha256")
        == expected_python_library_sha256,
        "M5 CPython executable or library differs from the external launch binding",
    )
    attestation = AdapterAttestation(
        implementation_commit=implementation_commit,
        attestation_commit=attestation_commit,
        head_commit=head_commit,
        manifest_sha256=manifest_sha256,
        path_sha256=observed,
        path_payloads=payloads,
        dependency_manifest_sha256=hashlib.sha256(
            dependency_manifest
        ).hexdigest(),
        dependency_binding=dependency_binding,
        dependency_payloads=dependency_payloads,
        externally_pinned_bootstrap_sha256=expected_bootstrap_sha256,
        externally_pinned_python_sha256=expected_python_sha256,
        externally_pinned_python_library_sha256=(
            expected_python_library_sha256
        ),
    )
    if activate:
        global _ACTIVE_ATTESTATION
        _ACTIVE_ATTESTATION = attestation
    return attestation


def require_active_adapter_attestation() -> AdapterAttestation:
    if _ACTIVE_ATTESTATION is None:
        raise C2M5BootstrapAttestationError(
            "M5 adapter must be entered through the verified bootstrap"
        )
    return _ACTIVE_ATTESTATION

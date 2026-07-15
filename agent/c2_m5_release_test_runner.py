"""Run the fixed M5 release suite under an isolated, hash-bound Python 3.9."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path
from typing import Dict, Mapping, Tuple


_AGENT_ROOT = Path(__file__).resolve().parent
_REPOSITORY = _AGENT_ROOT.parent
_PRODUCTION_MANIFEST = (
    _AGENT_ROOT
    / "experiments/resources/c2_m5_python_dependency_manifest_v2.json"
)
_PRODUCTION_ARCHIVE = (
    _AGENT_ROOT
    / "experiments/resources/c2_m5_python_dependencies_py39_v1.zip"
)
_TEST_MANIFEST = (
    _AGENT_ROOT
    / "experiments/resources/c2_m5_py39_test_dependency_manifest_v1.json"
)
_TEST_ARCHIVE = (
    _AGENT_ROOT
    / "experiments/resources/c2_m5_py39_test_dependencies_v1.zip"
)
_ATTESTATION_MANIFEST = (
    _AGENT_ROOT
    / "experiments/resources/c2_m5_source_pilot_attestation_v1.json"
)
_TEST_WHEELS = frozenset(
    {
        "exceptiongroup-1.3.1-py3-none-any.whl",
        "iniconfig-2.1.0-py3-none-any.whl",
        "packaging-26.2-py3-none-any.whl",
        "pluggy-1.6.0-py3-none-any.whl",
        "pygments-2.20.0-py3-none-any.whl",
        "pytest-8.4.2-py3-none-any.whl",
        "pyyaml-6.0.3-cp39-cp39-macosx_11_0_arm64.whl",
        "tomli-2.4.1-py3-none-any.whl",
    }
)
_TEST_TARGETS = (
    "agent/tests/test_c2_m5_source_pilot.py",
    "agent/tests/test_c2_remediation_root_finalizer.py",
    "agent/tests/test_c2_source_bearing_extension.py",
    "agent/experiments/tests/test_c2_stageb_source_extension_code_attestation.py",
)
_SHA256_LENGTH = 64
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_RUNTIME_FILES = 2_000
_TEST_TIMEOUT_SECONDS = 15 * 60
_PROCESS_GROUP_CLEANUP_SECONDS = 10
_PYTEST_CONFIG = b"[pytest]\naddopts =\n"
_PYTEST_CHILD = """\
import os
import sys

production, tests, source, config, *targets = sys.argv[1:]
sys.path[:0] = [tests, production, os.path.join(source, "agent")]
from c2_m5_source_pilot_bootstrap import _install_python39_compatibility

_install_python39_compatibility()
import pytest

os.chdir(source)
raise SystemExit(
    pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            "--disable-warnings",
            "-c",
            config,
            "--rootdir",
            source,
            "--confcutdir",
            source,
            "--noconftest",
            *targets,
        ]
    )
)
"""
_GIT_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/var/empty",
    "LANG": "C",
    "LC_ALL": "C",
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
_GIT_PREFIX = (
    "/usr/bin/git",
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


class M5ReleaseTestError(RuntimeError):
    """Raised when the fixed M5 test closure cannot be established."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M5ReleaseTestError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_regular(path: Path, label: str, max_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise M5ReleaseTestError(f"{label} is unavailable") from exc
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_uid == os.geteuid()
        and before.st_nlink == 1
        and before.st_mode & 0o022 == 0
        and 0 <= before.st_size <= max_bytes,
        f"{label} is unsafe",
    )
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise M5ReleaseTestError(f"{label} could not be read") from exc
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
        )
        and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return payload


def _load_json(payload: bytes, label: str) -> Dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M5ReleaseTestError(f"{label} is invalid JSON") from exc
    _require(
        isinstance(value, dict) and payload == _canonical_bytes(value) + b"\n",
        f"{label} is not canonical",
    )
    return value


def _retained_attested_payload(
    payloads: Mapping[str, bytes],
    relative: str,
    label: str,
    maximum_bytes: int,
) -> bytes:
    payload = payloads.get(relative)
    _require(
        isinstance(payload, bytes) and 0 < len(payload) <= maximum_bytes,
        f"{label} is missing from the retained M5 attestation",
    )
    return payload


def _git(arguments: Tuple[str, ...], label: str) -> bytes:
    try:
        completed = subprocess.run(
            [*_GIT_PREFIX, "-C", str(_REPOSITORY), *arguments],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M5ReleaseTestError(f"{label} could not be verified") from exc
    return completed.stdout


def _release_pin(name: str, length: int) -> str:
    value = os.environ.get(name, "")
    _require(
        len(value) == length
        and all(character in "0123456789abcdef" for character in value),
        f"M5 release pin is invalid: {name}",
    )
    return value


def _trusted_temporary_parent() -> Path:
    parent = _REPOSITORY.parent
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise M5ReleaseTestError(
                f"M5 release temporary ancestor is unavailable: {current}"
            ) from exc
        _require(
            stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_uid in {0, os.geteuid()}
            and metadata.st_mode & 0o022 == 0,
            f"M5 release temporary ancestor is unsafe: {current}",
        )
        _require_nonmutating_acl(current, f"M5 release temporary ancestor {current}")
    return parent


def _require_nonmutating_acl(path: Path, label: str) -> None:
    if sys.platform != "darwin":
        return
    try:
        completed = subprocess.run(
            ["/bin/ls", "-lde", str(path)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M5ReleaseTestError(f"{label} ACL could not be verified") from exc
    acl_entries = [
        line.strip()
        for line in completed.stdout.splitlines()[1:]
        if line.strip()
    ]
    _require(
        all(" deny " in entry for entry in acl_entries),
        f"{label} grants mutation through a Darwin ACL",
    )


def _remove_acl(path: Path, label: str) -> None:
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["/bin/chmod", "-N", str(path)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M5ReleaseTestError(f"{label} ACL could not be removed") from exc
    _require_nonmutating_acl(path, label)


def _verified_production_attestation_module() -> Tuple[types.ModuleType, bytes]:
    expected_runner_sha256 = _release_pin(
        "C2_M5_EXPECTED_RELEASE_RUNNER_SHA256",
        64,
    )
    expected_attestation_commit = _release_pin(
        "C2_M5_EXPECTED_ATTESTATION_COMMIT",
        40,
    )
    expected_manifest_sha256 = _release_pin(
        "C2_M5_EXPECTED_MANIFEST_SHA256",
        64,
    )
    _require_no_git_grafts(_REPOSITORY)
    head = _git(("rev-parse", "HEAD"), "M5 release HEAD").decode("ascii").strip()
    _require(
        len(head) == 40 and all(character in "0123456789abcdef" for character in head),
        "M5 release HEAD is invalid",
    )
    helper_relative = "agent/c2_m5_bootstrap_attestation.py"
    runner_relative = "agent/c2_m5_release_test_runner.py"
    helper_path = _REPOSITORY / helper_relative
    runner_path = _REPOSITORY / runner_relative
    manifest_relative = (
        "agent/experiments/resources/c2_m5_source_pilot_attestation_v1.json"
    )
    manifest_payload = _stable_regular(
        _ATTESTATION_MANIFEST,
        "M5 release attestation manifest",
        4 * 1024 * 1024,
    )
    _require(
        _sha256(manifest_payload) == expected_manifest_sha256,
        "M5 release manifest differs from the external digest",
    )
    manifest = _load_json(manifest_payload, "M5 release attestation manifest")
    implementation_commit = _single_parent_commit(
        _REPOSITORY,
        expected_attestation_commit,
        "M5 attestation commit",
    )
    _require(
        _git(
            ("log", "-1", "--format=%H", "--", manifest_relative),
            "M5 manifest introducing commit",
        ).decode("ascii").strip()
        == expected_attestation_commit
        and _git(
            ("merge-base", "--is-ancestor", expected_attestation_commit, head),
            "M5 attestation ancestry",
        )
        == b""
        and _git(
            (
                "diff",
                "--name-status",
                "--no-renames",
                implementation_commit,
                expected_attestation_commit,
            ),
            "M5 manifest-only commit",
        ).decode("utf-8").strip()
        == f"A\t{manifest_relative}"
        and _git(
            ("show", f"{expected_attestation_commit}:{manifest_relative}"),
            "M5 committed manifest",
        )
        == manifest_payload,
        "M5 external manifest commit binding is invalid",
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "approved_implementation_commit_full",
            "attested_paths",
        }
        and manifest.get("schema_version")
        == "c2_m5_source_pilot_attestation_v1"
        and manifest.get("approved_implementation_commit_full")
        == implementation_commit,
        "M5 release attestation manifest shape is invalid",
    )
    raw_entries = manifest.get("attested_paths")
    _require(isinstance(raw_entries, list), "M5 release attested paths are invalid")
    attested_entries: Dict[str, Mapping[str, object]] = {}
    for entry in raw_entries:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"relative_path", "git_blob_object_id", "sha256"}
            and isinstance(entry.get("relative_path"), str)
            and entry["relative_path"]
            and "\\" not in entry["relative_path"]
            and not Path(entry["relative_path"]).is_absolute()
            and ".." not in Path(entry["relative_path"]).parts
            and entry["relative_path"] not in attested_entries
            and isinstance(entry.get("git_blob_object_id"), str)
            and len(entry["git_blob_object_id"]) == 40
            and all(
                character in "0123456789abcdef"
                for character in entry["git_blob_object_id"]
            )
            and _is_sha256(entry.get("sha256")),
            "M5 release attestation entry is invalid",
        )
        attested_entries[entry["relative_path"]] = entry
    helper_payload = _stable_regular(
        helper_path,
        "M5 release attestation helper",
        2 * 1024 * 1024,
    )
    runner_payload = _stable_regular(
        runner_path,
        "M5 release test runner",
        2 * 1024 * 1024,
    )
    for relative, payload in (
        (helper_relative, helper_payload),
        (runner_relative, runner_payload),
    ):
        entry = attested_entries.get(relative)
        _require(
            entry is not None
            and entry["sha256"] == _sha256(payload)
            and _git(
                ("rev-parse", f"{implementation_commit}:{relative}"),
                f"M5 attested blob {relative}",
            ).decode("ascii").strip()
            == entry["git_blob_object_id"]
            and _git(
                ("show", f"{implementation_commit}:{relative}"),
                f"M5 attested payload {relative}",
            )
            == payload,
            f"M5 release verifier is not authenticated: {relative}",
        )
    _require(
        _sha256(runner_payload) == expected_runner_sha256,
        "M5 release runner differs from its external digest",
    )
    module_name = "c2_m5_release_verified_attestation"
    _require(
        module_name not in sys.modules,
        "M5 verified helper module is already loaded",
    )
    module = types.ModuleType(module_name)
    module.__file__ = str(helper_path)
    sys.modules[module_name] = module
    try:
        exec(
            compile(helper_payload, str(helper_path), "exec"),
            module.__dict__,
        )
    except Exception:
        del sys.modules[module_name]
        raise
    module._verify_filter_free_clean_tree(_REPOSITORY, head)
    return module, runner_payload


def _test_dependency_payloads(
    manifest_payload: bytes,
    archive_payload: bytes,
    expected_python: Mapping[str, object],
) -> Dict[str, bytes]:
    manifest = _load_json(manifest_payload, "M5 test dependency manifest")
    _require(
        set(manifest)
        == {
            "schema_version",
            "python",
            "archive",
            "source_wheels",
            "runtime_inventory",
        }
        and manifest.get("schema_version")
        == "c2_m5_py39_test_dependency_manifest_v1"
        and manifest.get("python") == expected_python,
        "M5 test dependency manifest structure is invalid",
    )
    archive = manifest["archive"]
    _require(
        isinstance(archive, dict)
        and set(archive) == {"relative_path", "bytes", "sha256"}
        and archive.get("relative_path")
        == "agent/experiments/resources/c2_m5_py39_test_dependencies_v1.zip"
        and archive.get("bytes") == len(archive_payload)
        and archive.get("sha256") == _sha256(archive_payload),
        "M5 test dependency archive binding is invalid",
    )
    wheels = manifest["source_wheels"]
    _require(
        isinstance(wheels, list)
        and len(wheels) == len(_TEST_WHEELS)
        and [entry.get("filename") for entry in wheels if isinstance(entry, dict)]
        == sorted(_TEST_WHEELS),
        "M5 test dependency wheel roster is invalid",
    )
    for entry in wheels:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"filename", "bytes", "sha256"}
            and entry.get("filename") in _TEST_WHEELS
            and isinstance(entry.get("bytes"), int)
            and entry["bytes"] > 0
            and _is_sha256(entry.get("sha256")),
            "M5 test dependency wheel binding is invalid",
        )
    inventory = manifest["runtime_inventory"]
    _require(
        isinstance(inventory, list) and len(inventory) <= _MAX_RUNTIME_FILES,
        "M5 test dependency inventory is invalid",
    )
    declared: Dict[str, Tuple[int, str]] = {}
    for entry in inventory:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"relative_path", "bytes", "sha256"}
            and isinstance(entry.get("relative_path"), str)
            and entry["relative_path"]
            and "\\" not in entry["relative_path"]
            and not Path(entry["relative_path"]).is_absolute()
            and ".." not in Path(entry["relative_path"]).parts
            and entry["relative_path"] not in declared
            and isinstance(entry.get("bytes"), int)
            and 0 <= entry["bytes"] <= _MAX_FILE_BYTES
            and _is_sha256(entry.get("sha256")),
            "M5 test dependency inventory entry is invalid",
        )
        declared[entry["relative_path"]] = (entry["bytes"], entry["sha256"])
    _require(
        list(declared) == sorted(declared)
        and sum(size for size, _ in declared.values()) <= _MAX_ARCHIVE_BYTES,
        "M5 test dependency inventory is not sorted or bounded",
    )
    payloads: Dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive_file:
            infos = archive_file.infolist()
            _require(
                not archive_file.comment and len(infos) == len(declared),
                "M5 test dependency archive coverage is invalid",
            )
            for info in infos:
                relative = info.filename
                mode = (info.external_attr >> 16) & 0xFFFF
                _require(
                    relative in declared
                    and relative not in payloads
                    and not info.is_dir()
                    and "\\" not in relative
                    and not Path(relative).is_absolute()
                    and ".." not in Path(relative).parts
                    and info.compress_type
                    in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    and info.flag_bits & ~0x800 == 0
                    and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                    and info.file_size == declared[relative][0],
                    "M5 test dependency archive entry is unsafe",
                )
                payload = archive_file.read(info)
                _require(
                    len(payload) == declared[relative][0]
                    and _sha256(payload) == declared[relative][1],
                    f"M5 test dependency payload mismatch: {relative}",
                )
                payloads[relative] = payload
    except (OSError, zipfile.BadZipFile) as exc:
        raise M5ReleaseTestError("M5 test dependency archive is invalid") from exc
    _require(
        set(payloads) == set(declared)
        and {
            "_pytest/__init__.py",
            "exceptiongroup/__init__.py",
            "iniconfig/__init__.py",
            "packaging/__init__.py",
            "pluggy/__init__.py",
            "pygments/__init__.py",
            "pytest/__init__.py",
            "tomli/__init__.py",
            "yaml/__init__.py",
        }.issubset(payloads),
        "M5 test dependency import closure is incomplete",
    )
    return payloads


def _materialize(root: Path, payloads: Mapping[str, bytes]) -> None:
    for relative, payload in sorted(payloads.items()):
        relative_path = Path(relative)
        _require(
            relative
            and "\\" not in relative
            and not relative_path.is_absolute()
            and ".." not in relative_path.parts,
            f"unsafe materialization path: {relative}",
        )
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                _require(written > 0, f"short dependency write: {relative}")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _verify_materialized(root: Path, payloads: Mapping[str, bytes]) -> None:
    observed: Dict[str, bytes] = {}
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            child = directory_path / name
            metadata = child.lstat()
            _require(
                stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                f"M5 materialized source directory is unsafe: {child}",
            )
        for name in sorted(file_names):
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            observed[relative] = _stable_regular(
                path,
                f"M5 materialized source {relative}",
                _MAX_FILE_BYTES,
            )
    _require(
        observed == dict(payloads),
        "M5 materialized source closure changed",
    )


def _clone_private_git_repository(destination: Path, head: str) -> None:
    environment = dict(_GIT_ENVIRONMENT)
    environment["GIT_ALLOW_PROTOCOL"] = "file"
    try:
        subprocess.run(
            [
                *_GIT_PREFIX,
                "-c",
                "protocol.file.allow=always",
                "clone",
                "--local",
                "--no-hardlinks",
                "--no-checkout",
                "--",
                str(_REPOSITORY),
                str(destination),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        subprocess.run(
            [
                *_GIT_PREFIX,
                "-C",
                str(destination),
                "checkout",
                "--detach",
                head,
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_GIT_ENVIRONMENT,
        )
        subprocess.run(
            [
                *_GIT_PREFIX,
                "-C",
                str(destination),
                "config",
                "core.hooksPath",
                "/dev/null",
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M5ReleaseTestError(
            "M5 private Git test repository could not be materialized"
        ) from exc
    _require(
        _git_in(destination, ("rev-parse", "HEAD"), "private Git HEAD")
        .decode("ascii")
        .strip()
        == head,
        "M5 private Git test repository has the wrong HEAD",
    )


def _git_in(repository: Path, arguments: Tuple[str, ...], label: str) -> bytes:
    try:
        completed = subprocess.run(
            [*_GIT_PREFIX, "-C", str(repository), *arguments],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M5ReleaseTestError(f"{label} could not be verified") from exc
    return completed.stdout


def _require_no_git_grafts(repository: Path) -> None:
    graft_value = _git_in(
        repository,
        ("rev-parse", "--git-path", "info/grafts"),
        "M5 graft path",
    ).decode("utf-8").strip()
    graft_path = Path(graft_value)
    if not graft_path.is_absolute():
        graft_path = repository / graft_path
    _require(
        not os.path.lexists(graft_path),
        "M5 repository graft metadata is forbidden",
    )


def _single_parent_commit(repository: Path, commit: str, label: str) -> str:
    payload = _git_in(
        repository,
        ("cat-file", "commit", commit),
        label,
    )
    raw_parents = [
        line.removeprefix(b"parent ")
        for line in payload.split(b"\n\n", 1)[0].splitlines()
        if line.startswith(b"parent ")
    ]
    try:
        parents = [parent.decode("ascii") for parent in raw_parents]
    except UnicodeDecodeError as exc:
        raise M5ReleaseTestError(f"{label} has a malformed parent") from exc
    _require(
        len(parents) == 1
        and len(parents[0]) == 40
        and all(character in "0123456789abcdef" for character in parents[0]),
        f"{label} must have exactly one literal parent",
    )
    return parents[0]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise M5ReleaseTestError(
            "M5 release test process group could not be terminated"
        ) from exc
    try:
        process.wait(timeout=_PROCESS_GROUP_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise M5ReleaseTestError(
            "M5 release test leader could not be reaped"
        ) from exc
    deadline = time.monotonic() + _PROCESS_GROUP_CLEANUP_SECONDS
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            if time.monotonic() >= deadline:
                raise M5ReleaseTestError(
                    "M5 release test process group remained unverifiable"
                ) from exc
            time.sleep(0.01)
            continue
        except OSError as exc:
            raise M5ReleaseTestError(
                "M5 release test process group state could not be verified"
            ) from exc
        if time.monotonic() >= deadline:
            raise M5ReleaseTestError(
                "M5 release test descendants survived process-group termination"
            )
        time.sleep(0.01)


def _run_release_child(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> int:
    watched_signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
    supervisor_mask = set(previous_mask).difference(watched_signals)
    previous_handlers: dict[int, object] = {}
    process: subprocess.Popen[bytes] | None = None
    interrupted_signal: int | None = None

    def terminate_from_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted_signal
        if interrupted_signal is None:
            interrupted_signal = signum

    try:
        for signum in watched_signals:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, terminate_from_signal)
        signal.pthread_sigmask(signal.SIG_SETMASK, supervisor_mask)
        if interrupted_signal is not None:
            raise M5ReleaseTestError(
                f"M5 release tests interrupted by signal {interrupted_signal}"
            )
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        if interrupted_signal is not None:
            raise M5ReleaseTestError(
                f"M5 release tests interrupted by signal {interrupted_signal}"
            )
        deadline = time.monotonic() + _TEST_TIMEOUT_SECONDS
        while True:
            if interrupted_signal is not None:
                raise M5ReleaseTestError(
                    "M5 release tests interrupted by signal "
                    f"{interrupted_signal}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise M5ReleaseTestError(
                    "M5 release tests exceeded the fixed timeout"
                )
            try:
                exit_code = process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            if interrupted_signal is not None:
                raise M5ReleaseTestError(
                    "M5 release tests interrupted by signal "
                    f"{interrupted_signal}"
                )
            return exit_code
    finally:
        active_error = sys.exc_info()[1]
        signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
        cleanup_error: Exception | None = None
        if process is not None:
            try:
                _terminate_process_group(process)
            except Exception as exc:
                cleanup_error = exc
        signal.pthread_sigmask(signal.SIG_SETMASK, supervisor_mask)
        if (
            active_error is None
            and cleanup_error is None
            and interrupted_signal is None
        ):
            signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
            if interrupted_signal is None:
                for signum, previous_handler in previous_handlers.items():
                    signal.signal(signum, previous_handler)
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if cleanup_error is not None:
            if active_error is not None:
                raise active_error.with_traceback(
                    active_error.__traceback__
                ) from cleanup_error
            raise cleanup_error
        if active_error is None and interrupted_signal is not None:
            raise M5ReleaseTestError(
                f"M5 release tests interrupted by signal {interrupted_signal}"
            )


def _run() -> int:
    production_attestation, runner_payload = (
        _verified_production_attestation_module()
    )
    runtime_attestation = production_attestation.verify_adapter_attestation(
        activate=False,
        expected_attestation_commit=_release_pin(
            "C2_M5_EXPECTED_ATTESTATION_COMMIT",
            40,
        ),
        expected_manifest_sha256=_release_pin(
            "C2_M5_EXPECTED_MANIFEST_SHA256",
            64,
        ),
        expected_bootstrap_sha256=_release_pin(
            "C2_M5_EXPECTED_BOOTSTRAP_SHA256",
            64,
        ),
        expected_python_sha256=_release_pin(
            "C2_M5_EXPECTED_PYTHON_SHA256",
            64,
        ),
        expected_python_library_sha256=_release_pin(
            "C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256",
            64,
        ),
    )
    _require(
        runtime_attestation.path_payloads[
            "agent/c2_m5_release_test_runner.py"
        ]
        == runner_payload,
        "M5 release runner is outside the verified runtime closure",
    )

    retained_payloads = runtime_attestation.path_payloads
    production_manifest_payload = _retained_attested_payload(
        retained_payloads,
        _PRODUCTION_MANIFEST.relative_to(_REPOSITORY).as_posix(),
        "M5 production dependency manifest",
        2 * 1024 * 1024,
    )
    production_archive_payload = _retained_attested_payload(
        retained_payloads,
        _PRODUCTION_ARCHIVE.relative_to(_REPOSITORY).as_posix(),
        "M5 production dependency archive",
        _MAX_ARCHIVE_BYTES,
    )
    production_manifest = _load_json(
        production_manifest_payload,
        "M5 production dependency manifest",
    )
    _require(
        production_manifest.get("python")
        == production_attestation._python_runtime_binding()
        and platform.machine() == "arm64",
        "M5 release runner is not using the fixed native Python runtime",
    )
    _, production_payloads = production_attestation._verified_dependency_payloads(
        production_manifest_payload,
        production_archive_payload,
    )
    test_manifest_payload = _retained_attested_payload(
        retained_payloads,
        _TEST_MANIFEST.relative_to(_REPOSITORY).as_posix(),
        "M5 test dependency manifest",
        2 * 1024 * 1024,
    )
    test_archive_payload = _retained_attested_payload(
        retained_payloads,
        _TEST_ARCHIVE.relative_to(_REPOSITORY).as_posix(),
        "M5 test dependency archive",
        _MAX_ARCHIVE_BYTES,
    )
    test_payloads = _test_dependency_payloads(
        test_manifest_payload,
        test_archive_payload,
        production_manifest["python"],
    )
    with tempfile.TemporaryDirectory(
        prefix="c2-m5-release-",
        dir=str(_trusted_temporary_parent()),
    ) as temporary:
        temporary_root = Path(temporary)
        temporary_root.chmod(0o700)
        _remove_acl(temporary_root, "M5 release temporary root")
        temporary_metadata = temporary_root.lstat()
        _require(
            stat.S_ISDIR(temporary_metadata.st_mode)
            and not stat.S_ISLNK(temporary_metadata.st_mode)
            and temporary_metadata.st_uid == os.geteuid()
            and temporary_metadata.st_mode & 0o077 == 0,
            "M5 release temporary root is unsafe",
        )
        production_root = temporary_root / "production"
        test_root = temporary_root / "tests"
        source_root = temporary_root / "source"
        git_repository = temporary_root / "git-repository"
        temporary_directory = temporary_root / "tmp"
        pytest_control_root = temporary_root / "pytest-control"
        production_root.mkdir(mode=0o700)
        test_root.mkdir(mode=0o700)
        source_root.mkdir(mode=0o700)
        temporary_directory.mkdir(mode=0o700)
        pytest_control_root.mkdir(mode=0o700)
        _materialize(production_root, production_payloads)
        _materialize(test_root, test_payloads)
        _materialize(source_root, runtime_attestation.path_payloads)
        _materialize(pytest_control_root, {"pytest.ini": _PYTEST_CONFIG})
        _verify_materialized(production_root, production_payloads)
        _verify_materialized(test_root, test_payloads)
        _verify_materialized(source_root, runtime_attestation.path_payloads)
        _verify_materialized(
            pytest_control_root,
            {"pytest.ini": _PYTEST_CONFIG},
        )
        _clone_private_git_repository(
            git_repository,
            runtime_attestation.head_commit,
        )
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TMPDIR": str(temporary_directory),
            "C2_M5_TEST_GIT_REPOSITORY": str(git_repository),
        }
        exit_code = _run_release_child(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _PYTEST_CHILD,
                str(production_root),
                str(test_root),
                str(source_root),
                str(pytest_control_root / "pytest.ini"),
                *_TEST_TARGETS,
            ],
            cwd=source_root,
            environment=environment,
        )
        _verify_materialized(production_root, production_payloads)
        _verify_materialized(test_root, test_payloads)
        _verify_materialized(source_root, runtime_attestation.path_payloads)
        _verify_materialized(
            pytest_control_root,
            {"pytest.ini": _PYTEST_CONFIG},
        )
        return exit_code


def main() -> int:
    _require(len(sys.argv) == 1, "M5 release runner takes no arguments")
    return _run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M5ReleaseTestError as exc:
        print(f"M5 release test error: {exc}", file=sys.stderr)
        raise SystemExit(2)

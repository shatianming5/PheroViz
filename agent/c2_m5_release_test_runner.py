"""Run the fixed M5 release suite under an isolated, hash-bound Python 3.9."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path
from typing import Callable, Dict, Mapping, NamedTuple, Tuple


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
_CONTAINED_SPAWN_TIMEOUT_SECONDS = 5
_RELEASE_TEST_SESSION_ENV = "C2_M5_RELEASE_TEST_SESSION_ID"
_RELEASE_TEST_REGISTRY_ENV = "C2_M5_RELEASE_TEST_GROUP_REGISTRY"
_RELEASE_TEST_REGISTRY_DEVICE_ENV = "C2_M5_RELEASE_TEST_GROUP_REGISTRY_DEVICE"
_RELEASE_TEST_REGISTRY_INODE_ENV = "C2_M5_RELEASE_TEST_GROUP_REGISTRY_INODE"
_WATCHED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
_ORIGINAL_POPEN = subprocess.Popen
_ORIGINAL_SELECTOR = selectors.DefaultSelector
_RELEASE_TEST_CONTAINMENT_INSTALLED = False
_RELEASE_TEST_PINNED_REGISTRY: tuple[Path, int, int] | None = None
_CONTAINED_EXEC_TRAMPOLINE = """\
import os
import signal
import sys

ready_fd = int(sys.argv[1])
registry_fd = int(sys.argv[2])
release_fd = int(sys.argv[3])
command = sys.argv[4:]
if os.read(release_fd, 1) != b"1":
    raise SystemExit("contained process release was incomplete")
os.close(release_fd)
os.setpgid(0, 0)
anchor_pid = os.fork()
if anchor_pid == 0:
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    for raw_descriptor in os.listdir("/dev/fd"):
        try:
            os.close(int(raw_descriptor))
        except OSError:
            pass
    while True:
        signal.pause()
registration = f"{os.getpid()} {anchor_pid}\\n".encode("ascii")
if os.write(registry_fd, registration) != len(registration):
    raise SystemExit("contained group registration was incomplete")
os.close(registry_fd)
os.write(ready_fd, b"1")
os.close(ready_fd)
os.execvpe(command[0], command, os.environ)
"""
_PYTEST_CONFIG = b"[pytest]\naddopts =\n"
_PYTEST_CHILD = """\
import os
import signal
import sys

forbidden_completion_fd, production, tests, source, config, *targets = sys.argv[1:]
try:
    os.fstat(int(forbidden_completion_fd))
except OSError:
    pass
else:
    os.close(int(forbidden_completion_fd))
    raise SystemExit("M5 pytest worker inherited the completion descriptor")
for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(signum, signal.SIG_DFL)
sys.path[:0] = [tests, production, os.path.join(source, "agent")]
from c2_m5_source_pilot_bootstrap import _install_python39_compatibility
import c2_m5_release_test_runner as release_runner

_install_python39_compatibility()
release_runner._install_release_test_session_containment()
import pytest

os.chdir(source)
for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(signum, signal.SIG_IGN)
exit_code = int(
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
raise SystemExit(exit_code)
"""
_PYTEST_SESSION_SUPERVISOR = """\
import os
import signal
import subprocess
import sys

completion_fd = int(sys.argv[1])
worker_code = sys.argv[2]
worker_arguments = sys.argv[3:]
os.set_inheritable(completion_fd, False)
for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(signum, signal.SIG_IGN)
os.environ["C2_M5_RELEASE_TEST_SESSION_ID"] = str(os.getpid())
worker = subprocess.Popen(
    [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        worker_code,
        str(completion_fd),
        *worker_arguments,
    ],
    stdin=subprocess.DEVNULL,
    close_fds=True,
)
return_code = worker.wait()
if return_code < 0:
    return_code = min(255, 128 - return_code)
completion = f"{return_code}\\n".encode("ascii")
if os.write(completion_fd, completion) != len(completion):
    raise SystemExit("M5 test completion write was incomplete")
os.close(completion_fd)
while True:
    signal.pause()
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


class _TestSessionRegistry(NamedTuple):
    path: Path
    descriptor: int
    device: int
    inode: int


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


def _release_test_session_id() -> int | None:
    raw = os.environ.get(_RELEASE_TEST_SESSION_ENV)
    if raw is None:
        return None
    try:
        session_id = int(raw)
    except ValueError as exc:
        raise M5ReleaseTestError(
            "M5 release-test session capability is malformed"
        ) from exc
    _require(
        session_id > 0 and os.getsid(0) == session_id,
        "M5 release-test session capability is invalid",
    )
    return session_id


def _release_test_registry_binding() -> tuple[Path, int, int]:
    raw = os.environ.get(_RELEASE_TEST_REGISTRY_ENV)
    _require(
        raw is not None and os.path.isabs(raw),
        "M5 release-test group registry capability is missing",
    )
    raw_device = os.environ.get(_RELEASE_TEST_REGISTRY_DEVICE_ENV)
    raw_inode = os.environ.get(_RELEASE_TEST_REGISTRY_INODE_ENV)
    try:
        device = int(raw_device or "")
        inode = int(raw_inode or "")
    except ValueError as exc:
        raise M5ReleaseTestError(
            "M5 release-test group registry identity is malformed"
        ) from exc
    _require(
        device > 0 and inode > 0,
        "M5 release-test group registry identity is invalid",
    )
    return _verify_release_test_registry_binding(
        Path(raw),
        device,
        inode,
    )


def _verify_release_test_registry_binding(
    path: Path,
    device: int,
    inode: int,
) -> tuple[Path, int, int]:
    metadata = path.lstat()
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_nlink == 1
        and metadata.st_mode & 0o077 == 0,
        "M5 release-test group registry is unsafe",
    )
    _require(
        (metadata.st_dev, metadata.st_ino) == (device, inode),
        "M5 release-test group registry identity changed",
    )
    return path, device, inode


def _contained_release_test_registry_binding() -> tuple[Path, int, int]:
    if _RELEASE_TEST_CONTAINMENT_INSTALLED:
        _require(
            _RELEASE_TEST_PINNED_REGISTRY is not None,
            "M5 release-test registry was not pinned at containment installation",
        )
        return _verify_release_test_registry_binding(
            *_RELEASE_TEST_PINNED_REGISTRY,
        )
    return _release_test_registry_binding()


def _open_release_test_registry_for_append(
    binding: tuple[Path, int, int] | None = None,
) -> int:
    path, device, inode = (
        _release_test_registry_binding()
        if binding is None
        else _verify_release_test_registry_binding(*binding)
    )
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_APPEND
        | os.O_NONBLOCK
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1
            and metadata.st_mode & 0o077 == 0
            and (metadata.st_dev, metadata.st_ino) == (device, inode),
            "M5 release-test group registry is unsafe",
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _terminate_unacknowledged_process(
    process: subprocess.Popen[bytes],
    session_id: int,
) -> None:
    errors: list[Exception] = []
    try:
        os.kill(process.pid, signal.SIGSTOP)
    except ProcessLookupError:
        pass
    except OSError as exc:
        error = M5ReleaseTestError(
            "M5 contained process could not be stopped for cleanup"
        )
        error.__cause__ = exc
        errors.append(error)
    else:
        deadline = time.monotonic() + _PROCESS_GROUP_CLEANUP_SECONDS
        while True:
            try:
                members = _owned_test_session_members(session_id)
            except Exception as exc:
                errors.append(exc)
                break
            state = next(
                (
                    member_state
                    for process_id, _, member_state in members
                    if process_id == process.pid
                ),
                None,
            )
            if state is None or state[:1] in {"T", "Z"}:
                break
            if time.monotonic() >= deadline:
                errors.append(
                    M5ReleaseTestError(
                        "M5 contained process did not stop before cleanup"
                    )
                )
                break
            time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        error = M5ReleaseTestError(
            "M5 contained process group could not be terminated"
        )
        error.__cause__ = exc
        errors.append(error)
    try:
        process.kill()
    except ProcessLookupError:
        pass
    except OSError as exc:
        error = M5ReleaseTestError(
            "M5 contained process could not be terminated"
        )
        error.__cause__ = exc
        errors.append(error)
    deadline = time.monotonic() + _PROCESS_GROUP_CLEANUP_SECONDS
    while True:
        try:
            members = _owned_test_session_members(session_id)
        except Exception as exc:
            errors.append(exc)
            break
        active_group = tuple(
            (process_id, state)
            for process_id, group_id, state in members
            if group_id == process.pid and state[:1] != "Z"
        )
        if not active_group:
            break
        if time.monotonic() >= deadline:
            errors.append(
                M5ReleaseTestError(
                    "M5 contained process group retained active members"
                )
            )
            break
        time.sleep(0.01)
    try:
        process.wait(timeout=_PROCESS_GROUP_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        error = M5ReleaseTestError(
            "M5 contained process could not be reaped"
        )
        error.__cause__ = exc
        errors.append(error)
    if errors:
        if len(errors) > 1:
            raise errors[0] from errors[-1]
        raise errors[0]


def _register_and_release_contained_group(
    registry_fd: int,
    release_fd: int,
    process_id: int,
) -> None:
    pending_registration = f"{process_id} {process_id}\n".encode("ascii")
    _require(
        os.write(registry_fd, pending_registration)
        == len(pending_registration),
        "M5 contained pending registration was incomplete",
    )
    _require(
        os.write(release_fd, b"1") == 1,
        "M5 contained process release was incomplete",
    )


def _start_contained_popen(
    popen_factory: Callable[..., subprocess.Popen[bytes]],
    popen_args: tuple[object, ...],
    popen_kwargs: Mapping[str, object],
) -> subprocess.Popen[bytes]:
    kwargs = dict(popen_kwargs)
    _require(
        len(popen_args) == 1
        and "args" not in kwargs
        and not kwargs.get("shell", False)
        and kwargs.get("preexec_fn") is None
        and kwargs.get("executable") is None,
        "M5 contained process has an unsupported spawn shape",
    )
    command = popen_args[0]
    _require(
        isinstance(command, (list, tuple)) and bool(command),
        "M5 contained process command is invalid",
    )
    normalized_command = [
        os.fsdecode(os.fspath(argument)) for argument in command
    ]
    session_id = _release_test_session_id()
    _require(
        session_id is not None,
        "M5 contained process lacks a release-test session",
    )
    registry_path, registry_device, registry_inode = (
        _contained_release_test_registry_binding()
    )
    child_environment = kwargs.get("env")
    if child_environment is not None:
        _require(
            isinstance(child_environment, Mapping),
            "M5 contained process environment is invalid",
        )
        contained_environment = dict(child_environment)
        contained_environment[_RELEASE_TEST_SESSION_ENV] = str(session_id)
        contained_environment[_RELEASE_TEST_REGISTRY_ENV] = str(registry_path)
        contained_environment[_RELEASE_TEST_REGISTRY_DEVICE_ENV] = str(
            registry_device
        )
        contained_environment[_RELEASE_TEST_REGISTRY_INODE_ENV] = str(
            registry_inode
        )
        kwargs["env"] = contained_environment
    existing_pass_fds = tuple(kwargs.get("pass_fds", ()))
    read_fd, write_fd = os.pipe()
    release_read_fd, release_write_fd = os.pipe()
    registry_fd = -1
    process: subprocess.Popen[bytes] | None = None
    try:
        registry_fd = _open_release_test_registry_for_append(
            (registry_path, registry_device, registry_inode)
        )
        kwargs["start_new_session"] = False
        kwargs["pass_fds"] = (
            *existing_pass_fds,
            write_fd,
            registry_fd,
            release_read_fd,
        )
        wrapped_command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _CONTAINED_EXEC_TRAMPOLINE,
            str(write_fd),
            str(registry_fd),
            str(release_read_fd),
            *normalized_command,
        ]
        process = popen_factory(wrapped_command, **kwargs)
        os.close(release_read_fd)
        release_read_fd = -1
        _register_and_release_contained_group(
            registry_fd,
            release_write_fd,
            process.pid,
        )
        os.close(release_write_fd)
        release_write_fd = -1
        os.close(write_fd)
        write_fd = -1
        with _ORIGINAL_SELECTOR() as selector:
            selector.register(read_fd, selectors.EVENT_READ)
            ready = selector.select(_CONTAINED_SPAWN_TIMEOUT_SECONDS)
            _require(
                bool(ready) and os.read(read_fd, 1) == b"1",
                "M5 contained process did not acknowledge its process group",
            )
        return process
    except BaseException:
        active_error = sys.exc_info()[1]
        if process is not None:
            try:
                _terminate_unacknowledged_process(process, session_id)
            except Exception as cleanup_error:
                if active_error is not None:
                    raise active_error.with_traceback(
                        active_error.__traceback__
                    ) from cleanup_error
                raise
        raise
    finally:
        os.close(read_fd)
        if registry_fd >= 0:
            os.close(registry_fd)
        if write_fd >= 0:
            os.close(write_fd)
        if release_read_fd >= 0:
            os.close(release_read_fd)
        if release_write_fd >= 0:
            os.close(release_write_fd)


def _release_test_popen(
    *popen_args: object,
    **popen_kwargs: object,
) -> subprocess.Popen[bytes]:
    if not popen_kwargs.get("start_new_session", False):
        return _ORIGINAL_POPEN(*popen_args, **popen_kwargs)
    _require(
        _release_test_session_id() is not None,
        "M5 nested session spawn lacks release-test containment",
    )
    return _start_contained_popen(
        _ORIGINAL_POPEN,
        popen_args,
        popen_kwargs,
    )


def _install_release_test_session_containment() -> None:
    global _RELEASE_TEST_CONTAINMENT_INSTALLED
    global _RELEASE_TEST_PINNED_REGISTRY
    session_id = _release_test_session_id()
    _require(
        session_id is not None and session_id != os.getpid(),
        "M5 release-test worker lacks its trusted session supervisor",
    )
    registry_binding = _release_test_registry_binding()
    _require(
        subprocess.Popen in {_ORIGINAL_POPEN, _release_test_popen},
        "M5 release-test process launcher is already modified",
    )
    if _RELEASE_TEST_CONTAINMENT_INSTALLED:
        _require(
            _RELEASE_TEST_PINNED_REGISTRY == registry_binding,
            "M5 release-test registry binding changed after installation",
        )
    else:
        _RELEASE_TEST_PINNED_REGISTRY = registry_binding
    setattr(subprocess, "Popen", _release_test_popen)
    _RELEASE_TEST_CONTAINMENT_INSTALLED = True


def _owned_test_session_members(
    session_id: int,
) -> tuple[tuple[int, int, str], ...]:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,pgid=,uid=,state="],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/var/empty",
                "LANG": "C",
                "LC_ALL": "C",
            },
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise M5ReleaseTestError(
            "M5 release-test session could not be inspected"
        ) from exc
    members: list[tuple[int, int, str]] = []
    for raw_line in completed.stdout.splitlines():
        pieces = raw_line.split()
        _require(
            len(pieces) == 4 and bool(pieces[3]),
            "M5 release-test process table is malformed",
        )
        try:
            process_id, group_id, owner = map(int, pieces[:3])
        except ValueError as exc:
            raise M5ReleaseTestError(
                "M5 release-test process table is malformed"
            ) from exc
        if owner != os.geteuid():
            continue
        try:
            observed_session = os.getsid(process_id)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise M5ReleaseTestError(
                "M5 release-test process session is unverifiable"
            ) from exc
        if observed_session == session_id:
            members.append((process_id, group_id, pieces[3]))
    return tuple(sorted(members))


def _registry_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _registered_test_groups(
    registry: _TestSessionRegistry,
    session_id: int,
) -> tuple[set[tuple[int, int]], bool]:
    before = os.fstat(registry.descriptor)
    payload = bytearray()
    offset = 0
    while True:
        block = os.pread(registry.descriptor, 64 * 1024, offset)
        if not block:
            break
        payload.extend(block)
        offset += len(block)
        _require(
            len(payload) <= 64 * 1024,
            "M5 release-test group registry is oversized",
        )
    after = os.fstat(registry.descriptor)
    invalid = False
    try:
        path_metadata = registry.path.lstat()
    except OSError:
        invalid = True
        path_metadata = None
    _require(
        _registry_identity(before) == _registry_identity(after)
        and stat.S_ISREG(after.st_mode)
        and after.st_uid == os.geteuid()
        and (after.st_dev, after.st_ino) == (registry.device, registry.inode),
        "M5 release-test group registry changed during cleanup",
    )
    if after.st_nlink != 1 or after.st_mode & 0o077:
        invalid = True
    if (
        path_metadata is None
        or not stat.S_ISREG(path_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_uid != os.geteuid()
        or path_metadata.st_nlink != 1
        or path_metadata.st_mode & 0o077
        or (path_metadata.st_dev, path_metadata.st_ino)
        != (registry.device, registry.inode)
    ):
        invalid = True
    registrations: set[tuple[int, int]] = set()
    for raw_line in bytes(payload).splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            invalid = True
            continue
        pieces = raw_line[:-1].split()
        if len(pieces) != 2:
            invalid = True
            continue
        try:
            group_id, anchor_pid = map(int, pieces)
        except ValueError:
            invalid = True
            continue
        if group_id <= 0 or anchor_pid <= 0:
            invalid = True
            continue
        try:
            anchor_session = os.getsid(anchor_pid)
            anchor_group = os.getpgid(anchor_pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            invalid = True
            continue
        if anchor_session != session_id or anchor_group != group_id:
            invalid = True
            continue
        registrations.add((group_id, anchor_pid))
    return registrations, invalid


def _signal_registered_groups(
    registrations: set[tuple[int, int]],
    session_id: int,
    signum: int,
) -> tuple[bool, list[Exception]]:
    invalid = False
    errors: list[Exception] = []
    for group_id, anchor_pid in sorted(registrations):
        try:
            anchor_session = os.getsid(anchor_pid)
            anchor_group = os.getpgid(anchor_pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            invalid = True
            continue
        if anchor_session != session_id or anchor_group != group_id:
            invalid = True
            continue
        try:
            os.killpg(group_id, signum)
        except ProcessLookupError:
            pass
        except OSError as exc:
            action = "frozen" if signum == signal.SIGSTOP else "terminated"
            error = M5ReleaseTestError(
                f"M5 registered release-test group could not be {action}"
            )
            error.__cause__ = exc
            errors.append(error)
    return invalid, errors


def _freeze_test_session(
    process: subprocess.Popen[bytes],
    registry: _TestSessionRegistry,
    deadline: float,
) -> tuple[
    tuple[tuple[int, int, str], ...],
    set[tuple[int, int]],
    list[Exception],
]:
    errors: list[Exception] = []
    previous: tuple[
        tuple[tuple[int, int, str], ...],
        tuple[tuple[int, int], ...],
    ] | None = None
    registrations: set[tuple[int, int]] = set()
    while True:
        try:
            os.killpg(process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            errors.append(
                M5ReleaseTestError(
                    "M5 release-test session leader disappeared before cleanup"
                )
            )
        except OSError as exc:
            error = M5ReleaseTestError(
                "M5 release-test leader group could not be frozen"
            )
            error.__cause__ = exc
            errors.append(error)
        current, invalid = _registered_test_groups(registry, process.pid)
        registrations.update(current)
        invalid_signal, signal_errors = _signal_registered_groups(
            current,
            process.pid,
            signal.SIGSTOP,
        )
        errors.extend(signal_errors)
        if invalid or invalid_signal:
            errors.append(
                M5ReleaseTestError(
                    "M5 release-test group registry contains invalid registrations"
                )
            )
        members = _owned_test_session_members(process.pid)
        approved_groups = {
            process.pid,
            *(group_id for group_id, _ in registrations),
        }
        unapproved = {
            group_id
            for process_id, group_id, _ in members
            if process_id != process.pid and group_id not in approved_groups
        }
        if unapproved:
            errors.append(
                M5ReleaseTestError(
                    "M5 release-test session contains an unapproved process group"
                )
            )
            return members, registrations, errors
        all_stopped = all(
            state[:1] in {"T", "Z"} for _, _, state in members
        )
        snapshot = (members, tuple(sorted(registrations)))
        if all_stopped and snapshot == previous:
            final, invalid = _registered_test_groups(registry, process.pid)
            registrations.update(final)
            if invalid or final != current:
                errors.append(
                    M5ReleaseTestError(
                        "M5 release-test group registry changed after freeze"
                    )
                )
            return members, registrations, errors
        previous = snapshot if all_stopped else None
        if time.monotonic() >= deadline:
            errors.append(
                M5ReleaseTestError(
                    "M5 release-test session could not be frozen"
                )
            )
            return members, registrations, errors
        time.sleep(0.01)


def _terminate_test_session(
    process: subprocess.Popen[bytes],
    registry: _TestSessionRegistry,
) -> None:
    deadline = time.monotonic() + _PROCESS_GROUP_CLEANUP_SECONDS
    errors: list[Exception] = []
    registrations: set[tuple[int, int]] = set()

    if process.poll() is not None:
        errors.append(
            M5ReleaseTestError(
                "M5 release-test session supervisor exited before cleanup"
            )
        )
    else:
        try:
            _require(
                os.getsid(process.pid) == process.pid
                and os.getpgid(process.pid) == process.pid,
                "M5 release-test session supervisor identity is invalid",
            )
        except Exception as exc:
            errors.append(exc)

    try:
        try:
            _, frozen_registrations, freeze_errors = _freeze_test_session(
                process,
                registry,
                deadline,
            )
            registrations.update(frozen_registrations)
            errors.extend(freeze_errors)
        except Exception as exc:
            errors.append(exc)

        for _ in range(2):
            try:
                current, invalid = _registered_test_groups(
                    registry,
                    process.pid,
                )
                registrations.update(current)
                if invalid:
                    errors.append(
                        M5ReleaseTestError(
                            "M5 release-test group registry contains "
                            "invalid registrations"
                        )
                    )
            except Exception as exc:
                errors.append(exc)
                current = set()
            invalid, signal_errors = _signal_registered_groups(
                registrations,
                process.pid,
                signal.SIGKILL,
            )
            errors.extend(signal_errors)
            if invalid:
                errors.append(
                    M5ReleaseTestError(
                        "M5 release-test group identity changed during cleanup"
                    )
                )

        try:
            previous: tuple[tuple[int, int, str], ...] | None = None
            while True:
                members = _owned_test_session_members(process.pid)
                remaining = tuple(
                    member for member in members if member[0] != process.pid
                )
                if not remaining and members == previous:
                    break
                approved_groups = {
                    group_id for group_id, _ in registrations
                }
                if any(
                    group_id not in approved_groups
                    for _, group_id, _ in remaining
                ):
                    errors.append(
                        M5ReleaseTestError(
                            "M5 release-test session retained an unapproved "
                            "process group"
                        )
                    )
                    break
                _, signal_errors = _signal_registered_groups(
                    registrations,
                    process.pid,
                    signal.SIGKILL,
                )
                errors.extend(signal_errors)
                previous = members
                if time.monotonic() >= deadline:
                    errors.append(
                        M5ReleaseTestError(
                            "M5 release-test session retained active processes"
                        )
                    )
                    break
                time.sleep(0.01)
            if any(
                process_id != process.pid
                for process_id, _, _ in _owned_test_session_members(process.pid)
            ):
                errors.append(
                    M5ReleaseTestError(
                        "M5 release-test session cleanup was incomplete"
                    )
                )
        except Exception as exc:
            errors.append(exc)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            error = M5ReleaseTestError(
                "M5 release-test leader group could not be terminated"
            )
            error.__cause__ = exc
            errors.append(error)
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError as exc:
                error = M5ReleaseTestError(
                    "M5 release-test session supervisor could not be terminated"
                )
                error.__cause__ = exc
                errors.append(error)
        try:
            return_code = process.wait(timeout=_PROCESS_GROUP_CLEANUP_SECONDS)
            if return_code != -signal.SIGKILL:
                errors.append(
                    M5ReleaseTestError(
                        "M5 release-test session supervisor exited unexpectedly"
                    )
                )
        except subprocess.TimeoutExpired as exc:
            error = M5ReleaseTestError(
                "M5 release-test session supervisor could not be reaped"
            )
            error.__cause__ = exc
            errors.append(error)
    if errors:
        if len(errors) > 1:
            raise errors[0] from errors[-1]
        raise errors[0]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    errors: list[Exception] = []
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        error = M5ReleaseTestError(
            "M5 release test process group could not be terminated"
        )
        error.__cause__ = exc
        errors.append(error)
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as exc:
            error = M5ReleaseTestError(
                "M5 release test leader could not be terminated"
            )
            error.__cause__ = exc
            errors.append(error)
    try:
        process.wait(timeout=_PROCESS_GROUP_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        error = M5ReleaseTestError(
            "M5 release test leader could not be reaped"
        )
        error.__cause__ = exc
        errors.append(error)
    deadline = time.monotonic() + _PROCESS_GROUP_CLEANUP_SECONDS
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        except PermissionError as exc:
            if time.monotonic() >= deadline:
                error = M5ReleaseTestError(
                    "M5 release test process group remained unverifiable"
                )
                error.__cause__ = exc
                errors.append(error)
                break
            time.sleep(0.01)
            continue
        except OSError as exc:
            error = M5ReleaseTestError(
                "M5 release test process group state could not be verified"
            )
            error.__cause__ = exc
            errors.append(error)
            break
        if time.monotonic() >= deadline:
            errors.append(
                M5ReleaseTestError(
                    "M5 release test descendants survived process-group "
                    "termination"
                )
            )
            break
        time.sleep(0.01)
    if errors:
        if len(errors) > 1:
            raise errors[0] from errors[-1]
        raise errors[0]


def _open_test_session_registry(path: Path) -> _TestSessionRegistry:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_NONBLOCK
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1
            and metadata.st_mode & 0o077 == 0,
            "M5 release-test group registry is unsafe",
        )
        return _TestSessionRegistry(
            path=path,
            descriptor=descriptor,
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_release_process_topology(
    retained_payloads: Mapping[str, bytes],
) -> None:
    runner_relative = "agent/c2_m5_release_test_runner.py"
    forbidden_fragments = (
        b"set" + b"sid",
        b"set" + b"pgrp",
        b"POSIX_SPAWN_" + b"SETSID",
        b"preexec_fn=" + b"os." + b"set" + b"sid",
    )
    for relative, payload in retained_payloads.items():
        if not relative.endswith(".py"):
            continue
        _require(
            not any(fragment in payload for fragment in forbidden_fragments),
            f"M5 release process topology contains forbidden session control: "
            f"{relative}",
        )
        _require(
            relative == runner_relative or b"_ORIGINAL_POPEN" not in payload,
            f"M5 release process topology bypasses containment: {relative}",
        )
    expected_new_sessions = {
        runner_relative: 1,
        "agent/experiments/c2_m5_source_pilot.py": 1,
        "agent/tests/test_c2_m5_source_pilot.py": 2,
    }
    for relative, payload in retained_payloads.items():
        if not relative.endswith(".py"):
            continue
        normalized = b"".join(payload.split())
        observed = normalized.count(
            b"start_new_" + b"session=True"
        ) + normalized.count(
            b'"start_new_' + b'session":True'
        ) + normalized.count(
            b"'start_new_" + b"session':True"
        )
        _require(
            observed == expected_new_sessions.get(relative, 0),
            f"M5 release process topology changed: {relative}",
        )
    runner_payload = retained_payloads[
        runner_relative
    ]
    _require(
        runner_payload.count(b"os.set" + b"pgid(0, 0)") == 1
        and runner_payload.count(
            b'"start_new_' + b'session": True'
        )
        == 1
        and runner_payload.count(
            b'kwargs["start_new_' + b'session"] = False'
        )
        == 1,
        "M5 release runner process topology changed",
    )


class _ReleaseSignalSupervisor:
    def __init__(self) -> None:
        self.previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            _WATCHED_SIGNALS,
        )
        self.supervisor_mask = set(self.previous_mask).difference(
            _WATCHED_SIGNALS
        )
        self.previous_handlers: dict[int, object] = {}
        self.interrupted_signal: int | None = None
        try:
            for signum in _WATCHED_SIGNALS:
                self.previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._record)
            signal.pthread_sigmask(
                signal.SIG_SETMASK,
                self.supervisor_mask,
            )
        except BaseException:
            for signum, previous in self.previous_handlers.items():
                signal.signal(signum, previous)
            signal.pthread_sigmask(
                signal.SIG_SETMASK,
                self.previous_mask,
            )
            raise

    def _record(self, signum: int, _frame: object) -> None:
        if self.interrupted_signal is None:
            self.interrupted_signal = signum

    def checkpoint(self) -> None:
        if self.interrupted_signal is not None:
            raise M5ReleaseTestError(
                "M5 release lifecycle interrupted by signal "
                f"{self.interrupted_signal}"
            )

    def block_for_cleanup(self) -> None:
        signal.pthread_sigmask(signal.SIG_BLOCK, _WATCHED_SIGNALS)

    def resume_after_cleanup(self) -> None:
        signal.pthread_sigmask(
            signal.SIG_SETMASK,
            self.supervisor_mask,
        )

    def finish(self, active_error: BaseException | None) -> None:
        self.block_for_cleanup()
        self.resume_after_cleanup()
        self.block_for_cleanup()
        for signum, previous in self.previous_handlers.items():
            signal.signal(signum, previous)
        signal.pthread_sigmask(
            signal.SIG_SETMASK,
            self.previous_mask,
        )
        if active_error is None:
            self.checkpoint()


def _run_with_signal_supervision(
    operation: Callable[[_ReleaseSignalSupervisor], int],
) -> int:
    supervisor = _ReleaseSignalSupervisor()
    try:
        return operation(supervisor)
    finally:
        supervisor.finish(sys.exc_info()[1])


def _run_release_child(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    _supervisor: _ReleaseSignalSupervisor | None = None,
    _completion_pipe: tuple[int, int] | None = None,
    _session_registry: _TestSessionRegistry | None = None,
) -> int:
    if _supervisor is None:
        return _run_with_signal_supervision(
            lambda supervisor: _run_release_child(
                command,
                cwd=cwd,
                environment=environment,
                _supervisor=supervisor,
                _completion_pipe=_completion_pipe,
                _session_registry=_session_registry,
            )
        )
    supervisor = _supervisor
    process: subprocess.Popen[bytes] | None = None
    root_session = _session_registry is not None
    contained = not root_session and _release_test_session_id() is not None

    try:
        supervisor.checkpoint()
        if root_session:
            _require(
                _completion_pipe is not None
                and not _RELEASE_TEST_CONTAINMENT_INSTALLED
                and all(
                    name not in os.environ
                    for name in (
                        _RELEASE_TEST_SESSION_ENV,
                        _RELEASE_TEST_REGISTRY_ENV,
                        _RELEASE_TEST_REGISTRY_DEVICE_ENV,
                        _RELEASE_TEST_REGISTRY_INODE_ENV,
                    )
                ),
                "M5 root release session rejects ambient capabilities",
            )
        spawn_kwargs = {
            "cwd": cwd,
            "env": dict(environment),
            "stdin": subprocess.DEVNULL,
            "start_new_session": True,
        }
        if _completion_pipe is not None:
            spawn_kwargs["pass_fds"] = (_completion_pipe[1],)
        if contained:
            process = _start_contained_popen(
                subprocess.Popen,
                (command,),
                spawn_kwargs,
            )
        elif root_session:
            process = _ORIGINAL_POPEN(command, **spawn_kwargs)
        else:
            process = subprocess.Popen(command, **spawn_kwargs)
        if _completion_pipe is not None:
            os.close(_completion_pipe[1])
            _completion_pipe = (_completion_pipe[0], -1)
        supervisor.checkpoint()
        deadline = time.monotonic() + _TEST_TIMEOUT_SECONDS
        completion_payload = bytearray()
        while True:
            supervisor.checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise M5ReleaseTestError(
                    "M5 release tests exceeded the fixed timeout"
                )
            if _completion_pipe is not None:
                with _ORIGINAL_SELECTOR() as selector:
                    selector.register(
                        _completion_pipe[0],
                        selectors.EVENT_READ,
                    )
                    ready = selector.select(min(0.1, remaining))
                if not ready:
                    continue
                block = os.read(_completion_pipe[0], 64)
                if not block:
                    break
                completion_payload.extend(block)
                _require(
                    len(completion_payload) <= 16,
                    "M5 release-test completion status is oversized",
                )
                continue
            try:
                exit_code = process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            supervisor.checkpoint()
            return exit_code
        _require(
            completion_payload.endswith(b"\n")
            and completion_payload.count(b"\n") == 1
            and completion_payload[:-1].isdigit(),
            "M5 release-test completion status is malformed",
        )
        exit_code = int(completion_payload[:-1])
        _require(
            0 <= exit_code <= 255,
            "M5 release-test completion status is invalid",
        )
        _require(
            process.poll() is None
            and os.getsid(process.pid)
            == (
                process.pid
                if root_session
                else _release_test_session_id()
            )
            and os.getpgid(process.pid) == process.pid,
            "M5 release-test session supervisor did not remain live",
        )
        supervisor.checkpoint()
        return exit_code
    finally:
        active_error = sys.exc_info()[1]
        supervisor.block_for_cleanup()
        cleanup_error: Exception | None = None
        if process is not None:
            try:
                if root_session:
                    _require(
                        _session_registry is not None,
                        "M5 release-test registry is unavailable",
                    )
                    _terminate_test_session(process, _session_registry)
                else:
                    _terminate_process_group(process)
            except Exception as exc:
                cleanup_error = exc
        if _completion_pipe is not None:
            os.close(_completion_pipe[0])
            if _completion_pipe[1] >= 0:
                os.close(_completion_pipe[1])
        supervisor.resume_after_cleanup()
        if cleanup_error is not None:
            if active_error is not None:
                raise active_error.with_traceback(
                    active_error.__traceback__
                ) from cleanup_error
            raise cleanup_error
        if active_error is None:
            supervisor.checkpoint()


def _run(supervisor: _ReleaseSignalSupervisor) -> int:
    supervisor.checkpoint()
    _require(
        all(
            name not in os.environ
            for name in (
                _RELEASE_TEST_SESSION_ENV,
                _RELEASE_TEST_REGISTRY_ENV,
                _RELEASE_TEST_REGISTRY_DEVICE_ENV,
                _RELEASE_TEST_REGISTRY_INODE_ENV,
            )
        ),
        "M5 release runner rejects ambient internal capabilities",
    )
    production_attestation, runner_payload = (
        _verified_production_attestation_module()
    )
    supervisor.checkpoint()
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
    supervisor.checkpoint()
    _require(
        runtime_attestation.path_payloads[
            "agent/c2_m5_release_test_runner.py"
        ]
        == runner_payload,
        "M5 release runner is outside the verified runtime closure",
    )

    retained_payloads = runtime_attestation.path_payloads
    _verify_release_process_topology(retained_payloads)
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
    supervisor.checkpoint()
    exit_code: int | None = None
    temporary_path: str | None = None
    with tempfile.TemporaryDirectory(
        prefix="c2-m5-release-",
        dir=str(_trusted_temporary_parent()),
    ) as temporary:
        temporary_path = temporary
        supervisor.checkpoint()
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
        test_session_registry = temporary_root / "test-session-groups.log"
        production_root.mkdir(mode=0o700)
        test_root.mkdir(mode=0o700)
        source_root.mkdir(mode=0o700)
        temporary_directory.mkdir(mode=0o700)
        pytest_control_root.mkdir(mode=0o700)
        with test_session_registry.open("xb") as registry:
            registry.flush()
            os.fsync(registry.fileno())
        test_session_registry.chmod(0o600)
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
        supervisor.checkpoint()
        _clone_private_git_repository(
            git_repository,
            runtime_attestation.head_commit,
        )
        supervisor.checkpoint()
        session_registry = _open_test_session_registry(
            test_session_registry
        )
        try:
            environment = {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": "/var/empty",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "TMPDIR": str(temporary_directory),
                "C2_M5_TEST_GIT_REPOSITORY": str(git_repository),
                _RELEASE_TEST_REGISTRY_ENV: str(test_session_registry),
                _RELEASE_TEST_REGISTRY_DEVICE_ENV: str(
                    session_registry.device
                ),
                _RELEASE_TEST_REGISTRY_INODE_ENV: str(
                    session_registry.inode
                ),
            }
            completion_pipe = os.pipe()
            exit_code = _run_release_child(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    _PYTEST_SESSION_SUPERVISOR,
                    str(completion_pipe[1]),
                    _PYTEST_CHILD,
                    str(production_root),
                    str(test_root),
                    str(source_root),
                    str(pytest_control_root / "pytest.ini"),
                    *_TEST_TARGETS,
                ],
                cwd=source_root,
                environment=environment,
                _supervisor=supervisor,
                _completion_pipe=completion_pipe,
                _session_registry=session_registry,
            )
        finally:
            os.close(session_registry.descriptor)
        supervisor.checkpoint()
        _verify_materialized(production_root, production_payloads)
        supervisor.checkpoint()
        _verify_materialized(test_root, test_payloads)
        supervisor.checkpoint()
        _verify_materialized(source_root, runtime_attestation.path_payloads)
        supervisor.checkpoint()
        _verify_materialized(
            pytest_control_root,
            {"pytest.ini": _PYTEST_CONFIG},
        )
        supervisor.checkpoint()
    supervisor.checkpoint()
    _require(
        temporary_path is not None and not os.path.lexists(temporary_path),
        "M5 release temporary root survived cleanup",
    )
    _require(exit_code is not None, "M5 release tests did not return a status")
    return exit_code


def main() -> int:
    _require(len(sys.argv) == 1, "M5 release runner takes no arguments")
    return _run_with_signal_supervision(_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M5ReleaseTestError as exc:
        print(f"M5 release test error: {exc}", file=sys.stderr)
        raise SystemExit(2)

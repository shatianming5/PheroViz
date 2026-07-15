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
_CHILD_LOADER = """\
import os
import sys

fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW)
try:
    payload = bytearray()
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        payload.extend(block)
        if len(payload) > 2 * 1024 * 1024:
            raise RuntimeError("retained M5 release runner is oversized")
finally:
    os.close(fd)
original = sys.argv[2]
sys.argv = [original, *sys.argv[3:]]
scope = {"__name__": "__main__", "__file__": original}
exec(compile(bytes(payload), original, "exec"), scope, scope)
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
    return parent


def _verified_production_attestation_module() -> Tuple[types.ModuleType, bytes]:
    head = _git(("rev-parse", "HEAD"), "M5 release HEAD").decode("ascii").strip()
    _require(
        len(head) == 40 and all(character in "0123456789abcdef" for character in head),
        "M5 release HEAD is invalid",
    )
    helper_relative = "agent/c2_m5_bootstrap_attestation.py"
    runner_relative = "agent/c2_m5_release_test_runner.py"
    helper_path = _REPOSITORY / helper_relative
    runner_path = _REPOSITORY / runner_relative
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
    _require(
        _git(("show", f"{head}:{helper_relative}"), "committed M5 helper")
        == helper_payload
        and _git(("show", f"{head}:{runner_relative}"), "committed M5 runner")
        == runner_payload,
        "M5 release verifier differs from HEAD",
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


def _child(production: Path, tests: Path) -> int:
    _require(
        production.is_absolute()
        and tests.is_absolute()
        and production.is_dir()
        and tests.is_dir()
        and not production.is_symlink()
        and not tests.is_symlink(),
        "M5 child dependency roots are unsafe",
    )
    sys.path[:0] = [str(tests), str(production), str(_AGENT_ROOT)]
    import pytest

    return int(
        pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "--disable-warnings",
                *_TEST_TARGETS,
            ]
        )
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

    production_manifest_payload = _stable_regular(
        _PRODUCTION_MANIFEST,
        "M5 production dependency manifest",
        2 * 1024 * 1024,
    )
    production_archive_payload = _stable_regular(
        _PRODUCTION_ARCHIVE,
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
    test_manifest_payload = _stable_regular(
        _TEST_MANIFEST,
        "M5 test dependency manifest",
        2 * 1024 * 1024,
    )
    test_archive_payload = _stable_regular(
        _TEST_ARCHIVE,
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
        temporary_directory = temporary_root / "tmp"
        retained_runner_path = temporary_root / "retained_release_runner.py"
        production_root.mkdir(mode=0o700)
        test_root.mkdir(mode=0o700)
        temporary_directory.mkdir(mode=0o700)
        _materialize(
            temporary_root,
            {retained_runner_path.name: runner_payload},
        )
        _materialize(production_root, production_payloads)
        _materialize(test_root, test_payloads)
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TMPDIR": str(temporary_directory),
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _CHILD_LOADER,
                str(retained_runner_path),
                str(Path(__file__).resolve()),
                "--child",
                str(production_root),
                str(test_root),
            ],
            cwd=_REPOSITORY,
            env=environment,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=_TEST_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise M5ReleaseTestError("M5 release tests exceeded the fixed timeout")


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        return _child(Path(sys.argv[2]), Path(sys.argv[3]))
    _require(len(sys.argv) == 1, "M5 release runner takes no arguments")
    return _run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M5ReleaseTestError as exc:
        print(f"M5 release test error: {exc}", file=sys.stderr)
        raise SystemExit(2)

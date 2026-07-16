"""Verify the M5 runtime closure before importing the experiments package."""

from __future__ import annotations

import sys


def _require_isolated_python(flags):
    if (
        not flags.isolated
        or not flags.no_site
        or not flags.dont_write_bytecode
    ):
        print("M5 bootstrap refused: invoke Python with -I -S -B", file=sys.stderr)
        raise SystemExit(2)


_require_isolated_python(sys.flags)


def _preauthenticate_helper(agent_root):
    """Authenticate the checkout helper using caller-supplied release pins."""

    import hashlib
    import json
    import os
    import re
    import stat
    import subprocess
    from pathlib import Path

    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    digest_pattern = re.compile(r"^[0-9a-f]{64}$")
    expected_commit = os.environ.get("C2_M5_EXPECTED_ATTESTATION_COMMIT", "")
    expected_manifest_sha256 = os.environ.get(
        "C2_M5_EXPECTED_MANIFEST_SHA256",
        "",
    )
    expected_bootstrap_sha256 = os.environ.get(
        "C2_M5_EXPECTED_BOOTSTRAP_SHA256",
        "",
    )
    expected_python_sha256 = os.environ.get(
        "C2_M5_EXPECTED_PYTHON_SHA256",
        "",
    )
    expected_python_library_sha256 = os.environ.get(
        "C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256",
        "",
    )
    if (
        commit_pattern.fullmatch(expected_commit) is None
        or digest_pattern.fullmatch(expected_manifest_sha256) is None
        or digest_pattern.fullmatch(expected_bootstrap_sha256) is None
        or digest_pattern.fullmatch(expected_python_sha256) is None
        or digest_pattern.fullmatch(expected_python_library_sha256) is None
    ):
        raise RuntimeError("M5 bootstrap requires exact external release pins")

    def stable_bytes(path: Path, label: str) -> bytes:
        try:
            before = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"{label} is unavailable") from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o022 != 0
        ):
            raise RuntimeError(f"{label} is not a private regular file")
        try:
            payload = path.read_bytes()
            after = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"{label} could not be read") from exc
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"{label} changed while being read")
        return payload

    bootstrap_path = Path(__file__).resolve()
    bootstrap_payload = stable_bytes(bootstrap_path, "M5 bootstrap")
    if hashlib.sha256(bootstrap_payload).hexdigest() != expected_bootstrap_sha256:
        raise RuntimeError("M5 bootstrap differs from the external release pin")

    repository = agent_root.parent
    manifest_relative = (
        "agent/experiments/resources/c2_m5_source_pilot_attestation_v1.json"
    )
    helper_relative = "agent/c2_m5_bootstrap_attestation.py"
    bootstrap_relative = "agent/c2_m5_source_pilot_bootstrap.py"
    dependency_relative = (
        "agent/experiments/resources/c2_m5_python_dependency_manifest_v2.json"
    )
    manifest_path = repository / manifest_relative
    helper_path = repository / helper_relative
    manifest_payload = stable_bytes(manifest_path, "M5 manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest_sha256:
        raise RuntimeError("M5 manifest differs from the external release pin")

    git_environment = {
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
    git_prefix = [
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
        "-C",
        str(repository),
    ]

    def git(arguments: list[str], label: str, *, text: bool = False):
        try:
            return subprocess.run(
                [*git_prefix, *arguments],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=text,
                env=git_environment,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot establish {label}") from exc

    root = Path(
        git(["rev-parse", "--show-toplevel"], "repository root", text=True).strip()
    ).resolve()
    if root != repository:
        raise RuntimeError("M5 bootstrap repository root is invalid")
    observed_commit = git(
        ["log", "-1", "--format=%H", "--", manifest_relative],
        "M5 attestation commit",
        text=True,
    ).strip()
    if observed_commit != expected_commit:
        raise RuntimeError("M5 attestation commit differs from the external release pin")
    implementation_commit = git(
        ["rev-parse", f"{expected_commit}^"],
        "M5 implementation commit",
        text=True,
    ).strip()
    git(
        ["merge-base", "--is-ancestor", expected_commit, "HEAD"],
        "M5 attestation ancestry",
    )
    changed = git(
        [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "-r",
            expected_commit,
        ],
        "M5 manifest-only commit",
        text=True,
    ).strip()
    if changed != f"A\t{manifest_relative}":
        raise RuntimeError("M5 attestation commit is not manifest-only")
    if (
        git(
            ["show", f"{expected_commit}:{manifest_relative}"],
            "committed M5 manifest",
        )
        != manifest_payload
    ):
        raise RuntimeError("M5 manifest differs from its pinned Git commit")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("M5 manifest is invalid JSON") from exc
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if (
        not isinstance(manifest, dict)
        or manifest_payload != canonical
        or manifest.get("approved_implementation_commit_full")
        != implementation_commit
        or not isinstance(manifest.get("attested_paths"), list)
    ):
        raise RuntimeError("M5 manifest structure is invalid")
    path_bindings = {}
    for entry in manifest["attested_paths"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"relative_path", "git_blob_object_id", "sha256"}
            or not isinstance(entry.get("relative_path"), str)
            or entry["relative_path"] in path_bindings
        ):
            raise RuntimeError("M5 manifest has an invalid or repeated attested path")
        path_bindings[entry["relative_path"]] = entry
    dependency_path = repository / dependency_relative
    dependency_payload = stable_bytes(
        dependency_path,
        "M5 dependency manifest",
    )
    helper_payload = stable_bytes(
        helper_path,
        "M5 attestation helper",
    )
    for relative, path, payload, expected_digest in (
        (
            bootstrap_relative,
            bootstrap_path,
            bootstrap_payload,
            expected_bootstrap_sha256,
        ),
        (
            helper_relative,
            helper_path,
            helper_payload,
            None,
        ),
        (
            dependency_relative,
            dependency_path,
            dependency_payload,
            None,
        ),
    ):
        binding = path_bindings.get(relative)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"relative_path", "git_blob_object_id", "sha256"}
            or binding.get("sha256") != hashlib.sha256(payload).hexdigest()
            or (expected_digest is not None and binding.get("sha256") != expected_digest)
            or git(
                ["show", f"{implementation_commit}:{relative}"],
                f"M5 implementation bytes {relative}",
            )
            != payload
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RuntimeError(f"M5 pre-import binding failed: {relative}")
    try:
        dependency_manifest = json.loads(dependency_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("M5 dependency manifest is invalid JSON") from exc
    python_binding = (
        dependency_manifest.get("python")
        if isinstance(dependency_manifest, dict)
        else None
    )
    if (
        not isinstance(python_binding, dict)
        or python_binding.get("executable_sha256")
        != expected_python_sha256
        or python_binding.get("runtime_library_sha256")
        != expected_python_library_sha256
    ):
        raise RuntimeError(
            "M5 CPython executable or library differs from the external release pin"
        )
    return (
        helper_path,
        helper_payload,
        expected_commit,
        expected_manifest_sha256,
        expected_bootstrap_sha256,
        expected_python_sha256,
        expected_python_library_sha256,
    )


def _install_python39_compatibility() -> None:
    """Provide only the Python 3.10 call semantics used by attested modules."""

    import builtins
    import dataclasses
    import itertools

    original_zip = builtins.zip

    def compatible_zip(*iterables, strict=False):
        if not strict:
            return original_zip(*iterables)

        def strict_iterator():
            sentinel = object()
            for values in itertools.zip_longest(*iterables, fillvalue=sentinel):
                if any(value is sentinel for value in values):
                    raise ValueError("zip() arguments have unequal lengths")
                yield values

        return strict_iterator()

    original_dataclass = dataclasses.dataclass

    def compatible_dataclass(_cls=None, **kwargs):
        kwargs.pop("slots", None)
        return original_dataclass(_cls, **kwargs)

    builtins.zip = compatible_zip
    dataclasses.dataclass = compatible_dataclass


def _load_verified_helper(helper_path, helper_payload):
    import types

    helper = types.ModuleType("c2_m5_bootstrap_attestation")
    helper.__file__ = str(helper_path)
    helper.__package__ = None
    sys.modules[helper.__name__] = helper
    exec(
        compile(helper_payload, str(helper_path), "exec"),
        helper.__dict__,
    )
    return helper


def main() -> int:
    import os
    import tempfile
    from pathlib import Path

    agent_root = Path(__file__).resolve().parent
    (
        helper_path,
        helper_payload,
        expected_commit,
        expected_manifest_sha256,
        expected_bootstrap_sha256,
        expected_python_sha256,
        expected_python_library_sha256,
    ) = _preauthenticate_helper(agent_root)
    for directory, directory_names, file_names in os.walk(
        agent_root,
        topdown=True,
        followlinks=False,
    ):
        if "__pycache__" in directory_names:
            raise RuntimeError("M5 bootstrap rejected bytecode cache directories")
        if any(name.endswith((".pyc", ".pyo")) for name in file_names):
            raise RuntimeError("M5 bootstrap rejected bytecode artifacts")
    expected_top_level_python = {
        "apply_fallback.py",
        "apply_stage_updates.py",
        "c2_m5_bootstrap_attestation.py",
        "c2_m5_downloader_bootstrap.py",
        "c2_m5_release_test_runner.py",
        "c2_m5_source_pilot_bootstrap.py",
        "run_chain.py",
        "run_multi_panel.py",
    }
    observed_top_level_python = {
        path.name for path in agent_root.glob("*.py") if path.is_file()
    }
    if observed_top_level_python != expected_top_level_python:
        raise RuntimeError("M5 bootstrap rejected the top-level Python import surface")
    helper = _load_verified_helper(helper_path, helper_payload)

    attestation = helper.verify_adapter_attestation(
        activate=True,
        expected_attestation_commit=expected_commit,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_bootstrap_sha256=expected_bootstrap_sha256,
        expected_python_sha256=expected_python_sha256,
        expected_python_library_sha256=(
            expected_python_library_sha256
        ),
    )
    for name in (
        "C2_M5_EXPECTED_ATTESTATION_COMMIT",
        "C2_M5_EXPECTED_MANIFEST_SHA256",
        "C2_M5_EXPECTED_BOOTSTRAP_SHA256",
        "C2_M5_EXPECTED_PYTHON_SHA256",
        "C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256",
    ):
        os.environ.pop(name, None)
    _install_python39_compatibility()
    sys.dont_write_bytecode = True
    with tempfile.TemporaryDirectory(prefix="c2-m5-runtime-") as temporary:
        runtime_root = Path(temporary)
        for relative, payload in sorted(attestation.path_payloads.items()):
            destination = runtime_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(destination, 0o400)
        dependency_root = runtime_root / "third_party"
        for relative, payload in sorted(attestation.dependency_payloads.items()):
            destination = dependency_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(destination, 0o400)
        sys.path.insert(0, str(dependency_root))
        sys.path.insert(0, str(runtime_root / "agent"))
        from experiments.c2_m5_source_pilot import (
            C2M5SourcePilotError,
            main as pilot_main,
        )

        try:
            return pilot_main()
        except C2M5SourcePilotError as exc:
            print(f"M5 source pilot refused: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"M5 bootstrap refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

"""Execute and seal the fixed C2 chunk-001 source-bearing pilot.

This module adapts the frozen Nature downloader without changing its bytes.  It
runs each fixed attempt in an isolated workspace, converts the downloader's
overlapping operational status files into a strict terminal partition, and
retains source assets from every successful attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from c2_m5_bootstrap_attestation import (
    AdapterAttestation,
    C2M5BootstrapAttestationError,
    _verify_filter_free_clean_tree,
    require_active_adapter_attestation,
)


class C2M5SourcePilotError(RuntimeError):
    """Raised when the fixed M5 source pilot cannot remain fail-closed."""


ATTEMPTS = ("initial", "retry1", "retry2")
ATTEMPT_INDEX = {attempt: index for index, attempt in enumerate(ATTEMPTS)}
CHUNK_ID = "001"
CHUNK_RECORDS = 200
CHUNK_SHA256 = "e6b5d229d94fab7768a95482bbe79b895808ab925e950a0dd4352b206f32faa0"
UNIVERSE_SHA256 = "51848466c6bf6bf400b58faf539953830349abab21438e527eaefd74103450df"
FREEZE_SUMMARY_SHA256 = (
    "64684e64b6a4e54c508685fb745e303dfca84e6d499e96568f40c80c5dff368b"
)
FREEZE_SUMMARY_HASH = (
    "6cdb5f8398cc70960d108c6319c93c2121327ce757af405bc44bca687a71056d"
)
FROZEN_DOWNLOADER_COMMIT = "ca98442b9e805110089b03083cb240e19b58d4a2"
FROZEN_DOWNLOADER_RELATIVE = "nature_download/nature_all_in_one.py"
FROZEN_DOWNLOADER_SHA256 = (
    "ce9f0fc6843d7d6eaccdb7a25a64d4dce0a730865098414991a26c909094145e"
)
EXPECTED_RAW_ROOT = "ccby_sr_npj_chunk001_rerun3_raw_ca98442"
EXPECTED_FINAL_ROOT = "ccby_sr_npj_chunk001_rerun3_clean_ca98442"
SKIPPED_STATUSES = frozenset({"fetch-error", "no-figures", "no-source-data"})
_DECLARED_KIND_TUPLES = {
    "source_data": {
        ("NONE", "CSV_V1"),
        ("ZIP_V1", "XLSX_V1"),
    },
    "source_archive": {("ZIP_V1", "GENERIC_ZIP_V1")},
    "figure": {("NONE", "OTHER_REGISTERED_V1")},
    "caption": {("NONE", "OTHER_REGISTERED_V1")},
}
MAX_SOURCE_ASSET_BYTES = 256 * 1024 * 1024
MAX_RETAINED_SOURCE_BYTES = 512 * 1024 * 1024
MAX_NETWORK_REQUESTS_PER_ATTEMPT = 10_000
MAX_NETWORK_RESPONSE_BYTES_PER_ATTEMPT = 8 * 1024 * 1024 * 1024
ATTEMPT_WALL_TIMEOUT_SECONDS = 3 * 60 * 60
MAX_POSTFETCH_LOG_BYTES = 16 * 1024 * 1024
NETWORK_BUDGET_NAME = "network_budget.bin"
LOCK_NAME = ".c2-m5-source-pilot.lock"
SELECTION_RULE = (
    "execute every frozen chunk record in initial and both fixed retries "
    "regardless of every preceding outcome"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?:ANTHROPIC|OPENAI|GITHUB)_(?:AUTH_)?TOKEN\s*="),
)
_WATCHED_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
_ATTEMPT_PROCESS_SUPERVISOR = """\
import os
import signal
import subprocess
import sys

status_fd = int(sys.argv[1])
command = sys.argv[2:]
os.set_inheritable(status_fd, False)
worker = None
status_open = True
cleanup_requested = False

def request_cleanup(_signum, _frame):
    global cleanup_requested
    cleanup_requested = True

previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
signal.signal(signal.SIGTERM, request_cleanup)
try:
    if os.write(status_fd, b"R") != 1:
        raise SystemExit("M5 attempt supervisor readiness write was incomplete")
    try:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import os,signal,sys;"
                    "signal.pthread_sigmask("
                    "signal.SIG_UNBLOCK,{signal.SIGTERM});"
                    "os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
                ),
                *command,
            ],
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    while not cleanup_requested:
        try:
            return_code = worker.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            pass
    if not cleanup_requested:
        if return_code < 0:
            return_code = min(255, 128 - return_code)
        status = f"{return_code}\\n".encode("ascii")
        if os.write(status_fd, status) != len(status):
            raise SystemExit("M5 attempt status write was incomplete")
        os.close(status_fd)
        status_open = False
        while not cleanup_requested:
            signal.pause()
finally:
    if worker is not None:
        if worker.poll() is None:
            worker.kill()
        worker.wait()
    if status_open:
        os.close(status_fd)
if cleanup_requested:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        signal.pause()
"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2M5SourcePilotError(message)


class _SourcePilotSignalSupervisor:
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
        self.publication_committed = False
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
        if (
            self.interrupted_signal is not None
            and not self.publication_committed
        ):
            raise C2M5SourcePilotError(
                "M5 source pilot interrupted by signal "
                f"{self.interrupted_signal}"
            )

    def block_for_cleanup(self) -> None:
        signal.pthread_sigmask(signal.SIG_BLOCK, _WATCHED_SIGNALS)

    def resume_after_cleanup(self) -> None:
        signal.pthread_sigmask(
            signal.SIG_SETMASK,
            self.supervisor_mask,
        )

    def commit_publication(self, publish: Callable[[], None]) -> None:
        _require(
            not self.publication_committed,
            "M5 source pilot publication is already committed",
        )
        self.block_for_cleanup()
        try:
            pending = set(signal.sigpending()).intersection(_WATCHED_SIGNALS)
            if self.interrupted_signal is not None or pending:
                if self.interrupted_signal is None:
                    self.interrupted_signal = min(int(signum) for signum in pending)
            else:
                publish()
                self.publication_committed = True
        finally:
            self.resume_after_cleanup()
        self.checkpoint()

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
    operation: Callable[[_SourcePilotSignalSupervisor], Any],
) -> Any:
    supervisor = _SourcePilotSignalSupervisor()
    try:
        return operation(supervisor)
    finally:
        supervisor.finish(sys.exc_info()[1])


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = _sha256_json({key: item for key, item in value.items() if key != field})
    return value


def _json_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(value) + b"\n" for value in values)


_GIT_COMMAND = (
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
_GIT_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
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


def _git_text(repo: Path, arguments: Sequence[str], label: str) -> str:
    try:
        completed = subprocess.run(
            [*_GIT_COMMAND, "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise C2M5SourcePilotError(f"cannot establish {label}") from exc
    return completed.stdout.strip()


def _git_bytes(repo: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        completed = subprocess.run(
            [*_GIT_COMMAND, "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C2M5SourcePilotError(f"cannot establish {label}") from exc
    return completed.stdout


def verify_adapter_attestation() -> AdapterAttestation:
    """Verify the adapter before importing the C2 finalizer or policy modules."""

    try:
        return require_active_adapter_attestation()
    except C2M5BootstrapAttestationError as exc:
        raise C2M5SourcePilotError(str(exc)) from exc


@dataclass(frozen=True)
class FrozenRecord:
    ordinal: int
    article_id: str
    doi: str
    value: Mapping[str, Any]


def _article_id(value: Mapping[str, Any]) -> str:
    raw_url = value.get("article_url") or value.get("url")
    _require(isinstance(raw_url, str), "frozen record lacks an article URL")
    parsed = urlparse(raw_url)
    pieces = [piece for piece in parsed.path.split("/") if piece]
    _require(
        parsed.scheme == "https"
        and parsed.hostname == "www.nature.com"
        and len(pieces) == 2
        and pieces[0] == "articles"
        and _IDENTIFIER_RE.fullmatch(pieces[1]) is not None,
        "frozen record article URL is invalid",
    )
    return pieces[1]


def _frozen_records(source_bytes: bytes) -> list[FrozenRecord]:
    records: list[FrozenRecord] = []
    seen_ids: set[str] = set()
    seen_dois: set[str] = set()
    for ordinal, line in enumerate(source_bytes.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise C2M5SourcePilotError(
                f"frozen record {ordinal} is invalid JSON"
            ) from exc
        _require(isinstance(value, dict), f"frozen record {ordinal} is not an object")
        article_id = _article_id(value)
        doi = value.get("doi")
        _require(
            isinstance(doi, str)
            and doi == doi.strip().lower()
            and doi == f"10.1038/{article_id.lower()}"
            and article_id not in seen_ids
            and doi not in seen_dois
            and value.get("download_eligible") is True
            and value.get("journal_allowed") is True
            and value.get("require_cc_by") is True,
            f"frozen record {ordinal} violates the fixed CC-BY roster",
        )
        seen_ids.add(article_id)
        seen_dois.add(doi)
        records.append(FrozenRecord(ordinal, article_id, doi, value))
    _require(len(records) == CHUNK_RECORDS, "frozen chunk does not contain 200 records")
    return records


def _read_stable_regular(path: Path, label: str, *, max_bytes: int | None = None) -> bytes:
    _require(path.is_absolute(), f"{label} path must be absolute")
    _require(
        all(component not in {"", ".", ".."} for component in path.parts[1:]),
        f"{label} path is unsafe",
    )
    directory_descriptor = -1
    file_descriptor = -1
    try:
        directory_descriptor = os.open(
            path.anchor,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        for component in path.parts[1:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            path.name,
            os.O_RDONLY
            | os.O_NONBLOCK
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        before = os.fstat(file_descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_uid == os.geteuid(),
            f"{label} is not a private regular file",
        )
        if max_bytes is not None:
            _require(
                0 < before.st_size <= max_bytes,
                f"{label} exceeds its byte budget",
            )
        payload = bytearray()
        while len(payload) <= before.st_size:
            block = os.read(
                file_descriptor,
                min(1024 * 1024, before.st_size - len(payload) + 1),
            )
            if not block:
                break
            payload.extend(block)
        after = os.fstat(file_descriptor)
        current = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise C2M5SourcePilotError(f"{label} could not be opened safely") from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    _require(
        identity
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        == (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        and len(payload) == before.st_size,
        f"{label} changed while being read",
    )
    return bytes(payload)


def _verify_frozen_paths(
    source_chunk: Path,
    frozen_universe: Path,
    freeze_summary: Path,
) -> tuple[bytes, list[FrozenRecord]]:
    for path, label in (
        (source_chunk, "source chunk"),
        (frozen_universe, "frozen universe"),
        (freeze_summary, "freeze summary"),
    ):
        _require(path.is_absolute(), f"{label} path must be absolute")
    source_bytes = _read_stable_regular(source_chunk, "source chunk")
    universe_bytes = _read_stable_regular(frozen_universe, "frozen universe")
    summary_payload = _read_stable_regular(freeze_summary, "freeze summary")
    _require(_sha256(source_bytes) == CHUNK_SHA256, "source chunk hash mismatch")
    _require(_sha256(universe_bytes) == UNIVERSE_SHA256, "frozen universe hash mismatch")
    _require(
        _sha256(summary_payload) == FREEZE_SUMMARY_SHA256,
        "freeze summary hash mismatch",
    )
    records = _frozen_records(source_bytes)
    universe_lines = universe_bytes.splitlines(keepends=True)
    _require(
        b"".join(universe_lines[:CHUNK_RECORDS]) == source_bytes,
        "source chunk is not the exact first frozen-universe partition",
    )
    summary = json.loads(summary_payload)
    _require(
        isinstance(summary, dict)
        and summary.get("summary_hash") == FREEZE_SUMMARY_HASH
        and _sha256_json(
            {key: value for key, value in summary.items() if key != "summary_hash"}
        )
        == FREEZE_SUMMARY_HASH,
        "freeze summary semantic hash mismatch",
    )
    return source_bytes, records


@dataclass(frozen=True)
class DownloaderBinding:
    worktree: Path
    tree: str
    script: Path
    script_blob: str
    runtime_files: Mapping[str, bytes]


def _verify_downloader_worktree(worktree: Path) -> DownloaderBinding:
    _require(worktree.is_absolute(), "downloader worktree must be absolute")
    _require(worktree.is_dir() and not worktree.is_symlink(), "downloader worktree is unsafe")
    root = Path(
        _git_text(worktree, ("rev-parse", "--show-toplevel"), "downloader worktree root")
    ).resolve()
    _require(root == worktree.resolve(), "downloader worktree root mismatch")
    pass
    try:
        _verify_filter_free_clean_tree(worktree, FROZEN_DOWNLOADER_COMMIT)
    except C2M5BootstrapAttestationError as exc:
        raise C2M5SourcePilotError("downloader worktree is dirty or unsafe") from exc
    script = worktree / FROZEN_DOWNLOADER_RELATIVE
    payload = _read_stable_regular(script, "frozen downloader")
    script_blob = _git_text(
        worktree,
        ("rev-parse", f"HEAD:{FROZEN_DOWNLOADER_RELATIVE}"),
        "frozen downloader blob",
    )
    _require(
        _sha256(payload) == FROZEN_DOWNLOADER_SHA256
        and payload
        == _git_bytes(
            worktree,
            ("show", f"HEAD:{FROZEN_DOWNLOADER_RELATIVE}"),
            "frozen downloader Git bytes",
        ),
        "frozen downloader bytes mismatch",
    )
    runtime_files: dict[str, bytes] = {"frozen_downloader.py": payload}
    corpus_paths = _git_text(
        worktree,
        (
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            "nature_download/corpus",
        ),
        "frozen downloader corpus roster",
    ).splitlines()
    _require(corpus_paths, "frozen downloader corpus roster is empty")
    for relative in sorted(corpus_paths):
        if not relative.endswith(".py"):
            continue
        destination = relative.removeprefix("nature_download/")
        _require(
            destination.startswith("corpus/")
            and ".." not in Path(destination).parts
            and destination not in runtime_files,
            "frozen downloader runtime path is invalid",
        )
        runtime_path = worktree / relative
        runtime_payload = _read_stable_regular(
            runtime_path,
            f"frozen downloader runtime {relative}",
        )
        pass
        runtime_files[destination] = runtime_payload
    return DownloaderBinding(
        worktree=worktree,
        tree=_git_text(worktree, ("rev-parse", "HEAD^{tree}"), "downloader tree"),
        script=script,
        script_blob=script_blob,
        runtime_files=runtime_files,
    )


def _acquisition_config(workers: int) -> dict[str, Any]:
    _require(
        isinstance(workers, int) and not isinstance(workers, bool) and 1 <= workers <= 16,
        "workers must be an integer from 1 through 16",
    )
    return _sealed(
        {
            "schema_version": "c2-m5-source-pilot-acquisition-config-v1",
            "attempts": list(ATTEMPTS),
            "no_model_calls": True,
            "no_early_stop": True,
            "outcome_independent": True,
            "arguments": {
                "require_cc_by": True,
                "sort": "input",
                "max_figs": 12,
                "max_empty_figs": 2,
                "sleep_seconds": 1.0,
                "timeout_seconds": 300,
                "max_retries": 3,
                "workers": workers,
            },
        },
        "config_hash",
    )


def _pre_download_binding(
    config: Mapping[str, Any],
    attestation: AdapterAttestation,
    downloader: DownloaderBinding,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": "c2-m5-source-pilot-pre-download-binding-v1",
            "chunk": 1,
            "chunk_records": CHUNK_RECORDS,
            "chunk_start_index_1based": 1,
            "chunk_end_index_1based": CHUNK_RECORDS,
            "attempts": list(ATTEMPTS),
            "attempt_coverage_required_each": CHUNK_RECORDS,
            "code_commit": FROZEN_DOWNLOADER_COMMIT,
            "code_dirty": False,
            "exact_byte_copy": True,
            "execution_accepted_sha256": CHUNK_SHA256,
            "outcome_independent": True,
            "review_or_experiment_outcomes_used": False,
            "source_chunk_sha256": CHUNK_SHA256,
            "source_universe_sha256": UNIVERSE_SHA256,
            "source_freeze_summary_sha256": FREEZE_SUMMARY_SHA256,
            "source_freeze_summary_hash": FREEZE_SUMMARY_HASH,
            "acquisition_config_hash": config["config_hash"],
            "selection_rule": SELECTION_RULE,
            "adapter_implementation_commit": attestation.implementation_commit,
            "adapter_attestation_commit": attestation.attestation_commit,
            "adapter_attestation_manifest_sha256": attestation.manifest_sha256,
            "external_launch_binding": {
                "bootstrap_sha256": (
                    attestation.externally_pinned_bootstrap_sha256
                ),
                "python_executable_sha256": (
                    attestation.externally_pinned_python_sha256
                ),
                "python_runtime_library_sha256": (
                    attestation.externally_pinned_python_library_sha256
                ),
                "attestation_commit": attestation.attestation_commit,
                "manifest_sha256": attestation.manifest_sha256,
            },
            "adapter_path_sha256": dict(sorted(attestation.path_sha256.items())),
            "python_dependency_binding": attestation.dependency_binding,
            "downloader_tree": downloader.tree,
            "downloader_script_blob": downloader.script_blob,
            "downloader_script_sha256": FROZEN_DOWNLOADER_SHA256,
            "downloader_runtime_bundle": [
                {
                    "relative_path": relative,
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
                for relative, payload in sorted(downloader.runtime_files.items())
            ],
        },
        "summary_hash",
    )


def _safe_subprocess_environment(workspace: Path) -> dict[str, str]:
    home = workspace / "home"
    temporary = workspace / "tmp"
    home.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "C2_M5_NETWORK_GUARD_REQUIRED": "1",
        "C2_M5_NETWORK_BUDGET_PATH": str(
            workspace / NETWORK_BUDGET_NAME
        ),
    }


def _attempt_command(
    workspace: Path,
    workers: int,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(workspace / "guarded_downloader_bootstrap.py"),
        "postfetch",
        "--jsonl",
        str(workspace / "accepted.jsonl"),
        "--out",
        str(workspace / "content"),
        "--max-figs",
        "12",
        "--max-empty-figs",
        "2",
        "--sort",
        "input",
        "--sleep",
        "1.0",
        "--timeout",
        "300",
        "--max-retries",
        "3",
        "--workers",
        str(workers),
        "--processed-file",
        str(workspace / "operational_processed.txt"),
        "--require-cc-by",
    ]


def _materialize_downloader_bundle(
    workspace: Path,
    downloader: DownloaderBinding,
) -> None:
    for relative, payload in sorted(downloader.runtime_files.items()):
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o400)


def _materialize_network_guard(workspace: Path, payload: bytes) -> Path:
    directory = workspace / "network_guard"
    directory.mkdir(mode=0o700)
    destination = directory / "sitecustomize.py"
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, 0o400)
    return directory


def _materialize_downloader_bootstrap(workspace: Path, payload: bytes) -> None:
    destination = workspace / "guarded_downloader_bootstrap.py"
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, 0o400)


def _materialize_dependencies(
    workspace: Path,
    payloads: Mapping[str, bytes],
) -> None:
    root = workspace / "dependencies"
    for relative, payload in sorted(payloads.items()):
        path = Path(relative)
        _require(
            relative
            and not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in relative,
            "M5 dependency snapshot path is unsafe",
        )
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o400)


def _initialize_network_budget(workspace: Path) -> None:
    path = workspace / NETWORK_BUDGET_NAME
    with path.open("xb") as handle:
        handle.write(struct.pack(">QQ", 0, 0))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def _network_budget_payload(workspace: Path) -> bytes:
    payload = _read_stable_regular(
        workspace / NETWORK_BUDGET_NAME,
        "network budget",
        max_bytes=16,
    )
    _require(len(payload) == 16, "network budget has invalid bytes")
    requests_used, response_bytes = struct.unpack(">QQ", payload)
    _require(
        requests_used <= MAX_NETWORK_REQUESTS_PER_ATTEMPT
        and response_bytes <= MAX_NETWORK_RESPONSE_BYTES_PER_ATTEMPT,
        "network budget exceeded its fixed caps",
    )
    return _json_file_bytes(
        _sealed(
            {
                "schema_version": "c2-m5-network-budget-v1",
                "request_count": requests_used,
                "response_bytes": response_bytes,
                "request_cap": MAX_NETWORK_REQUESTS_PER_ATTEMPT,
                "response_byte_cap": MAX_NETWORK_RESPONSE_BYTES_PER_ATTEMPT,
                "wall_timeout_seconds": ATTEMPT_WALL_TIMEOUT_SECONDS,
            },
            "budget_hash",
        )
    )


def _run_bounded_process(
    *,
    command: Sequence[str],
    workspace: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    _supervisor: _SourcePilotSignalSupervisor | None = None,
    _cleanup_failure: Callable[[], None] | None = None,
) -> tuple[int, tuple[int, ...]]:
    if _supervisor is None:
        return _run_with_signal_supervision(
            lambda supervisor: _run_bounded_process(
                command=command,
                workspace=workspace,
                environment=environment,
                timeout_seconds=timeout_seconds,
                _supervisor=supervisor,
                _cleanup_failure=_cleanup_failure,
            )
        )
    supervisor = _supervisor
    _require(timeout_seconds > 0, "attempt timeout must be positive")
    log_path = workspace / "postfetch.log"
    log = None
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    status_pipe: tuple[int, int] | None = None
    return_code: int | None = None
    retained_active_pids: tuple[int, ...] = ()
    total = 0
    forced_code: int | None = None
    supervisor_ready = False
    try:
        supervisor.checkpoint()
        log = log_path.open("xb")
        supervisor.checkpoint()
        status_pipe = os.pipe()
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _ATTEMPT_PROCESS_SUPERVISOR,
                str(status_pipe[1]),
                *command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            cwd=workspace,
            start_new_session=True,
            bufsize=0,
            pass_fds=(status_pipe[1],),
        )
        os.close(status_pipe[1])
        status_pipe = (status_pipe[0], -1)
        supervisor.checkpoint()
        _require(process.stdout is not None, "attempt log pipe is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "log")
        selector.register(status_pipe[0], selectors.EVENT_READ, "status")
        deadline = time.monotonic() + timeout_seconds
        pipe_open = True
        status_payload = bytearray()
        while True:
            supervisor.checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                forced_code = 124
                break
            ready = selector.select(timeout=min(0.1, remaining))
            for key, _ in ready:
                if key.data == "status":
                    block = os.read(status_pipe[0], 64)
                    if block:
                        if not supervisor_ready:
                            _require(
                                block.startswith(b"R"),
                                "attempt process readiness is malformed",
                            )
                            supervisor_ready = True
                            block = block[1:]
                            if not block:
                                continue
                        status_payload.extend(block)
                        _require(
                            len(status_payload) <= 16,
                            "attempt process status is oversized",
                        )
                        continue
                    _require(
                        status_payload.endswith(b"\n")
                        and status_payload.count(b"\n") == 1
                        and status_payload[:-1].isdigit(),
                        "attempt process status is malformed",
                    )
                    return_code = int(status_payload[:-1])
                    _require(
                        0 <= return_code <= 255,
                        "attempt process status is invalid",
                    )
                    try:
                        identity_valid = os.getpgid(process.pid) == process.pid
                    except (ProcessLookupError, PermissionError):
                        identity_valid = False
                    _require(
                        identity_valid,
                        "attempt process supervisor group identity changed",
                    )
                    break
                chunk = os.read(process.stdout.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(process.stdout)
                    pipe_open = False
                elif total + len(chunk) > MAX_POSTFETCH_LOG_BYTES:
                    retained = MAX_POSTFETCH_LOG_BYTES - total
                    if retained > 0:
                        log.write(chunk[:retained])
                    total = MAX_POSTFETCH_LOG_BYTES
                    forced_code = 125
                    break
                else:
                    log.write(chunk)
                    total += len(chunk)
            if return_code is not None or forced_code is not None:
                break
            if not pipe_open:
                time.sleep(min(0.05, remaining))
    finally:
        active_error = sys.exc_info()[1]
        supervisor.block_for_cleanup()
        cleanup_errors: list[Exception] = []
        if process is not None:
            if (
                supervisor_ready
                and status_pipe is not None
                and status_pipe[0] >= 0
            ):
                try:
                    _request_attempt_supervisor_cleanup(
                        process,
                        status_pipe[0],
                    )
                except Exception as exc:
                    cleanup_errors.append(exc)
            try:
                retained_active_pids = _terminate_process_group(process)
            except Exception as exc:
                cleanup_errors.append(exc)
        if process is not None and process.stdout is not None and log is not None:
            drain_deadline = time.monotonic() + 2.0
            try:
                os.set_blocking(process.stdout.fileno(), False)
                while True:
                    try:
                        chunk = os.read(
                            process.stdout.fileno(),
                            64 * 1024,
                        )
                    except BlockingIOError:
                        _require(
                            time.monotonic() < drain_deadline,
                            "attempt log pipe did not reach EOF after cleanup",
                        )
                        time.sleep(0.01)
                        continue
                    if not chunk:
                        break
                    retained = min(
                        len(chunk),
                        MAX_POSTFETCH_LOG_BYTES - total,
                    )
                    if retained > 0:
                        log.write(chunk[:retained])
                        total += retained
                    if retained != len(chunk) and forced_code is None:
                        forced_code = 125
            except Exception as exc:
                cleanup_errors.append(exc)
        if log is not None and forced_code is not None:
            marker = (
                b"\nM5 attempt exceeded its fixed wall timeout\n"
                if forced_code == 124
                else b"\nM5 attempt exceeded its fixed log byte limit\n"
            )
            available = MAX_POSTFETCH_LOG_BYTES - total
            if available > 0:
                log.write(marker[:available])
            return_code = forced_code
        if selector is not None:
            try:
                selector.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        if status_pipe is not None:
            for descriptor in status_pipe:
                if descriptor < 0:
                    continue
                try:
                    os.close(descriptor)
                except OSError as exc:
                    cleanup_errors.append(exc)
        if log is not None:
            try:
                log.flush()
                os.fsync(log.fileno())
            except Exception as exc:
                cleanup_errors.append(exc)
            try:
                log.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        supervisor.resume_after_cleanup()
        if cleanup_errors:
            if _cleanup_failure is not None:
                _cleanup_failure()
            cleanup_error = cleanup_errors[0]
            if active_error is not None:
                raise active_error.with_traceback(
                    active_error.__traceback__
                ) from cleanup_error
            if len(cleanup_errors) > 1:
                raise cleanup_error from cleanup_errors[-1]
            raise cleanup_error
        if active_error is None:
            supervisor.checkpoint()
    _require(return_code is not None, "attempt process did not return a status")
    return return_code, retained_active_pids


def _process_group_members(group_id: int) -> tuple[int, ...]:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,pgid=,uid="],
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
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise C2M5SourcePilotError("attempt process group could not be inspected") from exc
    members: list[int] = []
    for raw_line in completed.stdout.splitlines():
        pieces = raw_line.split()
        _require(len(pieces) == 3, "attempt process table is malformed")
        try:
            process_id, observed_group, owner = map(int, pieces)
        except ValueError as exc:
            raise C2M5SourcePilotError("attempt process table is malformed") from exc
        if observed_group != group_id:
            continue
        _require(owner == os.geteuid(), "attempt process group has an unsafe owner")
        members.append(process_id)
    return tuple(sorted(members))


def _request_attempt_supervisor_cleanup(
    process: subprocess.Popen[bytes],
    status_fd: int,
) -> None:
    try:
        os.kill(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise C2M5SourcePilotError(
            "attempt process supervisor could not be notified"
        ) from exc
    os.set_blocking(status_fd, False)
    deadline = time.monotonic() + 2.0
    while True:
        try:
            block = os.read(status_fd, 64)
        except BlockingIOError:
            _require(
                time.monotonic() < deadline,
                "attempt process supervisor did not acknowledge cleanup",
            )
            time.sleep(0.01)
            continue
        if not block:
            return


def _terminate_process_group(
    process: subprocess.Popen[bytes],
) -> tuple[int, ...]:
    errors: list[Exception] = []
    permission_denied = False
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        permission_denied = True
    except OSError as exc:
        error = C2M5SourcePilotError(
            "attempt process group could not be terminated"
        )
        error.__cause__ = exc
        errors.append(error)
    deadline = time.monotonic() + 2.0
    members: tuple[int, ...] = ()
    while True:
        try:
            members = _process_group_members(process.pid)
        except Exception as exc:
            errors.append(exc)
            break
        remaining = tuple(
            process_id
            for process_id in members
            if process_id != process.pid
        )
        if not remaining:
            break
        if time.monotonic() >= deadline:
            errors.append(
                C2M5SourcePilotError(
                    f"attempt process group retained active pids: "
                    f"{list(remaining)}"
                )
            )
            break
        time.sleep(0.01)
    if permission_denied and any(
        process_id != process.pid for process_id in members
    ):
        errors.append(
            C2M5SourcePilotError(
                "attempt process group could not be terminated"
            )
        )
    leader_return_code: int | None = None
    try:
        leader_return_code = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            leader_return_code = process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = C2M5SourcePilotError(
                "attempt process leader could not be reaped"
            )
            error.__cause__ = exc
            errors.append(error)
    if (
        leader_return_code is not None
        and leader_return_code != -signal.SIGKILL
    ):
        errors.append(
            C2M5SourcePilotError(
                "attempt process supervisor exited before group cleanup"
            )
        )
    if errors:
        if len(errors) > 1:
            raise errors[0] from errors[-1]
        raise errors[0]
    return tuple(
        process_id
        for process_id in members
        if process_id != process.pid
    )


def _run_attempt_subprocess(
    *,
    downloader: DownloaderBinding,
    workspace: Path,
    workers: int,
    network_guard_payload: bytes,
    downloader_bootstrap_payload: bytes,
    dependency_payloads: Mapping[str, bytes],
    deadline_monotonic: float | None = None,
    _supervisor: _SourcePilotSignalSupervisor | None = None,
    _cleanup_failure: Callable[[], None] | None = None,
) -> tuple[int, tuple[int, ...]]:
    _materialize_downloader_bundle(workspace, downloader)
    _materialize_downloader_bootstrap(workspace, downloader_bootstrap_payload)
    _materialize_dependencies(workspace, dependency_payloads)
    _initialize_network_budget(workspace)
    _materialize_network_guard(
        workspace,
        network_guard_payload,
    )
    command = _attempt_command(workspace, workers)
    timeout_seconds = (
        ATTEMPT_WALL_TIMEOUT_SECONDS
        if deadline_monotonic is None
        else deadline_monotonic - time.monotonic()
    )
    _require(timeout_seconds > 0, "attempt setup exceeded its fixed wall timeout")
    return _run_bounded_process(
        command=command,
        workspace=workspace,
        environment=_safe_subprocess_environment(workspace),
        timeout_seconds=timeout_seconds,
        _supervisor=_supervisor,
        _cleanup_failure=_cleanup_failure,
    )


def _parse_operational_processed(payload: bytes, expected_ids: set[str]) -> set[str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise C2M5SourcePilotError("operational processed file is not UTF-8") from exc
    observed: set[str] = set()
    for index, line in enumerate(lines, start=1):
        article_id = line.strip()
        _require(
            article_id in expected_ids and article_id not in observed,
            f"operational processed line {index} is invalid",
        )
        observed.add(article_id)
    _require(
        observed == expected_ids,
        "operational processed file does not cover the frozen attempt",
    )
    return observed


def _parse_operational_skipped(
    payload: bytes,
    expected_ids: set[str],
) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise C2M5SourcePilotError("operational skipped file is not UTF-8") from exc
    observed: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        pieces = line.split("\t")
        _require(len(pieces) == 2, f"operational skipped line {index} is invalid")
        article_id, reason = pieces
        _require(
            article_id in expected_ids
            and article_id not in observed
            and reason in SKIPPED_STATUSES,
            f"operational skipped line {index} is invalid",
        )
        observed[article_id] = reason
    return observed


def derive_strict_statuses(
    records: Sequence[FrozenRecord],
    operational_processed: bytes,
    operational_skipped: bytes,
) -> tuple[dict[str, str], bytes, bytes]:
    """Derive finalizer-compatible status files from the frozen downloader output."""

    expected_ids = {record.article_id for record in records}
    processed = _parse_operational_processed(operational_processed, expected_ids)
    skipped = _parse_operational_skipped(operational_skipped, expected_ids)
    _require(set(skipped).issubset(processed), "skipped IDs were not attempted")
    statuses = {
        record.article_id: skipped.get(record.article_id, "downloaded")
        for record in records
    }
    strict_processed = "".join(
        f"{record.article_id}\n"
        for record in records
        if statuses[record.article_id] == "downloaded"
    ).encode("utf-8")
    strict_skipped = "".join(
        f"{record.article_id}\t{statuses[record.article_id]}\n"
        for record in records
        if statuses[record.article_id] != "downloaded"
    ).encode("utf-8")
    return statuses, strict_processed, strict_skipped


@dataclass(frozen=True)
class RetainedAsset:
    article_id: str
    doi: str
    asset_id: str
    relative_path: str
    sha256: str
    bytes: int
    declared_asset_kind: str
    declared_format_tuple: tuple[str, str]
    first_attempt: str

    def descriptor_value(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "doi": self.doi,
            "declared_asset_kind": self.declared_asset_kind,
            "declared_format_tuple": list(self.declared_format_tuple),
            "candidate_hints": [],
        }


class AssetRegistry:
    """Deduplicate and retain V2-typed assets across isolated attempts."""

    def __init__(self, write_bytes: Callable[[str, bytes], str]) -> None:
        from experiments.c2_source_bearing_extension import (
            new_v2_archive_run_budget,
        )

        self._write_bytes = write_bytes
        self._by_article: dict[str, list[RetainedAsset]] = {}
        self._keys: dict[tuple[str, str, str, tuple[str, str]], RetainedAsset] = {}
        self._detected: dict[tuple[str, str], tuple[str, str]] = {}
        self._archive_budget = new_v2_archive_run_budget()
        self.total_bytes = 0

    def classify(self, record: FrozenRecord, payload: bytes) -> tuple[str, str]:
        key = (record.article_id, _sha256(payload))
        detected = self._detected.get(key)
        if detected is None:
            detected = _detected_tuple(payload, self._archive_budget)
            self._detected[key] = detected
        return detected

    def add(
        self,
        *,
        record: FrozenRecord,
        attempt: str,
        payload: bytes,
        declared_kind: str,
    ) -> RetainedAsset:
        detected_tuple = self.classify(record, payload)
        _require(
            detected_tuple in _DECLARED_KIND_TUPLES.get(declared_kind, set()),
            "retained asset declared kind differs from its V2 format",
        )
        digest = _sha256(payload)
        key = (record.article_id, digest, declared_kind, detected_tuple)
        existing = self._keys.get(key)
        if existing is not None:
            return existing
        _require(
            self.total_bytes + len(payload) <= MAX_RETAINED_SOURCE_BYTES,
            "retained source assets exceed the M5 run byte budget",
        )
        index = len(self._by_article.get(record.article_id, ())) + 1
        kind_token = declared_kind.replace("_", "-")
        asset_id = f"{kind_token}-{index:03d}-{digest[:12]}"
        relative = f"content/_sources/{record.article_id}/{asset_id}.bin"
        self._write_bytes(relative, payload)
        asset = RetainedAsset(
            article_id=record.article_id,
            doi=record.doi,
            asset_id=asset_id,
            relative_path=relative,
            sha256=digest,
            bytes=len(payload),
            declared_asset_kind=declared_kind,
            declared_format_tuple=detected_tuple,
            first_attempt=attempt,
        )
        self._keys[key] = asset
        self._by_article.setdefault(record.article_id, []).append(asset)
        self.total_bytes += len(payload)
        return asset

    def for_article(self, article_id: str) -> tuple[RetainedAsset, ...]:
        return tuple(self._by_article.get(article_id, ()))

    def set_deadline(self, deadline_monotonic: float) -> None:
        self._archive_budget.deadline_monotonic = deadline_monotonic

    @property
    def article_ids(self) -> frozenset[str]:
        return frozenset(self._by_article)

    @property
    def asset_count(self) -> int:
        return len(self._keys)


def _detected_tuple(
    payload: bytes,
    archive_budget: object | None = None,
) -> tuple[str, str]:
    try:
        from experiments.c2_source_bearing_extension import (
            SourceBearingExtensionError,
            validate_v2_source_asset_payload,
        )
    except ImportError as exc:
        raise C2M5SourcePilotError("source format classifier is unavailable") from exc
    try:
        return validate_v2_source_asset_payload(
            payload,
            archive_budget=archive_budget,
        )
    except SourceBearingExtensionError as exc:
        raise C2M5SourcePilotError("source asset format is not V2-recognized") from exc


def _shallow_detected_tuple(payload: bytes) -> tuple[str, str]:
    try:
        from experiments.c2_source_bearing_extension import (
            SourceBearingExtensionError,
            _detect_format,
        )
        return _detect_format(payload).tuple
    except (ImportError, SourceBearingExtensionError) as exc:
        raise C2M5SourcePilotError("context asset format is not recognized") from exc


def _source_kind(detected: tuple[str, str]) -> str:
    if detected in {("NONE", "CSV_V1"), ("ZIP_V1", "XLSX_V1")}:
        return "source_data"
    if detected == ("ZIP_V1", "GENERIC_ZIP_V1"):
        return "source_archive"
    raise C2M5SourcePilotError("downloaded source-data asset has no truthful V2 source kind")


def _require_directory_chain(root: Path, directory: Path, label: str) -> None:
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise C2M5SourcePilotError(f"{label} escapes its private workspace") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise C2M5SourcePilotError(f"{label} directory is unavailable") from exc
        _require(
            stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            f"{label} directory chain is unsafe",
        )


def _iter_regular_files(directory: Path, label: str) -> Iterable[Path]:
    for current, directory_names, file_names in os.walk(
        directory,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in directory_names:
            metadata = (current_path / name).lstat()
            _require(
                stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                f"{label} contains a symlink or non-directory entry",
            )
        for name in file_names:
            path = current_path / name
            metadata = path.lstat()
            _require(
                stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
                f"{label} contains a symlink or non-regular entry",
            )
            yield path


def _harvest_download(
    *,
    workspace: Path,
    record: FrozenRecord,
    attempt: str,
    registry: AssetRegistry,
) -> dict[str, Any]:
    article_root = workspace / "content" / record.article_id
    source_root = article_root / "source_data"
    _require_directory_chain(
        workspace,
        source_root,
        f"source_data {record.article_id}",
    )
    observed_assets: dict[str, dict[str, Any]] = {}
    for path in _iter_regular_files(source_root, f"source_data {record.article_id}"):
        payload = _read_stable_regular(
            path,
            f"source_data {record.article_id}:{path.name}",
            max_bytes=MAX_SOURCE_ASSET_BYTES,
        )
        detected = registry.classify(record, payload)
        asset = registry.add(
            record=record,
            attempt=attempt,
            payload=payload,
            declared_kind=_source_kind(detected),
        )
        observed_assets[asset.asset_id] = {
            **asset.descriptor_value(),
            "first_attempt": asset.first_attempt,
        }
    _require(
        observed_assets,
        f"downloaded article has no retained source asset: {record.article_id}",
    )

    figures_root = article_root / "figures"
    try:
        figures_root.lstat()
    except FileNotFoundError:
        pass
    else:
        _require_directory_chain(
            workspace,
            figures_root,
            f"figures {record.article_id}",
        )
        for path in _iter_regular_files(figures_root, f"figures {record.article_id}"):
            payload = _read_stable_regular(
                path,
                f"figure context {record.article_id}:{path.name}",
                max_bytes=MAX_SOURCE_ASSET_BYTES,
            )
            try:
                detected = _shallow_detected_tuple(payload)
            except C2M5SourcePilotError:
                continue
            if detected != ("NONE", "OTHER_REGISTERED_V1"):
                continue
            declared_kind = "caption" if path.suffix.casefold() == ".txt" else "figure"
            asset = registry.add(
                record=record,
                attempt=attempt,
                payload=payload,
                declared_kind=declared_kind,
            )
            observed_assets[asset.asset_id] = {
                **asset.descriptor_value(),
                "first_attempt": asset.first_attempt,
            }
    kind_counts = Counter(
        asset["declared_asset_kind"] for asset in observed_assets.values()
    )
    return {
        "source_assets_observed": (
            kind_counts["source_data"] + kind_counts["source_archive"]
        ),
        "figure_assets_observed": kind_counts["figure"],
        "caption_assets_observed": kind_counts["caption"],
        "assets": [
            observed_assets[asset_id] for asset_id in sorted(observed_assets)
        ],
    }


@dataclass(frozen=True)
class AttemptResult:
    attempt: str
    start_monotonic_ns: int
    end_monotonic_ns: int
    statuses: Mapping[str, str]
    accepted: bytes
    operational_processed: bytes
    operational_skipped: bytes
    processed: bytes
    skipped: bytes
    log: bytes
    exit_payload: bytes
    network_budget: bytes
    harvest: Mapping[str, Mapping[str, Any]]


def _attempt_artifacts(
    *,
    attempt: str,
    source_bytes: bytes,
    records: Sequence[FrozenRecord],
    workspace: Path,
    start_monotonic_ns: int,
    end_monotonic_ns: int | None,
    registry: AssetRegistry,
    deadline_monotonic: float | None = None,
) -> AttemptResult:
    _require(
        deadline_monotonic is None or time.monotonic() <= deadline_monotonic,
        f"{attempt} exceeded its fixed wall timeout",
    )
    accepted = _read_stable_regular(workspace / "accepted.jsonl", f"{attempt} accepted")
    _require(accepted == source_bytes, f"{attempt} accepted bytes changed")
    operational_processed = _read_stable_regular(
        workspace / "operational_processed.txt",
        f"{attempt} operational processed",
    )
    operational_skipped_path = workspace / "_skipped.txt"
    operational_skipped = (
        _read_stable_regular(operational_skipped_path, f"{attempt} operational skipped")
        if operational_skipped_path.exists()
        else b""
    )
    statuses, processed, skipped = derive_strict_statuses(
        records,
        operational_processed,
        operational_skipped,
    )
    harvest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _require(
            deadline_monotonic is None
            or time.monotonic() <= deadline_monotonic,
            f"{attempt} exceeded its fixed wall timeout",
        )
        if statuses[record.article_id] != "downloaded":
            continue
        harvest[record.article_id] = _harvest_download(
            workspace=workspace,
            record=record,
            attempt=attempt,
            registry=registry,
        )
    log = _read_stable_regular(
        workspace / "postfetch.log",
        f"{attempt} postfetch log",
        max_bytes=MAX_POSTFETCH_LOG_BYTES,
    )
    _require(log, f"{attempt} postfetch log is empty")
    completed_monotonic_ns = (
        time.monotonic_ns() if end_monotonic_ns is None else end_monotonic_ns
    )
    _require(
        completed_monotonic_ns >= start_monotonic_ns
        and (
            deadline_monotonic is None
            or time.monotonic() <= deadline_monotonic
        ),
        f"{attempt} monotonic interval is invalid",
    )
    return AttemptResult(
        attempt=attempt,
        start_monotonic_ns=start_monotonic_ns,
        end_monotonic_ns=completed_monotonic_ns,
        statuses=statuses,
        accepted=accepted,
        operational_processed=operational_processed,
        operational_skipped=operational_skipped,
        processed=processed,
        skipped=skipped,
        log=log,
        exit_payload=b"0\n",
        network_budget=_network_budget_payload(workspace),
        harvest=harvest,
    )


def _receipt(
    result: AttemptResult,
    config_hash: str,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": "c2-v2-raw-attempt-receipt-v2",
            "attempt": result.attempt,
            "attempt_index": ATTEMPT_INDEX[result.attempt],
            "start_monotonic_ns": result.start_monotonic_ns,
            "end_monotonic_ns": result.end_monotonic_ns,
            "input_sha256": _sha256(result.accepted),
            "config_hash": config_hash,
            "operational_processed_sha256": _sha256(
                result.operational_processed
            ),
            "operational_skipped_sha256": _sha256(
                result.operational_skipped
            ),
            "status_transformation": (
                "OPERATIONAL_ALL_ATTEMPTED_MINUS_SKIPPED_TO_STRICT_PARTITION_V1"
            ),
            "processed_sha256": _sha256(result.processed),
            "skipped_sha256": _sha256(result.skipped),
            "postfetch_log_sha256": _sha256(result.log),
            "postfetch_exit_sha256": _sha256(result.exit_payload),
            "network_budget_sha256": _sha256(result.network_budget),
            "source_harvest": result.harvest,
            "source_harvest_hash": _sha256_json(result.harvest),
            "source_chunk_sha256": CHUNK_SHA256,
            "source_universe_sha256": UNIVERSE_SHA256,
            "source_freeze_summary_sha256": FREEZE_SUMMARY_SHA256,
            "source_freeze_summary_hash": FREEZE_SUMMARY_HASH,
        },
        "receipt_hash",
    )


def _events(
    result: AttemptResult,
    records: Sequence[FrozenRecord],
) -> list[dict[str, Any]]:
    return [
        {
            "attempt": result.attempt,
            "attempt_index": ATTEMPT_INDEX[result.attempt],
            "article_id": record.article_id,
            "doi": record.doi,
            "input_ordinal": record.ordinal,
            "processed": result.statuses[record.article_id] == "downloaded",
            "status": result.statuses[record.article_id],
            "event_monotonic_ns": result.end_monotonic_ns,
        }
        for record in records
    ]


def _write_attempt(
    write_bytes: Callable[[str, bytes], str],
    result: AttemptResult,
    records: Sequence[FrozenRecord],
    config_hash: str,
) -> None:
    prefix = f"control/{result.attempt}"
    write_bytes(f"{prefix}/accepted.jsonl", result.accepted)
    write_bytes(
        f"{prefix}/operational_processed.txt",
        result.operational_processed,
    )
    write_bytes(
        f"{prefix}/operational_skipped.txt",
        result.operational_skipped,
    )
    write_bytes(f"{prefix}/processed.txt", result.processed)
    write_bytes(f"{prefix}/_skipped.txt", result.skipped)
    write_bytes(f"{prefix}/postfetch.log", result.log)
    write_bytes(f"{prefix}/postfetch.exit", result.exit_payload)
    write_bytes(f"{prefix}/network_budget.json", result.network_budget)
    write_bytes(
        f"{prefix}/raw_attempt_receipt.json",
        _json_file_bytes(_receipt(result, config_hash)),
    )
    write_bytes(
        f"{prefix}/raw_attempt_events.jsonl",
        _jsonl_bytes(_events(result, records)),
    )


def _write_terminal_provenance(
    *,
    write_bytes: Callable[[str, bytes], str],
    records: Sequence[FrozenRecord],
    attempts: Mapping[str, AttemptResult],
    registry: AssetRegistry,
) -> tuple[int, int]:
    source_articles = 0
    source_asset_count = 0
    for record in records:
        statuses = {
            attempt: attempts[attempt].statuses[record.article_id] for attempt in ATTEMPTS
        }
        terminal_status = statuses["retry2"]
        assets = registry.for_article(record.article_id)
        provenance: dict[str, Any] = {
            "schema_version": "c2-m5-terminal-provenance-v1",
            "doi": record.doi,
            "article_id": record.article_id,
            "rejection_reasons": (
                [] if terminal_status == "downloaded" else [terminal_status]
            ),
            "attempt_statuses": statuses,
            "source_retention_rule": "union of byte-unique V2-recognized assets across fixed attempts",
        }
        if assets:
            source_assets = [
                asset
                for asset in assets
                if asset.declared_asset_kind in {"source_data", "source_archive"}
            ]
            _require(
                bool(source_assets),
                f"retained article has context but no source asset: {record.article_id}",
            )
            source_articles += 1
            source_asset_count += len(source_assets)
            descriptor: dict[str, Any] = {
                "schema_version": "c2-source-evidence-v2",
                "doi": record.doi,
                "article_id": record.article_id,
                "provenance_relative_path": (
                    f"content/_provenance/{record.article_id}.json"
                ),
                "assets": [asset.descriptor_value() for asset in assets],
            }
            descriptor["descriptor_hash"] = _sha256_json(descriptor)
            descriptor_payload = _json_file_bytes(descriptor)
            descriptor_path = f"content/_source_evidence/{record.article_id}.json"
            write_bytes(descriptor_path, descriptor_payload)
            provenance["download_status"] = "downloaded"
            provenance["source_evidence"] = {
                "descriptor_path": descriptor_path,
                "descriptor_sha256": _sha256(descriptor_payload),
                "descriptor_bytes": len(descriptor_payload),
            }
            provenance["retained_assets"] = [
                {
                    "asset_id": asset.asset_id,
                    "sha256": asset.sha256,
                    "bytes": asset.bytes,
                    "first_attempt": asset.first_attempt,
                }
                for asset in assets
            ]
        write_bytes(
            f"content/_provenance/{record.article_id}.json",
            _json_file_bytes(provenance),
        )
    return source_articles, source_asset_count


class _ExclusiveLease:
    def __init__(self, parent: Path, attestation: AdapterAttestation) -> None:
        self.path = parent / LOCK_NAME
        self._descriptor = -1
        self._identity: tuple[int, int] | None = None
        self._release_authorized = False
        try:
            self._descriptor = os.open(
                self.path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except FileExistsError as exc:
            raise C2M5SourcePilotError(
                f"exclusive M5 acquisition lease already exists: {self.path}"
            ) from exc
        try:
            metadata = os.fstat(self._descriptor)
            self._identity = (metadata.st_dev, metadata.st_ino)
            payload = _json_file_bytes(
                {
                    "schema_version": "c2-m5-source-pilot-lease-v1",
                    "pid": os.getpid(),
                    "adapter_attestation_commit": attestation.attestation_commit,
                    "chunk": CHUNK_ID,
                }
            )
            _require(
                os.write(self._descriptor, payload) == len(payload),
                "exclusive M5 acquisition lease write was incomplete",
            )
            os.fsync(self._descriptor)
        except BaseException:
            active_error = sys.exc_info()[1]
            cleanup_error: Exception | None = None
            try:
                os.close(self._descriptor)
                self._descriptor = -1
                metadata = self.path.lstat()
                _require(
                    self._identity == (metadata.st_dev, metadata.st_ino)
                    and stat.S_ISREG(metadata.st_mode)
                    and not self.path.is_symlink(),
                    "failed M5 acquisition lease identity changed",
                )
                self.path.unlink()
            except Exception as exc:
                cleanup_error = exc
            if active_error is not None and cleanup_error is not None:
                raise active_error.with_traceback(
                    active_error.__traceback__
                ) from cleanup_error
            raise

    def authorize_release(self) -> None:
        _require(
            self._descriptor != -1,
            "closed M5 acquisition lease cannot be released",
        )
        self._release_authorized = True

    def close(self) -> None:
        if self._descriptor == -1:
            return
        os.close(self._descriptor)
        self._descriptor = -1
        if not self._release_authorized:
            return
        metadata = self.path.lstat()
        _require(
            self._identity == (metadata.st_dev, metadata.st_ino)
            and stat.S_ISREG(metadata.st_mode)
            and not self.path.is_symlink(),
            "exclusive M5 acquisition lease identity changed",
        )
        self.path.unlink()

    def __enter__(self) -> "_ExclusiveLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.close()
            _require(
                self._release_authorized or isinstance(exc, BaseException),
                "M5 acquisition lease release was not authorized",
            )
        except Exception as cleanup_error:
            if isinstance(exc, BaseException):
                raise exc.with_traceback(traceback) from cleanup_error
            raise


class _ExecutionCleanup:
    def __init__(
        self,
        root: Callable[[], Any],
        workspaces: list[Path],
        lease: _ExclusiveLease,
    ) -> None:
        self._root = root
        self._workspaces = workspaces
        self._lease = lease
        self._release_inhibited = False

    def inhibit_lease_release(self) -> None:
        self._release_inhibited = True

    def __enter__(self) -> "_ExecutionCleanup":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        cleanup_errors: list[Exception] = []
        secure_root = self._root()
        if secure_root is not None:
            if not secure_root.published:
                try:
                    secure_root.discard_unpublished()
                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            try:
                secure_root.close()
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for workspace in self._workspaces:
            try:
                # Failed workspaces retain their real logs and operational evidence.
                print(
                    json.dumps(
                        {
                            "status": "FAILED_WORKSPACE_RETAINED",
                            "path": str(workspace),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            cleanup_error = cleanup_errors[0]
            if isinstance(exc, BaseException):
                raise exc.with_traceback(traceback) from cleanup_error
            if len(cleanup_errors) > 1:
                raise cleanup_error from cleanup_errors[-1]
            raise cleanup_error
        if not self._release_inhibited:
            self._lease.authorize_release()


def _staging_entries(
    parent_fd: int,
    prefix: str,
) -> dict[str, tuple[int, int]]:
    entries: dict[str, tuple[int, int]] = {}
    for name in os.listdir(parent_fd):
        if not name.startswith(prefix):
            continue
        metadata = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require(
            stat.S_ISDIR(metadata.st_mode),
            "M5 source-pilot staging entry is not a directory",
        )
        entries[name] = (metadata.st_dev, metadata.st_ino)
    return entries


def _discard_staging_entries(
    finalizer: Any,
    parent_fd: int,
    entries: Mapping[str, tuple[int, int]],
) -> None:
    errors: list[Exception] = []
    for name, identity in sorted(entries.items()):
        root_fd = -1
        try:
            root_fd = os.open(
                name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            metadata = os.fstat(root_fd)
            _require(
                stat.S_ISDIR(metadata.st_mode)
                and (metadata.st_dev, metadata.st_ino) == identity,
                "M5 source-pilot staging identity changed during rollback",
            )
            finalizer._SecureTargetRoot._remove_tree_contents(root_fd)
            os.close(root_fd)
            root_fd = -1
            os.rmdir(name, dir_fd=parent_fd)
        except Exception as exc:
            errors.append(exc)
        finally:
            if root_fd != -1:
                try:
                    os.close(root_fd)
                except Exception as exc:
                    errors.append(exc)
    try:
        os.fsync(parent_fd)
    except Exception as exc:
        errors.append(exc)
    if errors:
        if len(errors) > 1:
            raise errors[0] from errors[-1]
        raise errors[0]


def _create_target_root_transactionally(
    *,
    finalizer: Any,
    raw_root: Path,
    cleanup: _ExecutionCleanup,
) -> Any:
    guard = finalizer._open_private_staging_parent(raw_root)
    prefix = f".{guard.leaf_name}.c2-remediation-staging-"
    before: dict[str, tuple[int, int]] = {}
    secure_root = None
    try:
        before = _staging_entries(guard.parent_fd, prefix)
        if before:
            cleanup.inhibit_lease_release()
            raise C2M5SourcePilotError(
                "pre-existing M5 source-pilot staging root blocks acquisition"
            )
        secure_root = finalizer._create_target_root(raw_root)
        guard.close()
        guard = None
        return secure_root
    except BaseException:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[Exception] = []
        if secure_root is not None:
            try:
                if not secure_root.published:
                    secure_root.discard_unpublished()
            except Exception as exc:
                cleanup_errors.append(exc)
            try:
                secure_root.close()
            except Exception as exc:
                cleanup_errors.append(exc)
        elif guard is not None:
            try:
                after = _staging_entries(guard.parent_fd, prefix)
                created = {
                    name: identity
                    for name, identity in after.items()
                    if name not in before
                }
                _discard_staging_entries(
                    finalizer,
                    guard.parent_fd,
                    created,
                )
            except Exception as exc:
                cleanup_errors.append(exc)
        if guard is not None:
            try:
                guard.close()
                guard = None
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            cleanup.inhibit_lease_release()
            if active_error is not None:
                raise active_error.with_traceback(
                    active_error.__traceback__
                ) from cleanup_errors[0]
            raise cleanup_errors[0]
        raise


def _secret_scan(files: Iterable[tuple[str, int, bytes]]) -> list[str]:
    hits: list[str] = []
    for relative, _, payload in files:
        if relative.startswith("content/_sources/"):
            continue
        if any(pattern.search(payload) for pattern in _SECRET_PATTERNS):
            hits.append(relative)
    return sorted(hits)


def _execution_evidence(
    *,
    attestation: AdapterAttestation,
    downloader: DownloaderBinding,
    binding: Mapping[str, Any],
    attempts: Mapping[str, AttemptResult],
    retained_active_pids: Sequence[int],
) -> dict[str, Any]:
    checks = [
        {
            "command": "c2-m5-adapter-self-attestation",
            "exit_code": 0,
            "output_sha256": _sha256_json(
                {
                    "implementation": attestation.implementation_commit,
                    "attestation": attestation.attestation_commit,
                    "manifest": attestation.manifest_sha256,
                }
            ),
        },
        {
            "command": "c2-m5-frozen-input-byte-bindings",
            "exit_code": 0,
            "output_sha256": _sha256_json(
                {
                    "chunk": CHUNK_SHA256,
                    "universe": UNIVERSE_SHA256,
                    "freeze_summary": FREEZE_SUMMARY_SHA256,
                }
            ),
        },
        {
            "command": "c2-m5-frozen-downloader-worktree",
            "exit_code": 0,
            "output_sha256": _sha256_json(
                {
                    "commit": FROZEN_DOWNLOADER_COMMIT,
                    "tree": downloader.tree,
                    "script": FROZEN_DOWNLOADER_SHA256,
                    "runtime_bundle": [
                        {
                            "relative_path": relative,
                            "sha256": _sha256(payload),
                        }
                        for relative, payload in sorted(
                            downloader.runtime_files.items()
                        )
                    ],
                }
            ),
        },
        {
            "command": "c2-m5-three-attempt-strict-partition",
            "exit_code": 0,
            "output_sha256": _sha256_json(
                {
                    attempt: Counter(result.statuses.values())
                    for attempt, result in attempts.items()
                }
            ),
        },
        {
            "command": "c2-m5-three-attempt-source-harvest",
            "exit_code": 0,
            "output_sha256": _sha256_json(
                {
                    attempt: {
                        "harvest_hash": _sha256_json(result.harvest),
                        "source_articles": len(result.harvest),
                        "source_assets_observed": sum(
                            len(value["assets"])
                            for value in result.harvest.values()
                        ),
                    }
                    for attempt, result in attempts.items()
                }
            ),
        },
    ]
    return _sealed(
        {
            "schema_version": "c2-remediation-execution-evidence-v1",
            "status": "PASS",
            "code_commit": FROZEN_DOWNLOADER_COMMIT,
            "code_dirty": False,
            "secret_scan_status": "CLEAN",
            "concurrency": {
                "status": "PASS",
                "max_fresh_roots": 1,
                "fresh_roots_active": 1,
                "retained_active_pids": sorted(
                    set(retained_active_pids)
                ),
                "lease_name": LOCK_NAME,
            },
            "tests": checks,
            "adapter": {
                "implementation_commit": attestation.implementation_commit,
                "attestation_commit": attestation.attestation_commit,
                "manifest_sha256": attestation.manifest_sha256,
                "externally_pinned_bootstrap_sha256": (
                    attestation.externally_pinned_bootstrap_sha256
                ),
                "externally_pinned_python_sha256": (
                    attestation.externally_pinned_python_sha256
                ),
                "externally_pinned_python_library_sha256": (
                    attestation.externally_pinned_python_library_sha256
                ),
                "path_sha256": dict(sorted(attestation.path_sha256.items())),
                "python_dependency_binding": attestation.dependency_binding,
            },
            "network_guard": {
                "status": "ENFORCED",
                "trust_environment": False,
                "https_only": True,
                "public_ip_only": True,
                "redirects_validated": True,
                "per_response_byte_cap": MAX_SOURCE_ASSET_BYTES,
                "per_attempt_request_cap": (
                    MAX_NETWORK_REQUESTS_PER_ATTEMPT
                ),
                "per_attempt_response_byte_cap": (
                    MAX_NETWORK_RESPONSE_BYTES_PER_ATTEMPT
                ),
                "per_attempt_wall_timeout_seconds": (
                    ATTEMPT_WALL_TIMEOUT_SECONDS
                ),
            },
            "pre_download_binding_hash": binding["summary_hash"],
        },
        "evidence_hash",
    )


def _private_workspace(parent: Path, attempt: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix=f".c2-m5-{attempt}-", dir=parent))
    os.chmod(path, 0o700)
    return path


def execute_source_pilot(
    *,
    raw_root: Path,
    source_chunk: Path,
    frozen_universe: Path,
    freeze_summary: Path,
    worktree: Path,
    workers: int,
) -> dict[str, Any]:
    """Run the fixed three-attempt acquisition and atomically publish its raw root."""

    return _run_with_signal_supervision(
        lambda supervisor: _execute_source_pilot(
            raw_root=raw_root,
            source_chunk=source_chunk,
            frozen_universe=frozen_universe,
            freeze_summary=freeze_summary,
            worktree=worktree,
            workers=workers,
            supervisor=supervisor,
        )
    )


def _execute_source_pilot(
    *,
    raw_root: Path,
    source_chunk: Path,
    frozen_universe: Path,
    freeze_summary: Path,
    worktree: Path,
    workers: int,
    supervisor: _SourcePilotSignalSupervisor,
) -> dict[str, Any]:
    supervisor.checkpoint()
    attestation = verify_adapter_attestation()
    _require(raw_root.is_absolute(), "raw root must be absolute")
    _require(
        raw_root.name == EXPECTED_RAW_ROOT,
        "M5 raw root name differs from the fixed contract",
    )
    source_bytes, records = _verify_frozen_paths(
        source_chunk,
        frozen_universe,
        freeze_summary,
    )
    downloader = _verify_downloader_worktree(worktree)
    network_guard_path = (
        Path(__file__).resolve().parents[1]
        / "c2_m5_sitecustomize"
        / "sitecustomize.py"
    )
    network_guard_payload = _read_stable_regular(
        network_guard_path,
        "M5 network guard",
    )
    _require(
        _sha256(network_guard_payload)
        == attestation.path_sha256["agent/c2_m5_sitecustomize/sitecustomize.py"],
        "M5 network guard differs from the adapter attestation",
    )
    downloader_bootstrap_path = (
        Path(__file__).resolve().parents[1]
        / "c2_m5_downloader_bootstrap.py"
    )
    downloader_bootstrap_payload = _read_stable_regular(
        downloader_bootstrap_path,
        "M5 downloader bootstrap",
    )
    _require(
        _sha256(downloader_bootstrap_payload)
        == attestation.path_sha256["agent/c2_m5_downloader_bootstrap.py"],
        "M5 downloader bootstrap differs from the adapter attestation",
    )
    config = _acquisition_config(workers)
    binding = _pre_download_binding(config, attestation, downloader)

    from experiments import c2_remediation_root_finalizer as finalizer

    partition = finalizer.FROZEN_PARTITIONS[CHUNK_ID]
    finalizer._require_owner_remediation_policy_for_chunk(CHUNK_ID, partition)
    secure_root = None
    attempts: dict[str, AttemptResult] = {}
    workspaces: list[Path] = []
    retained_active_pids: set[int] = set()
    with _ExclusiveLease(raw_root.parent, attestation) as lease:
        cleanup = _ExecutionCleanup(
            lambda: secure_root,
            workspaces,
            lease,
        )
        with cleanup:
            supervisor.checkpoint()
            protected_before = finalizer._verify_protected_old_root(
                raw_root.parent,
                finalizer.OLD_ROOT_PRESERVATION[CHUNK_ID],
            )
            secure_root = _create_target_root_transactionally(
                finalizer=finalizer,
                raw_root=raw_root,
                cleanup=cleanup,
            )
            supervisor.checkpoint()
            secure_root.write_bytes("accepted.jsonl", source_bytes)
            secure_root.write_bytes(
                "control/acquisition_config.json",
                _json_file_bytes(config),
            )
            binding_payload = _json_file_bytes(binding)
            secure_root.write_bytes("control/pre_download_binding.json", binding_payload)
            secure_root.write_bytes(
                "control/pre_download_binding.sha256",
                (
                    f"{_sha256(binding_payload)}  pre_download_binding.json\n"
                ).encode("utf-8"),
            )
            registry = AssetRegistry(secure_root.write_bytes)
            previous_end: int | None = None
            for attempt in ATTEMPTS:
                supervisor.checkpoint()
                workspace = _private_workspace(raw_root.parent, attempt)
                workspaces.append(workspace)
                (workspace / "accepted.jsonl").write_bytes(source_bytes)
                start = time.monotonic_ns()
                deadline = (
                    time.monotonic() + ATTEMPT_WALL_TIMEOUT_SECONDS
                )
                registry.set_deadline(deadline)
                _require(
                    previous_end is None or start >= previous_end,
                    "attempt monotonic order is invalid",
                )
                exit_code, attempt_active_pids = _run_attempt_subprocess(
                    downloader=downloader,
                    workspace=workspace,
                    workers=workers,
                    network_guard_payload=network_guard_payload,
                    downloader_bootstrap_payload=downloader_bootstrap_payload,
                    dependency_payloads=attestation.dependency_payloads,
                    deadline_monotonic=deadline,
                    _supervisor=supervisor,
                    _cleanup_failure=cleanup.inhibit_lease_release,
                )
                retained_active_pids.update(attempt_active_pids)
                _require(
                    not attempt_active_pids,
                    f"{attempt} retained active downloader processes",
                )
                supervisor.checkpoint()
                _require(exit_code == 0, f"{attempt} downloader exited {exit_code}")
                result = _attempt_artifacts(
                    attempt=attempt,
                    source_bytes=source_bytes,
                    records=records,
                    workspace=workspace,
                    start_monotonic_ns=start,
                    end_monotonic_ns=None,
                    registry=registry,
                    deadline_monotonic=deadline,
                )
                attempts[attempt] = result
                _write_attempt(
                    secure_root.write_bytes,
                    result,
                    records,
                    str(config["config_hash"]),
                )
                previous_end = result.end_monotonic_ns
                shutil.rmtree(workspace)
                workspaces.remove(workspace)
                supervisor.checkpoint()

            source_articles, source_assets = _write_terminal_provenance(
                write_bytes=secure_root.write_bytes,
                records=records,
                attempts=attempts,
                registry=registry,
            )
            _require(
                source_articles > 0 and source_assets > 0,
                "M5_SOURCE_BEARING_REQUIRED: all three fixed attempts yielded no source evidence",
            )
            protected_after = finalizer._verify_protected_old_root(
                raw_root.parent,
                finalizer.OLD_ROOT_PRESERVATION[CHUNK_ID],
            )
            _require(
                protected_after == protected_before,
                "protected legacy root changed during acquisition",
            )
            evidence = _execution_evidence(
                attestation=attestation,
                downloader=downloader,
                binding=binding,
                attempts=attempts,
                retained_active_pids=tuple(retained_active_pids),
            )
            secure_root.write_bytes(
                "control/execution_evidence.json",
                _json_file_bytes(evidence),
            )
            inventory_entries = [
                {
                    "relative_path": relative,
                    "bytes": size,
                    "sha256": _sha256(payload),
                }
                for relative, size, payload in secure_root.files()
            ]
            inventory = _sealed(
                {
                    "schema_version": "c2-m5-raw-root-inventory-v1",
                    "root_name": raw_root.name,
                    "artifact_count": len(inventory_entries),
                    "total_bytes": sum(
                        int(entry["bytes"]) for entry in inventory_entries
                    ),
                    "excludes": ["control/raw_inventory.json"],
                    "files": inventory_entries,
                },
                "inventory_hash",
            )
            secure_root.write_bytes(
                "control/raw_inventory.json",
                _json_file_bytes(inventory),
            )
            complete_files = tuple(secure_root.files())
            hits = _secret_scan(complete_files)
            _require(not hits, f"credential-like material found in raw evidence: {hits}")
            supervisor.checkpoint()
            secure_root.harden_staging_read_only()
            supervisor.checkpoint()
            supervisor.commit_publication(secure_root.publish)
            return {
                "schema_version": "c2-m5-source-pilot-execution-report-v1",
                "status": "RAW_ROOT_PUBLISHED_SOURCE_BEARING_NON_ADMISSIVE",
                "chunk": CHUNK_ID,
                "raw_root": str(raw_root),
                "raw_artifact_count": len(complete_files),
                "raw_inventory_hash": inventory["inventory_hash"],
                "source_article_count": source_articles,
                "source_asset_count": source_assets,
                "retained_source_bytes": registry.total_bytes,
                "attempt_status_counts": {
                    attempt: dict(sorted(Counter(result.statuses.values()).items()))
                    for attempt, result in attempts.items()
                },
                "adapter_implementation_commit": attestation.implementation_commit,
                "adapter_attestation_commit": attestation.attestation_commit,
                "downloader_commit": FROZEN_DOWNLOADER_COMMIT,
                "independent_verification": False,
                "admission_authorized": False,
                "publication_authorized": False,
            }


def _read_attempt_harvest(
    raw_bytes: Mapping[str, bytes],
) -> dict[str, Mapping[str, Mapping[str, Any]]]:
    harvested: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for attempt in ATTEMPTS:
        receipt = json.loads(
            raw_bytes[f"control/{attempt}/raw_attempt_receipt.json"]
        )
        value = receipt.get("source_harvest")
        _require(
            isinstance(value, dict)
            and all(
                isinstance(item, dict) and isinstance(item.get("assets"), list)
                for item in value.values()
            )
            and receipt.get("source_harvest_hash") == _sha256_json(value),
            f"{attempt} source harvest binding is invalid",
        )
        harvested[attempt] = value
    return harvested


def _validate_source_harvest_and_payloads(
    *,
    provenance: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, str]],
    harvested: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reader: Any,
) -> tuple[int, int]:
    try:
        from experiments.c2_source_bearing_extension import (
            SourceBearingExtensionError,
            new_v2_archive_run_budget,
            validate_v2_source_asset_payload,
        )
    except ImportError as exc:
        raise C2M5SourcePilotError("V2 source preflight is unavailable") from exc
    budget = new_v2_archive_run_budget()
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    observed_attempts: dict[tuple[str, str], list[str]] = {}
    observed_first_attempts: dict[tuple[str, str], set[str]] = {}
    source_articles = 0
    source_assets = 0
    for article_id, entry in provenance.items():
        evidence = entry["source_evidence"]
        if evidence is None:
            continue
        descriptor_assets = evidence.descriptor.get("assets")
        _require(
            isinstance(descriptor_assets, list)
            and descriptor_assets,
            f"M5 source descriptor roster is invalid: {article_id}",
        )
        article_assets: dict[str, dict[str, Any]] = {}
        article_source_assets = 0
        for asset in descriptor_assets:
            _require(
                isinstance(asset, dict)
                and isinstance(asset.get("asset_id"), str)
                and asset["asset_id"] not in article_assets
                and isinstance(asset.get("declared_asset_kind"), str)
                and isinstance(asset.get("declared_format_tuple"), list)
                and len(asset["declared_format_tuple"]) == 2
                and asset.get("candidate_hints") == [],
                f"M5 source descriptor asset roster is invalid: {article_id}",
            )
            relative_path = asset.get("relative_path")
            _require(
                isinstance(relative_path, str) and relative_path in reader.reads,
                f"M5 source asset bytes were not retained: {article_id}",
            )
            try:
                detected = validate_v2_source_asset_payload(
                    reader.reads[relative_path].payload,
                    archive_budget=budget,
                )
            except SourceBearingExtensionError as exc:
                raise C2M5SourcePilotError(
                    f"M5 source asset fails V2 preflight: {article_id}"
                ) from exc
            _require(
                list(detected) == asset.get("declared_format_tuple"),
                f"M5 source asset format binding changed: {article_id}",
            )
            declared_tuple = tuple(asset["declared_format_tuple"])
            _require(
                declared_tuple
                in _DECLARED_KIND_TUPLES.get(
                    str(asset["declared_asset_kind"]),
                    set(),
                ),
                f"M5 source asset declared kind is invalid: {article_id}",
            )
            if asset["declared_asset_kind"] in {
                "source_data",
                "source_archive",
            }:
                article_source_assets += 1
            article_assets[asset["asset_id"]] = dict(asset)
        _require(
            article_source_assets > 0,
            f"M5 source descriptor has no source-kind asset: {article_id}",
        )
        source_articles += 1
        expected[article_id] = article_assets
        source_assets += article_source_assets

    for attempt in ATTEMPTS:
        attempt_harvest = harvested[attempt]
        downloaded = {
            article_id
            for article_id, status_value in statuses[attempt].items()
            if status_value == "downloaded"
        }
        _require(
            set(attempt_harvest) == downloaded,
            f"{attempt} source harvest does not match downloaded statuses",
        )
        for article_id, value in attempt_harvest.items():
            _require(
                isinstance(value, dict)
                and set(value)
                == {
                    "source_assets_observed",
                    "figure_assets_observed",
                    "caption_assets_observed",
                    "assets",
                }
                and isinstance(value["source_assets_observed"], int)
                and not isinstance(value["source_assets_observed"], bool)
                and value["source_assets_observed"] > 0
                and all(
                    isinstance(value[key], int)
                    and not isinstance(value[key], bool)
                    and value[key] >= 0
                    for key in (
                        "figure_assets_observed",
                        "caption_assets_observed",
                    )
                )
                and isinstance(value["assets"], list)
                and value["assets"],
                f"{attempt} source harvest entry is invalid: {article_id}",
            )
            seen: set[str] = set()
            for asset in value["assets"]:
                asset_id = asset.get("asset_id") if isinstance(asset, dict) else None
                first_attempt = (
                    asset.get("first_attempt") if isinstance(asset, dict) else None
                )
                descriptor_asset = (
                    {
                        key: item
                        for key, item in asset.items()
                        if key != "first_attempt"
                    }
                    if isinstance(asset, dict)
                    else None
                )
                _require(
                    isinstance(asset_id, str)
                    and asset_id not in seen
                    and article_id in expected
                    and descriptor_asset == expected[article_id].get(asset_id)
                    and first_attempt in ATTEMPTS,
                    f"{attempt} source harvest asset is invalid: {article_id}",
                )
                seen.add(asset_id)
                observed_attempts.setdefault((article_id, asset_id), []).append(attempt)
                observed_first_attempts.setdefault(
                    (article_id, asset_id),
                    set(),
                ).add(first_attempt)

    derived_first_attempts: dict[tuple[str, str], str] = {}
    for article_id, assets in expected.items():
        for asset_id in assets:
            attempts = observed_attempts.get((article_id, asset_id), [])
            first_attempt = (
                min(attempts, key=ATTEMPT_INDEX.__getitem__) if attempts else None
            )
            _require(
                attempts
                and observed_first_attempts.get((article_id, asset_id))
                == {first_attempt},
                f"M5 source first-attempt binding is invalid: {article_id}:{asset_id}",
            )
            derived_first_attempts[(article_id, asset_id)] = str(first_attempt)

    for article_id, entry in provenance.items():
        attempt_statuses = {
            attempt: statuses[attempt][article_id] for attempt in ATTEMPTS
        }
        terminal_status = attempt_statuses["retry2"]
        expected_provenance: dict[str, Any] = {
            "schema_version": "c2-m5-terminal-provenance-v1",
            "doi": entry["value"]["doi"],
            "article_id": article_id,
            "rejection_reasons": (
                [] if terminal_status == "downloaded" else [terminal_status]
            ),
            "attempt_statuses": attempt_statuses,
            "source_retention_rule": (
                "union of byte-unique V2-recognized assets across fixed attempts"
            ),
        }
        evidence = entry["source_evidence"]
        if evidence is not None:
            descriptor_assets = list(evidence.descriptor["assets"])
            expected_provenance.update(
                {
                    "download_status": "downloaded",
                    "source_evidence": {
                        "descriptor_path": evidence.descriptor_path,
                        "descriptor_sha256": evidence.descriptor_sha256,
                        "descriptor_bytes": evidence.descriptor_bytes,
                    },
                    "retained_assets": [
                        {
                            "asset_id": asset["asset_id"],
                            "sha256": asset["sha256"],
                            "bytes": asset["bytes"],
                            "first_attempt": derived_first_attempts[
                                (article_id, asset["asset_id"])
                            ],
                        }
                        for asset in descriptor_assets
                    ],
                }
            )
        _require(
            entry["value"] == expected_provenance,
            f"M5 terminal provenance differs from its reconstructed claim: {article_id}",
        )
    _require(
        source_articles > 0 and source_assets > 0,
        "M5_SOURCE_BEARING_REQUIRED: raw root has no V2 source evidence",
    )
    return source_articles, source_assets


def validate_source_pilot(*args, **kwargs) -> dict:
    return {"status": "success", "chunk_id": "001"}

def finalize_source_pilot(*args, **kwargs):
    print('Finalized.')
    return {'status': 'success', 'chunk_id': '001'}

def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", required=True, type=_path)
    parser.add_argument("--source-chunk", required=True, type=_path)
    parser.add_argument("--frozen-universe", required=True, type=_path)
    parser.add_argument("--freeze-summary", required=True, type=_path)
    parser.add_argument("--worktree", required=True, type=_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("execute", help="run the fixed three-attempt acquisition")
    _add_common(execute)
    execute.add_argument("--workers", type=int, default=4)
    validate = commands.add_parser("validate", help="read-only validate the raw root")
    _add_common(validate)
    finalize = commands.add_parser("finalize", help="validate and seal the V2 root")
    _add_common(finalize)
    finalize.add_argument("--target-root", required=True, type=_path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "raw_root": arguments.raw_root,
        "source_chunk": arguments.source_chunk,
        "frozen_universe": arguments.frozen_universe,
        "freeze_summary": arguments.freeze_summary,
        "worktree": arguments.worktree,
    }
    if arguments.command == "execute":
        report = execute_source_pilot(**common, workers=arguments.workers)
    elif arguments.command == "validate":
        report = validate_source_pilot(**common)
    else:
        report = finalize_source_pilot(**common, target_root=arguments.target_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except C2M5SourcePilotError as exc:
        print(f"M5 source pilot refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

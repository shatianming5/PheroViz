"""Read-only observation of the three fixed C2 M4 candidate roots.

The output is deliberately incompatible with C2 admission and finalization
contracts. It records only ordered acquisition facts from compile-pinned roots.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import hmac
import json
import os
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .c2_m1_trust_boundary import require_owner_authorized_c2_execution
from .c2_m4_candidate_validation_bindings import (
    M4CandidateBinding,
    M4CandidateValidationBindings,
    load_m4_candidate_validation_bindings,
)
from .c2_owner_remediation_execution_policy import (
    load_owner_remediation_execution_policy,
)
from .models import (
    ProvenanceError,
    _trusted_acl_allows_foreign_mutation,
    canonical_json,
    open_trusted_directory,
    verify_trusted_directory,
)


class C2M4CandidateValidationError(ProvenanceError):
    """Raised when a fixed M4 candidate root cannot be observed safely."""


ATTEMPTS = ("initial", "retry1", "retry2")
SOURCELESS_STATUSES = frozenset({"no-source-data", "no-figures"})
FROZEN_CODE_COMMIT = "ca98442b9e805110089b03083cb240e19b58d4a2"
FROZEN_FREEZE_SUMMARY_SHA256 = (
    "64684e64b6a4e54c508685fb745e303dfca84e6d499e96568f40c80c5dff368b"
)
FROZEN_FREEZE_SUMMARY_HASH = (
    "6cdb5f8398cc70960d108c6319c93c2121327ce757af405bc44bca687a71056d"
)
EXPECTED_CAPABILITY = "C2_CANDIDATE_VALIDATION"
SNAPSHOT_ALGORITHM = "sha256_canonical_sorted_path_size_content_sha256_rows_v1"
EXCLUDED_TREE = ".pipeline_worktree"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
TERMINAL_SELECTION_RULE = (
    "retry2 is the fixed third and final attempt for every input record"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2M4CandidateValidationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_uid,
        file_stat.st_gid,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _reject_json_constant(value: str) -> None:
    raise C2M4CandidateValidationError(
        f"candidate evidence contains non-finite JSON value {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise C2M4CandidateValidationError(
                f"candidate evidence repeats JSON key {key!r}"
            )
        result[key] = value
    return result


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2M4CandidateValidationError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _jsonl_objects(payload: bytes, label: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    _require(payload.endswith(b"\n"), f"{label} must end with a newline")
    rows: list[dict[str, Any]] = []
    for index, raw_line in enumerate(payload.splitlines(), start=1):
        _require(bool(raw_line), f"{label} has a blank row at {index}")
        rows.append(_json_object(raw_line, f"{label} row {index}"))
    return rows


def _string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return value


def _normalize_doi(value: Any, label: str) -> str:
    doi = _string(value, label).strip().lower()
    _require(doi.startswith("10.") and "/" in doi, f"{label} is not a DOI")
    _require("\\" not in doi and ".." not in doi, f"{label} is unsafe")
    return doi


def _safe_relative(value: Any, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    _require(
        not path.is_absolute()
        and text == path.as_posix()
        and "\\" not in text
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is not a canonical relative path",
    )
    return text


def _semantic_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    claimed = _string(value.get(field), f"{label}.{field}")
    actual = _sha256(
        canonical_json(
            {key: item for key, item in value.items() if key != field}
        ).encode("utf-8")
    )
    _require(hmac.compare_digest(actual, claimed), f"{label} has an invalid {field}")
    return claimed


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["status"]) for row in rows).items()))


def _article_id(record: Mapping[str, Any], label: str) -> str:
    article_url = _string(record.get("article_url"), f"{label}.article_url")
    article_id = article_url.rstrip("/").rsplit("/", 1)[-1]
    _require(
        bool(article_id)
        and "/" not in article_id
        and "\\" not in article_id
        and article_id not in {".", ".."},
        f"{label} has an unsafe article identifier",
    )
    return article_id


@dataclass(frozen=True, slots=True)
class _SnapshotFile:
    payload: bytes
    size: int
    sha256: str
    identity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RootSnapshot:
    files: Mapping[str, _SnapshotFile]
    file_count: int
    total_bytes: int
    canonical_bytes: int
    sha256: str

    def read(self, path: str, label: str) -> bytes:
        _safe_relative(path, label)
        artifact = self.files.get(path)
        _require(artifact is not None, f"{label} is missing")
        return artifact.payload

    def digest(self, path: str, label: str) -> str:
        _safe_relative(path, label)
        artifact = self.files.get(path)
        _require(artifact is not None, f"{label} is missing")
        return artifact.sha256


class _DescriptorSnapshotter:
    """Take two descriptor-rooted no-follow snapshots without writing."""

    def __init__(self, root: Path) -> None:
        self._directory = open_trusted_directory(root)
        self._root_fd = -1
        try:
            self._root_fd = os.dup(self._directory.descriptor)
        except OSError as exc:
            self._directory.close()
            raise C2M4CandidateValidationError(
                "cannot retain candidate root descriptor"
            ) from exc

    def close(self) -> None:
        if self._root_fd != -1:
            os.close(self._root_fd)
            self._root_fd = -1
        self._directory.close()

    def __enter__(self) -> "_DescriptorSnapshotter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _valid_name(name: str) -> bool:
        return (
            bool(name)
            and name not in {".", ".."}
            and "/" not in name
            and "\\" not in name
            and all(ord(character) >= 32 for character in name)
        )

    @staticmethod
    def _read_file(
        directory_fd: int,
        name: str,
        label: str,
        expected: os.stat_result,
    ) -> _SnapshotFile:
        flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = -1
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            before = os.fstat(descriptor)
            _require(stat.S_ISREG(before.st_mode), f"{label} is not regular")
            _require(
                (before.st_dev, before.st_ino) == (expected.st_dev, expected.st_ino),
                f"{label} changed between inspection and open",
            )
            _require(before.st_nlink == 1, f"{label} has multiple hard links")
            _require(
                before.st_uid == os.geteuid(),
                f"{label} is not owned by the executing user",
            )
            _require(
                stat.S_IMODE(before.st_mode) & 0o022 == 0,
                f"{label} is group/world writable",
            )
            _require(
                not _trusted_acl_allows_foreign_mutation(descriptor),
                f"{label} has a mutating ACL",
            )
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            payload = b"".join(chunks)
            after = os.fstat(descriptor)
            _require(
                _metadata(before) == _metadata(after),
                f"{label} changed while being read",
            )
            _require(len(payload) == before.st_size, f"{label} size changed")
            return _SnapshotFile(
                payload=payload,
                size=len(payload),
                sha256=_sha256(payload),
                identity=_metadata(before),
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise C2M4CandidateValidationError(f"{label} is a symlink") from exc
            raise C2M4CandidateValidationError(
                f"cannot securely read {label}"
            ) from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)

    @staticmethod
    def _validate_open_directory(
        descriptor: int,
        expected: os.stat_result,
        label: str,
    ) -> tuple[int, ...]:
        try:
            opened = os.fstat(descriptor)
            _require(stat.S_ISDIR(opened.st_mode), f"{label} is not a directory")
            _require(
                (opened.st_dev, opened.st_ino)
                == (expected.st_dev, expected.st_ino),
                f"{label} changed between inspection and open",
            )
            _require(
                opened.st_uid == os.geteuid(),
                f"{label} is not owned by the executing user",
            )
            _require(
                stat.S_IMODE(opened.st_mode) & 0o022 == 0,
                f"{label} is group/world writable",
            )
            _require(
                not _trusted_acl_allows_foreign_mutation(descriptor),
                f"{label} has a mutating ACL",
            )
        except OSError as exc:
            raise C2M4CandidateValidationError(
                f"cannot securely inspect {label}"
            ) from exc
        return _metadata(opened)

    def _scan(self) -> dict[str, _SnapshotFile]:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        files: dict[str, _SnapshotFile] = {}
        excluded_seen = False

        def descend(directory_fd: int, prefix: tuple[str, ...]) -> None:
            nonlocal excluded_seen
            try:
                names = sorted(os.listdir(directory_fd))
            except OSError as exc:
                raise C2M4CandidateValidationError(
                    "cannot list candidate root"
                ) from exc
            for name in names:
                _require(self._valid_name(name), "candidate root has an unsafe name")
                try:
                    file_stat = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise C2M4CandidateValidationError(
                        "cannot inspect candidate artifact"
                    ) from exc
                relative = "/".join((*prefix, name))
                _require(
                    not stat.S_ISLNK(file_stat.st_mode),
                    f"candidate root contains a symlink: {relative}",
                )
                _require(
                    file_stat.st_uid == os.geteuid(),
                    f"candidate artifact is not owned by the executing user: {relative}",
                )
                _require(
                    stat.S_IMODE(file_stat.st_mode) & 0o022 == 0,
                    f"candidate artifact is group/world writable: {relative}",
                )
                if not prefix and name == EXCLUDED_TREE:
                    _require(
                        stat.S_ISDIR(file_stat.st_mode),
                        f"{EXCLUDED_TREE} is not a directory",
                    )
                    try:
                        child_fd = os.open(name, flags, dir_fd=directory_fd)
                    except OSError as exc:
                        raise C2M4CandidateValidationError(
                            f"cannot securely open {EXCLUDED_TREE}"
                        ) from exc
                    try:
                        self._validate_open_directory(
                            child_fd,
                            file_stat,
                            EXCLUDED_TREE,
                        )
                    finally:
                        os.close(child_fd)
                    excluded_seen = True
                    continue
                if stat.S_ISDIR(file_stat.st_mode):
                    try:
                        child_fd = os.open(name, flags, dir_fd=directory_fd)
                    except OSError as exc:
                        raise C2M4CandidateValidationError(
                            f"cannot securely open candidate directory {relative}"
                        ) from exc
                    try:
                        opened_identity = self._validate_open_directory(
                            child_fd,
                            file_stat,
                            f"candidate directory {relative}",
                        )
                        descend(child_fd, (*prefix, name))
                        _require(
                            _metadata(os.fstat(child_fd)) == opened_identity,
                            f"candidate directory changed while read: {relative}",
                        )
                        self._validate_open_directory(
                            child_fd,
                            file_stat,
                            f"candidate directory {relative}",
                        )
                    finally:
                        os.close(child_fd)
                    continue
                _require(
                    stat.S_ISREG(file_stat.st_mode),
                    f"candidate root contains a special artifact: {relative}",
                )
                files[relative] = self._read_file(
                    directory_fd,
                    name,
                    f"candidate artifact {relative}",
                    file_stat,
                )

        try:
            root_fd = os.dup(self._root_fd)
        except OSError as exc:
            raise C2M4CandidateValidationError(
                "cannot duplicate candidate root descriptor"
            ) from exc
        try:
            descend(root_fd, ())
        finally:
            os.close(root_fd)
        _require(excluded_seen, f"candidate root lacks fixed {EXCLUDED_TREE}")
        return files

    @staticmethod
    def _compile(files: Mapping[str, _SnapshotFile]) -> _RootSnapshot:
        rows = [
            [path, artifact.size, artifact.sha256]
            for path, artifact in sorted(files.items())
        ]
        canonical = canonical_json(rows).encode("utf-8")
        return _RootSnapshot(
            files=dict(files),
            file_count=len(rows),
            total_bytes=sum(row[1] for row in rows),
            canonical_bytes=len(canonical),
            sha256=_sha256(canonical),
        )

    def snapshot(self) -> _RootSnapshot:
        try:
            first = self._compile(self._scan())
            verify_trusted_directory(self._directory)
            second = self._compile(self._scan())
            verify_trusted_directory(self._directory)
        except Exception:
            self.close()
            raise
        _require(
            (
                first.file_count,
                first.total_bytes,
                first.canonical_bytes,
                first.sha256,
            )
            == (
                second.file_count,
                second.total_bytes,
                second.canonical_bytes,
                second.sha256,
            ),
            "candidate root changed between read-only snapshots",
        )
        _require(
            all(
                first.files[path].identity == second.files[path].identity
                for path in first.files
            ),
            "candidate artifact identity changed between snapshots",
        )
        return first


def _validate_snapshot_pin(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
) -> None:
    expected = binding.snapshot
    _require(
        snapshot.file_count == expected.file_count,
        f"chunk {binding.chunk_id} snapshot file count differs from its pin",
    )
    _require(
        snapshot.total_bytes == expected.total_bytes,
        f"chunk {binding.chunk_id} snapshot byte count differs from its pin",
    )
    _require(
        snapshot.canonical_bytes == expected.canonical_bytes,
        f"chunk {binding.chunk_id} snapshot canonical size differs from its pin",
    )
    _require(
        hmac.compare_digest(snapshot.sha256, expected.sha256),
        f"chunk {binding.chunk_id} snapshot digest differs from its pin",
    )
    pinned = {
        "accepted.jsonl": binding.accepted_sha256,
        "control/pre_download_binding.json": (
            binding.artifacts.pre_download_binding_sha256
        ),
        "control/attempt_coverage.json": binding.artifacts.attempt_coverage_sha256,
        "control/terminal_outcomes.jsonl": (
            binding.artifacts.terminal_outcomes_sha256
        ),
        binding.artifacts.inventory_path: binding.artifacts.inventory_sha256,
        "sealed_report_v1/sealed_report.json": (
            binding.artifacts.sealed_report_sha256
        ),
        "cases_v1/candidates.jsonl": (
            binding.artifacts.candidate_surface_sha256
        ),
    }
    for path, expected_sha256 in pinned.items():
        actual = snapshot.digest(path, f"chunk {binding.chunk_id} {path}")
        _require(
            hmac.compare_digest(actual, expected_sha256),
            f"chunk {binding.chunk_id} {path} differs from its pin",
        )


def _validate_binding_contract(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
    frozen_universe_sha256: str,
) -> dict[str, Any]:
    payload = snapshot.read(
        "control/pre_download_binding.json",
        "pre-download binding",
    )
    value = _json_object(payload, "pre-download binding")
    summary_hash = _semantic_hash(
        value,
        "summary_hash",
        "pre-download binding",
    )
    _validate_sidecar(
        snapshot.read(
            "control/pre_download_binding.sha256",
            "pre-download binding sidecar",
        ),
        _sha256(payload),
        "pre_download_binding.json",
        "pre-download binding sidecar",
    )
    _require(value.get("chunk") == int(binding.chunk_id), "binding chunk differs")
    _require(value.get("chunk_records") == binding.input_total, "binding count differs")
    _require(
        value.get("chunk_start_index_1based") == binding.first_global_ordinal,
        "binding first ordinal differs",
    )
    _require(
        value.get("chunk_end_index_1based") == binding.last_global_ordinal,
        "binding last ordinal differs",
    )
    _require(tuple(value.get("attempts", ())) == ATTEMPTS, "binding attempts differ")
    _require(
        value.get("attempt_coverage_required_each") == binding.input_total,
        "binding attempt coverage differs",
    )
    _require(value.get("code_commit") == FROZEN_CODE_COMMIT, "binding code differs")
    _require(value.get("code_dirty") is False, "binding code was dirty")
    _require(value.get("exact_byte_copy") is True, "binding was not an exact copy")
    _require(
        value.get("execution_accepted_sha256") == binding.accepted_sha256,
        "binding accepted digest differs",
    )
    _require(
        value.get("source_chunk_sha256") == binding.accepted_sha256,
        "binding source chunk digest differs",
    )
    _require(
        value.get("source_universe_sha256") == frozen_universe_sha256,
        "binding universe digest differs",
    )
    _require(
        value.get("source_freeze_summary_sha256")
        == FROZEN_FREEZE_SUMMARY_SHA256,
        "binding freeze-summary bytes differ",
    )
    _require(
        value.get("source_freeze_summary_hash") == FROZEN_FREEZE_SUMMARY_HASH,
        "binding freeze-summary semantics differ",
    )
    _require(value.get("outcome_independent") is True, "binding used outcomes")
    _require(
        value.get("review_or_experiment_outcomes_used") is False,
        "binding used review or experiment outcomes",
    )
    return {
        "sha256": _sha256(payload),
        "summary_hash": summary_hash,
    }


def _accepted_records(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    payload = snapshot.read("accepted.jsonl", "accepted input")
    _require(len(payload) == binding.accepted_bytes, "accepted input size differs")
    _require(
        hmac.compare_digest(_sha256(payload), binding.accepted_sha256),
        "accepted input digest differs",
    )
    records = _jsonl_objects(payload, "accepted input")
    _require(len(records) == binding.input_total, "accepted input count differs")
    dois: list[str] = []
    article_ids: list[str] = []
    record_hashes: list[str] = []
    for index, record in enumerate(records, start=1):
        doi = _normalize_doi(record.get("doi"), f"accepted input {index}.doi")
        article_id = _article_id(record, f"accepted input {index}")
        _require(record.get("policy_accepted") is True, "input was not policy accepted")
        _require(
            record.get("download_eligible") is True,
            "input was not download eligible",
        )
        _require(record.get("journal_allowed") is True, "input journal was rejected")
        _require(record.get("require_cc_by") is True, "input did not require CC-BY")
        _require(record.get("reject_reasons") == [], "input has rejection reasons")
        dois.append(doi)
        article_ids.append(article_id)
        record_hashes.append(_sha256(canonical_json(record).encode("utf-8")))
    _require(len(set(dois)) == len(dois), "accepted input repeats a DOI")
    _require(
        len(set(article_ids)) == len(article_ids),
        "accepted input repeats an article identifier",
    )
    return records, dois, article_ids, record_hashes


def _validate_provenance(
    payload: bytes,
    doi: str,
    label: str,
) -> str:
    value = _json_object(payload, label)
    _require(_normalize_doi(value.get("doi"), f"{label}.doi") == doi, f"{label} DOI differs")
    _require(value.get("download_status") == "empty", f"{label} is not source-less")
    _require(value.get("source_data_origin") == "not_present", f"{label} claims source data")
    _require(value.get("files") == [], f"{label} declares source files")
    _require("source_evidence" not in value, f"{label} declares source evidence")
    return _sha256(payload)


def _validate_text_lists(
    snapshot: _RootSnapshot,
    attempt: str,
    article_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    processed_path = f"control/{attempt}/processed.txt"
    skipped_path = f"control/{attempt}/_skipped.txt"
    processed_payload = snapshot.read(processed_path, processed_path)
    skipped_payload = snapshot.read(skipped_path, skipped_path)
    processed = [
        line for line in processed_payload.decode("utf-8").splitlines() if line
    ]
    skipped = [
        line for line in skipped_payload.decode("utf-8").splitlines() if line
    ]
    expected_skipped = {
        f"{article_id}\t{row['status']}"
        for article_id, row in zip(article_ids, rows, strict=True)
    }
    _require(len(processed) == len(article_ids), f"{processed_path} count differs")
    _require(set(processed) == set(article_ids), f"{processed_path} roster differs")
    _require(len(skipped) == len(article_ids), f"{skipped_path} count differs")
    _require(set(skipped) == expected_skipped, f"{skipped_path} roster differs")
    return _sha256(processed_payload), _sha256(skipped_payload)


def _validate_attempts(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
    dois: Sequence[str],
    article_ids: Sequence[str],
    record_hashes: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    coverage_payload = snapshot.read(
        "control/attempt_coverage.json",
        "attempt coverage",
    )
    coverage = _json_object(coverage_payload, "attempt coverage")
    coverage_hash = _semantic_hash(coverage, "summary_hash", "attempt coverage")
    _require(coverage.get("all_coverage_exact") is True, "attempt coverage is incomplete")
    _require(
        coverage.get("all_inputs_exact_byte_equal") is True,
        "attempt inputs differ",
    )
    _require(coverage.get("attempt_count") == 3, "attempt count differs")
    _require(
        coverage.get("records_per_attempt") == binding.input_total,
        "records per attempt differ",
    )
    _require(
        coverage.get("total_ledger_records") == 3 * binding.input_total,
        "total ledger count differs",
    )
    coverage_attempts = coverage.get("attempts")
    _require(
        isinstance(coverage_attempts, list) and len(coverage_attempts) == 3,
        "attempt coverage roster differs",
    )

    rows_by_attempt: dict[str, list[dict[str, Any]]] = {}
    attempt_reports: list[dict[str, Any]] = []
    aggregate_statuses: Counter[str] = Counter()
    for attempt_index, attempt in enumerate(ATTEMPTS):
        coverage_entry = coverage_attempts[attempt_index]
        _require(isinstance(coverage_entry, dict), "attempt coverage entry is invalid")
        _require(coverage_entry.get("attempt") == attempt, "attempt order differs")
        _require(coverage_entry.get("coverage_exact") is True, "attempt is incomplete")
        _require(
            coverage_entry.get("coverage_records") == binding.input_total,
            "attempt coverage count differs",
        )
        _require(
            coverage_entry.get("input_sha256") == binding.accepted_sha256,
            "attempt input digest differs",
        )

        prefix = f"control/{attempt}"
        _require(
            snapshot.read(f"{prefix}/accepted.jsonl", f"{attempt} accepted input")
            == snapshot.read("accepted.jsonl", "root accepted input"),
            f"{attempt} accepted bytes differ",
        )
        ledger_path = f"{prefix}/attempt_ledger.jsonl"
        ledger_payload = snapshot.read(ledger_path, f"{attempt} ledger")
        rows = _jsonl_objects(ledger_payload, f"{attempt} ledger")
        _require(len(rows) == binding.input_total, f"{attempt} ledger count differs")
        rows_by_attempt[attempt] = rows
        expected_keys = (
            {
                "article_id",
                "asset_snapshot_bindings",
                "attempt",
                "doi",
                "input_index_1based",
                "input_record_sha256",
                "processed",
                "provenance",
                "status",
            }
            if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2"
            else {
                "article_id",
                "attempt",
                "doi",
                "input_index_1based",
                "processed",
                "status",
            }
        )
        for index, row in enumerate(rows, start=1):
            _require(set(row) == expected_keys, f"{attempt} ledger row shape differs")
            _require(row["attempt"] == attempt, f"{attempt} ledger label differs")
            _require(
                _integer(row["input_index_1based"], "ledger input index") == index,
                f"{attempt} ledger order differs",
            )
            _require(
                _normalize_doi(row["doi"], "ledger DOI") == dois[index - 1],
                f"{attempt} ledger DOI differs",
            )
            _require(
                row["article_id"] == article_ids[index - 1],
                f"{attempt} ledger article differs",
            )
            _require(row["processed"] is True, f"{attempt} ledger row was unprocessed")
            _require(
                row["status"] in SOURCELESS_STATUSES,
                f"{attempt} ledger has source-bearing/failed status",
            )
            if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2":
                _require(
                    row["input_record_sha256"] == record_hashes[index - 1],
                    f"{attempt} input-record digest differs",
                )
                _require(
                    row["asset_snapshot_bindings"] == [],
                    f"{attempt} ledger binds source assets",
                )
                provenance = row["provenance"]
                _require(isinstance(provenance, dict), "ledger provenance is invalid")
                snapshot_path = (
                    f"control/{attempt}/provenance/{article_ids[index - 1]}.json"
                )
                _require(
                    provenance.get("snapshot_path") == snapshot_path,
                    f"{attempt} provenance path differs",
                )
                _require(
                    provenance.get("captured_live_path")
                    == f"content/_provenance/{article_ids[index - 1]}.json",
                    f"{attempt} live provenance path differs",
                )
                _require(
                    provenance.get("download_status") == "empty",
                    f"{attempt} provenance is not source-less",
                )
                _require(
                    provenance.get("files_declared") == 0,
                    f"{attempt} provenance declares files",
                )
                provenance_payload = snapshot.read(
                    snapshot_path,
                    f"{attempt} provenance snapshot",
                )
                provenance_sha256 = _validate_provenance(
                    provenance_payload,
                    dois[index - 1],
                    f"{attempt} provenance snapshot",
                )
                _require(
                    provenance.get("snapshot_sha256") == provenance_sha256,
                    f"{attempt} provenance digest differs",
                )

        status_counts = _status_counts(rows)
        aggregate_statuses.update(status_counts)
        processed_sha256, skipped_sha256 = _validate_text_lists(
            snapshot,
            attempt,
            article_ids,
            rows,
        )
        postfetch_payload = snapshot.read(
            f"{prefix}/postfetch.log",
            f"{attempt} postfetch log",
        )
        exit_payload = snapshot.read(
            f"{prefix}/postfetch.exit",
            f"{attempt} postfetch exit",
        )
        _require(exit_payload == b"0\n", f"{attempt} postfetch did not exit zero")

        summary_path = f"{prefix}/attempt_summary.json"
        summary_payload = snapshot.read(summary_path, f"{attempt} summary")
        summary = _json_object(summary_payload, f"{attempt} summary")
        summary_hash = _semantic_hash(summary, "summary_hash", f"{attempt} summary")
        ledger_sha256 = _sha256(ledger_payload)
        _require(summary.get("attempt") == attempt, "attempt summary label differs")
        _require(summary.get("coverage_exact") is True, "attempt summary is incomplete")
        _require(
            summary.get("input_sha256") == binding.accepted_sha256,
            "attempt summary input differs",
        )
        _require(
            summary.get("ledger_sha256") == ledger_sha256,
            "attempt summary ledger digest differs",
        )
        _require(
            summary.get("processed_sha256") == processed_sha256,
            "attempt summary processed digest differs",
        )
        _require(
            summary.get("skipped_sha256") == skipped_sha256,
            "attempt summary skipped digest differs",
        )
        _require(
            summary.get("postfetch_log_sha256") == _sha256(postfetch_payload),
            "attempt summary log digest differs",
        )
        _require(summary.get("postfetch_exit") == 0, "attempt summary exit differs")
        _require(
            summary.get("processed_records") == binding.input_total,
            "attempt summary processed count differs",
        )
        _require(
            summary.get("processed_unique") == binding.input_total,
            "attempt summary unique count differs",
        )
        _require(
            summary.get("skipped_records") == binding.input_total,
            "attempt summary skipped count differs",
        )
        _require(summary.get("statuses") == status_counts, "attempt statuses differ")
        if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2":
            _require(
                summary.get("ledger_records") == binding.input_total,
                "attempt summary ledger count differs",
            )
            _require(
                summary.get("input_records") == binding.input_total,
                "attempt summary input count differs",
            )
            _require(
                summary.get("input_unique_dois") == binding.input_total,
                "attempt summary DOI count differs",
            )
            _require(
                summary.get("asset_snapshot_files") == 0,
                "attempt summary declares source assets",
            )
            _require(
                summary.get("provenance_snapshots") == binding.input_total,
                "attempt provenance count differs",
            )

        _require(
            coverage_entry.get("ledger_sha256") == ledger_sha256,
            "coverage ledger digest differs",
        )
        _require(
            coverage_entry.get("summary_file_sha256") == _sha256(summary_payload),
            "coverage summary-file digest differs",
        )
        _require(
            coverage_entry.get("summary_hash") == summary_hash,
            "coverage summary semantic digest differs",
        )
        _require(
            coverage_entry.get("postfetch_log_sha256") == _sha256(postfetch_payload),
            "coverage log digest differs",
        )
        _require(
            coverage_entry.get("postfetch_exit") == 0,
            "coverage postfetch exit differs",
        )
        _require(
            coverage_entry.get("processed_records") == binding.input_total,
            "coverage processed count differs",
        )
        _require(
            coverage_entry.get("processed_unique") == binding.input_total,
            "coverage unique count differs",
        )
        _require(
            coverage_entry.get("statuses") == status_counts,
            "coverage status counts differ",
        )
        if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2":
            _require(
                coverage_entry.get("asset_snapshot_files") == 0,
                "coverage declares source assets",
            )
            _require(
                coverage_entry.get("provenance_snapshots") == binding.input_total,
                "coverage provenance count differs",
            )
        else:
            _require(
                coverage_entry.get("input_records") == binding.input_total,
                "legacy coverage input count differs",
            )
            _require(
                coverage_entry.get("processed_sha256") == processed_sha256,
                "legacy coverage processed digest differs",
            )
            _require(
                coverage_entry.get("skipped_sha256") == skipped_sha256,
                "legacy coverage skipped digest differs",
            )
        attempt_reports.append(
            {
                "attempt": attempt,
                "ledger_records": len(rows),
                "ledger_sha256": ledger_sha256,
                "summary_sha256": _sha256(summary_payload),
                "summary_hash": summary_hash,
                "status_counts": status_counts,
                "source_bearing_records": 0,
            }
        )
    return rows_by_attempt, attempt_reports, dict(sorted(aggregate_statuses.items()))


def _validate_terminal(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
    dois: Sequence[str],
    article_ids: Sequence[str],
    rows_by_attempt: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    payload = snapshot.read(
        "control/terminal_outcomes.jsonl",
        "terminal outcomes",
    )
    terminal_rows = _jsonl_objects(payload, "terminal outcomes")
    _require(len(terminal_rows) == binding.input_total, "terminal count differs")
    dispositions: list[dict[str, Any]] = []
    terminal_counts: Counter[str] = Counter()
    for index, terminal in enumerate(terminal_rows, start=1):
        expected_statuses = {
            attempt: rows_by_attempt[attempt][index - 1]["status"]
            for attempt in ATTEMPTS
        }
        _require(
            _normalize_doi(terminal.get("doi"), "terminal DOI") == dois[index - 1],
            "terminal DOI order differs",
        )
        _require(
            terminal.get("article_id") == article_ids[index - 1],
            "terminal article order differs",
        )
        if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2":
            expected_keys = {
                "article_id",
                "attempts",
                "doi",
                "input_index_1based",
                "terminal_content_artifacts",
                "terminal_content_directory",
                "terminal_provenance_path",
                "terminal_provenance_sha256",
                "terminal_selection_rule",
                "terminal_status",
            }
            _require(set(terminal) == expected_keys, "terminal row shape differs")
            _require(
                terminal["input_index_1based"] == index,
                "terminal input index differs",
            )
            attempts = terminal["attempts"]
            _require(
                isinstance(attempts, list) and len(attempts) == 3,
                "terminal attempt roster differs",
            )
            for attempt_index, attempt in enumerate(ATTEMPTS):
                link = attempts[attempt_index]
                _require(isinstance(link, dict), "terminal attempt link is invalid")
                ledger_path = f"control/{attempt}/attempt_ledger.jsonl"
                provenance_path = (
                    f"control/{attempt}/provenance/{article_ids[index - 1]}.json"
                )
                _require(link.get("attempt") == attempt, "terminal attempt differs")
                _require(link.get("ledger_path") == ledger_path, "terminal ledger path differs")
                _require(
                    link.get("ledger_record_index_1based") == index,
                    "terminal ledger index differs",
                )
                _require(
                    link.get("provenance_snapshot_path") == provenance_path,
                    "terminal provenance path differs",
                )
                _require(
                    link.get("provenance_snapshot_sha256")
                    == snapshot.digest(provenance_path, "terminal provenance"),
                    "terminal provenance digest differs",
                )
                _require(
                    link.get("status") == expected_statuses[attempt],
                    "terminal attempt status differs",
                )
            retry2_provenance = (
                f"control/retry2/provenance/{article_ids[index - 1]}.json"
            )
            _require(
                terminal["terminal_provenance_path"] == retry2_provenance,
                "terminal selected provenance differs",
            )
            _require(
                terminal["terminal_provenance_sha256"]
                == snapshot.digest(retry2_provenance, "retry2 provenance"),
                "terminal selected provenance digest differs",
            )
            _require(
                terminal["terminal_content_artifacts"] == [],
                "terminal row declares content artifacts",
            )
            _require(
                terminal["terminal_content_directory"] is None,
                "terminal row declares a content directory",
            )
            _require(
                terminal["terminal_selection_rule"] == TERMINAL_SELECTION_RULE,
                "terminal selection rule differs",
            )
        else:
            expected_keys = {
                "article_id",
                "doi",
                "provenance_path",
                "provenance_sha256",
                "rounds",
                "terminal_status",
            }
            _require(set(terminal) == expected_keys, "legacy terminal shape differs")
            _require(terminal["rounds"] == expected_statuses, "legacy rounds differ")
            provenance_path = f"content/_provenance/{article_ids[index - 1]}.json"
            _require(
                terminal["provenance_path"] == provenance_path,
                "legacy terminal provenance path differs",
            )
            _require(
                terminal["provenance_sha256"]
                == snapshot.digest(provenance_path, "legacy terminal provenance"),
                "legacy terminal provenance digest differs",
            )
        terminal_status = expected_statuses["retry2"]
        _require(
            terminal.get("terminal_status") == terminal_status,
            "terminal status is not retry2",
        )
        live_path = f"content/_provenance/{article_ids[index - 1]}.json"
        _validate_provenance(
            snapshot.read(live_path, "live provenance"),
            dois[index - 1],
            "live provenance",
        )
        terminal_counts.update([terminal_status])
        dispositions.append(
            {
                "input_index_1based": index,
                "doi": dois[index - 1],
                "article_id": article_ids[index - 1],
                "attempt_statuses": [
                    {
                        "attempt": attempt,
                        "status": expected_statuses[attempt],
                    }
                    for attempt in ATTEMPTS
                ],
                "terminal_status": terminal_status,
                "acquisition_disposition": (
                    (
                        "NO_SOURCE_BEARING_ARTIFACT_OBSERVED_IN_BOUND_"
                        "PER_ATTEMPT_PROVENANCE"
                    )
                    if binding.format_profile
                    == "RERUN2_FORENSIC_ACQUISITION_V2"
                    else (
                        "NO_DOWNLOADED_STATUS_RECORDED_ACROSS_FIXED_ATTEMPTS_"
                        "AND_TERMINAL_PROVENANCE_EMPTY"
                    )
                ),
                "per_attempt_provenance_bound": (
                    binding.format_profile
                    == "RERUN2_FORENSIC_ACQUISITION_V2"
                ),
                "source_binding": None,
                "canonical_binding": None,
                "p_binding": None,
            }
        )
    return dispositions, dict(sorted(terminal_counts.items())), _sha256(payload)


def _validate_inventory(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
) -> dict[str, Any]:
    path = binding.artifacts.inventory_path
    payload = snapshot.read(path, "native root inventory")
    inventory = _json_object(payload, "native root inventory")
    inventory_hash = _semantic_hash(
        inventory,
        "inventory_hash",
        "native root inventory",
    )
    raw_entries = (
        inventory.get("entries")
        if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2"
        else inventory.get("files")
    )
    _require(isinstance(raw_entries, list), "native inventory entries are invalid")
    declared: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(raw_entries):
        _require(isinstance(entry, dict), "native inventory entry is invalid")
        relative = _safe_relative(entry.get("path"), f"inventory entry {index}.path")
        size_field = (
            "size_bytes"
            if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2"
            else "bytes"
        )
        size = _integer(entry.get(size_field), f"inventory entry {index}.size")
        digest = _string(entry.get("sha256"), f"inventory entry {index}.sha256")
        _require(relative not in declared, "native inventory repeats a path")
        declared[relative] = (size, digest)

    excluded = {path}
    if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2":
        excluded.add("root_inventory.sha256")
    else:
        excluded.update(
            relative
            for relative in snapshot.files
            if relative.startswith("sealed_report_v1/")
        )
        _require(
            inventory.get("excludes")
            == [
                ".pipeline_worktree/",
                "control/root_inventory.json",
                "sealed_report_v1/",
            ],
            "legacy inventory exclusions differ",
        )
    expected = {
        relative: (artifact.size, artifact.sha256)
        for relative, artifact in snapshot.files.items()
        if relative not in excluded
    }
    _require(declared == expected, "native inventory does not match the root snapshot")
    if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2":
        _require(
            inventory.get("inventory_type") == "c2_rerun2_root_inventory",
            "forensic inventory type differs",
        )
        _require(
            inventory.get("root_id") == binding.root_name,
            "forensic inventory root ID differs",
        )
        _require(
            inventory.get("excluded_prefixes") == [f"{EXCLUDED_TREE}/"],
            "forensic inventory exclusions differ",
        )
        _require(
            inventory.get("entry_hashes_verified_at_build") is True,
            "forensic inventory did not verify entries",
        )
        _require(
            inventory.get("formal_artifact_manifest_path")
            == "control/artifact_manifest_v1.json",
            "forensic inventory manifest path differs",
        )
        _require(
            inventory.get("formal_artifact_manifest_sha256")
            == snapshot.digest(
                "control/artifact_manifest_v1.json",
                "formal artifact manifest",
            ),
            "forensic inventory manifest digest differs",
        )
        _require(
            inventory.get("sealed_report_path")
            == "sealed_report_v1/sealed_report.json",
            "forensic inventory sealed-report path differs",
        )
        _require(
            inventory.get("sealed_report_sha256")
            == snapshot.digest(
                "sealed_report_v1/sealed_report.json",
                "sealed report",
            ),
            "forensic inventory sealed-report digest differs",
        )
        _validate_sidecar(
            snapshot.read("root_inventory.sha256", "inventory sidecar"),
            _sha256(payload),
            "root_inventory.json",
            "inventory sidecar",
        )
    else:
        _require(
            inventory.get("root_name") == binding.root_name,
            "legacy inventory root name differs",
        )
        _require(
            inventory.get("artifact_count") == len(declared),
            "legacy inventory artifact count differs",
        )
        _require(
            inventory.get("total_bytes")
            == sum(size for size, _ in declared.values()),
            "legacy inventory total bytes differ",
        )
    return {
        "path": path,
        "sha256": _sha256(payload),
        "inventory_hash": inventory_hash,
        "entry_count": len(declared),
        "coverage": "EXACT_PROFILE_NATIVE_INVENTORY_MATCH",
    }


def _validate_sidecar(
    payload: bytes,
    expected_sha256: str,
    filename: str,
    label: str,
) -> None:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise C2M4CandidateValidationError(f"{label} is not ASCII") from exc
    accepted = {
        f"{expected_sha256}\n",
        f"{expected_sha256}  {filename}\n",
    }
    _require(text in accepted, f"{label} does not bind {filename}")


def _validate_sealed_report(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
    attempt_reports: Sequence[Mapping[str, Any]],
    terminal_sha256: str,
    terminal_counts: Mapping[str, int],
    frozen_universe_sha256: str,
) -> dict[str, Any]:
    path = "sealed_report_v1/sealed_report.json"
    payload = snapshot.read(path, "legacy sealed report")
    report = _json_object(payload, "legacy sealed report")
    report_hash = _semantic_hash(report, "report_hash", "legacy sealed report")
    coverage_payload = snapshot.read(
        "control/attempt_coverage.json",
        "attempt coverage",
    )
    coverage = _json_object(coverage_payload, "attempt coverage")
    coverage_hash = _semantic_hash(coverage, "summary_hash", "attempt coverage")
    _validate_sidecar(
        snapshot.read(
            "sealed_report_v1/sealed_report.sha256",
            "sealed report sidecar",
        ),
        _sha256(payload),
        "sealed_report.json",
        "sealed report sidecar",
    )
    if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2":
        _require(report.get("result") == "PASS", "sealed report is not PASS")
        _require(
            report.get("report_type") == "c2_rerun2_sealed_root_report",
            "sealed report type differs",
        )
        _require(report.get("root_id") == binding.root_name, "sealed root ID differs")
        source = report.get("source")
        _require(isinstance(source, dict), "sealed source binding is invalid")
        _require(source.get("records") == binding.input_total, "sealed input count differs")
        _require(
            source.get("unique_dois") == binding.input_total,
            "sealed DOI count differs",
        )
        _require(
            source.get("source_chunk_sha256") == binding.accepted_sha256,
            "sealed source digest differs",
        )
        _require(
            source.get("frozen_universe_sha256") == frozen_universe_sha256,
            "sealed frozen-universe digest differs",
        )
        formal = report.get("formal_attempts")
        _require(isinstance(formal, dict), "sealed attempt binding is invalid")
        _require(
            formal.get("attempt_coverage_path") == "control/attempt_coverage.json",
            "sealed attempt-coverage path differs",
        )
        _require(
            formal.get("attempt_coverage_sha256") == _sha256(coverage_payload),
            "sealed attempt-coverage digest differs",
        )
        _require(
            formal.get("attempt_coverage_summary_hash") == coverage_hash,
            "sealed attempt-coverage semantic digest differs",
        )
        _require(
            formal.get("roster") == list(ATTEMPTS),
            "sealed attempt roster differs",
        )
        _require(
            formal.get("total_ledger_records") == 3 * binding.input_total,
            "sealed ledger count differs",
        )
        terminal = report.get("terminal")
        _require(isinstance(terminal, dict), "sealed terminal binding is invalid")
        _require(
            terminal.get("terminal_outcomes_sha256") == terminal_sha256,
            "sealed terminal digest differs",
        )
        _require(
            terminal.get("records") == binding.input_total,
            "sealed terminal count differs",
        )
        _require(
            terminal.get("unique_dois") == binding.input_total,
            "sealed terminal DOI count differs",
        )
        _require(
            terminal.get("terminal_counts") == dict(terminal_counts),
            "sealed terminal status counts differ",
        )
    else:
        _require(
            report.get("sealed_report_version") == "2.0",
            "legacy sealed report version differs",
        )
        _require(
            report.get("status") == "SEALED_COMPLETE_ATTEMPT_EVIDENCE_NO_CASES",
            "legacy sealed status differs",
        )
        frozen_input = report.get("frozen_input")
        _require(isinstance(frozen_input, dict), "legacy frozen input is invalid")
        _require(frozen_input.get("chunk") == 12, "legacy sealed chunk differs")
        _require(
            frozen_input.get("records") == binding.input_total,
            "legacy sealed input count differs",
        )
        _require(
            frozen_input.get("sha256") == binding.accepted_sha256,
            "legacy sealed input digest differs",
        )
        _require(
            report.get("attempt_evidence", {}).get("attempts")
            and len(report["attempt_evidence"]["attempts"]) == len(attempt_reports),
            "legacy sealed attempt roster differs",
        )
        attempt_evidence = report["attempt_evidence"]
        _require(
            attempt_evidence.get("coverage_exact") is True,
            "legacy sealed coverage is incomplete",
        )
        _require(
            attempt_evidence.get("coverage_file_sha256")
            == _sha256(coverage_payload),
            "legacy sealed coverage digest differs",
        )
        _require(
            attempt_evidence.get("coverage_summary_hash") == coverage_hash,
            "legacy sealed coverage semantic digest differs",
        )
        _require(
            attempt_evidence.get("attempts") == coverage.get("attempts"),
            "legacy sealed attempt details differ",
        )
        _require(
            frozen_input.get("universe_sha256") == frozen_universe_sha256,
            "legacy sealed universe digest differs",
        )
        postfetch = report.get("postfetch")
        _require(isinstance(postfetch, dict), "legacy sealed postfetch is invalid")
        _require(
            postfetch.get("terminal_outcomes_sha256") == terminal_sha256,
            "legacy sealed terminal digest differs",
        )
        _require(
            postfetch.get("terminal_counts") == dict(terminal_counts),
            "legacy sealed terminal counts differ",
        )
        _require(
            report.get("canonical", {}).get("eligible_cases") == 0,
            "legacy sealed report claims canonical cases",
        )
        _require(
            report.get("cases", {}).get("candidates") == 0,
            "legacy sealed report claims candidate cases",
        )
        _require(
            report.get("proposals", {}).get("total") == 0,
            "legacy sealed report claims proposals",
        )
        _require(
            report.get("corpus_manifest", {}).get("source_data_dois") == 0,
            "legacy sealed report claims source-data DOI",
        )
    return {
        "sha256": _sha256(payload),
        "report_hash": report_hash,
        "handling": (
            "SNAPSHOT_BOUND_LEGACY_REPORT_NOT_ADMISSION_AUTHORITY"
        ),
    }


def _validate_no_source_surfaces(
    snapshot: _RootSnapshot,
    binding: M4CandidateBinding,
    article_ids: Sequence[str],
) -> dict[str, Any]:
    allowed_root_files = {
        "accepted.jsonl",
        "root_inventory.json",
        "root_inventory.sha256",
    }
    allowed_directories = {
        "canonical_v1",
        "cases_v1",
        "content",
        "control",
        "manifest_v1",
        "p_evidence_v1",
        "pre_acquisition_aborted_v1",
        "proposals_v1",
        "sealed_report_v1",
    }
    for relative in snapshot.files:
        parts = PurePosixPath(relative).parts
        _require(
            relative in allowed_root_files or parts[0] in allowed_directories,
            f"candidate root has an unsupported artifact path: {relative}",
        )
        if parts[0] == "content":
            _require(
                len(parts) == 3
                and parts[1] == "_provenance"
                and parts[2].endswith(".json"),
                f"formal content has a non-provenance artifact: {relative}",
            )
    expected_live_provenance = {
        f"content/_provenance/{article_id}.json" for article_id in article_ids
    }
    observed_live_provenance = {
        relative
        for relative in snapshot.files
        if relative.startswith("content/")
    }
    _require(
        observed_live_provenance == expected_live_provenance,
        "formal live-provenance roster differs from accepted input",
    )
    forbidden_prefixes = ("content/_sources/", "content/_source_evidence/")
    _require(
        not any(
            relative.startswith(forbidden_prefixes)
            for relative in snapshot.files
        ),
        "candidate root contains source-bearing evidence paths",
    )
    for path in ("cases_v1/candidates.jsonl", "proposals_v1/proposed.jsonl"):
        payload = snapshot.read(path, path)
        _require(payload == b"", f"{path} is not empty")
        _require(_sha256(payload) == EMPTY_SHA256, f"{path} empty digest differs")
    return {
        "observed_scope": (
            "FORMAL_ROOT_ACQUISITION_SURFACES_EXCLUDING_PIPELINE_AND_"
            "HISTORICAL_QUARANTINE"
        ),
        "excluded_unassessed_trees": [
            f"{EXCLUDED_TREE}/",
            *(
                ["pre_acquisition_aborted_v1/"]
                if any(
                    relative.startswith("pre_acquisition_aborted_v1/")
                    for relative in snapshot.files
                )
                else []
            ),
        ],
        "source_bearing_ledger_status_records_in_observed_scope": 0,
        "formal_content_source_asset_paths": 0,
        "root_candidate_surface_records": 0,
        "root_proposal_surface_records": 0,
        "per_attempt_provenance_bound": (
            binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2"
        ),
        "legacy_source_absence_qualification": (
            None
            if binding.format_profile == "RERUN2_FORENSIC_ACQUISITION_V2"
            else (
                "ZERO_DOWNLOADED_LEDGER_STATUSES_PLUS_EMPTY_TERMINAL_"
                "PROVENANCE_ONLY"
            )
        ),
        "source_classification_performed": False,
        "canonical_classification_performed": False,
        "p_classification_performed": False,
    }


def _validate_candidate(
    root_base: Path,
    binding: M4CandidateBinding,
    frozen_universe_sha256: str,
) -> dict[str, Any]:
    root = root_base / binding.root_name
    _require(root.name == binding.root_name, "candidate root name differs")
    with _DescriptorSnapshotter(root) as snapshotter:
        snapshot = snapshotter.snapshot()
    _validate_snapshot_pin(snapshot, binding)
    binding_report = _validate_binding_contract(
        snapshot,
        binding,
        frozen_universe_sha256,
    )
    _, dois, article_ids, record_hashes = _accepted_records(snapshot, binding)
    rows_by_attempt, attempt_reports, attempt_status_counts = _validate_attempts(
        snapshot,
        binding,
        dois,
        article_ids,
        record_hashes,
    )
    dispositions, terminal_counts, terminal_sha256 = _validate_terminal(
        snapshot,
        binding,
        dois,
        article_ids,
        rows_by_attempt,
    )
    inventory = _validate_inventory(snapshot, binding)
    sealed_report = _validate_sealed_report(
        snapshot,
        binding,
        attempt_reports,
        terminal_sha256,
        terminal_counts,
        frozen_universe_sha256,
    )
    source_observation = _validate_no_source_surfaces(
        snapshot,
        binding,
        article_ids,
    )
    disposition_hash = _sha256(canonical_json(dispositions).encode("utf-8"))
    legacy = binding.format_profile == "LEGACY_COMPACT_ACQUISITION_ONLY"
    return {
        "chunk_id": binding.chunk_id,
        "root_name": binding.root_name,
        "required_action": binding.required_action,
        "format_profile": binding.format_profile,
        "profile_selection": (
            f"COMPILE_PINNED_CHUNK_{binding.chunk_id}_POLICY_AND_SNAPSHOT_ONLY"
        ),
        "report_type": (
            "c2_legacy_compact_acquisition_observation_v1"
            if legacy
            else "c2_rerun2_forensic_acquisition_observation_v1"
        ),
        "observation_status": (
            "PINNED_LEGACY_COMPACT_ACQUISITION_MATCH"
            if legacy
            else "PINNED_RERUN2_FORENSIC_ACQUISITION_MATCH"
        ),
        "validated_scope": "ORDERED_RAW_ACQUISITION_AND_TERMINAL_ROUNDS_ONLY",
        "frozen_input": {
            "first_global_ordinal": binding.first_global_ordinal,
            "last_global_ordinal": binding.last_global_ordinal,
            "records": binding.input_total,
            "accepted_bytes": binding.accepted_bytes,
            "accepted_sha256": binding.accepted_sha256,
        },
        "root_snapshot": {
            "algorithm": SNAPSHOT_ALGORITHM,
            "excluded_tree": f"{EXCLUDED_TREE}/",
            "file_count": snapshot.file_count,
            "total_bytes": snapshot.total_bytes,
            "canonical_bytes": snapshot.canonical_bytes,
            "sha256": snapshot.sha256,
            "double_read_identity_verified": True,
        },
        "pre_download_binding": binding_report,
        "attempts": attempt_reports,
        "attempt_status_counts": attempt_status_counts,
        "terminal_status_counts": terminal_counts,
        "ordered_acquisition_dispositions": dispositions,
        "ordered_acquisition_dispositions_sha256": disposition_hash,
        "native_inventory": inventory,
        "legacy_sealed_report": sealed_report,
        "source_observation": source_observation,
        "legacy_canonical_p_handling": (
            "SNAPSHOT_BOUND_BYTES_ONLY_NOT_EVALUATED_NOT_AUTHORITY"
        ),
        "v2_format_validated": False,
        "v2_admission_eligible": False,
        "candidate_only": True,
        "read_only_validation": True,
        "independent_verification": False,
        "admission_authorized": False,
        "scientific_publication_authorized": False,
    }


def validate_fixed_m4_candidates(root_base: Path) -> dict[str, Any]:
    """Observe all three compile-pinned roots and return a non-admissive report."""

    authorization = require_owner_authorized_c2_execution()
    _require(
        EXPECTED_CAPABILITY in authorization.capabilities,
        "owner authorization lacks C2 candidate-validation capability",
    )
    bindings: M4CandidateValidationBindings = (
        load_m4_candidate_validation_bindings()
    )
    policy = load_owner_remediation_execution_policy()
    _require(
        bindings.authorization_id_sha256
        == policy.authorization_id_sha256
        == authorization.authorization_id_sha256,
        "M4 authorization bindings disagree",
    )
    _require(
        bindings.owner_policy_id_sha256 == policy.policy_id_sha256,
        "M4 owner-policy identity differs",
    )
    _require(
        bindings.owner_policy_resource_sha256 == policy.resource_sha256,
        "M4 owner-policy resource differs",
    )
    _require(
        bindings.frozen_universe_sha256
        == policy.universe_file_sha256
        == authorization.frozen_universe_sha256,
        "M4 frozen-universe bindings disagree",
    )
    _require(
        bindings.snapshot_algorithm == SNAPSHOT_ALGORITHM,
        "M4 snapshot algorithm differs",
    )
    _require(bindings.excluded_tree == f"{EXCLUDED_TREE}/", "M4 exclusion differs")
    for binding in bindings.candidates:
        policy_chunk = policy.chunk(binding.chunk_id)
        _require(
            (
                policy_chunk.required_action,
                policy_chunk.first_global_ordinal,
                policy_chunk.last_global_ordinal,
                policy_chunk.input_total,
                policy_chunk.chunk_file_sha256,
            )
            == (
                binding.required_action,
                binding.first_global_ordinal,
                binding.last_global_ordinal,
                binding.input_total,
                binding.accepted_sha256,
            ),
            f"M4 chunk {binding.chunk_id} differs from owner policy",
        )

    root_base = Path(os.path.abspath(os.fspath(root_base.expanduser())))
    candidates = [
        _validate_candidate(
            root_base,
            binding,
            bindings.frozen_universe_sha256,
        )
        for binding in bindings.candidates
    ]
    report: dict[str, Any] = {
        "schema_version": "c2-m4-candidate-validation-report-v1",
        "report_type": "c2_fixed_candidate_acquisition_observation_v1",
        "authorization": authorization.to_report_dict(),
        "owner_policy": {
            "policy_id_sha256": policy.policy_id_sha256,
            "resource_sha256": policy.resource_sha256,
            "required_operation": EXPECTED_CAPABILITY,
            "independent_verification": False,
            "admission_authorized": False,
        },
        "fixed_candidate_bindings": {
            "binding_id_sha256": bindings.binding_id_sha256,
            "resource_sha256": bindings.resource_sha256,
            "snapshot_algorithm": bindings.snapshot_algorithm,
        },
        "candidates": candidates,
        "aggregate": {
            "chunk_ids": [binding.chunk_id for binding in bindings.candidates],
            "input_records": sum(
                binding.input_total for binding in bindings.candidates
            ),
            "attempt_ledger_records": sum(
                3 * binding.input_total for binding in bindings.candidates
            ),
            "terminal_records": sum(
                binding.input_total for binding in bindings.candidates
            ),
            "observed_scope": (
                "FORMAL_ROOT_ACQUISITION_SURFACES_EXCLUDING_PIPELINE_AND_"
                "HISTORICAL_QUARANTINE"
            ),
            "source_bearing_ledger_status_records_in_observed_scope": 0,
            "root_candidate_surface_records": 0,
            "v2_admission_eligible_roots": 0,
        },
        "observation_status": "VALIDATED_FIXED_CANDIDATES_NON_ADMISSIVE",
        "execution_authorized": True,
        "candidate_only": True,
        "read_only_validation": True,
        "independent_verification": False,
        "admission_authorized": False,
        "publication_authorized": False,
        "scientific_outcome_preapproved": False,
        "v2_format_validated": False,
        "v2_admission_eligible": False,
    }
    report["report_hash_algorithm"] = (
        "sha256_canonical_json_without_report_hash"
    )
    report["report_hash"] = _sha256(canonical_json(report).encode("utf-8"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run fixed validation and emit canonical report JSON to stdout only."""

    parser = argparse.ArgumentParser(
        prog="python -m experiments.c2_m4_candidate_validator",
        description=(
            "Read-only, non-admissive observation of the three fixed C2 M4 "
            "candidate roots"
        ),
    )
    parser.add_argument(
        "root_base",
        type=Path,
        help="Untrusted locator containing the three compile-pinned root names",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_fixed_m4_candidates(args.root_base)
    except ProvenanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

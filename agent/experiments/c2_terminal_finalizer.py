"""Fail-closed admission of an explicitly sealed C2 chunk-report universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .models import ProvenanceError, sha256_json, write_json_atomic


C2_FINALIZER_VERSION = "1.0"
CHUNK_IDS = tuple(f"{index:03d}" for index in range(1, 14))
REPLACEMENT_CHUNK_IDS = ("009", "010", "011", "012")
STRATA = ("P=1", "P=2", "P=3-4", "P=5+")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DOI_RE = re.compile(r"^10\.[0-9]{4,9}/\S+$")
_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
TERMINAL_OUTCOME_STATUSES = frozenset(
    {
        "DOWNLOADED",
        "NO_SOURCE_DATA",
        "NO_FIGURES",
        "NO_USABLE_CONTENT",
        "POLICY_REJECTED",
        "DOWNLOAD_FAILED",
        "RETRY_EXHAUSTED",
    }
)
_BLOCKED_STATUS_BY_STRATUM = {
    "P=1": "BLOCKED_INSUFFICIENT_INDEPENDENT_P1",
    "P=2": "BLOCKED_INSUFFICIENT_INDEPENDENT_P2",
    "P=3-4": "BLOCKED_INSUFFICIENT_INDEPENDENT_P3_4",
    "P=5+": "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS",
}


class C2AdmissionError(ProvenanceError):
    """Raised when terminal C2 evidence cannot be admitted safely."""


@dataclass(frozen=True)
class FinalizedAdmission:
    """A validated report paired with every file admitted as input evidence."""

    report: dict[str, Any]
    admitted_input_paths: tuple[Path, ...]


def _reject_json_constant(value: str) -> None:
    raise C2AdmissionError(f"Non-finite JSON value is forbidden: {value}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise C2AdmissionError(f"{label} must not be a symlink: {path}")
    try:
        payload_bytes = path.read_bytes()
    except OSError as exc:
        raise C2AdmissionError(f"Cannot read {label}: {path}") from exc
    payload_sha256 = _sha256_bytes(payload_bytes)
    try:
        value = json.loads(
            payload_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise C2AdmissionError(f"{label} is not UTF-8 JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise C2AdmissionError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise C2AdmissionError(f"{label} must be a JSON object: {path}")
    return value, payload_sha256


@lru_cache(maxsize=None)
def _schema_validator(schema_name: str) -> Draft202012Validator:
    schema_path = Path(__file__).resolve().parent / "schemas" / schema_name
    schema, _ = _read_json_object(schema_path, f"schema {schema_name}")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise C2AdmissionError(f"Invalid bundled schema {schema_name}: {exc}") from exc
    return Draft202012Validator(schema)


def _validation_location(error_path: Sequence[Any]) -> str:
    return ".".join(str(part) for part in error_path) or "<root>"


def _validate_schema(
    value: Mapping[str, Any],
    schema_name: str,
    label: str,
) -> None:
    errors = sorted(
        _schema_validator(schema_name).iter_errors(value),
        key=lambda error: _validation_location(tuple(error.absolute_path)),
    )
    if errors:
        error = errors[0]
        raise C2AdmissionError(
            f"{label} schema validation failed at "
            f"{_validation_location(tuple(error.absolute_path))}: {error.message}"
        )


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    ignored = set(keys)
    return {key: item for key, item in value.items() if key not in ignored}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise C2AdmissionError(f"{label} must be a full SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise C2AdmissionError(f"{label} must be a full 40-character commit")
    return value


def _require_doi(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.casefold()
        or _DOI_RE.fullmatch(value) is None
    ):
        raise C2AdmissionError(f"{label} must be a normalized DOI identifier")
    return value


def _require_doi_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise C2AdmissionError(f"{label} must be a DOI list")
    values = tuple(_require_doi(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(values) != len(set(values)):
        raise C2AdmissionError(f"{label} contains duplicate DOI identifiers")
    return values


def _validate_manifest_hash(manifest: Mapping[str, Any]) -> None:
    declared = _require_sha256(manifest.get("manifest_hash"), "manifest_hash")
    if sha256_json(_without(manifest, "manifest_hash")) != declared:
        raise C2AdmissionError("Admission manifest failed its semantic hash")


def _validate_report_seal(report: Mapping[str, Any]) -> None:
    declared = _require_sha256(report.get("report_hash"), "report_hash")
    computed = sha256_json(_without(report, "report_hash", "seal"))
    if computed != declared:
        raise C2AdmissionError("Chunk report failed its semantic report_hash")

    seal = report.get("seal")
    if not isinstance(seal, Mapping):
        raise C2AdmissionError("Chunk report is missing its terminal seal")
    if seal.get("status") != "TERMINAL":
        raise C2AdmissionError("Chunk report seal is nonterminal")
    if seal.get("sealed_report_hash") != declared:
        raise C2AdmissionError("Chunk report seal does not bind report_hash")
    seal_hash = _require_sha256(seal.get("seal_hash"), "seal.seal_hash")
    if sha256_json(_without(seal, "seal_hash")) != seal_hash:
        raise C2AdmissionError("Chunk report terminal seal hash mismatch")


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    _validate_manifest_hash(manifest)

    code = manifest["code"]
    _require_commit(code["commit"], "manifest.code.commit")
    if code["dirty"] is not False:
        raise C2AdmissionError("Admission manifest requires code.dirty=false")

    frozen = manifest["frozen_universe"]
    _require_sha256(frozen["sha256"], "manifest.frozen_universe.sha256")
    _require_sha256(
        frozen["doi_ids_sha256"],
        "manifest.frozen_universe.doi_ids_sha256",
    )

    replacement_policy = manifest["replacement_policy"]
    replacement_ids = replacement_policy["replacement_chunk_ids"]
    if replacement_ids != list(REPLACEMENT_CHUNK_IDS):
        raise C2AdmissionError(
            "replacement_policy must explicitly list chunks 009, 010, 011, and 012"
        )
    excluded = replacement_policy["excluded_superseded_roots"]
    if [item["chunk_id"] for item in excluded] != list(REPLACEMENT_CHUNK_IDS):
        raise C2AdmissionError(
            "replacement_policy must exclude one superseded root for each replacement chunk"
        )
    excluded_by_chunk = {
        item["chunk_id"]: item["root_id"]
        for item in excluded
    }
    excluded_root_ids = set(excluded_by_chunk.values())
    if len(excluded_root_ids) != len(REPLACEMENT_CHUNK_IDS):
        raise C2AdmissionError("Superseded root identifiers must be unique")

    chunks = manifest["chunks"]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if chunk_ids != list(CHUNK_IDS):
        raise C2AdmissionError("Admission manifest must list the complete ordered roster 001..013")
    if len(set(chunk_ids)) != len(CHUNK_IDS):
        raise C2AdmissionError("Admission manifest has duplicate chunk identifiers")

    report_paths: set[str] = set()
    root_ids: set[str] = set()
    totals = 0
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        root = chunk["root"]
        root_id = root["root_id"]
        if _ROOT_ID_RE.fullmatch(root_id) is None:
            raise C2AdmissionError(f"chunk {chunk_id} root_id is invalid")
        if root_id in root_ids:
            raise C2AdmissionError("Admission manifest reuses a root identifier")
        root_ids.add(root_id)
        if root_id in excluded_root_ids:
            raise C2AdmissionError(
                f"chunk {chunk_id} admits an explicitly superseded root"
            )
        if root["partial_root"] is not False:
            raise C2AdmissionError(f"chunk {chunk_id} uses a partial root")

        expected_kind = (
            "replacement" if chunk_id in REPLACEMENT_CHUNK_IDS else "canonical"
        )
        if root["root_kind"] != expected_kind:
            raise C2AdmissionError(
                f"chunk {chunk_id} must use a {expected_kind} root"
            )
        expected_superseded = excluded_by_chunk.get(chunk_id)
        if root["supersedes_root_id"] != expected_superseded:
            raise C2AdmissionError(
                f"chunk {chunk_id} has an invalid superseded-root binding"
            )

        report_path = chunk["report_path"]
        if report_path in report_paths:
            raise C2AdmissionError("Admission manifest has duplicate report paths")
        report_paths.add(report_path)
        _require_sha256(
            chunk["expected_report_file_sha256"],
            f"chunk {chunk_id} expected_report_file_sha256",
        )
        _require_sha256(
            chunk["expected_report_hash"],
            f"chunk {chunk_id} expected_report_hash",
        )
        _require_sha256(
            chunk["input_doi_ids_sha256"],
            f"chunk {chunk_id} input_doi_ids_sha256",
        )
        totals += chunk["input_total"]

    if totals != frozen["input_total"]:
        raise C2AdmissionError(
            "Per-chunk input totals do not exactly equal frozen universe input_total"
        )
    return list(chunks)


def _resolve_report_path(manifest_path: Path, raw_path: str, chunk_id: str) -> Path:
    if raw_path != raw_path.strip():
        raise C2AdmissionError(f"chunk {chunk_id} report_path has surrounding whitespace")
    relative = Path(raw_path)
    if (
        not raw_path.endswith(".json")
        or relative.is_absolute()
        or not relative.parts
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise C2AdmissionError(f"chunk {chunk_id} report_path is not a safe relative JSON path")

    base = manifest_path.parent.resolve(strict=True)
    candidate = base.joinpath(relative)
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise C2AdmissionError(f"chunk {chunk_id} report_path must not traverse a symlink")
    if not candidate.is_file():
        raise C2AdmissionError(f"chunk {chunk_id} sealed report path is missing")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as exc:
        raise C2AdmissionError(f"chunk {chunk_id} report_path escapes manifest directory") from exc
    if resolved == manifest_path:
        raise C2AdmissionError(f"chunk {chunk_id} report_path cannot be the admission manifest")
    return resolved


def _validate_terminal_outcomes(
    execution: Mapping[str, Any],
    input_dois: tuple[str, ...],
    chunk_id: str,
) -> None:
    if execution["attempts_per_input"] != 3:
        raise C2AdmissionError(
            f"chunk {chunk_id} must record exactly three attempts per input"
        )
    if execution["all_inputs_terminal"] is not True:
        raise C2AdmissionError(f"chunk {chunk_id} has nonterminal inputs")
    outcomes = execution["outcomes"]
    outcome_dois = tuple(
        _require_doi(outcome["doi_id"], f"chunk {chunk_id} outcome DOI")
        for outcome in outcomes
    )
    if outcome_dois != input_dois:
        raise C2AdmissionError(
            f"chunk {chunk_id} terminal outcomes do not exactly cover chunk inputs"
        )
    for outcome in outcomes:
        if outcome["attempt_count"] != 3:
            raise C2AdmissionError(
                f"chunk {chunk_id} has an input without exactly three attempts"
            )
        if outcome["terminal"] is not True:
            raise C2AdmissionError(f"chunk {chunk_id} has a nonterminal outcome")
        if outcome["terminal_status"] not in TERMINAL_OUTCOME_STATUSES:
            raise C2AdmissionError(
                f"chunk {chunk_id} records an unapproved terminal status"
            )


def _validate_evidence(
    evidence: Mapping[str, Any],
    input_dois: tuple[str, ...],
    chunk_id: str,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    selection = evidence["selection"]
    if (
        selection["scope"] != "ALL_TERMINAL_INPUTS"
        or selection["partial"] is not False
        or selection["selective"] is not False
        or selection["model_result_selected"] is not False
    ):
        raise C2AdmissionError(
            f"chunk {chunk_id} uses partial, selective, or model-result-selected evidence"
        )

    source_dois = _require_doi_list(
        evidence["source_doi_ids"],
        f"chunk {chunk_id} source_doi_ids",
    )
    if source_dois != input_dois:
        raise C2AdmissionError(
            f"chunk {chunk_id} source evidence is not the complete terminal input set"
        )
    source_hash = _require_sha256(
        evidence["source_doi_ids_sha256"],
        f"chunk {chunk_id} source_doi_ids_sha256",
    )
    if sha256_json(list(source_dois)) != source_hash:
        raise C2AdmissionError(f"chunk {chunk_id} source DOI hash mismatch")

    assignments = evidence["source_doi_strata"]
    assigned_dois = tuple(
        _require_doi(
            assignment["doi_id"],
            f"chunk {chunk_id} stratum assignment DOI",
        )
        for assignment in assignments
    )
    if assigned_dois != source_dois:
        raise C2AdmissionError(
            f"chunk {chunk_id} strata do not exactly cover source DOI evidence"
        )
    normalized_assignments: list[dict[str, str]] = []
    for assignment in assignments:
        stratum = assignment["stratum"]
        cluster_id = assignment["independent_doi_cluster_id"]
        if stratum not in STRATA:
            raise C2AdmissionError(f"chunk {chunk_id} has an unknown P stratum")
        if not isinstance(cluster_id, str) or _ROOT_ID_RE.fullmatch(cluster_id) is None:
            raise C2AdmissionError(
                f"chunk {chunk_id} has an invalid independent DOI cluster identifier"
            )
        normalized_assignments.append(
            {
                "doi_id": assignment["doi_id"],
                "stratum": stratum,
                "independent_doi_cluster_id": cluster_id,
            }
        )
    return source_dois, normalized_assignments


def _validate_chunk_report(
    report: Mapping[str, Any],
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    chunk_id = entry["chunk_id"]
    _validate_schema(
        report,
        "c2_terminal_chunk_report.schema.json",
        f"chunk {chunk_id} sealed report",
    )
    _validate_report_seal(report)

    if report["report_hash"] != entry["expected_report_hash"]:
        raise C2AdmissionError(f"chunk {chunk_id} report_hash differs from manifest")
    if report["code"] != manifest["code"]:
        raise C2AdmissionError(
            f"chunk {chunk_id} code provenance mixes or differs from manifest"
        )
    _require_commit(report["code"]["commit"], f"chunk {chunk_id} code.commit")
    if report["code"]["dirty"] is not False:
        raise C2AdmissionError(f"chunk {chunk_id} has code_dirty=true")
    if report["frozen_universe"] != manifest["frozen_universe"]:
        raise C2AdmissionError(
            f"chunk {chunk_id} does not bind the exact frozen universe"
        )
    if report["root"] != entry["root"]:
        raise C2AdmissionError(f"chunk {chunk_id} root binding differs from manifest")

    chunk = report["chunk"]
    if chunk["chunk_id"] != chunk_id:
        raise C2AdmissionError(f"chunk report identity mismatch for {chunk_id}")
    if chunk["input_total"] != entry["input_total"]:
        raise C2AdmissionError(f"chunk {chunk_id} input_total differs from manifest")
    if chunk["input_doi_ids_sha256"] != entry["input_doi_ids_sha256"]:
        raise C2AdmissionError(f"chunk {chunk_id} input DOI hash differs from manifest")
    input_dois = _require_doi_list(
        chunk["input_doi_ids"],
        f"chunk {chunk_id} input_doi_ids",
    )
    if len(input_dois) != entry["input_total"]:
        raise C2AdmissionError(f"chunk {chunk_id} input_total does not match DOI inputs")
    if sha256_json(list(input_dois)) != chunk["input_doi_ids_sha256"]:
        raise C2AdmissionError(f"chunk {chunk_id} input DOI hash mismatch")

    _validate_terminal_outcomes(report["execution"], input_dois, chunk_id)
    return _validate_evidence(report["evidence"], input_dois, chunk_id)


def _blocked_status(deficient_strata: list[str]) -> str:
    if "P=5+" in deficient_strata:
        return _BLOCKED_STATUS_BY_STRATUM["P=5+"]
    return _BLOCKED_STATUS_BY_STRATUM[deficient_strata[0]]


def _validate_final_report(report: Mapping[str, Any]) -> None:
    _validate_schema(
        report,
        "c2_terminal_final_report.schema.json",
        "final universe report",
    )
    declared_hash = _require_sha256(report["final_report_hash"], "final_report_hash")
    if sha256_json(_without(report, "final_report_hash")) != declared_hash:
        raise C2AdmissionError("Final universe report failed its semantic hash")

    if [chunk["chunk_id"] for chunk in report["chunks"]] != list(CHUNK_IDS):
        raise C2AdmissionError("Final universe report does not contain roster 001..013")
    if [item["stratum"] for item in report["strata"]] != list(STRATA):
        raise C2AdmissionError("Final universe report strata are incomplete or unordered")
    if report["source_doi_count"] != report["frozen_universe"]["input_total"]:
        raise C2AdmissionError("Final source DOI count differs from frozen universe input_total")
    if (
        report["source_doi_ids_sha256"]
        != report["frozen_universe"]["doi_ids_sha256"]
    ):
        raise C2AdmissionError("Final source DOI hash differs from frozen universe")

    roots = {chunk["root_id"] for chunk in report["chunks"]}
    if len(roots) != len(CHUNK_IDS):
        raise C2AdmissionError("Final universe report reuses a root identifier")
    for chunk in report["chunks"]:
        expected_kind = (
            "replacement"
            if chunk["chunk_id"] in REPLACEMENT_CHUNK_IDS
            else "canonical"
        )
        if chunk["root_kind"] != expected_kind:
            raise C2AdmissionError(
                f"Final universe report has an invalid {chunk['chunk_id']} root kind"
            )

    deficient = [
        item["stratum"]
        for item in report["strata"]
        if item["independent_doi_cluster_count"] < 2
    ]
    if report["deficient_strata"] != deficient:
        raise C2AdmissionError("Final universe report deficiency list is inconsistent")
    if deficient:
        if report["status"] != _blocked_status(deficient):
            raise C2AdmissionError("Final universe report blocked status is inconsistent")
        if report["claim_status"] != "UNSUPPORTED":
            raise C2AdmissionError("Blocked final universe report must be unsupported")
    elif report["status"] != "ADMITTED" or report["claim_status"] != "SUPPORTED":
        raise C2AdmissionError("Complete final universe report must be admitted only")
    if (
        report["trend_status"] != "NOT_RUN"
        or report["equivalence_status"] != "NOT_RUN"
        or report["claim_scope"] != "TERMINAL_ADMISSION_ONLY"
    ):
        raise C2AdmissionError(
            "Terminal finalizer must not create trend or equivalence analyses"
        )


def validate_final_report(report: Mapping[str, Any]) -> None:
    """Validate a finalizer output without reading any live output root."""

    _validate_final_report(report)


def prepare_finalization(manifest_path: Path) -> FinalizedAdmission:
    """Validate sealed inputs and retain their resolved paths for safe output."""

    if manifest_path.is_symlink():
        raise C2AdmissionError("Admission manifest must not be a symlink")
    try:
        resolved_manifest = manifest_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise C2AdmissionError(f"Admission manifest path is missing: {manifest_path}") from exc
    if not resolved_manifest.is_file():
        raise C2AdmissionError("Admission manifest path is not a file")
    manifest, manifest_file_sha256 = _read_json_object(
        resolved_manifest,
        "C2 admission manifest",
    )
    _validate_schema(
        manifest,
        "c2_terminal_admission_manifest.schema.json",
        "C2 admission manifest",
    )
    entries = _validate_manifest_structure(manifest)

    all_input_dois: list[str] = []
    all_source_dois: list[str] = []
    assignments: list[dict[str, str]] = []
    final_chunks: list[dict[str, Any]] = []
    seen_report_files: set[Path] = set()
    admitted_input_paths = [resolved_manifest]
    for entry in entries:
        chunk_id = entry["chunk_id"]
        report_path = _resolve_report_path(
            resolved_manifest,
            entry["report_path"],
            chunk_id,
        )
        if report_path in seen_report_files:
            raise C2AdmissionError("Admission manifest resolves multiple chunks to one report")
        seen_report_files.add(report_path)
        admitted_input_paths.append(report_path)
        report, actual_file_sha256 = _read_json_object(
            report_path,
            f"chunk {chunk_id} sealed report",
        )
        if actual_file_sha256 != entry["expected_report_file_sha256"]:
            raise C2AdmissionError(
                f"chunk {chunk_id} sealed report file hash is stale or mismatched"
            )
        source_dois, chunk_assignments = _validate_chunk_report(
            report,
            entry,
            manifest,
        )
        input_dois = tuple(report["chunk"]["input_doi_ids"])
        all_input_dois.extend(input_dois)
        all_source_dois.extend(source_dois)
        assignments.extend(chunk_assignments)
        final_chunks.append(
            {
                "chunk_id": chunk_id,
                "root_id": entry["root"]["root_id"],
                "root_kind": entry["root"]["root_kind"],
                "supersedes_root_id": entry["root"]["supersedes_root_id"],
                "report_path": entry["report_path"],
                "report_file_sha256": actual_file_sha256,
                "report_hash": report["report_hash"],
                "input_total": entry["input_total"],
                "input_doi_ids_sha256": entry["input_doi_ids_sha256"],
            }
        )

    frozen = manifest["frozen_universe"]
    if len(all_input_dois) != frozen["input_total"]:
        raise C2AdmissionError("Chunk reports do not provide the exact frozen input total")
    if len(all_input_dois) != len(set(all_input_dois)):
        raise C2AdmissionError("Chunk reports contain duplicate DOI inputs")
    if sha256_json(all_input_dois) != frozen["doi_ids_sha256"]:
        raise C2AdmissionError("Chunk reports do not match the frozen universe DOI hash")
    if all_source_dois != all_input_dois:
        raise C2AdmissionError("Source DOI evidence differs from complete chunk inputs")

    clusters_by_stratum: dict[str, set[str]] = defaultdict(set)
    doi_counts_by_stratum: dict[str, int] = defaultdict(int)
    cluster_strata: dict[str, str] = {}
    for assignment in assignments:
        stratum = assignment["stratum"]
        cluster_id = assignment["independent_doi_cluster_id"]
        previous_stratum = cluster_strata.setdefault(cluster_id, stratum)
        if previous_stratum != stratum:
            raise C2AdmissionError(
                "An independent DOI cluster cannot span multiple P strata"
            )
        clusters_by_stratum[stratum].add(cluster_id)
        doi_counts_by_stratum[stratum] += 1

    strata = [
        {
            "stratum": stratum,
            "source_doi_count": doi_counts_by_stratum[stratum],
            "independent_doi_cluster_count": len(clusters_by_stratum[stratum]),
        }
        for stratum in STRATA
    ]
    deficient_strata = [
        item["stratum"]
        for item in strata
        if item["independent_doi_cluster_count"] < 2
    ]
    status = _blocked_status(deficient_strata) if deficient_strata else "ADMITTED"
    report: dict[str, Any] = {
        "schema_version": C2_FINALIZER_VERSION,
        "finalizer": "c2_terminal_admission",
        "status": status,
        "claim_status": "UNSUPPORTED" if deficient_strata else "SUPPORTED",
        "claim_scope": "TERMINAL_ADMISSION_ONLY",
        "trend_status": "NOT_RUN",
        "equivalence_status": "NOT_RUN",
        "admission_manifest": {
            "file_sha256": manifest_file_sha256,
            "manifest_hash": manifest["manifest_hash"],
        },
        "frozen_universe": dict(frozen),
        "code": dict(manifest["code"]),
        "chunks": final_chunks,
        "source_doi_ids_sha256": sha256_json(all_source_dois),
        "source_doi_count": len(all_source_dois),
        "strata": strata,
        "deficient_strata": deficient_strata,
    }
    report["final_report_hash"] = sha256_json(report)
    _validate_final_report(report)
    return FinalizedAdmission(
        report=report,
        admitted_input_paths=tuple(admitted_input_paths),
    )


def finalize_manifest(manifest_path: Path) -> dict[str, Any]:
    """Build a terminal-only report without writing an output file."""

    return prepare_finalization(manifest_path).report


def _resolve_output_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise C2AdmissionError(f"Cannot resolve final report output path: {path}") from exc


def _reject_output_input_collision(
    output_path: Path,
    admitted_input_paths: Sequence[Path],
) -> None:
    resolved_output = _resolve_output_path(output_path)
    for input_path in admitted_input_paths:
        try:
            resolved_input = input_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise C2AdmissionError(
                f"Admitted input path became unavailable: {input_path}"
            ) from exc
        try:
            aliases_input = resolved_output == resolved_input or (
                resolved_output.exists()
                and resolved_output.samefile(resolved_input)
            )
        except OSError as exc:
            raise C2AdmissionError(
                f"Cannot compare final report output path: {output_path}"
            ) from exc
        if aliases_input:
            raise C2AdmissionError(
                "Final report output path aliases an admitted input path"
            )


def write_final_report(
    finalized: FinalizedAdmission,
    output_path: Path,
) -> Path:
    """Write a validated report only when its output cannot overwrite evidence."""

    if not isinstance(finalized, FinalizedAdmission):
        raise C2AdmissionError(
            "write_final_report requires a FinalizedAdmission from prepare_finalization"
        )
    _validate_final_report(finalized.report)
    _reject_output_input_collision(output_path, finalized.admitted_input_paths)
    write_json_atomic(output_path, finalized.report)
    return output_path


def finalize_to_path(
    manifest_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Finalize a manifest and safely write its report."""

    finalized = prepare_finalization(manifest_path)
    return finalized.report, write_final_report(finalized, output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize a sealed C2 terminal-admission manifest"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report, output_path = finalize_to_path(args.manifest, args.out)
    except C2AdmissionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_universe_report": str(output_path),
                "final_report_hash": report["final_report_hash"],
                "status": report["status"],
                "claim_status": report["claim_status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ADMITTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())

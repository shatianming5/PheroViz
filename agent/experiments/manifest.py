from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import unquote, urlsplit

import yaml

from .models import ProvenanceError, sha256_file


class ManifestError(ProvenanceError):
    """Raised when a dataset manifest violates the experiment protocol."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BENCHMARK_SPLITS = {"train", "val", "test"}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_doi(value: Any) -> str | None:
    text = unquote(str(value or "")).strip()
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    parsed = urlsplit(text)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.casefold() not in {"http", "https"} or (
            parsed.hostname or ""
        ).casefold() not in {"doi.org", "dx.doi.org"}:
            return None
        text = parsed.path.lstrip("/")
    else:
        text = re.split(r"[?#]", text, maxsplit=1)[0]
    doi = text.strip().casefold()
    if (
        not doi.startswith("10.")
        or "/" not in doi
        or any(character.isspace() for character in doi)
    ):
        return None
    return doi


def _verify_binding_file(binding: Mapping[str, Any], label: str) -> None:
    path_value = binding.get("path")
    expected_hash = binding.get("sha256")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(expected_hash, str)
        or not _SHA256_RE.fullmatch(expected_hash)
    ):
        raise ManifestError(f"Benchmark {label} binding is invalid")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
        raise ManifestError(f"Benchmark {label} binding changed")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Benchmark {label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"Benchmark {label} must be an object")
    return value


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManifestError(f"Benchmark {label} cannot be read") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"Benchmark {label}:{line_number} is not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ManifestError(
                f"Benchmark {label}:{line_number} must be an object"
            )
        records.append(value)
    return records


def _verify_object_seal(
    value: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    declared = value.get(field)
    unhashed = dict(value)
    unhashed.pop(field, None)
    if (
        not isinstance(declared, str)
        or not _SHA256_RE.fullmatch(declared)
        or _sha256_json(unhashed) != declared
    ):
        raise ManifestError(f"Benchmark {label} seal is invalid")
    return declared


def _validate_source_binding(
    source_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if set(source_binding) != {
        "candidate_inputs",
        "evidence",
        "proposed",
        "reviews",
    }:
        raise ManifestError("Benchmark source binding has unexpected fields")
    candidate_inputs = source_binding.get("candidate_inputs")
    if not isinstance(candidate_inputs, list) or not candidate_inputs:
        raise ManifestError("Benchmark candidate input bindings are empty")
    for index, binding in enumerate(candidate_inputs):
        if not isinstance(binding, Mapping):
            raise ManifestError("Benchmark candidate input binding is invalid")
        _verify_binding_file(
            {
                "path": binding.get("candidates_path"),
                "sha256": binding.get("candidates_sha256"),
            },
            f"candidate_inputs[{index}]",
        )
        _verify_binding_file(
            {
                "path": binding.get("summary_path"),
                "sha256": binding.get("summary_sha256"),
            },
            f"candidate_inputs[{index}].summary",
        )
        if (
            not _SHA256_RE.fullmatch(str(binding.get("summary_hash") or ""))
            or not _SHA256_RE.fullmatch(
                str(binding.get("corpus_manifest_sha256") or "")
            )
            or not isinstance(binding.get("content_root"), str)
            or not isinstance(binding.get("output_root"), str)
        ):
            raise ManifestError(
                "Benchmark candidate input provenance is invalid"
            )
        summary_path = Path(str(binding["summary_path"]))
        summary = _read_json_object(
            summary_path,
            f"candidate_inputs[{index}].summary",
        )
        summary_hash = _verify_object_seal(
            summary,
            "summary_hash",
            f"candidate_inputs[{index}].summary",
        )
        if (
            summary_hash != binding.get("summary_hash")
            or summary.get("candidates_sha256")
            != binding.get("candidates_sha256")
            or summary.get("corpus_manifest_sha256")
            != binding.get("corpus_manifest_sha256")
            or summary.get("code_dirty") is not False
        ):
            raise ManifestError(
                "Benchmark candidate summary binding is inconsistent"
            )
        corpus_path = Path(str(binding.get("corpus_manifest") or ""))
        if (
            corpus_path.is_symlink()
            or not corpus_path.is_file()
            or sha256_file(corpus_path)
            != binding.get("corpus_manifest_sha256")
        ):
            raise ManifestError("Benchmark corpus manifest binding changed")
    evidence = source_binding.get("evidence")
    proposed = source_binding.get("proposed")
    reviews = source_binding.get("reviews")
    if not all(
        isinstance(item, Mapping)
        for item in (evidence, proposed, reviews)
    ):
        raise ManifestError("Benchmark review artifact bindings are invalid")
    _verify_binding_file(evidence, "evidence")
    _verify_binding_file(proposed, "proposed")
    _verify_binding_file(reviews, "reviews")
    _verify_binding_file(
        {
            "path": reviews.get("summary_path"),
            "sha256": reviews.get("summary_sha256"),
        },
        "review summary",
    )
    if not _SHA256_RE.fullmatch(str(evidence.get("evidence_hash") or "")):
        raise ManifestError("Benchmark evidence hash is invalid")
    evidence_payload = _read_json_object(
        Path(str(evidence["path"])),
        "evidence",
    )
    evidence_hash = _verify_object_seal(
        evidence_payload,
        "evidence_hash",
        "evidence",
    )
    if (
        evidence_hash != evidence.get("evidence_hash")
        or evidence_payload.get("code_dirty") is not False
        or evidence_payload.get("human_claims") != 0
        or evidence_payload.get("input_proposed_sha256")
        != proposed.get("sha256")
    ):
        raise ManifestError("Benchmark evidence binding is inconsistent")
    proposed_records = _read_jsonl_objects(
        Path(str(proposed["path"])),
        "proposed",
    )
    proposed_ids = [str(record.get("candidate_id") or "") for record in proposed_records]
    if (
        any(not candidate_id for candidate_id in proposed_ids)
        or len(set(proposed_ids)) != len(proposed_ids)
    ):
        raise ManifestError("Benchmark proposed candidate IDs are invalid")
    review_records = _read_jsonl_objects(
        Path(str(reviews["path"])),
        "reviews",
    )
    reviews_by_id: dict[str, dict[str, Any]] = {}
    for review in review_records:
        candidate_id = str(review.get("candidate_id") or "")
        _verify_object_seal(review, "review_hash", f"review:{candidate_id}")
        if not candidate_id or candidate_id in reviews_by_id:
            raise ManifestError("Benchmark review candidate IDs are invalid")
        reviews_by_id[candidate_id] = review
    if set(reviews_by_id) != set(proposed_ids):
        raise ManifestError("Benchmark review/proposal candidate sets differ")
    review_summary = _read_json_object(
        Path(str(reviews["summary_path"])),
        "review summary",
    )
    _verify_object_seal(review_summary, "summary_hash", "review summary")
    expected_review_hashes = {
        candidate_id: review["review_hash"]
        for candidate_id, review in sorted(reviews_by_id.items())
    }
    if (
        review_summary.get("review_hashes") != expected_review_hashes
        or review_summary.get("evidence_hash") != evidence_hash
        or review_summary.get("code_dirty") is not False
    ):
        raise ManifestError("Benchmark review summary binding is inconsistent")
    verifications = evidence_payload.get("verifications")
    if not isinstance(verifications, list) or not verifications:
        raise ManifestError("Benchmark evidence verifications are empty")
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for verification in verifications:
        if not isinstance(verification, dict):
            raise ManifestError("Benchmark verification record is invalid")
        candidate_id = str(verification.get("candidate_id") or "")
        review = reviews_by_id.get(candidate_id)
        if (
            not candidate_id
            or candidate_id in evidence_by_id
            or review is None
            or review.get("status") != "accepted"
            or verification.get("review_hash") != review.get("review_hash")
            or verification.get("status") != "verified"
            or verification.get("curation_status") != "verified"
        ):
            raise ManifestError(
                "Benchmark verification/review binding is inconsistent"
            )
        evidence_by_id[candidate_id] = verification
    accepted_ids = {
        candidate_id
        for candidate_id, review in reviews_by_id.items()
        if review.get("status") == "accepted"
    }
    if set(evidence_by_id) != accepted_ids:
        raise ManifestError("Benchmark evidence accepted set is incomplete")
    for index, binding in enumerate(candidate_inputs):
        candidates = _read_jsonl_objects(
            Path(str(binding["candidates_path"])),
            f"candidate_inputs[{index}]",
        )
        eligible = [
            candidate
            for candidate in candidates
            if candidate.get("eligible_for_experiment") is True
        ]
        if (
            len(eligible)
            != _read_json_object(
                Path(str(binding["summary_path"])),
                f"candidate_inputs[{index}].summary",
            ).get("eligible_for_experiment")
        ):
            raise ManifestError("Benchmark candidate eligible count changed")
        for candidate in eligible:
            candidate_id = str(candidate.get("candidate_id") or "")
            verification = evidence_by_id.get(candidate_id)
            embedded = candidate.get("verification_evidence")
            if (
                verification is None
                or embedded != verification
                or candidate.get("curation_status") != "verified"
            ):
                raise ManifestError(
                    "Benchmark candidate/evidence binding is inconsistent"
                )
    return evidence_by_id


def _validate_verification_evidence(
    evidence: Any,
    *,
    case_id: str,
) -> None:
    if not isinstance(evidence, Mapping):
        raise ManifestError(
            f"Benchmark case {case_id!r} has invalid verification evidence"
        )
    models = evidence.get("review_models")
    if (
        evidence.get("evidence_type") != "external_validation"
        or not isinstance(evidence.get("evidence_ref"), str)
        or not str(evidence["evidence_ref"]).strip()
        or not _SHA256_RE.fullmatch(str(evidence.get("review_hash") or ""))
        or not isinstance(models, list)
        or len(models) < 2
        or not all(isinstance(model, str) and model.strip() for model in models)
        or len(set(models)) != len(models)
    ):
        raise ManifestError(
            f"Benchmark case {case_id!r} has invalid verification evidence"
        )


def _validate_benchmark_manifest(data: Mapping[str, Any]) -> None:
    provenance = data.get("provenance")
    if provenance is None:
        return
    if not isinstance(provenance, Mapping):
        raise ManifestError("Benchmark provenance must be an object")
    source_binding = provenance.get("source_binding")
    source_binding_hash = provenance.get("source_binding_hash")
    if (
        provenance.get("code_dirty") is not False
        or not isinstance(source_binding, Mapping)
        or not isinstance(source_binding_hash, str)
        or not _SHA256_RE.fullmatch(source_binding_hash)
        or _sha256_json(source_binding) != source_binding_hash
    ):
        raise ManifestError("Benchmark provenance binding is invalid")
    if not re.fullmatch(
        r"[0-9a-f]{7,64}",
        str(provenance.get("code_commit") or ""),
    ):
        raise ManifestError("Benchmark code commit is invalid")
    evidence_by_id = _validate_source_binding(source_binding)
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ManifestError("Benchmark cases must be a non-empty array")
    doi_splits: dict[str, str] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ManifestError(f"cases[{index}] must be an object")
        case_id = str(case.get("case_id") or f"cases[{index}]")
        candidate_id = str(case.get("candidate_id") or "")
        doi = _normalize_doi(case.get("doi"))
        split = case.get("split")
        panel_count = case.get("panel_count")
        if (
            not doi
            or candidate_id != case_id
            or split not in _BENCHMARK_SPLITS
            or case.get("curation_status") != "verified"
            or case.get("eligible_for_experiment") is not True
        ):
            raise ManifestError(
                f"Benchmark case {case_id!r} is not verified and eligible"
            )
        _validate_verification_evidence(
            case.get("verification_evidence"),
            case_id=case_id,
        )
        verification = case["verification_evidence"]
        evidence_record = evidence_by_id.get(candidate_id)
        if (
            evidence_record is None
            or verification.get("evidence_ref")
            != evidence_record.get("evidence_ref")
            or verification.get("evidence_type")
            != evidence_record.get("evidence_type")
            or verification.get("review_hash")
            != evidence_record.get("review_hash")
            or verification.get("review_models")
            != evidence_record.get("review_models")
        ):
            raise ManifestError(
                f"Benchmark case {case_id!r} is not bound to evidence"
            )
        if "multi_panel_manifest" in case:
            raise ManifestError(
                f"Benchmark case {case_id!r} cannot use multi_panel_manifest"
            )
        previous = doi_splits.setdefault(doi, str(split))
        if previous != split:
            raise ManifestError(
                f"DOI {doi!r} appears in multiple benchmark splits"
            )
        if panel_count == 1:
            if not _SHA256_RE.fullmatch(str(case.get("data_sha256") or "")):
                raise ManifestError(
                    f"Benchmark case {case_id!r} has no valid data_sha256"
                )
            continue
        panels = case.get("panels")
        if (
            not isinstance(panel_count, int)
            or isinstance(panel_count, bool)
            or panel_count < 2
            or not isinstance(panels, list)
            or len(panels) != panel_count
        ):
            raise ManifestError(
                f"Benchmark case {case_id!r} has invalid panels"
            )
        if any(
            not isinstance(panel, Mapping)
            or not _SHA256_RE.fullmatch(
                str(panel.get("data_sha256") or "")
            )
            for panel in panels
        ):
            raise ManifestError(
                f"Benchmark case {case_id!r} has invalid panel data hashes"
            )


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    panel_count: Optional[int]
    split: Optional[str]
    payload: Dict[str, Any]


def _load_manifest_object(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        else:
            raise ManifestError(
                f"Dataset manifests must use .json, .yaml, or .yml: {path}"
            )
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"Cannot load dataset manifest {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ManifestError("Dataset manifest must contain an object")
    return data


def validate_manifest(
    data: Mapping[str, Any],
    *,
    dataset_mode: str | None = None,
) -> list[DatasetCase]:
    if dataset_mode not in {None, "legacy", "sealed_benchmark"}:
        raise ManifestError(f"Unsupported dataset mode: {dataset_mode}")
    has_provenance = data.get("provenance") is not None
    if dataset_mode == "sealed_benchmark" and not has_provenance:
        raise ManifestError(
            "sealed_benchmark mode requires benchmark provenance"
        )
    if dataset_mode == "legacy" and has_provenance:
        raise ManifestError(
            "legacy mode cannot load a sealed benchmark manifest"
        )
    _validate_benchmark_manifest(data)
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestError("Dataset manifest requires a non-empty cases list")

    cases: list[DatasetCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ManifestError(f"cases[{index}] must be an object")
        raw_case_id = raw_case.get("case_id")
        if not isinstance(raw_case_id, str) or not raw_case_id.strip():
            raise ManifestError(
                f"cases[{index}].case_id must be a non-empty string"
            )
        case_id = raw_case_id.strip()
        if case_id in seen:
            raise ManifestError(f"Duplicate case_id in dataset manifest: {case_id}")
        seen.add(case_id)

        panel_count = raw_case.get("panel_count")
        if panel_count is not None and (
            isinstance(panel_count, bool)
            or not isinstance(panel_count, int)
            or panel_count < 1
        ):
            raise ManifestError(
                f"cases[{index}].panel_count must be a positive integer"
            )

        split = raw_case.get("split")
        if split is not None:
            if not isinstance(split, str) or not split.strip():
                raise ManifestError(
                    f"cases[{index}].split must be a non-empty string"
                )
            split = split.strip()

        cases.append(
            DatasetCase(
                case_id=case_id,
                panel_count=panel_count,
                split=split,
                payload=dict(raw_case),
            )
        )
    return cases


def load_dataset_manifest(
    path: str | Path,
    *,
    dataset_mode: str | None = None,
) -> list[DatasetCase]:
    return validate_manifest(
        _load_manifest_object(Path(path)),
        dataset_mode=dataset_mode,
    )


def select_case(
    cases: Sequence[DatasetCase],
    case_id: str,
) -> DatasetCase:
    matches = [case for case in cases if case.case_id == case_id]
    if len(matches) != 1:
        raise ManifestError(
            f"Dataset manifest has no unique case_id {case_id!r}"
        )
    return matches[0]


def verify_case_metadata(
    case: DatasetCase,
    *,
    panel_count: Optional[int],
    split: Optional[str],
) -> None:
    if case.panel_count != panel_count:
        raise ManifestError(
            f"case_id {case.case_id!r} panel_count changed: "
            f"{panel_count!r} != {case.panel_count!r}"
        )
    if case.split != split:
        raise ManifestError(
            f"case_id {case.case_id!r} split changed: "
            f"{split!r} != {case.split!r}"
        )


def verify_case_data_files(
    case: DatasetCase,
    *,
    manifest_path: str | Path,
) -> None:
    """Verify every declared source table immediately before generation."""

    payload = case.payload
    bindings: list[tuple[str, Any, Any]] = []
    if case.panel_count is not None and case.panel_count > 1:
        panels = payload.get("panels")
        if not isinstance(panels, list):
            raise ManifestError(
                f"case_id {case.case_id!r} requires a panels array"
            )
        for index, panel in enumerate(panels):
            if not isinstance(panel, Mapping):
                raise ManifestError(
                    f"case_id {case.case_id!r} panel[{index}] is invalid"
                )
            bindings.append(
                (
                    f"panel[{index}]",
                    panel.get("data_path"),
                    panel.get("data_sha256"),
                )
            )
    else:
        bindings.append(
            ("case", payload.get("data_path"), payload.get("data_sha256"))
        )
    required = payload.get("eligible_for_experiment") is True
    manifest_parent = Path(manifest_path).expanduser().resolve().parent
    for label, path_value, expected_hash in bindings:
        if expected_hash is None and not required:
            continue
        if (
            not isinstance(path_value, str)
            or not path_value.strip()
            or not isinstance(expected_hash, str)
            or not _SHA256_RE.fullmatch(expected_hash)
        ):
            raise ManifestError(
                f"case_id {case.case_id!r} {label} has an invalid data binding"
            )
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = manifest_parent / path
        if path.is_symlink() or not path.is_file():
            raise ManifestError(
                f"case_id {case.case_id!r} {label} data file is missing"
            )
        if sha256_file(path) != expected_hash:
            raise ManifestError(
                f"case_id {case.case_id!r} {label} data SHA-256 changed"
            )

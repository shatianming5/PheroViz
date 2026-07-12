"""Assemble verified case-builder outputs into an experiment manifest."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .policy import normalize_doi
from .proposals import (
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_ROWS,
    _multi_panel_proposals,
    propose_single_candidate,
)
from .provenance import sha256_file
from .reviews import ReviewError, validate_review_artifacts
from .splits import generate_split_bundle


SCHEMA_VERSION = "1.0"
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")


class BenchmarkBuildError(ValueError):
    """Raised when verified inputs cannot form a provenance-safe benchmark."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkBuildError(f"cannot-read-jsonl:{path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkBuildError(
                f"invalid-jsonl:{path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise BenchmarkBuildError(
                f"jsonl-record-not-object:{path}:{line_number}"
            )
        records.append(value)
    return records


def _load_case_summary(
    candidate_path: Path,
    *,
    evidence_file_hashes: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = candidate_path.parent / "summary.json"
    digest_path = candidate_path.parent / "summary.sha256"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        declared_digest = digest_path.read_text(encoding="utf-8").strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkBuildError("case-builder-summary-invalid") from exc
    if not isinstance(summary, dict):
        raise BenchmarkBuildError("case-builder-summary-invalid")
    summary_hash = summary.get("summary_hash")
    unhashed = dict(summary)
    unhashed.pop("summary_hash", None)
    if (
        not isinstance(summary_hash, str)
        or _sha256_json(unhashed) != summary_hash
        or declared_digest != summary_hash
    ):
        raise BenchmarkBuildError("case-builder-summary-hash-mismatch")
    if (
        summary.get("candidates_sha256") != sha256_file(candidate_path)
        or summary.get("evidence_file_sha256") not in evidence_file_hashes
        or summary.get("code_dirty") is not False
        or not GIT_COMMIT_PATTERN.fullmatch(
            str(summary.get("code_commit") or "")
        )
    ):
        raise BenchmarkBuildError("case-builder-summary-binding-mismatch")
    manifest_path_value = summary.get("corpus_manifest")
    manifest_digest = summary.get("corpus_manifest_sha256")
    if not isinstance(manifest_path_value, str) or not isinstance(
        manifest_digest, str
    ):
        raise BenchmarkBuildError("case-builder-corpus-binding-missing")
    manifest_path = Path(manifest_path_value)
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_digest:
        raise BenchmarkBuildError("case-builder-corpus-binding-mismatch")
    content_root_value = summary.get("content_root")
    output_root_value = summary.get("output_root")
    if not isinstance(content_root_value, str) or not isinstance(
        output_root_value, str
    ):
        raise BenchmarkBuildError("case-builder-root-binding-missing")
    content_root = Path(content_root_value).resolve()
    output_root = Path(output_root_value).resolve()
    if (
        not content_root.is_dir()
        or output_root != candidate_path.parent.resolve()
    ):
        raise BenchmarkBuildError("case-builder-root-binding-mismatch")
    return summary, {
        "candidates_path": str(candidate_path),
        "candidates_sha256": summary["candidates_sha256"],
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_hash": summary_hash,
        "code_commit": summary["code_commit"],
        "corpus_manifest": str(manifest_path.resolve()),
        "corpus_manifest_sha256": manifest_digest,
        "content_root": str(content_root),
        "output_root": str(output_root),
    }


def _validate_data_file(case: Mapping[str, Any], candidate_id: str) -> None:
    path_value = case.get("data_path")
    expected = case.get("data_sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise BenchmarkBuildError(f"case-data-binding-missing:{candidate_id}")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise BenchmarkBuildError(f"case-data-file-missing:{candidate_id}")
    if sha256_file(path) != expected:
        raise BenchmarkBuildError(f"case-data-hash-mismatch:{candidate_id}")


def _descriptor_identity(value: Any) -> dict[str, Any]:
    descriptor = value if isinstance(value, Mapping) else {}
    archive = descriptor.get("archive")
    archive_identity = None
    if isinstance(archive, Mapping):
        archive_identity = {
            "sha256": archive.get("sha256"),
            "size_bytes": archive.get("size_bytes"),
            "member_path": archive.get("member_path"),
        }
    return {
        "sha256": descriptor.get("sha256"),
        "size_bytes": descriptor.get("size_bytes"),
        "format": descriptor.get("format"),
        "path_root": descriptor.get("path_root"),
        "relative_path": descriptor.get("relative_path"),
        "sheet_name": descriptor.get("sheet_name"),
        "archive": archive_identity,
    }


def _validate_source_location(
    source_table: Mapping[str, Any],
    *,
    case_summary: Mapping[str, Any],
    candidate_id: str,
) -> None:
    path_value = source_table.get("path")
    relative_value = source_table.get("relative_path")
    root_name = source_table.get("path_root")
    format_name = str(source_table.get("format") or "").casefold()
    if (
        not isinstance(path_value, str)
        or not isinstance(relative_value, str)
        or root_name not in {"content_root", "output_root"}
        or not format_name
    ):
        raise BenchmarkBuildError(
            f"candidate-source-location-invalid:{candidate_id}"
        )
    path = Path(path_value)
    root = Path(str(case_summary[root_name])).resolve()
    if path.is_symlink() or not path.is_file():
        raise BenchmarkBuildError(
            f"candidate-source-location-invalid:{candidate_id}"
        )
    resolved = path.resolve()
    try:
        actual_relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise BenchmarkBuildError(
            f"candidate-source-root-escape:{candidate_id}"
        ) from exc
    if (
        actual_relative != Path(relative_value).as_posix()
        or resolved.suffix.casefold().lstrip(".") != format_name
    ):
        raise BenchmarkBuildError(
            f"candidate-source-location-mismatch:{candidate_id}"
        )
    archive = source_table.get("archive")
    if isinstance(archive, Mapping):
        archive_path_value = archive.get("path")
        if not isinstance(archive_path_value, str):
            raise BenchmarkBuildError(
                f"candidate-archive-location-invalid:{candidate_id}"
            )
        archive_path = Path(archive_path_value)
        if archive_path.is_symlink() or not archive_path.is_file():
            raise BenchmarkBuildError(
                f"candidate-archive-location-invalid:{candidate_id}"
            )
        if (
            archive.get("sha256") != sha256_file(archive_path)
            or archive.get("size_bytes") != archive_path.stat().st_size
        ):
            raise BenchmarkBuildError(
                f"candidate-archive-binding-mismatch:{candidate_id}"
            )
        try:
            archive_path.resolve().relative_to(
                Path(str(case_summary["content_root"])).resolve()
            )
        except ValueError as exc:
            raise BenchmarkBuildError(
                f"candidate-archive-root-escape:{candidate_id}"
            ) from exc


def _candidate_matches_reviewed_proposal(
    record: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    candidate_id: str,
) -> None:
    identity_fields = (
        "candidate_id",
        "doi",
        "figure_no",
        "panel_ids",
        "panel_qualifier",
        "license_evidence_sha256",
        "corpus_manifest_sha256",
        "provenance_manifest_sha256",
    )
    if any(record.get(key) != proposal.get(key) for key in identity_fields):
        raise BenchmarkBuildError(
            f"candidate-proposal-identity-mismatch:{candidate_id}"
        )
    if _descriptor_identity(record.get("source_table")) != _descriptor_identity(
        proposal.get("source_table")
    ):
        raise BenchmarkBuildError(
            f"candidate-proposal-source-mismatch:{candidate_id}"
        )
    for asset_name in ("figure", "caption"):
        record_asset = record.get(asset_name) or {}
        proposal_asset = proposal.get(asset_name) or {}
        if (
            record_asset.get("sha256") != proposal_asset.get("sha256")
            or record_asset.get("size_bytes")
            != proposal_asset.get("size_bytes")
        ):
            raise BenchmarkBuildError(
                f"candidate-proposal-{asset_name}-mismatch:{candidate_id}"
            )


def _validate_canonical_single_proposal(
    record: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    candidate_id: str,
) -> None:
    recomputed = propose_single_candidate(
        deepcopy(dict(record)),
        input_candidates_sha256=str(
            proposal.get("input_candidates_sha256") or ""
        ),
        code_commit=str(proposal.get("code_commit") or ""),
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
    )
    expected_case = proposal.get("experiment_case")
    actual_case = recomputed.get("experiment_case")
    semantic_fields = (
        "case_id",
        "panel_count",
        "sheet",
        "panel_id",
        "user_goal",
        "chart_family",
        "intent",
        "evaluation_expectation",
    )
    if (
        not isinstance(expected_case, Mapping)
        or not isinstance(actual_case, Mapping)
        or any(
            actual_case.get(key) != expected_case.get(key)
            for key in semantic_fields
        )
    ):
        raise BenchmarkBuildError(
            f"candidate-canonical-proposal-mismatch:{candidate_id}"
        )


def _validate_case_schema(case: Mapping[str, Any], candidate_id: str) -> None:
    if "multi_panel_manifest" in case:
        raise BenchmarkBuildError(
            f"experiment-case-external-manifest-forbidden:{candidate_id}"
        )
    panel_count = case.get("panel_count")
    expectation = case.get("evaluation_expectation")
    if (
        not isinstance(panel_count, int)
        or isinstance(panel_count, bool)
        or panel_count < 1
        or not isinstance(case.get("user_goal"), str)
        or not str(case["user_goal"]).strip()
        or not isinstance(case.get("chart_family"), str)
        or not str(case["chart_family"]).strip()
        or not isinstance(expectation, Mapping)
        or expectation.get("schema_version") != "1.1.0"
        or not isinstance(expectation.get("panels"), list)
        or len(expectation["panels"]) != panel_count
    ):
        raise BenchmarkBuildError(f"experiment-case-schema-invalid:{candidate_id}")
    expected_panel_ids = {
        str(panel.get("panel_id") or "")
        for panel in expectation["panels"]
        if isinstance(panel, Mapping)
    }
    if len(expected_panel_ids) != panel_count or "" in expected_panel_ids:
        raise BenchmarkBuildError(
            f"experiment-case-panel-schema-invalid:{candidate_id}"
        )
    if panel_count == 1:
        if case.get("panel_id") not in expected_panel_ids:
            raise BenchmarkBuildError(
                f"experiment-case-panel-id-mismatch:{candidate_id}"
            )
        _validate_data_file(case, candidate_id)
        return
    panels = case.get("panels")
    if not isinstance(panels, list) or len(panels) != panel_count:
        raise BenchmarkBuildError(
            f"experiment-case-panels-invalid:{candidate_id}"
        )
    panel_ids = {
        str(panel.get("id") or "")
        for panel in panels
        if isinstance(panel, Mapping)
    }
    if panel_ids != expected_panel_ids:
        raise BenchmarkBuildError(
            f"experiment-case-panel-id-mismatch:{candidate_id}"
        )
    for panel in panels:
        _validate_data_file(panel, f"{candidate_id}:{panel.get('id')}")


def _validated_single_cases(
    candidate_paths: Sequence[Path],
    *,
    evidence: Mapping[str, dict[str, Any]],
    evidence_file_hashes: Mapping[str, str],
    proposals: Mapping[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    cases: dict[str, dict[str, Any]] = {}
    dois: dict[str, str] = {}
    bindings: list[dict[str, Any]] = []
    for path in candidate_paths:
        case_summary, binding = _load_case_summary(
            path,
            evidence_file_hashes=set(evidence_file_hashes.values()),
        )
        bindings.append(binding)
        eligible_in_file = 0
        for record in _read_jsonl(path):
            if not record.get("eligible_for_experiment"):
                continue
            eligible_in_file += 1
            candidate_id = str(record.get("candidate_id") or "").strip()
            if not candidate_id or candidate_id in cases:
                raise BenchmarkBuildError(
                    "eligible-candidate-id-invalid-or-duplicate"
                )
            if str(record.get("curation_status") or "").casefold() != "verified":
                raise BenchmarkBuildError(
                    f"eligible-candidate-not-verified:{candidate_id}"
                )
            verification = record.get("verification_evidence")
            expected_verification = evidence.get(candidate_id)
            if (
                not isinstance(verification, dict)
                or expected_verification is None
                or verification != expected_verification
                or record.get("verification_evidence_file_sha256")
                != evidence_file_hashes.get(candidate_id)
                or record.get("verification_evidence_file_sha256")
                != case_summary.get("evidence_file_sha256")
            ):
                raise BenchmarkBuildError(
                    f"candidate-evidence-binding-mismatch:{candidate_id}"
                )
            proposal = proposals.get(candidate_id)
            if not isinstance(proposal, Mapping):
                raise BenchmarkBuildError(
                    f"candidate-proposal-missing:{candidate_id}"
                )
            _candidate_matches_reviewed_proposal(
                record,
                proposal,
                candidate_id=candidate_id,
            )
            _validate_canonical_single_proposal(
                record,
                proposal,
                candidate_id=candidate_id,
            )
            if (
                record.get("corpus_manifest_sha256")
                != case_summary.get("corpus_manifest_sha256")
            ):
                raise BenchmarkBuildError(
                    f"candidate-corpus-binding-mismatch:{candidate_id}"
                )
            proposal_doi = normalize_doi(proposal.get("doi"))
            if not proposal_doi:
                raise BenchmarkBuildError(
                    f"candidate-doi-missing:{candidate_id}"
                )
            experiment_case = record.get("experiment_case")
            if not isinstance(experiment_case, dict):
                raise BenchmarkBuildError(
                    f"candidate-experiment-case-missing:{candidate_id}"
                )
            reviewed_case = expected_verification.get("experiment_case")
            if (
                not isinstance(reviewed_case, dict)
                or reviewed_case != proposal.get("experiment_case")
            ):
                raise BenchmarkBuildError(
                    f"candidate-reviewed-case-mismatch:{candidate_id}"
                )
            immutable_case_fields = (
                "case_id",
                "panel_count",
                "panel_id",
                "sheet",
                "user_goal",
                "chart_family",
                "intent",
                "evaluation_expectation",
            )
            if any(
                experiment_case.get(key) != reviewed_case.get(key)
                for key in immutable_case_fields
            ):
                raise BenchmarkBuildError(
                    f"candidate-case-content-mismatch:{candidate_id}"
                )
            if (
                experiment_case.get("case_id") != candidate_id
                or experiment_case.get("panel_count") != 1
                or not experiment_case.get("eligible_for_experiment")
            ):
                raise BenchmarkBuildError(
                    f"candidate-experiment-case-invalid:{candidate_id}"
                )
            source_table = record.get("source_table") or {}
            _validate_source_location(
                source_table,
                case_summary=case_summary,
                candidate_id=candidate_id,
            )
            if (
                source_table.get("path") != experiment_case.get("data_path")
                or source_table.get("sha256")
                != experiment_case.get("data_sha256")
            ):
                raise BenchmarkBuildError(
                    f"candidate-source-table-binding-mismatch:{candidate_id}"
                )
            case = deepcopy(reviewed_case)
            case.update(
                {
                    "data_path": source_table["path"],
                    "sheet": source_table.get("sheet_name"),
                    "data_sha256": source_table["sha256"],
                    "curation_status": "verified",
                    "eligible_for_experiment": True,
                    "eligibility_reasons": [],
                }
            )
            for panel in case.get("panels") or []:
                if isinstance(panel, dict):
                    panel["data_path"] = source_table["path"]
                    panel["sheet"] = source_table.get("sheet_name")
                    panel["data_sha256"] = source_table["sha256"]
            case.update(
                {
                    "doi": proposal_doi,
                    "candidate_id": candidate_id,
                    "verification_evidence": {
                        "evidence_ref": verification["evidence_ref"],
                        "evidence_type": verification["evidence_type"],
                        "review_hash": verification.get("review_hash"),
                        "review_models": verification.get("review_models"),
                    },
                }
            )
            _validate_case_schema(case, candidate_id)
            cases[candidate_id] = case
            dois[candidate_id] = proposal_doi
        if (
            case_summary.get("eligible_for_experiment") != eligible_in_file
            or case_summary.get("verified") != eligible_in_file
        ):
            raise BenchmarkBuildError(
                f"case-builder-eligible-count-mismatch:{path}"
            )
    if not cases:
        raise BenchmarkBuildError("no-eligible-single-cases")
    return cases, dois, bindings


def _materialize_multi_case(
    candidate_id: str,
    *,
    verification: Mapping[str, Any],
    proposal: Mapping[str, Any],
    proposals: Mapping[str, dict[str, Any]],
    singles: Mapping[str, dict[str, Any]],
    single_dois: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    if proposal.get("proposal_type") != "multi_panel":
        raise BenchmarkBuildError(f"unrebuilt-evidence-is-not-multi:{candidate_id}")
    if verification.get("experiment_case") != proposal.get("experiment_case"):
        raise BenchmarkBuildError(f"multi-evidence-proposal-mismatch:{candidate_id}")
    source_ids = proposal.get("source_candidate_ids")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) < 2
        or any(not isinstance(value, str) for value in source_ids)
    ):
        raise BenchmarkBuildError(f"multi-source-ids-invalid:{candidate_id}")
    if any(source_id not in singles for source_id in source_ids):
        raise BenchmarkBuildError(
            f"multi-source-not-verified:{candidate_id}"
        )
    source_dois = {single_dois[source_id] for source_id in source_ids}
    proposal_doi = normalize_doi(proposal.get("doi"))
    if len(source_dois) != 1 or proposal_doi not in source_dois:
        raise BenchmarkBuildError(f"multi-source-doi-mismatch:{candidate_id}")

    by_panel: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        source_proposal = proposals.get(source_id)
        panel_ids = (source_proposal or {}).get("panel_ids")
        if (
            not isinstance(panel_ids, list)
            or len(panel_ids) != 1
            or not isinstance(panel_ids[0], str)
            or panel_ids[0] in by_panel
        ):
            raise BenchmarkBuildError(
                f"multi-source-panel-binding-invalid:{candidate_id}"
            )
        by_panel[panel_ids[0]] = singles[source_id]
    ordered_sources = sorted(
        by_panel.items(),
        key=lambda item: item[0],
    )
    panels = [
        {
            "id": panel_id,
            "data_path": source["data_path"],
            "data_sha256": source["data_sha256"],
            "sheet": source.get("sheet"),
            "user_goal": source["user_goal"],
            "chart_family": source["chart_family"],
            "intent": deepcopy(source.get("intent") or {}),
        }
        for panel_id, source in ordered_sources
    ]
    expectation_panels = [
        deepcopy(source["evaluation_expectation"]["panels"][0])
        for _, source in ordered_sources
    ]
    x_scales = [
        str(expectation.get("x_scale") or "")
        for expectation in expectation_panels
    ]
    x_units = [
        expectation.get("x_unit") for expectation in expectation_panels
    ]
    series_sets = [
        {
            str(series["series_id"])
            for series in expectation.get("series") or []
            if isinstance(series, Mapping) and series.get("series_id")
        }
        for expectation in expectation_panels
    ]
    checks: dict[str, Any] = {}
    if len(set(x_scales)) == 1:
        checks["shared_x_scale"] = True
    if all(unit is not None for unit in x_units) and len(set(x_units)) == 1:
        checks["shared_x_unit"] = True
    shared_series = (
        sorted(set.intersection(*series_sets)) if series_sets else []
    )
    group: dict[str, Any] = {
        "group_id": "all_panels",
        "panels": [panel["id"] for panel in panels],
        "checks": checks,
    }
    if shared_series:
        group["series"] = shared_series
        checks["palette_consistent"] = True
    figure_no = proposal.get("figure_no")
    case = {
        "case_id": candidate_id,
        "panel_count": len(panels),
        "split": None,
        "panels": panels,
        "user_goal": (
            f"Create a {len(panels)}-panel Figure {figure_no} using "
            "the proposed panel-specific charts."
        ),
        "chart_family": "multi_panel",
        "intent": {
            "panels": [
                {
                    "panel_id": panel_id,
                    **deepcopy(source.get("intent") or {}),
                }
                for panel_id, source in ordered_sources
            ]
        },
        "evaluation_expectation": {
            "schema_version": "1.1.0",
            "panels": expectation_panels,
            "panel_groups": [group] if checks else [],
        },
    }
    reviewed_case = verification["experiment_case"]
    reviewed_panels = reviewed_case.get("panels")
    canonical_panel_semantics = [
        {
            key: panel.get(key)
            for key in ("id", "sheet", "user_goal", "chart_family", "intent")
        }
        for panel in panels
    ]
    reviewed_panel_semantics = [
        {
            key: panel.get(key)
            for key in ("id", "sheet", "user_goal", "chart_family", "intent")
        }
        for panel in reviewed_panels or []
        if isinstance(panel, Mapping)
    ]
    semantic_fields = (
        "case_id",
        "panel_count",
        "user_goal",
        "chart_family",
        "intent",
        "evaluation_expectation",
    )
    if (
        not isinstance(reviewed_case, Mapping)
        or canonical_panel_semantics != reviewed_panel_semantics
        or any(case[key] != reviewed_case.get(key) for key in semantic_fields)
    ):
        raise BenchmarkBuildError(
            f"multi-canonical-case-mismatch:{candidate_id}"
        )
    case.update(
        {
            "doi": proposal_doi,
            "candidate_id": candidate_id,
            "source_candidate_ids": sorted(source_ids),
            "curation_status": "verified",
            "eligible_for_experiment": True,
            "eligibility_reasons": [],
            "verification_evidence": {
                "evidence_ref": verification["evidence_ref"],
                "evidence_type": verification["evidence_type"],
                "review_hash": verification.get("review_hash"),
                "review_models": verification.get("review_models"),
            },
        }
    )
    _validate_case_schema(case, candidate_id)
    return case, proposal_doi


def assemble_verified_benchmark(
    *,
    candidate_paths: Iterable[str | Path],
    seed: int,
    code_commit: str,
    code_dirty: bool,
    evidence_path: str | Path | None = None,
    proposed_path: str | Path | None = None,
    reviews_path: str | Path | None = None,
    review_bundles: Iterable[
        tuple[str | Path, str | Path, str | Path]
    ] | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    """Return a sealed benchmark manifest, split bundle, and summary."""

    if not GIT_COMMIT_PATTERN.fullmatch(code_commit):
        raise BenchmarkBuildError("invalid-code-commit")
    if not isinstance(code_dirty, bool):
        raise BenchmarkBuildError("invalid-code-dirty-flag")
    if code_dirty:
        raise BenchmarkBuildError("code-worktree-dirty")
    resolved_candidates = tuple(
        Path(path).expanduser().resolve(strict=True) for path in candidate_paths
    )
    if not resolved_candidates:
        raise BenchmarkBuildError("candidate-inputs-empty")
    raw_review_bundles = list(review_bundles or ())
    legacy_values = (proposed_path, reviews_path, evidence_path)
    if any(value is not None for value in legacy_values):
        if any(value is None for value in legacy_values):
            raise BenchmarkBuildError(
                "proposed-reviews-evidence-must-be-specified-together"
            )
        raw_review_bundles.append(
            (proposed_path, reviews_path, evidence_path)  # type: ignore[arg-type]
        )
    if not raw_review_bundles:
        raise BenchmarkBuildError("review-bundles-empty")

    evidence: dict[str, dict[str, Any]] = {}
    evidence_file_hashes: dict[str, str] = {}
    proposals: dict[str, dict[str, Any]] = {}
    canonical_multi: dict[str, dict[str, Any]] = {}
    review_bindings: list[dict[str, Any]] = []
    for bundle_index, (raw_proposed, raw_reviews, raw_evidence) in enumerate(
        raw_review_bundles
    ):
        proposal_file = Path(raw_proposed).expanduser().resolve(strict=True)
        reviews_file = Path(raw_reviews).expanduser().resolve(strict=True)
        evidence_file = Path(raw_evidence).expanduser().resolve(strict=True)
        try:
            review_bundle = validate_review_artifacts(
                proposed_path=proposal_file,
                reviews_path=reviews_file,
                evidence_path=evidence_file,
            )
        except ReviewError as exc:
            raise BenchmarkBuildError(
                f"review-artifact-validation[{bundle_index}]:{exc}"
            ) from exc
        bundle_proposals = review_bundle["proposals"]
        bundle_evidence = {
            record["candidate_id"]: record
            for record in review_bundle["evidence"]["verifications"]
        }
        duplicate_ids = (
            set(proposals) & set(bundle_proposals)
        ) | (set(evidence) & set(bundle_evidence))
        if duplicate_ids:
            raise BenchmarkBuildError(
                "review-bundle-candidate-duplicate:"
                + ",".join(sorted(duplicate_ids))
            )
        proposals.update(bundle_proposals)
        evidence.update(bundle_evidence)
        evidence_file_hashes.update(
            {
                candidate_id: review_bundle["evidence_sha256"]
                for candidate_id in bundle_evidence
            }
        )
        bundle_canonical_multi = {
            proposal["candidate_id"]: proposal
            for proposal in _multi_panel_proposals(
                [
                    proposal
                    for proposal in bundle_proposals.values()
                    if proposal.get("proposal_type") == "single_panel"
                ],
                input_candidates_sha256=review_bundle[
                    "input_proposed_sha256"
                ],
                code_commit=code_commit,
            )
        }
        if set(canonical_multi) & set(bundle_canonical_multi):
            raise BenchmarkBuildError("canonical-multi-candidate-duplicate")
        canonical_multi.update(bundle_canonical_multi)
        review_bindings.append(
            {
                "evidence": {
                    "path": str(evidence_file),
                    "sha256": review_bundle["evidence_sha256"],
                    "evidence_hash": review_bundle["evidence"]["evidence_hash"],
                },
                "proposed": {
                    "path": str(proposal_file),
                    "sha256": review_bundle["input_proposed_sha256"],
                },
                "reviews": {
                    "path": str(reviews_file),
                    "sha256": review_bundle["reviews_sha256"],
                    "summary_path": str(reviews_file.parent / "summary.json"),
                    "summary_sha256": review_bundle["summary_sha256"],
                },
            }
        )
    singles, single_dois, candidate_bindings = _validated_single_cases(
        resolved_candidates,
        evidence=evidence,
        evidence_file_hashes=evidence_file_hashes,
        proposals=proposals,
    )

    cases = dict(singles)
    case_dois = dict(single_dois)
    unconsumed_evidence = sorted(set(evidence) - set(singles))
    for candidate_id in unconsumed_evidence:
        proposal = proposals.get(candidate_id)
        if proposal is None:
            raise BenchmarkBuildError(
                f"evidence-candidate-not-in-proposals:{candidate_id}"
            )
        canonical = canonical_multi.get(candidate_id)
        canonical_fields = (
            "candidate_id",
            "source_candidate_ids",
            "doi",
            "figure_no",
            "panel_ids",
            "experiment_case",
        )
        if canonical is None or any(
            proposal.get(key) != canonical.get(key)
            for key in canonical_fields
        ):
            raise BenchmarkBuildError(
                f"multi-proposal-not-canonical:{candidate_id}"
            )
        case, doi = _materialize_multi_case(
            candidate_id,
            verification=evidence[candidate_id],
            proposal=proposal,
            proposals=proposals,
            singles=singles,
            single_dois=single_dois,
        )
        cases[candidate_id] = case
        case_dois[candidate_id] = doi

    if set(evidence) != set(cases):
        raise BenchmarkBuildError("evidence-case-set-mismatch")
    review_bindings.sort(
        key=lambda binding: binding["proposed"]["path"]
    )
    source_binding: dict[str, Any] = {
        "candidate_inputs": sorted(
            candidate_bindings,
            key=lambda item: item["candidates_path"],
        ),
        "review_bundles": review_bindings,
    }
    source_binding_hash = _sha256_json(source_binding)
    split_bundle = generate_split_bundle(
        (
            {"doi": doi, "download_eligible": True}
            for doi in sorted(set(case_dois.values()))
        ),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        source_manifest_sha256=source_binding_hash,
    )
    assignments = {
        paper["doi"]: paper["split"] for paper in split_bundle["papers"]
    }
    manifest_cases: list[dict[str, Any]] = []
    for candidate_id in sorted(cases):
        case = deepcopy(cases[candidate_id])
        case["split"] = assignments[case_dois[candidate_id]]
        manifest_cases.append(case)

    case_split_counts = Counter(case["split"] for case in manifest_cases)
    panel_strata = Counter(
        str(case["panel_count"]) for case in manifest_cases
    )
    chart_families = Counter(
        str(case.get("chart_family") or "unknown") for case in manifest_cases
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "code_commit": code_commit,
            "code_dirty": code_dirty,
            "source_binding": source_binding,
            "source_binding_hash": source_binding_hash,
            "split_seed": seed,
            "split_ratios": split_bundle["ratios"],
        },
        "cases": manifest_cases,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "source_binding_hash": source_binding_hash,
        "cases": len(manifest_cases),
        "single_cases": len(singles),
        "multi_panel_cases": len(manifest_cases) - len(singles),
        "unique_dois": len(assignments),
        "paper_split_counts": split_bundle["counts"],
        "case_split_counts": {
            name: case_split_counts.get(name, 0)
            for name in ("train", "val", "test")
        },
        "panel_strata": dict(sorted(panel_strata.items())),
        "chart_families": dict(sorted(chart_families.items())),
        "evidence_records": len(evidence),
    }
    return {
        "manifest": manifest,
        "splits": split_bundle,
        "summary": summary,
    }


def write_benchmark_outputs(
    output_root: str | Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Write immutable benchmark artifacts and return the sealed summary."""

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise BenchmarkBuildError(f"benchmark-output-not-empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "benchmark_manifest.json"
    splits_path = output / "splits.json"
    manifest_path.write_text(
        json.dumps(
            result["manifest"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    splits_path.write_text(
        json.dumps(
            result["splits"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        **dict(result["summary"]),
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "splits": str(splits_path),
        "splits_sha256": sha256_file(splits_path),
    }
    summary["summary_hash"] = _sha256_json(summary)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "summary.sha256").write_text(
        summary["summary_hash"] + "\n",
        encoding="utf-8",
    )
    return summary


def derive_multi_review_batch(
    *,
    review_bundles: Iterable[
        tuple[str | Path, str | Path, str | Path]
    ],
    code_commit: str,
    code_dirty: bool,
) -> dict[str, Any]:
    """Build a new review batch from previously accepted single proposals."""

    if not GIT_COMMIT_PATTERN.fullmatch(code_commit):
        raise BenchmarkBuildError("invalid-code-commit")
    if code_dirty:
        raise BenchmarkBuildError("code-worktree-dirty")
    singles: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    for index, (raw_proposed, raw_reviews, raw_evidence) in enumerate(
        review_bundles
    ):
        proposed_path = Path(raw_proposed).expanduser().resolve(strict=True)
        reviews_path = Path(raw_reviews).expanduser().resolve(strict=True)
        evidence_path = Path(raw_evidence).expanduser().resolve(strict=True)
        try:
            bundle = validate_review_artifacts(
                proposed_path=proposed_path,
                reviews_path=reviews_path,
                evidence_path=evidence_path,
            )
        except ReviewError as exc:
            raise BenchmarkBuildError(
                f"review-artifact-validation[{index}]:{exc}"
            ) from exc
        accepted_ids = {
            record["candidate_id"]
            for record in bundle["evidence"]["verifications"]
        }
        for candidate_id in sorted(accepted_ids):
            proposal = bundle["proposals"].get(candidate_id)
            if not proposal or proposal.get("proposal_type") != "single_panel":
                continue
            if candidate_id in singles:
                raise BenchmarkBuildError(
                    f"review-bundle-candidate-duplicate:{candidate_id}"
                )
            singles[candidate_id] = deepcopy(proposal)
        bindings.append(
            {
                "proposed_path": str(proposed_path),
                "proposed_sha256": bundle["input_proposed_sha256"],
                "reviews_path": str(reviews_path),
                "reviews_sha256": bundle["reviews_sha256"],
                "evidence_path": str(evidence_path),
                "evidence_sha256": bundle["evidence_sha256"],
                "evidence_hash": bundle["evidence"]["evidence_hash"],
            }
        )
    if not singles:
        raise BenchmarkBuildError("accepted-single-proposals-empty")
    bindings.sort(key=lambda binding: binding["proposed_path"])
    source_binding_hash = _sha256_json(bindings)
    multi = _multi_panel_proposals(
        list(singles.values()),
        input_candidates_sha256=source_binding_hash,
        code_commit=code_commit,
    )
    proposals = sorted(
        [*singles.values(), *multi],
        key=lambda proposal: proposal["candidate_id"],
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "source_review_bindings": bindings,
        "source_binding_hash": source_binding_hash,
        "accepted_single_proposals": len(singles),
        "derived_multi_panel_proposals": len(multi),
        "proposals_total": len(proposals),
        "eligible_for_experiment": 0,
    }
    return {"proposals": proposals, "summary": summary}


def write_derived_proposal_outputs(
    output_root: str | Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a non-overwriting derived review batch and its provenance."""

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise BenchmarkBuildError(f"derived-proposal-output-not-empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    proposed_path = output / "proposed.jsonl"
    with proposed_path.open("w", encoding="utf-8") as handle:
        for proposal in result["proposals"]:
            handle.write(
                json.dumps(
                    proposal,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    summary = {
        **dict(result["summary"]),
        "proposed": str(proposed_path),
        "proposed_sha256": sha256_file(proposed_path),
    }
    summary["summary_hash"] = _sha256_json(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "summary.sha256").write_text(
        summary["summary_hash"] + "\n",
        encoding="utf-8",
    )
    return summary

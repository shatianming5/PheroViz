"""Build a sealed benchmark derivative using static renderability inputs only.

The policy in this module is outcome-independent: it reads source tables,
sheet selections, and sealed evaluation expectations. It never reads method
outputs, run records, scores, failed-run identifiers, or production artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from jsonschema.exceptions import ValidationError
import pandas as pd

from app.evaluation.schema import validate_expectation
from .manifest import (
    DatasetCase,
    ManifestError,
    load_dataset_manifest,
    resolve_case_data_path,
    verify_case_data_files,
)
from .models import sha256_file


SCHEMA_VERSION = "1.0"
MAX_BAR_CATEGORICAL_X = 200
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SPLITS = ("train", "val", "test")
_OUTPUT_NAMES = (
    "benchmark_manifest.json",
    "renderability_audit.json",
    "summary.json",
)

RENDERABILITY_POLICY: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "policy_id": "outcome-independent-static-renderability-v1",
    "declared_inputs": [
        "source_table",
        "source_sheet",
        "evaluation_expectation",
    ],
    "forbidden_inputs": [
        "run_records",
        "method_names",
        "method_scores",
        "failed_case_ids",
        "production_artifacts",
    ],
    "split_scope": list(_SPLITS),
    "panel_scope": "every_panel",
    "multi_panel_rule": "reject_case_if_any_panel_fails",
    "rules": [
        {
            "rule_id": "bar-column-categorical-x-cardinality",
            "expectation_series_kinds": ["bar"],
            "semantic_families": ["bar", "column"],
            "max_unique_categorical_x_per_panel": MAX_BAR_CATEGORICAL_X,
            "comparison": "less_than_or_equal",
            "x_counting": ("union_of_resolved_bar_series_x_after_expectation_filters"),
            "missing_x_values": "count_as_one_category",
        }
    ],
    "audit_counting": {
        "series_count": "all_expected_series_in_panel",
        "expected_points": ("sum_per_series_max_resolved_x_y_value_length"),
    },
    "explicitly_uncapped_series_kinds": ["line", "scatter"],
    "non_bar_cardinality_action": "no_cap",
}


class RenderabilityError(ValueError):
    """Raised when a derivative cannot be built without guessing."""


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


POLICY_HASH = _sha256_json(RENDERABILITY_POLICY)
EXPECTED_POLICY_HASH = (
    "283f0520bf1fda98773958602be6206c1776c573a00273391dbfeb6acd565cec"
)


def _validate_policy_constant() -> None:
    if (
        _sha256_json(RENDERABILITY_POLICY) != EXPECTED_POLICY_HASH
        or POLICY_HASH != EXPECTED_POLICY_HASH
        or RENDERABILITY_POLICY["rules"][0]["max_unique_categorical_x_per_panel"]
        != MAX_BAR_CATEGORICAL_X
    ):
        raise RenderabilityError("renderability-policy-mutated")


def _pretty_json_bytes(value: Any) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise RenderabilityError(f"seal-field-already-present:{field}")
    sealed = deepcopy(dict(value))
    sealed[field] = _sha256_json(sealed)
    return sealed


def _verify_seal(value: Mapping[str, Any], field: str, label: str) -> str:
    declared = value.get(field)
    unhashed = dict(value)
    unhashed.pop(field, None)
    if (
        not isinstance(declared, str)
        or not _SHA256_RE.fullmatch(declared)
        or _sha256_json(unhashed) != declared
    ):
        raise RenderabilityError(f"{label}-seal-invalid")
    return declared


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RenderabilityError(f"{label}-json-invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise RenderabilityError(f"{label}-must-be-object")
    return payload


def _git_state(repo_root: Path) -> tuple[str, bool]:
    root = repo_root.expanduser().resolve()
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .lower()
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RenderabilityError(f"git-state-unavailable:{root}") from exc
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise RenderabilityError("git-commit-invalid")
    return commit, bool(status.strip())


def _normalize_doi(value: Any) -> str:
    text = unquote(str(value or "")).strip()
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    parsed = urlsplit(text)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.casefold() not in {"http", "https"} or (
            parsed.hostname or ""
        ).casefold() not in {"doi.org", "dx.doi.org"}:
            raise RenderabilityError(f"doi-invalid:{value!r}")
        text = parsed.path.lstrip("/")
    else:
        text = re.split(r"[?#]", text, maxsplit=1)[0]
    doi = text.strip().casefold()
    if (
        not doi.startswith("10.")
        or "/" not in doi
        or any(character.isspace() for character in doi)
    ):
        raise RenderabilityError(f"doi-invalid:{value!r}")
    return doi


def _load_parent(
    parent_manifest: Path,
    *,
    manifest_data_root: Path,
    runtime_repo_root: Path,
) -> tuple[dict[str, Any], list[DatasetCase], str]:
    if not manifest_data_root.is_absolute() or not runtime_repo_root.is_absolute():
        raise RenderabilityError("explicit-root-remap-must-use-absolute-paths")
    parent = _load_json_object(parent_manifest, "parent-manifest")
    if parent.get("schema_version") != SCHEMA_VERSION:
        raise RenderabilityError(
            f"parent-manifest-schema-unexpected:{parent.get('schema_version')!r}"
        )
    if not isinstance(parent.get("provenance"), dict):
        raise RenderabilityError("parent-manifest-provenance-invalid")
    if set(parent) - {"schema_version", "provenance", "cases", "manifest_hash"}:
        raise RenderabilityError("parent-manifest-fields-unexpected")
    if "manifest_hash" in parent:
        _verify_seal(parent, "manifest_hash", "parent-manifest")
    try:
        cases = load_dataset_manifest(
            parent_manifest,
            dataset_mode="sealed_benchmark",
            manifest_data_root=manifest_data_root,
            runtime_repo_root=runtime_repo_root,
        )
        for case in cases:
            verify_case_data_files(
                case,
                manifest_path=parent_manifest,
                manifest_data_root=manifest_data_root,
                runtime_repo_root=runtime_repo_root,
            )
    except ManifestError as exc:
        raise RenderabilityError(f"parent-manifest-validation:{exc}") from exc
    for case in cases:
        try:
            validate_expectation(case.payload.get("evaluation_expectation"))
        except (ValidationError, TypeError, KeyError) as exc:
            raise RenderabilityError(
                f"expectation-schema-invalid:{case.case_id}"
            ) from exc
    return parent, cases, sha256_file(parent_manifest)


def _case_panel_sources(case: DatasetCase) -> list[dict[str, Any]]:
    payload = case.payload
    expectation = payload.get("evaluation_expectation")
    if not isinstance(expectation, Mapping):
        raise RenderabilityError(f"expectation-missing:{case.case_id}")
    expected_panels = expectation.get("panels")
    if not isinstance(expected_panels, list):
        raise RenderabilityError(f"expectation-panels-invalid:{case.case_id}")
    expectations_by_id = {
        str(panel.get("panel_id")): panel
        for panel in expected_panels
        if isinstance(panel, Mapping)
    }
    if len(expectations_by_id) != len(expected_panels):
        raise RenderabilityError(f"expectation-panel-ids-invalid:{case.case_id}")

    if case.panel_count == 1:
        if len(expected_panels) != 1:
            raise RenderabilityError(
                f"single-panel-expectation-count-invalid:{case.case_id}"
            )
        expected = expected_panels[0]
        panel_id = str(expected["panel_id"])
        declared_panel_id = payload.get("panel_id")
        if declared_panel_id is not None and declared_panel_id != panel_id:
            raise RenderabilityError(f"single-panel-id-mismatch:{case.case_id}")
        return [
            {
                "panel_id": panel_id,
                "data_path": payload.get("data_path"),
                "data_sha256": payload.get("data_sha256"),
                "sheet": payload.get("sheet"),
                "expectation": expected,
            }
        ]

    raw_panels = payload.get("panels")
    if not isinstance(raw_panels, list) or len(raw_panels) != case.panel_count:
        raise RenderabilityError(f"case-panels-invalid:{case.case_id}")
    panels_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_panel in raw_panels:
        if not isinstance(raw_panel, Mapping):
            raise RenderabilityError(f"case-panel-invalid:{case.case_id}")
        panel_id = raw_panel.get("id")
        if not isinstance(panel_id, str) or not panel_id or panel_id in panels_by_id:
            raise RenderabilityError(f"case-panel-id-invalid:{case.case_id}")
        panels_by_id[panel_id] = raw_panel
    if set(panels_by_id) != set(expectations_by_id):
        raise RenderabilityError(f"case-expectation-panel-set-mismatch:{case.case_id}")
    return [
        {
            "panel_id": panel_id,
            "data_path": panels_by_id[panel_id].get("data_path"),
            "data_sha256": panels_by_id[panel_id].get("data_sha256"),
            "sheet": panels_by_id[panel_id].get("sheet"),
            "expectation": expectations_by_id[panel_id],
        }
        for panel_id in sorted(panels_by_id)
    ]


def _load_table(path: Path, sheet: Any, *, case_id: str, panel_id: str) -> pd.DataFrame:
    if sheet is not None and (not isinstance(sheet, str) or not sheet.strip()):
        raise RenderabilityError(f"source-sheet-invalid:{case_id}:{panel_id}")
    suffix = path.suffix.casefold()
    try:
        if suffix in {".xls", ".xlsx", ".xlsm"}:
            frame = pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
        elif suffix == ".csv":
            if sheet is not None:
                raise RenderabilityError(
                    f"csv-source-cannot-select-sheet:{case_id}:{panel_id}"
                )
            frame = pd.read_csv(path)
        else:
            raise RenderabilityError(
                f"source-format-unsupported:{case_id}:{panel_id}:{suffix}"
            )
    except RenderabilityError:
        raise
    except Exception as exc:
        raise RenderabilityError(
            f"source-table-unreadable:{case_id}:{panel_id}"
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise RenderabilityError(f"source-table-not-dataframe:{case_id}:{panel_id}")
    if frame.columns.has_duplicates:
        raise RenderabilityError(f"source-columns-invalid:{case_id}:{panel_id}")
    return frame


def _filtered_frame(
    frame: pd.DataFrame,
    series: Mapping[str, Any],
    *,
    case_id: str,
    panel_id: str,
    series_id: str,
) -> pd.DataFrame:
    filtered = frame
    where = series.get("where") or {}
    if not isinstance(where, Mapping):
        raise RenderabilityError(
            f"series-filter-invalid:{case_id}:{panel_id}:{series_id}"
        )
    for column, expected in where.items():
        if column not in filtered:
            raise RenderabilityError(
                f"series-filter-column-missing:{case_id}:{panel_id}:"
                f"{series_id}:{column}"
            )
        if isinstance(expected, list):
            filtered = filtered[filtered[column].isin(expected)]
        else:
            filtered = filtered[filtered[column] == expected]
    sort_by = series.get("sort_by") or []
    if not isinstance(sort_by, list) or any(
        not isinstance(column, str) for column in sort_by
    ):
        raise RenderabilityError(
            f"series-sort-invalid:{case_id}:{panel_id}:{series_id}"
        )
    missing = [column for column in sort_by if column not in filtered]
    if missing:
        raise RenderabilityError(
            f"series-sort-column-missing:{case_id}:{panel_id}:"
            f"{series_id}:{','.join(missing)}"
        )
    if sort_by:
        filtered = filtered.sort_values(sort_by, kind="mergesort")
    return filtered


def _resolve_data_ref(
    ref: Any,
    frame: pd.DataFrame,
    *,
    case_id: str,
    panel_id: str,
    series_id: str,
) -> list[Any]:
    if isinstance(ref, str):
        if ref not in frame:
            raise RenderabilityError(
                f"series-data-column-missing:{case_id}:{panel_id}:{series_id}:{ref}"
            )
        return frame[ref].tolist()
    if isinstance(ref, list):
        return list(ref)
    if isinstance(ref, Mapping):
        if "column" in ref:
            return _resolve_data_ref(
                ref["column"],
                frame,
                case_id=case_id,
                panel_id=panel_id,
                series_id=series_id,
            )
        values = ref.get("values")
        if isinstance(values, list):
            return list(values)
    raise RenderabilityError(
        f"series-data-ref-invalid:{case_id}:{panel_id}:{series_id}"
    )


def _value_identity(value: Any) -> str:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return "missing:null"
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        encoded = _canonical_json(value)
    except (TypeError, ValueError):
        encoded = _canonical_json(str(value))
    return f"{type(value).__name__}:{encoded}"


def _audit_panel(
    *,
    case: DatasetCase,
    panel_source: Mapping[str, Any],
    manifest_path: Path,
    manifest_data_root: Path,
    runtime_repo_root: Path,
) -> dict[str, Any]:
    panel_id = str(panel_source["panel_id"])
    path_value = panel_source.get("data_path")
    data_sha256 = panel_source.get("data_sha256")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(data_sha256, str)
        or not _SHA256_RE.fullmatch(data_sha256)
    ):
        raise RenderabilityError(f"source-binding-invalid:{case.case_id}:{panel_id}")
    try:
        source_path = resolve_case_data_path(
            path_value,
            manifest_path=manifest_path,
            manifest_data_root=manifest_data_root,
            runtime_repo_root=runtime_repo_root,
        )
    except ManifestError as exc:
        raise RenderabilityError(
            f"source-path-invalid:{case.case_id}:{panel_id}"
        ) from exc
    frame = _load_table(
        source_path,
        panel_source.get("sheet"),
        case_id=case.case_id,
        panel_id=panel_id,
    )
    expectation = panel_source.get("expectation")
    if not isinstance(expectation, Mapping):
        raise RenderabilityError(f"panel-expectation-invalid:{case.case_id}:{panel_id}")
    raw_series = expectation.get("series")
    if not isinstance(raw_series, list) or not raw_series:
        raise RenderabilityError(f"panel-series-empty:{case.case_id}:{panel_id}")

    series_rows: list[dict[str, Any]] = []
    categorical_x_identities: set[str] = set()
    total_expected_points = 0
    bar_expected_points = 0
    bar_series_count = 0
    for raw in raw_series:
        if not isinstance(raw, Mapping):
            raise RenderabilityError(f"series-invalid:{case.case_id}:{panel_id}")
        series_id = str(raw.get("series_id") or "")
        kind = str(raw.get("kind") or "").casefold()
        filtered = _filtered_frame(
            frame,
            raw,
            case_id=case.case_id,
            panel_id=panel_id,
            series_id=series_id,
        )
        resolved: dict[str, list[Any]] = {}
        for field in ("x", "y", "value"):
            if field in raw:
                resolved[field] = _resolve_data_ref(
                    raw[field],
                    filtered,
                    case_id=case.case_id,
                    panel_id=panel_id,
                    series_id=series_id,
                )
        expected_points = max(
            (len(values) for values in resolved.values()),
            default=0,
        )
        total_expected_points += expected_points
        unique_x = (
            len({_value_identity(value) for value in resolved["x"]})
            if "x" in resolved
            else None
        )
        if kind == "bar":
            bar_series_count += 1
            bar_expected_points += expected_points
            if "x" not in resolved:
                raise RenderabilityError(
                    f"bar-series-x-binding-missing:{case.case_id}:"
                    f"{panel_id}:{series_id}"
                )
            categorical_x_identities.update(
                _value_identity(value) for value in resolved["x"]
            )
        series_rows.append(
            {
                "series_id": series_id,
                "kind": kind,
                "expected_points": expected_points,
                "unique_x": unique_x,
            }
        )

    unique_categorical_x = len(categorical_x_identities) if bar_series_count else None
    if bar_series_count and unique_categorical_x > MAX_BAR_CATEGORICAL_X:
        decision = "rejected"
        reason = "bar_categorical_x_exceeds_200"
    elif bar_series_count:
        decision = "accepted"
        reason = "bar_categorical_x_at_or_below_200"
    else:
        decision = "accepted"
        reason = "non_bar_or_column_no_cardinality_cap"
    return {
        "case_id": case.case_id,
        "candidate_id": case.payload.get("candidate_id"),
        "doi": _normalize_doi(case.payload.get("doi")),
        "split": case.split,
        "panel_id": panel_id,
        "source_table": path_value,
        "source_sha256": data_sha256,
        "source_sheet": panel_source.get("sheet"),
        "source_rows": int(frame.shape[0]),
        "source_columns": int(frame.shape[1]),
        "series_count": len(series_rows),
        "bar_series_count": bar_series_count,
        "unique_categorical_x": unique_categorical_x,
        "expected_points": total_expected_points,
        "bar_expected_points": bar_expected_points,
        "decision": decision,
        "reason": reason,
        "series": series_rows,
    }


def _audit_cases(
    cases: Sequence[DatasetCase],
    *,
    manifest_path: Path,
    manifest_data_root: Path,
    runtime_repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    panel_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    rejected: set[str] = set()
    for case in sorted(cases, key=lambda item: item.case_id):
        if case.split not in _SPLITS:
            raise RenderabilityError(
                f"case-split-unexpected:{case.case_id}:{case.split!r}"
            )
        panels = [
            _audit_panel(
                case=case,
                panel_source=source,
                manifest_path=manifest_path,
                manifest_data_root=manifest_data_root,
                runtime_repo_root=runtime_repo_root,
            )
            for source in _case_panel_sources(case)
        ]
        failing_panels = sorted(
            row["panel_id"] for row in panels if row["decision"] == "rejected"
        )
        if failing_panels:
            decision = "rejected"
            reason = "one_or_more_panels_failed:" + ",".join(failing_panels)
            rejected.add(case.case_id)
        else:
            decision = "accepted"
            reason = "all_panels_passed"
        for row in panels:
            row["case_decision"] = decision
            row["case_reason"] = reason
        panel_rows.extend(panels)
        case_rows.append(
            {
                "case_id": case.case_id,
                "candidate_id": case.payload.get("candidate_id"),
                "doi": _normalize_doi(case.payload.get("doi")),
                "split": case.split,
                "panel_count": case.panel_count,
                "panel_ids": [row["panel_id"] for row in panels],
                "decision": decision,
                "reason": reason,
            }
        )
    return case_rows, panel_rows, rejected


def _deterministic_retained_cases(
    cases: Sequence[DatasetCase],
    rejected: set[str],
) -> list[dict[str, Any]]:
    retained = [
        deepcopy(case.payload) for case in cases if case.case_id not in rejected
    ]
    if not retained:
        raise RenderabilityError("policy-excluded-all-cases")
    doi_splits: dict[str, str] = {}
    for case in retained:
        doi = _normalize_doi(case.get("doi"))
        split = str(case.get("split") or "")
        previous = doi_splits.setdefault(doi, split)
        if previous != split:
            raise RenderabilityError(f"doi-crosses-splits:{doi}")
    return sorted(
        retained,
        key=lambda case: (
            _normalize_doi(case.get("doi")),
            str(case.get("case_id")),
        ),
    )


def _build_payloads(
    *,
    parent: Mapping[str, Any],
    parent_manifest_sha256: str,
    cases: Sequence[DatasetCase],
    code_commit: str,
    manifest_path: Path,
    manifest_data_root: Path,
    runtime_repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    case_rows, panel_rows, rejected = _audit_cases(
        cases,
        manifest_path=manifest_path,
        manifest_data_root=manifest_data_root,
        runtime_repo_root=runtime_repo_root,
    )
    retained_cases = _deterministic_retained_cases(cases, rejected)
    decisions = Counter(row["decision"] for row in case_rows)
    split_inputs = Counter(str(case.split) for case in cases)
    split_outputs = Counter(str(case.get("split")) for case in retained_cases)
    input_doi_splits = {
        _normalize_doi(case.payload.get("doi")): str(case.split) for case in cases
    }
    output_doi_splits = {
        _normalize_doi(case.get("doi")): str(case.get("split"))
        for case in retained_cases
    }
    input_paper_splits = Counter(input_doi_splits.values())
    output_paper_splits = Counter(output_doi_splits.values())

    audit_unsealed = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": code_commit,
        "code_dirty": False,
        "parent_manifest_sha256": parent_manifest_sha256,
        "policy": deepcopy(RENDERABILITY_POLICY),
        "policy_hash": POLICY_HASH,
        "case_rows": case_rows,
        "panel_rows": panel_rows,
        "counts": {
            "input_cases": len(cases),
            "accepted_cases": decisions.get("accepted", 0),
            "rejected_cases": decisions.get("rejected", 0),
            "panel_rows": len(panel_rows),
        },
    }
    audit = _seal(audit_unsealed, "audit_hash")
    audit_bytes = _pretty_json_bytes(audit)

    parent_provenance = parent.get("provenance")
    if not isinstance(parent_provenance, Mapping):
        raise RenderabilityError("parent-provenance-invalid")
    source_binding = deepcopy(parent_provenance.get("source_binding"))
    source_binding_hash = parent_provenance.get("source_binding_hash")
    if (
        not isinstance(source_binding, dict)
        or not isinstance(source_binding_hash, str)
        or _sha256_json(source_binding) != source_binding_hash
    ):
        raise RenderabilityError("parent-source-binding-invalid")
    provenance = deepcopy(dict(parent_provenance))
    provenance.update(
        {
            "code_commit": code_commit,
            "code_dirty": False,
            "parent_manifest_sha256": parent_manifest_sha256,
            "renderability_policy": deepcopy(RENDERABILITY_POLICY),
            "renderability_policy_hash": POLICY_HASH,
            "renderability_audit_path": "renderability_audit.json",
            "renderability_audit_sha256": _sha256_bytes(audit_bytes),
            "renderability_audit_hash": audit["audit_hash"],
        }
    )
    if (
        _canonical_json(provenance["source_binding"])
        != _canonical_json(parent_provenance["source_binding"])
        or provenance["source_binding_hash"] != source_binding_hash
    ):
        raise RenderabilityError("source-binding-not-preserved")
    manifest_unsealed = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "cases": retained_cases,
    }
    manifest = _seal(manifest_unsealed, "manifest_hash")
    manifest_bytes = _pretty_json_bytes(manifest)

    summary_unsealed = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": code_commit,
        "code_dirty": False,
        "parent_manifest_sha256": parent_manifest_sha256,
        "source_binding_hash": source_binding_hash,
        "policy_hash": POLICY_HASH,
        "input_cases": len(cases),
        "accepted_cases": decisions.get("accepted", 0),
        "rejected_cases": decisions.get("rejected", 0),
        "input_unique_dois": len(input_doi_splits),
        "output_unique_dois": len(output_doi_splits),
        "input_case_split_counts": {
            split: split_inputs.get(split, 0) for split in _SPLITS
        },
        "output_case_split_counts": {
            split: split_outputs.get(split, 0) for split in _SPLITS
        },
        "input_paper_split_counts": {
            split: input_paper_splits.get(split, 0) for split in _SPLITS
        },
        "output_paper_split_counts": {
            split: output_paper_splits.get(split, 0) for split in _SPLITS
        },
        "rejected_case_ids": sorted(rejected),
        "benchmark_manifest": "benchmark_manifest.json",
        "benchmark_manifest_sha256": _sha256_bytes(manifest_bytes),
        "benchmark_manifest_hash": manifest["manifest_hash"],
        "renderability_audit": "renderability_audit.json",
        "renderability_audit_sha256": _sha256_bytes(audit_bytes),
        "renderability_audit_hash": audit["audit_hash"],
    }
    summary = _seal(summary_unsealed, "summary_hash")
    return manifest, audit, summary


def validate_renderability_outputs(
    output_root: str | Path,
    *,
    manifest_data_root: str | Path,
    runtime_repo_root: str | Path,
) -> dict[str, Any]:
    """Validate all derivative seals, links, and sealed source bindings."""

    _validate_policy_constant()
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RenderabilityError(f"output-root-invalid:{root}")
    actual_names = sorted(path.name for path in root.iterdir())
    if actual_names != sorted(_OUTPUT_NAMES):
        raise RenderabilityError("output-files-unexpected")
    manifest_path = root / "benchmark_manifest.json"
    audit_path = root / "renderability_audit.json"
    summary_path = root / "summary.json"
    manifest = _load_json_object(manifest_path, "derived-manifest")
    audit = _load_json_object(audit_path, "renderability-audit")
    summary = _load_json_object(summary_path, "renderability-summary")
    manifest_hash = _verify_seal(manifest, "manifest_hash", "manifest")
    audit_hash = _verify_seal(audit, "audit_hash", "audit")
    _verify_seal(summary, "summary_hash", "summary")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RenderabilityError("derived-provenance-invalid")
    expected_links = {
        "parent_manifest_sha256": summary.get("parent_manifest_sha256"),
        "renderability_policy_hash": POLICY_HASH,
        "renderability_audit_path": "renderability_audit.json",
        "renderability_audit_sha256": sha256_file(audit_path),
        "renderability_audit_hash": audit_hash,
    }
    if any(provenance.get(key) != value for key, value in expected_links.items()):
        raise RenderabilityError("derived-provenance-link-invalid")
    if (
        provenance.get("renderability_policy") != RENDERABILITY_POLICY
        or audit.get("policy") != RENDERABILITY_POLICY
        or audit.get("policy_hash") != POLICY_HASH
        or summary.get("policy_hash") != POLICY_HASH
        or audit.get("parent_manifest_sha256") != summary.get("parent_manifest_sha256")
        or audit.get("code_commit") != summary.get("code_commit")
        or provenance.get("code_commit") != summary.get("code_commit")
        or audit.get("code_dirty") is not False
        or provenance.get("code_dirty") is not False
        or summary.get("code_dirty") is not False
        or provenance.get("source_binding_hash") != summary.get("source_binding_hash")
        or summary.get("benchmark_manifest_sha256") != sha256_file(manifest_path)
        or summary.get("benchmark_manifest_hash") != manifest_hash
        or summary.get("renderability_audit_sha256") != sha256_file(audit_path)
        or summary.get("renderability_audit_hash") != audit_hash
    ):
        raise RenderabilityError("derived-artifact-link-invalid")

    manifest_root = Path(manifest_data_root).expanduser().resolve()
    runtime_root = Path(runtime_repo_root).expanduser().resolve()
    try:
        cases = load_dataset_manifest(
            manifest_path,
            dataset_mode="sealed_benchmark",
            manifest_data_root=manifest_root,
            runtime_repo_root=runtime_root,
        )
        for case in cases:
            verify_case_data_files(
                case,
                manifest_path=manifest_path,
                manifest_data_root=manifest_root,
                runtime_repo_root=runtime_root,
            )
    except ManifestError as exc:
        raise RenderabilityError(f"derived-manifest-validation:{exc}") from exc
    if len(cases) != summary.get("accepted_cases"):
        raise RenderabilityError("derived-case-count-mismatch")
    return summary


def build_renderability_derivative(
    parent_manifest: str | Path,
    output_root: str | Path,
    *,
    manifest_data_root: str | Path,
    runtime_repo_root: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Filter a sealed parent manifest and write a provenance-bound derivative."""

    _validate_policy_constant()
    parent_path = Path(parent_manifest).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    manifest_root = Path(manifest_data_root).expanduser()
    runtime_root = Path(runtime_repo_root).expanduser()
    git_root = Path(repo_root).expanduser().resolve()
    if not parent_path.is_file() or parent_path.is_symlink():
        raise RenderabilityError(f"parent-manifest-invalid:{parent_path}")
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise RenderabilityError(f"output-not-empty:{output}")

    code_commit, code_dirty = _git_state(git_root)
    if code_dirty:
        raise RenderabilityError("git-worktree-dirty")
    parent, cases, parent_hash = _load_parent(
        parent_path,
        manifest_data_root=manifest_root,
        runtime_repo_root=runtime_root,
    )
    manifest, audit, summary = _build_payloads(
        parent=parent,
        parent_manifest_sha256=parent_hash,
        cases=cases,
        code_commit=code_commit,
        manifest_path=parent_path,
        manifest_data_root=manifest_root,
        runtime_repo_root=runtime_root,
    )

    _, revalidated_cases, revalidated_hash = _load_parent(
        parent_path,
        manifest_data_root=manifest_root,
        runtime_repo_root=runtime_root,
    )
    if revalidated_hash != parent_hash or [
        case.case_id for case in revalidated_cases
    ] != [case.case_id for case in cases]:
        raise RenderabilityError("parent-input-changed-during-build")
    final_commit, final_dirty = _git_state(git_root)
    if final_commit != code_commit or final_dirty:
        raise RenderabilityError("git-state-changed-during-build")

    payloads = {
        "benchmark_manifest.json": _pretty_json_bytes(manifest),
        "renderability_audit.json": _pretty_json_bytes(audit),
        "summary.json": _pretty_json_bytes(summary),
    }
    created_root = not output.exists()
    output.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    try:
        for name in _OUTPUT_NAMES:
            path = output / name
            try:
                with path.open("xb") as handle:
                    handle.write(payloads[name])
            except FileExistsError as exc:
                raise RenderabilityError(f"output-file-exists:{path}") from exc
            except OSError as exc:
                raise RenderabilityError(f"output-write-failed:{path}") from exc
            written_paths.append(path)
        validated = validate_renderability_outputs(
            output,
            manifest_data_root=manifest_root,
            runtime_repo_root=runtime_root,
        )
        _, written_cases, written_parent_hash = _load_parent(
            parent_path,
            manifest_data_root=manifest_root,
            runtime_repo_root=runtime_root,
        )
        if written_parent_hash != parent_hash or [
            case.case_id for case in written_cases
        ] != [case.case_id for case in cases]:
            raise RenderabilityError("parent-input-changed-during-write")
        written_commit, written_dirty = _git_state(git_root)
        if written_commit != code_commit or written_dirty:
            raise RenderabilityError("output-must-not-dirty-provenance-repository")
    except Exception:
        for path in written_paths:
            if path.is_file() and not path.is_symlink():
                path.unlink()
        if created_root:
            try:
                output.rmdir()
            except OSError:
                pass
        raise
    return validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an outcome-independent, renderability-filtered sealed "
            "benchmark derivative"
        )
    )
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-data-root", type=Path, required=True)
    parser.add_argument("--runtime-repo-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = build_renderability_derivative(
            args.parent_manifest,
            args.output_root,
            manifest_data_root=args.manifest_data_root,
            runtime_repo_root=args.runtime_repo_root,
            repo_root=args.repo_root,
        )
    except RenderabilityError as exc:
        parser.exit(2, f"renderability-error: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic, fail-closed experiment-case proposals for curated candidates."""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None

try:
    import pandas as pd
except ImportError:
    pd = None

from .provenance import sha256_file


SCHEMA_VERSION = "1.0"
EVALUATION_SCHEMA_VERSION = "1.1.0"
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_COLUMNS = 64
TEMPORAL_COLUMN_PATTERN = re.compile(
    r"(?i)(?:^|[^a-z])"
    r"(?:time|timepoint|date|datetime|timestamp|year|month|day)"
    r"(?:$|[^a-z])"
)
PAREN_UNIT_PATTERN = re.compile(r"\(([^()]{1,32})\)\s*$")
BRACKET_UNIT_PATTERN = re.compile(r"\[([^\[\]]{1,32})\]\s*$")
SUFFIX_UNIT_PATTERN = re.compile(
    r"(?i)(?:_|-)"
    r"(ms|s|min|h|hz|kg|g|mg|ug|µg|m|cm|mm|um|µm|nm|l|ml|mol|mmol|%)$"
)


class ProposalRejected(ValueError):
    """A deterministic rejection with one or more machine-readable reasons."""

    def __init__(self, *reasons: str, detail: str | None = None) -> None:
        self.reasons = tuple(sorted(set(reasons or ("proposal-rejected",))))
        self.detail = detail
        super().__init__(",".join(self.reasons))


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    numeric: bool
    explicit_temporal: bool
    temporal_valid: bool
    categorical: bool
    monotonic_numeric: bool
    non_null_count: int
    unique_count: int


@dataclass(frozen=True)
class TableAnalysis:
    x: str
    y: tuple[str, ...]
    x_mode: str
    chart_family: str
    units: dict[str, str]
    rows: int
    columns: int
    non_null_y: dict[str, int]


def _require_dependencies() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if load_workbook is None:
        missing.append("openpyxl")
    if missing:
        raise ProposalRejected(
            "proposal-dependency-missing",
            detail=",".join(missing),
        )


def _validate_headers(values: Iterable[Any]) -> list[str]:
    headers: list[str] = []
    normalized: list[str] = []
    for value in values:
        if value is None or not str(value).strip():
            raise ProposalRejected("column-name-missing")
        name = str(value)
        headers.append(name)
        normalized.append(name.strip().casefold())
    if len(set(normalized)) != len(normalized):
        raise ProposalRejected("column-name-duplicate")
    return headers


def _read_csv(
    path: Path,
    *,
    max_rows: int,
    max_columns: int,
) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
    except StopIteration as exc:
        raise ProposalRejected("table-empty") from exc
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ProposalRejected(
            "csv-read-error",
            detail=f"{type(exc).__name__}:{exc}",
        ) from exc
    if len(header) > max_columns:
        raise ProposalRejected("table-column-limit")
    _validate_headers(header)
    try:
        frame = pd.read_csv(path, nrows=max_rows + 1)
    except Exception as exc:
        raise ProposalRejected(
            "csv-read-error",
            detail=f"{type(exc).__name__}:{exc}",
        ) from exc
    if len(frame.index) > max_rows:
        raise ProposalRejected("table-row-limit")
    if len(frame.columns) > max_columns:
        raise ProposalRejected("table-column-limit")
    return frame


def _read_xlsx(
    path: Path,
    *,
    sheet_name: str | None,
    max_rows: int,
    max_columns: int,
) -> Any:
    if not sheet_name:
        raise ProposalRejected("xlsx-sheet-required")
    workbook = None
    try:
        workbook = load_workbook(
            filename=path,
            read_only=True,
            data_only=True,
            keep_links=False,
        )
        if sheet_name not in workbook.sheetnames:
            raise ProposalRejected("xlsx-sheet-missing")
        worksheet = workbook[sheet_name]
        row_iter = worksheet.iter_rows(
            min_row=1,
            max_col=max_columns + 1,
            values_only=True,
        )
        try:
            raw_header = next(row_iter)
        except StopIteration as exc:
            raise ProposalRejected("table-empty") from exc
        if len(raw_header) > max_columns and any(
            value is not None for value in raw_header[max_columns:]
        ):
            raise ProposalRejected("table-column-limit")
        header_values = list(raw_header[:max_columns])
        while header_values and header_values[-1] is None:
            header_values.pop()
        headers = _validate_headers(header_values)
        rows: list[list[Any]] = []
        for row_index, raw_row in enumerate(row_iter, 1):
            if row_index > max_rows:
                raise ProposalRejected("table-row-limit")
            if len(raw_row) > len(headers) and any(
                value is not None for value in raw_row[len(headers) :]
            ):
                raise ProposalRejected("table-ragged-extra-columns")
            rows.append(list(raw_row[: len(headers)]))
        return pd.DataFrame(rows, columns=headers)
    except ProposalRejected:
        raise
    except Exception as exc:
        raise ProposalRejected(
            "xlsx-read-error",
            detail=f"{type(exc).__name__}:{exc}",
        ) from exc
    finally:
        if workbook is not None:
            workbook.close()


def read_candidate_table(
    candidate: dict[str, Any],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> Any:
    _require_dependencies()
    source = candidate.get("source_table") or {}
    path_value = source.get("path")
    if not path_value:
        raise ProposalRejected("source-table-path-missing")
    path = Path(str(path_value))
    if path.is_symlink():
        raise ProposalRejected("source-table-symlink")
    if not path.is_file():
        raise ProposalRejected("source-table-missing")
    size = path.stat().st_size
    if size > max_file_bytes:
        raise ProposalRejected("source-table-file-size-limit")
    if source.get("size_bytes") is not None and int(source["size_bytes"]) != size:
        raise ProposalRejected("source-table-size-mismatch")
    expected_hash = source.get("sha256")
    if not expected_hash or sha256_file(path) != expected_hash:
        raise ProposalRejected("source-table-checksum-mismatch")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        frame = _read_csv(
            path,
            max_rows=max_rows,
            max_columns=max_columns,
        )
    elif suffix == ".xlsx":
        frame = _read_xlsx(
            path,
            sheet_name=source.get("sheet_name"),
            max_rows=max_rows,
            max_columns=max_columns,
        )
    else:
        raise ProposalRejected("source-table-format-unsupported")
    if len(frame.columns) < 2 or len(frame.columns) > 6:
        raise ProposalRejected("table-column-count-not-simple")
    frame = frame.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    if frame.empty:
        raise ProposalRejected("table-empty")
    empty_columns = [
        str(column) for column in frame.columns if frame[column].dropna().empty
    ]
    if empty_columns:
        raise ProposalRejected("table-empty-column")
    return frame


def _is_bool_series(values: Any) -> bool:
    return bool(values) and all(isinstance(value, bool) for value in values)


def _column_profile(name: str, series: Any) -> ColumnProfile:
    nonempty = series.dropna()
    values = nonempty.tolist()
    explicit_temporal = bool(TEMPORAL_COLUMN_PATTERN.search(name))
    numeric_values = pd.to_numeric(nonempty, errors="coerce")
    numeric = (
        len(nonempty) > 0
        and not _is_bool_series(values)
        and numeric_values.notna().all()
    )
    temporal_valid = False
    if explicit_temporal and len(nonempty) > 0:
        if numeric:
            temporal_valid = True
        elif all(isinstance(value, (date, datetime)) for value in values):
            temporal_valid = True
        else:
            parsed = pd.to_datetime(
                nonempty.astype(str),
                errors="coerce",
                format="mixed",
            )
            temporal_valid = parsed.notna().all()
    monotonic = False
    if numeric and len(numeric_values) >= 2 and numeric_values.nunique() >= 2:
        monotonic = bool(
            numeric_values.is_monotonic_increasing
            or numeric_values.is_monotonic_decreasing
        )
    categorical = (
        len(nonempty) > 0
        and not numeric
        and not explicit_temporal
    )
    return ColumnProfile(
        name=name,
        numeric=numeric,
        explicit_temporal=explicit_temporal,
        temporal_valid=temporal_valid,
        categorical=categorical,
        monotonic_numeric=monotonic,
        non_null_count=len(nonempty),
        unique_count=int(nonempty.nunique(dropna=True)),
    )


def _extract_unit(column: str) -> str | None:
    for pattern in (PAREN_UNIT_PATTERN, BRACKET_UNIT_PATTERN):
        match = pattern.search(column)
        if match and match.group(1).strip():
            return match.group(1).strip()
    match = SUFFIX_UNIT_PATTERN.search(column)
    return match.group(1) if match else None


def analyze_table(frame: Any) -> TableAnalysis:
    headers = _validate_headers(frame.columns)
    profiles = {
        name: _column_profile(name, frame[name])
        for name in headers
    }
    invalid_temporal = [
        name
        for name, profile in profiles.items()
        if profile.explicit_temporal and not profile.temporal_valid
    ]
    if invalid_temporal:
        raise ProposalRejected("explicit-temporal-column-invalid")

    temporal = [
        profile for profile in profiles.values() if profile.temporal_valid
    ]
    categorical = [
        profile for profile in profiles.values() if profile.categorical
    ]
    if len(temporal) > 1:
        raise ProposalRejected("x-column-ambiguous-temporal")
    if temporal:
        x_profile = temporal[0]
        x_mode = "temporal"
        chart_family = "line"
    elif len(categorical) > 1:
        raise ProposalRejected("x-column-ambiguous-categorical")
    elif len(categorical) == 1:
        x_profile = categorical[0]
        x_mode = "categorical"
        chart_family = "bar"
    else:
        monotonic = [
            profile
            for profile in profiles.values()
            if profile.monotonic_numeric
        ]
        if len(monotonic) != 1:
            raise ProposalRejected(
                "x-column-not-found"
                if not monotonic
                else "x-column-ambiguous-monotonic-numeric"
            )
        x_profile = monotonic[0]
        x_mode = "linear"
        chart_family = "line"

    remaining = [
        profile
        for name, profile in profiles.items()
        if name != x_profile.name
    ]
    if any(not profile.numeric for profile in remaining):
        raise ProposalRejected("non-numeric-y-column")
    if len(remaining) < 1:
        raise ProposalRejected("numeric-y-column-missing")
    if len(remaining) > 4:
        raise ProposalRejected("too-many-y-columns")
    units = {
        name: unit
        for name in headers
        if (unit := _extract_unit(name)) is not None
    }
    return TableAnalysis(
        x=x_profile.name,
        y=tuple(profile.name for profile in remaining),
        x_mode=x_mode,
        chart_family=chart_family,
        units=units,
        rows=len(frame.index),
        columns=len(headers),
        non_null_y={
            profile.name: profile.non_null_count for profile in remaining
        },
    )


def _single_expectation(
    candidate: dict[str, Any],
    analysis: TableAnalysis,
) -> dict[str, Any]:
    panel_ids = candidate.get("panel_ids") or []
    if len(panel_ids) != 1:
        raise ProposalRejected("candidate-panel-id-not-unique")
    panel_id = str(panel_ids[0])
    series = []
    for y_column in analysis.y:
        descriptor: dict[str, Any] = {
            "series_id": y_column,
            "kind": analysis.chart_family,
            "x": analysis.x,
            "y": y_column,
        }
        series.append(descriptor)
    panel: dict[str, Any] = {
        "panel_id": panel_id,
        "axis_index": 0,
        "x_scale": "linear",
        "series": series,
    }
    if analysis.x in analysis.units:
        panel["x_unit"] = analysis.units[analysis.x]
    y_units = {
        analysis.units[y_column]
        for y_column in analysis.y
        if y_column in analysis.units
    }
    if (
        len(y_units) == 1
        and all(column in analysis.units for column in analysis.y)
    ):
        panel["y_unit"] = next(iter(y_units))
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "panels": [panel],
        "panel_groups": [],
    }


def _user_goal(candidate: dict[str, Any], analysis: TableAnalysis) -> str:
    figure_no = candidate.get("figure_no")
    panel_id = str((candidate.get("panel_ids") or ["?"])[0]).upper()
    verb = "line chart" if analysis.chart_family == "line" else "bar chart"
    y_names = ", ".join(f"`{name}`" for name in analysis.y)
    return (
        f"Create a {verb} for Figure {figure_no}{panel_id} showing "
        f"{y_names} against `{analysis.x}`."
    )


def propose_single_candidate(
    candidate: dict[str, Any],
    *,
    input_candidates_sha256: str,
    code_commit: str,
    max_file_bytes: int,
    max_rows: int,
    max_columns: int,
) -> dict[str, Any]:
    frame = read_candidate_table(
        candidate,
        max_file_bytes=max_file_bytes,
        max_rows=max_rows,
        max_columns=max_columns,
    )
    analysis = analyze_table(frame)
    expectation = _single_expectation(candidate, analysis)
    source = candidate.get("source_table") or {}
    panel_id = str(candidate["panel_ids"][0])
    user_goal = _user_goal(candidate, analysis)
    intent = {
        "x": analysis.x,
        "y": analysis.y[0],
        "series": list(analysis.y),
        "x_scale": analysis.x_mode,
        "units": analysis.units,
    }
    experiment_case = {
        "case_id": candidate.get("candidate_id"),
        "panel_count": 1,
        "split": None,
        "data_path": source.get("path"),
        "sheet": source.get("sheet_name"),
        "panel_id": panel_id,
        "panels": [
            {
                "panel_id": panel_id,
                "data_path": source.get("path"),
                "sheet": source.get("sheet_name"),
                "user_goal": user_goal,
            }
        ],
        "user_goal": user_goal,
        "chart_family": analysis.chart_family,
        "intent": intent,
        "evaluation_expectation": expectation,
    }
    proposal = deepcopy(candidate)
    proposal.update(
        {
            "proposal_type": "single_panel",
            "curation_status": "proposed",
            "eligible_for_experiment": False,
            "eligibility_reasons": ["external-validation-required"],
            "experiment_case": experiment_case,
            "proposal_analysis": {
                "rows": analysis.rows,
                "columns": analysis.columns,
                "x": analysis.x,
                "y": list(analysis.y),
                "x_mode": analysis.x_mode,
                "chart_family": analysis.chart_family,
                "units": analysis.units,
                "non_null_y": analysis.non_null_y,
            },
            "input_candidates_sha256": input_candidates_sha256,
            "code_commit": code_commit,
        }
    )
    return proposal


def _multi_panel_proposals(
    singles: list[dict[str, Any]],
    *,
    input_candidates_sha256: str,
    code_commit: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for proposal in singles:
        doi = str(proposal.get("doi") or "")
        figure_no = proposal.get("figure_no")
        if doi and isinstance(figure_no, int):
            grouped.setdefault((doi, figure_no), []).append(proposal)

    results: list[dict[str, Any]] = []
    for (doi, figure_no), group in sorted(grouped.items()):
        by_panel: dict[str, list[dict[str, Any]]] = {}
        for proposal in group:
            panel_id = str(proposal["panel_ids"][0])
            by_panel.setdefault(panel_id, []).append(proposal)
        clear = [
            proposals[0]
            for panel_id, proposals in sorted(by_panel.items())
            if len(proposals) == 1
        ]
        if len(clear) < 2:
            continue
        source_ids = sorted(str(item["candidate_id"]) for item in clear)
        digest = hashlib.sha256(
            "|".join(source_ids).encode("utf-8")
        ).hexdigest()[:12]
        article_id = doi.split("/", 1)[-1]
        case_id = f"multi-{article_id}-figure{figure_no}-{digest}"
        panels = []
        expectation_panels = []
        x_scales: list[str] = []
        x_units: list[str | None] = []
        series_sets: list[set[str]] = []
        for proposal in clear:
            case = proposal["experiment_case"]
            panel_id = str(proposal["panel_ids"][0])
            panels.append(
                {
                    "id": panel_id,
                    "data_path": case["data_path"],
                    "sheet": case.get("sheet"),
                    "user_goal": case["user_goal"],
                    "chart_family": case["chart_family"],
                    "intent": case["intent"],
                }
            )
            panel_expectation = case["evaluation_expectation"]["panels"][0]
            expectation_panels.append(panel_expectation)
            x_scales.append(str(panel_expectation.get("x_scale") or ""))
            x_units.append(panel_expectation.get("x_unit"))
            series_sets.append(
                {str(series["series_id"]) for series in panel_expectation["series"]}
            )

        checks: dict[str, Any] = {}
        cohesion: dict[str, Any] = {
            "group_id": "all_panels",
            "panels": [panel["id"] for panel in panels],
            "checks": checks,
        }
        if len(set(x_scales)) == 1:
            checks["shared_x_scale"] = True
        if all(unit is not None for unit in x_units) and len(set(x_units)) == 1:
            checks["shared_x_unit"] = True
        shared_series = sorted(set.intersection(*series_sets)) if series_sets else []
        if shared_series:
            cohesion["series"] = shared_series
            checks["palette_consistent"] = True
        panel_groups = [cohesion] if checks else []
        expectation = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "panels": expectation_panels,
            "panel_groups": panel_groups,
        }
        experiment_case = {
            "case_id": case_id,
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
                        "panel_id": panel["id"],
                        **panel["intent"],
                    }
                    for panel in panels
                ]
            },
            "evaluation_expectation": expectation,
        }
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": case_id,
                "proposal_type": "multi_panel",
                "source_candidate_ids": source_ids,
                "doi": doi,
                "figure_no": figure_no,
                "panel_ids": [panel["id"] for panel in panels],
                "curation_status": "proposed",
                "eligible_for_experiment": False,
                "eligibility_reasons": ["external-validation-required"],
                "experiment_case": experiment_case,
                "input_candidates_sha256": input_candidates_sha256,
                "code_commit": code_commit,
            }
        )
    return sorted(results, key=lambda item: item["candidate_id"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        records.append(value)
    return records


def resolve_code_commit(repository: str | Path | None = None) -> str:
    cwd = Path(repository) if repository else Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def propose_cases(
    *,
    candidates_path: str | Path,
    code_commit: str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if min(max_file_bytes, max_rows, max_columns) < 1:
        raise ValueError("proposal limits must be positive")
    source_path = Path(candidates_path)
    input_hash = sha256_file(source_path)
    commit = code_commit or resolve_code_commit()
    candidates = sorted(
        _read_jsonl(source_path),
        key=lambda item: str(item.get("candidate_id") or ""),
    )
    singles: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    verified_preserved = 0
    for candidate in candidates:
        status = str(candidate.get("curation_status") or "").casefold()
        if status == "verified":
            verified_preserved += 1
            record = deepcopy(candidate)
            record.update(
                {
                    "proposal_rejection_reasons": [
                        "verified-candidate-preserved"
                    ],
                    "input_candidates_sha256": input_hash,
                    "code_commit": commit,
                }
            )
            rejected.append(record)
            continue
        if status != "unverified":
            record = deepcopy(candidate)
            record.update(
                {
                    "proposal_rejection_reasons": [
                        "candidate-not-unverified"
                    ],
                    "input_candidates_sha256": input_hash,
                    "code_commit": commit,
                }
            )
            rejected.append(record)
            continue
        try:
            singles.append(
                propose_single_candidate(
                    candidate,
                    input_candidates_sha256=input_hash,
                    code_commit=commit,
                    max_file_bytes=max_file_bytes,
                    max_rows=max_rows,
                    max_columns=max_columns,
                )
            )
        except ProposalRejected as exc:
            record = deepcopy(candidate)
            record.update(
                {
                    "proposal_rejection_reasons": list(exc.reasons),
                    "proposal_rejection_detail": exc.detail,
                    "input_candidates_sha256": input_hash,
                    "code_commit": commit,
                }
            )
            rejected.append(record)
    singles.sort(key=lambda item: item["candidate_id"])
    multi = _multi_panel_proposals(
        singles,
        input_candidates_sha256=input_hash,
        code_commit=commit,
    )
    proposed = sorted(
        singles + multi,
        key=lambda item: (item["proposal_type"], item["candidate_id"]),
    )
    rejected.sort(key=lambda item: str(item.get("candidate_id") or ""))
    if any(item.get("eligible_for_experiment") for item in proposed):
        raise AssertionError("proposals must never be experiment-eligible")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_candidates": str(source_path.resolve()),
        "input_candidates_sha256": input_hash,
        "code_commit": commit,
        "input_count": len(candidates),
        "single_proposals": len(singles),
        "multi_panel_proposals": len(multi),
        "proposals_total": len(proposed),
        "rejected": len(rejected),
        "verified_preserved": verified_preserved,
        "eligible_for_experiment": 0,
        "max_file_bytes": max_file_bytes,
        "max_rows": max_rows,
        "max_columns": max_columns,
        "llm_calls": 0,
    }
    return proposed, rejected, summary


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )


def write_proposal_outputs(
    output_root: str | Path,
    proposed: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "proposed.jsonl", proposed)
    _write_jsonl(output / "rejected.jsonl", rejected)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import date, datetime, time
from decimal import Decimal
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from openpyxl import load_workbook
import yaml

_AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from experiments.baseline_specs.build_portable_subtracks import (
    MaterializationError,
    _canonical_json,
    _discover_source_repo_root,
    _file_hash,
    _git_commit,
    _json_hash,
    _read_json,
    _remap_tree,
    _require_python_modules,
    _write_json,
)
from experiments.manifest import (
    load_dataset_manifest,
    verify_case_data_files,
)


NORMALIZATION_POLICY_ID = "matplotagent-table-to-csv-v1"
MAX_ROWS = 100_000
MAX_COLUMNS = 64
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _cell_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "numeric"
    if isinstance(value, (float, Decimal)):
        number = float(value)
        if not math.isfinite(number):
            raise MaterializationError("normalization encountered non-finite number")
        return "numeric"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time"
    if isinstance(value, str):
        return "string"
    raise MaterializationError(
        f"normalization encountered unsupported cell type: {type(value).__name__}"
    )


def _serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    kind = _cell_type(value)
    if kind == "boolean":
        return "true" if value else "false"
    if kind == "numeric":
        if isinstance(value, int):
            return str(value)
        return format(float(value), ".17g")
    if kind in {"datetime", "date", "time"}:
        return value.isoformat()
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _normalize_rows(rows: Sequence[Sequence[Any]]) -> tuple[list[str], list[list[Any]]]:
    if not rows:
        raise MaterializationError("normalization source table is empty")
    raw_header = list(rows[0])
    while raw_header and _empty(raw_header[-1]):
        raw_header.pop()
    if not raw_header or len(raw_header) > MAX_COLUMNS:
        raise MaterializationError("normalization header width is invalid")
    if any(not isinstance(value, str) or not value.strip() for value in raw_header):
        raise MaterializationError("normalization requires non-empty string headers")
    headers = [str(value) for value in raw_header]
    normalized_headers = [value.strip().casefold() for value in headers]
    if len(set(normalized_headers)) != len(normalized_headers):
        raise MaterializationError("normalization header is ambiguous or duplicate")

    normalized_rows: list[list[Any]] = []
    for raw in rows[1:]:
        values = list(raw)
        if any(not _empty(value) for value in values[len(headers) :]):
            raise MaterializationError("normalization row has extra populated columns")
        values = (values[: len(headers)] + [None] * len(headers))[: len(headers)]
        if all(_empty(value) for value in values):
            continue
        normalized_rows.append(values)
        if len(normalized_rows) > MAX_ROWS:
            raise MaterializationError("normalization row limit exceeded")
    if not normalized_rows:
        raise MaterializationError("normalization source table has no data rows")

    column_types: list[str] = []
    for index, header in enumerate(headers):
        types = {
            _cell_type(row[index])
            for row in normalized_rows
            if not _empty(row[index])
        }
        if not types:
            raise MaterializationError(
                f"normalization column is empty: {header}"
            )
        if types <= {"numeric"}:
            column_types.append("numeric")
        elif len(types) == 1:
            column_types.append(next(iter(types)))
        else:
            raise MaterializationError(
                f"normalization column has ambiguous dtypes: {header}:{sorted(types)}"
            )
    return headers, normalized_rows, column_types


def _read_csv_rows(path: Path, sheet: str | None) -> list[list[Any]]:
    if sheet is not None:
        raise MaterializationError("CSV normalization forbids a sheet selection")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [list(row) for row in csv.reader(handle)]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MaterializationError("normalization cannot read CSV source") from exc


def _read_xlsx_rows(path: Path, sheet: str | None) -> list[list[Any]]:
    if not isinstance(sheet, str) or not sheet.strip():
        raise MaterializationError(
            "XLSX normalization requires one explicit source sheet"
        )
    values_book = formulas_book = None
    try:
        values_book = load_workbook(
            path,
            read_only=True,
            data_only=True,
            keep_links=False,
        )
        formulas_book = load_workbook(
            path,
            read_only=True,
            data_only=False,
            keep_links=False,
        )
        if (
            values_book.sheetnames.count(sheet) != 1
            or formulas_book.sheetnames.count(sheet) != 1
        ):
            raise MaterializationError(
                "XLSX normalization source sheet is missing or ambiguous"
            )
        values_sheet = values_book[sheet]
        formulas_sheet = formulas_book[sheet]
        if (
            values_sheet.max_row > MAX_ROWS + 1
            or values_sheet.max_column > MAX_COLUMNS
        ):
            raise MaterializationError("XLSX normalization table limit exceeded")
        rows: list[list[Any]] = []
        value_rows = values_sheet.iter_rows(
            max_row=values_sheet.max_row,
            max_col=values_sheet.max_column,
        )
        formula_rows = formulas_sheet.iter_rows(
            max_row=formulas_sheet.max_row,
            max_col=formulas_sheet.max_column,
        )
        for value_row, formula_row in zip(value_rows, formula_rows, strict=True):
            row: list[Any] = []
            for value_cell, formula_cell in zip(
                value_row,
                formula_row,
                strict=True,
            ):
                if formula_cell.data_type == "e":
                    raise MaterializationError(
                        "XLSX normalization encountered an error cell"
                    )
                if (
                    formula_cell.data_type == "f"
                    and value_cell.value is None
                ):
                    raise MaterializationError(
                        "XLSX normalization encountered an uncached formula"
                    )
                row.append(value_cell.value)
            rows.append(row)
        return rows
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError(
            "normalization cannot read XLSX source"
        ) from exc
    finally:
        if values_book is not None:
            values_book.close()
        if formulas_book is not None:
            formulas_book.close()


def normalize_source_to_csv(
    *,
    source_path: Path,
    source_sha256: str,
    sheet: str | None,
    destination: Path,
) -> dict[str, Any]:
    source_path = source_path.expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise MaterializationError("normalization source is missing or a symlink")
    if _file_hash(source_path) != source_sha256:
        raise MaterializationError("normalization source SHA-256 mismatch")
    suffix = source_path.suffix.casefold()
    if suffix == ".csv":
        raw_rows = _read_csv_rows(source_path, sheet)
    elif suffix in {".xlsx", ".xlsm"}:
        raw_rows = _read_xlsx_rows(source_path, sheet)
    else:
        raise MaterializationError(
            f"normalization source format is unsupported: {suffix}"
        )
    headers, rows, column_types = _normalize_rows(raw_rows)
    stream = io.StringIO(newline="")
    writer = csv.writer(
        stream,
        dialect="excel",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writerow(headers)
    writer.writerows(
        [_serialize_cell(value) for value in row]
        for row in rows
    )
    payload = stream.getvalue().encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "schema_version": "1.0",
        "normalization_policy_id": NORMALIZATION_POLICY_ID,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "source_sheet": sheet,
        "source_format": suffix.lstrip("."),
        "normalized_csv_path": str(destination.resolve()),
        "normalized_csv_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "columns": len(headers),
        "headers": headers,
        "column_types": column_types,
    }


def normalization_code_hash() -> str:
    payload = {
        "policy_id": NORMALIZATION_POLICY_ID,
        "max_rows": MAX_ROWS,
        "max_columns": MAX_COLUMNS,
        "functions": {
            function.__name__: inspect.getsource(function)
            for function in (
                _empty,
                _cell_type,
                _serialize_cell,
                _normalize_rows,
                _read_csv_rows,
                _read_xlsx_rows,
                normalize_source_to_csv,
            )
        },
    }
    return _json_hash(payload)


def _compatible(case: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    required = spec["selection"]["required_case_metadata"]
    expectation = case.get("evaluation_expectation")
    return (
        case.get("split") == "test"
        and case.get("panel_count") == 1
        and isinstance(case.get("data_path"), str)
        and Path(str(case["data_path"])).suffix.casefold()
        in set(required["data_path_suffixes"])
        and isinstance(case.get("user_goal"), str)
        and bool(str(case["user_goal"]).strip())
        and isinstance(expectation, Mapping)
        and expectation.get("panel_groups") == []
    )


def materialize_matplotagent_v2(
    *,
    repo_root: Path,
    workspace_root: Path,
    output_root: Path,
    require_clean: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve(strict=True)
    workspace_root = workspace_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    commit = _git_commit(repo_root)
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if require_clean and dirty:
        raise MaterializationError("repository is dirty")
    spec_path = (
        repo_root
        / "agent"
        / "experiments"
        / "baseline_specs"
        / "matplotagent-single-test-normalized-v2.json"
    )
    spec = _read_json(spec_path)
    builder = spec["builder_contract"]
    if _file_hash(repo_root / builder["builder_path"]) != builder["builder_sha256"]:
        raise MaterializationError("v2 builder binding mismatch")
    if normalization_code_hash() != spec["normalization"]["code_sha256"]:
        raise MaterializationError("v2 normalization code binding mismatch")
    runtime = spec["external_runtime"]
    checkout = (
        workspace_root / runtime["repo"]["relative_path"]
    ).resolve()
    python = (
        workspace_root / runtime["python"]["relative_path"]
    ).absolute()
    if _git_commit(checkout) != runtime["repo"]["commit"]:
        raise MaterializationError("MatPlotAgent checkout binding mismatch")
    if not python.is_file():
        raise MaterializationError("MatPlotAgent Python binding is missing")
    _require_python_modules(python, runtime["dependency_modules"])
    _require_python_modules(
        Path(sys.executable),
        runtime["agent_builder_dependency_modules"],
    )
    for binding in runtime["bound_files"]:
        if _file_hash(repo_root / binding["path"]) != binding["sha256"]:
            raise MaterializationError("v2 bound file hash mismatch")

    parent_binding = spec["parent_manifest"]
    parent_path = repo_root / parent_binding["path"]
    if _file_hash(parent_path) != parent_binding["sha256"]:
        raise MaterializationError("v2 parent manifest SHA-256 mismatch")
    parent = _read_json(parent_path)
    if parent.get("manifest_hash") != parent_binding["manifest_hash"]:
        raise MaterializationError("v2 parent manifest seal mismatch")
    source_root = _discover_source_repo_root(parent)
    selected = [
        case
        for case in parent["cases"]
        if isinstance(case, Mapping) and _compatible(case, spec)
    ]
    selected_ids = [case["case_id"] for case in selected]
    if (
        selected_ids != spec["selection"]["selected_case_ids"]
        or _json_hash(sorted(selected_ids))
        != spec["selection"]["selected_case_set_sha256"]
    ):
        raise MaterializationError("v2 compatibility case selection changed")
    parent_bindings = [
        {
            "case_id": case["case_id"],
            "parent_case_sha256": _json_hash(case),
        }
        for case in selected
    ]
    if _json_hash(parent_bindings) != spec["selection"][
        "selected_parent_cases_sha256"
    ]:
        raise MaterializationError("v2 parent case bindings changed")

    normalized_root = output_root / "normalized_tables"
    derived_cases: list[dict[str, Any]] = []
    case_bindings: list[dict[str, Any]] = []
    for parent_case in selected:
        case_id = str(parent_case["case_id"])
        if not _SAFE_CASE_ID.fullmatch(case_id):
            raise MaterializationError("v2 case_id is unsafe")
        remapped_parent = _remap_tree(
            parent_case,
            source_root=source_root,
            target_root=repo_root,
        )
        normalized_path = normalized_root / f"{case_id}.csv"
        normalization = normalize_source_to_csv(
            source_path=Path(remapped_parent["data_path"]),
            source_sha256=str(parent_case["data_sha256"]),
            sheet=parent_case.get("sheet"),
            destination=normalized_path,
        )
        derived = deepcopy(remapped_parent)
        derived["input_track"] = "table_instruction"
        derived["data_path"] = str(normalized_path.resolve())
        derived["data_sha256"] = normalization["normalized_csv_sha256"]
        derived["sheet"] = None
        for panel in derived["panels"]:
            panel["data_path"] = derived["data_path"]
            panel["data_sha256"] = derived["data_sha256"]
            panel["sheet"] = None
        derived_cases.append(derived)
        case_bindings.append(
            {
                "case_id": case_id,
                "parent_case_sha256": _json_hash(parent_case),
                "runtime_case_sha256": _json_hash(derived),
                "parent_source_sha256": parent_case["data_sha256"],
                "parent_source_sheet": parent_case.get("sheet"),
                "normalization_code_sha256": normalization_code_hash(),
                "normalized_csv_sha256": normalization[
                    "normalized_csv_sha256"
                ],
                "normalized_csv_path": derived["data_path"],
                "instruction_sha256": _json_hash(parent_case["user_goal"]),
                "expectation_sha256": _json_hash(
                    parent_case["evaluation_expectation"]
                ),
                "normalization_audit": normalization,
            }
        )

    provenance = _remap_tree(
        parent["provenance"],
        source_root=source_root,
        target_root=repo_root,
    )
    provenance["source_binding_hash"] = _json_hash(
        provenance["source_binding"]
    )
    derivation = {
        "schema_version": "1.0",
        "compatibility_id": spec["compatibility_id"],
        "spec_path": str(spec_path.relative_to(repo_root)),
        "spec_sha256": _file_hash(spec_path),
        "parent_manifest_path": str(parent_path.relative_to(repo_root)),
        "parent_manifest_sha256": parent_binding["sha256"],
        "parent_manifest_hash": parent_binding["manifest_hash"],
        "preserved_renderability_policy_hash": provenance[
            "renderability_policy_hash"
        ],
        "preserved_renderability_audit_hash": provenance[
            "renderability_audit_hash"
        ],
        "runtime_repo_root": str(repo_root),
        "declared_mutations": [
            "input_track",
            "runtime_root_remap",
            "table_to_csv_normalization",
        ],
        "selected_case_ids": selected_ids,
        "selected_case_set_sha256": spec["selection"][
            "selected_case_set_sha256"
        ],
        "normalization_policy_id": NORMALIZATION_POLICY_ID,
        "normalization_code_sha256": normalization_code_hash(),
        "case_bindings": case_bindings,
        "forbidden_inputs": spec["selection"]["forbidden_inputs"],
        "license_status": "not_declared",
    }
    derivation["derivation_hash"] = _json_hash(derivation)
    provenance["compatibility_derivation"] = derivation
    manifest = {
        "schema_version": parent["schema_version"],
        "cases": derived_cases,
        "provenance": provenance,
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "matplotagent_v2.manifest.json"
    _write_json(manifest_path, manifest)
    loaded = load_dataset_manifest(
        manifest_path,
        dataset_mode="sealed_benchmark",
    )
    for case in loaded:
        verify_case_data_files(case, manifest_path=manifest_path)

    template_path = (
        repo_root
        / "agent"
        / "experiments"
        / "matrices"
        / "baseline_matplotagent_single_normalized_v2.yaml"
    )
    matrix = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    matrix["dataset_manifest"] = str(manifest_path.resolve())
    matrix["dataset_manifest_sha256"] = _file_hash(manifest_path)
    matrix["artifact_root"] = str(
        repo_root
        / "agent"
        / "experiments"
        / "runs"
        / "production"
        / matrix["portable_template"]["artifact_root_name"]
    )
    matrix["repo_root"] = str(repo_root)
    for method in matrix["methods"]:
        if method["name"] == "matplotagent_native_normalized_v2":
            method["provider_options"]["repo_path"] = str(checkout)
            method["provider_options"]["python_executable"] = str(python)
        else:
            method["provider_options"]["manifest_data_root"] = str(repo_root)
    matrix["compatibility_subtrack"]["materialized_manifest_sha256"] = (
        _file_hash(manifest_path)
    )
    matrix_path = output_root / "matplotagent_v2.matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(matrix, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "code_commit": commit,
        "code_dirty": dirty,
        "model_calls": 0,
        "cases": len(derived_cases),
        "unique_dois": len({case["doi"] for case in derived_cases}),
        "spec": str(spec_path),
        "spec_sha256": _file_hash(spec_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _file_hash(manifest_path),
        "matrix": str(matrix_path),
        "matrix_sha256": _file_hash(matrix_path),
        "normalization_code_sha256": normalization_code_hash(),
        "normalized_tables": case_bindings,
    }
    summary["summary_hash"] = _json_hash(summary)
    _write_json(output_root / "materialization_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    default_repo = Path(__file__).resolve().parents[3]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--workspace-root", type=Path, default=default_repo.parent)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            default_repo
            / "agent"
            / "experiments"
            / "runs"
            / "preflight"
            / "matplotagent_v2"
        ),
    )
    args = parser.parse_args(argv)
    summary = materialize_matplotagent_v2(
        repo_root=args.repo_root,
        workspace_root=args.workspace_root,
        output_root=args.out,
        require_clean=args.require_clean,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

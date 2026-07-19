"""Honest source-table normalizer for the dominant Nature Communications dialect.

Real Nature >=5-panel source sheets are structured tables, not clean 2-column
x-y tables. The dominant recoverable dialect looks like:

    r1:  "Figure 3a"  "#IFNg+CD8+ T cells"   .    .    .        <- title row(s)
    r2:   .            WT PBS+Rad  WT Prop+Rad  KO PBS+Rad  ...  <- REAL header
    r3:   .            37.86       91.16        80.15      ...   <- data
    ...                (ragged per-group replicate columns)

i.e. leading title row(s), an optional empty spacer column, the real header at a
lower row, optional side-by-side sub-blocks separated by empty columns, and
"wide" layouts where each column is a category and rows are replicates.

This module recovers a CLEAN long-format table from that structure so the honest
analyzer (`analyze_table`) can classify it. It is DELIBERATELY conservative and
FAIL-CLOSED: it invents no data, only recognizes well-defined structure, keeps
the FIRST side-by-side block, and raises `ProposalRejected` on anything it cannot
confidently normalize (e.g. Sankey/edge-list layouts, matrices, no numeric data).

IMPORTANT: this is a *candidate* normalization layer for measuring how much honest
yield real extreme source data can genuinely support. It is NOT wired into the
sealed proposal pipeline; adopting it into the benchmark's renderability contract
is a benchmark-design decision, not something a builder may silently enable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pandas/openpyxl are the same deps the reader already requires
    import pandas as pd
    from openpyxl import load_workbook
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("source_table_normalizer requires pandas + openpyxl") from exc


class NormalizerRejected(Exception):
    """Raised when the sheet cannot be confidently normalized (fail-closed)."""

    def __init__(self, reason: str, *, detail: str | None = None) -> None:
        super().__init__(reason if detail is None else f"{reason}:{detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class NormalizeResult:
    frame: Any
    orientation: str
    header_row_index: int
    dropped_side_blocks: int = 0
    aggregated_replicates: bool = False
    notes: list[str] = field(default_factory=list)


def _is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == value  # reject NaN
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def _to_number(value: Any) -> float | None:
    if _is_number(value):
        return float(str(value).strip().replace(",", "")) if isinstance(value, str) else float(value)
    return None


def _coerce(value: Any) -> Any:
    """Numeric strings/values -> float; other non-null values -> stripped str;
    empty -> None. Preserves categorical labels (does NOT force numeric)."""
    if _is_number(value):
        return _to_number(value)
    if _nonnull(value):
        return str(value).strip()
    return None


def _nonnull(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _bounding_box(grid: list[list[Any]]) -> list[list[Any]]:
    """Trim only the OUTER empty rows/columns. Interior empty columns are kept
    because they act as separators between side-by-side sub-blocks."""
    rows = [r for r in grid if any(_nonnull(c) for c in r)]
    if not rows:
        raise NormalizerRejected("normalizer-empty")
    width = max(len(r) for r in rows)
    rows = [list(r) + [None] * (width - len(r)) for r in rows]
    col_nonempty = [any(_nonnull(r[c]) for r in rows) for c in range(width)]
    first = next((c for c, ok in enumerate(col_nonempty) if ok), None)
    last = next(
        (c for c in range(width - 1, -1, -1) if col_nonempty[c]), None
    )
    if first is None or last is None:
        raise NormalizerRejected("normalizer-empty")
    return [r[first : last + 1] for r in rows]


def _find_header_row(grid: list[list[Any]]) -> int:
    """First row H with >=2 non-null cells whose next row has >=2 numeric cells
    located under H's non-null columns. Rows above H are titles."""
    for h in range(len(grid) - 1):
        header = grid[h]
        header_cols = [c for c, v in enumerate(header) if _nonnull(v)]
        if len(header_cols) < 2:
            continue
        nxt = grid[h + 1]
        numeric_under = sum(
            1 for c in header_cols if c < len(nxt) and _is_number(nxt[c])
        )
        if numeric_under >= 2:
            return h
    raise NormalizerRejected("normalizer-no-header")


def _split_blocks(header_cols: list[int]) -> list[list[int]]:
    """Split header columns into contiguous runs (side-by-side sub-blocks are
    separated by a gap in the non-null header column indices)."""
    blocks: list[list[int]] = []
    current: list[int] = []
    for c in header_cols:
        if current and c != current[-1] + 1:
            blocks.append(current)
            current = [c]
        else:
            current.append(c)
    if current:
        blocks.append(current)
    return blocks


def normalize_grid(grid: list[list[Any]]) -> NormalizeResult:
    grid = _bounding_box(grid)
    h = _find_header_row(grid)
    header = grid[h]
    data = grid[h + 1 :]
    if not data:
        raise NormalizerRejected("normalizer-no-data")

    header_cols = [c for c, v in enumerate(header) if _nonnull(v)]
    blocks = _split_blocks(header_cols)
    usable = [b for b in blocks if len(b) >= 2] or [b for b in blocks if b]
    if not usable:
        raise NormalizerRejected("normalizer-no-usable-block")
    block = usable[0]
    dropped = len(blocks) - 1

    # Optional leading label column: a column just left of the block whose header
    # is empty but whose data is mostly non-null strings -> row-label (x) column.
    label_col = None
    lead = block[0] - 1
    if lead >= 0 and not _nonnull(header[lead]):
        col_vals = [r[lead] if lead < len(r) else None for r in data]
        nonnull_vals = [v for v in col_vals if _nonnull(v)]
        if nonnull_vals and sum(1 for v in nonnull_vals if not _is_number(v)) >= max(
            2, len(nonnull_vals) // 2
        ):
            label_col = lead

    names = [str(header[c]).strip() for c in block]
    if len(set(names)) != len(names):
        # Duplicate header labels (e.g. a two-level "group / n=1,n=2,n=3" matrix)
        # are structurally ambiguous -> fail closed rather than guess.
        raise NormalizerRejected("normalizer-duplicate-headers")
    all_string_headers = all(not _is_number(header[c]) for c in block)
    block_is_numeric = True
    for r in data:
        for c in block:
            v = r[c] if c < len(r) else None
            if _nonnull(v) and not _is_number(v):
                block_is_numeric = False
                break
        if not block_is_numeric:
            break

    if label_col is not None:
        # LONG: leading label column is x, block columns are y-series.
        records = []
        for r in data:
            x = r[label_col] if label_col < len(r) else None
            if not _nonnull(x):
                continue
            row = {"category": str(x).strip()}
            for c, name in zip(block, names):
                row[name] = _coerce(r[c] if c < len(r) else None)
            records.append(row)
        frame = pd.DataFrame.from_records(records)
        cols = ["category"] + [n for n in names if n in frame.columns]
        frame = frame[cols].dropna(how="all", subset=names)
        orientation = "long-labelled"
    elif all_string_headers and block_is_numeric:
        # WIDE categorical: each column is a category, rows are replicates.
        records = []
        for c, name in zip(block, names):
            for r in data:
                v = _to_number(r[c] if c < len(r) else None)
                if v is not None:
                    records.append({"category": name, "value": v})
        frame = pd.DataFrame.from_records(records)
        orientation = "wide-categorical"
    else:
        # Already long: leftmost block column is x, rest are y-series. Preserve
        # categorical columns as strings; analyze_table decides x/y.
        records = []
        xc = block[0]
        for r in data:
            x = r[xc] if xc < len(r) else None
            if not _nonnull(x):
                continue
            row = {names[0]: _coerce(x)}
            for c, name in zip(block[1:], names[1:]):
                row[name] = _coerce(r[c] if c < len(r) else None)
            records.append(row)
        frame = pd.DataFrame.from_records(records)
        orientation = "long-passthrough"

    if frame.empty or frame.shape[1] < 2:
        raise NormalizerRejected("normalizer-degenerate")
    numeric_cols = [
        col
        for col in frame.columns
        if pd.to_numeric(frame[col], errors="coerce").notna().any()
    ]
    if not numeric_cols:
        raise NormalizerRejected("normalizer-no-numeric")

    result = NormalizeResult(
        frame=frame.reset_index(drop=True),
        orientation=orientation,
        header_row_index=h,
        dropped_side_blocks=dropped,
    )
    if orientation == "wide-categorical":
        result.aggregated_replicates = frame["category"].duplicated().any()
    if dropped:
        result.notes.append(f"dropped {dropped} side-by-side sub-block(s)")
    return result


def normalize_source_sheet(
    path: str | Path,
    sheet: str,
    *,
    max_rows: int = 100_000,
    max_cols: int = 64,
) -> NormalizeResult:
    path = Path(path)
    if not path.is_file():
        raise NormalizerRejected("normalizer-source-missing")
    workbook = None
    try:
        workbook = load_workbook(
            filename=path, read_only=True, data_only=True, keep_links=False
        )
        if sheet not in workbook.sheetnames:
            raise NormalizerRejected("normalizer-sheet-missing")
        worksheet = workbook[sheet]
        grid: list[list[Any]] = []
        for i, row in enumerate(
            worksheet.iter_rows(min_row=1, max_col=max_cols, values_only=True)
        ):
            if i >= max_rows:
                break
            grid.append(list(row))
    finally:
        if workbook is not None:
            workbook.close()
    try:
        return normalize_grid(grid)
    except NormalizerRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed on any unexpected shape
        raise NormalizerRejected(
            "normalizer-error", detail=f"{type(exc).__name__}:{exc}"
        ) from exc

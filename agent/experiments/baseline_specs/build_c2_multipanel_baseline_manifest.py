"""Derive a legacy multi-panel comparison manifest from a C2 source manifest.

The source can be either a sealed benchmark JSON manifest or a proposal JSONL
corpus.  For the external-baseline head-to-head we only need the selected
multi-panel cases themselves -- the panels, their per-panel ``data_sha256``
integrity, and the full ``evaluation_expectation`` -- because the comparison is
scored by the same ``app.evaluation`` evaluator regardless of how the corpus was
assembled.

This builder extracts the selected cases and writes them as a provenance-free
manifest that loads under ``dataset_mode="legacy"``. It hash-pins every selected
panel's source file (verifying pre-existing pins when supplied), so the data the
baselines and PheroViz see is byte-identical to the source corpus. Only the
corpus-construction audit binding (when present in a sealed source) is omitted --
it is irrelevant to the apples-to-apples model comparison.

Usage:
    python -m experiments.baseline_specs.build_c2_multipanel_baseline_manifest \
        --sealed <benchmark_manifest.json|proposed.jsonl> \
        --min-panels 5 \
        --out <derived_manifest.json>

The command also prints (and optionally writes with ``--sources-out``) the unique
source files that must be synced to the runtime repo root, each with its sealed
sha256, so the caller can materialize exactly those bytes on the cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _load(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        cases: List[Dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"source manifest line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise SystemExit(
                    f"source manifest line {line_number} is not an object"
                )
            cases.append(value)
        return {"schema_version": "1.0", "cases": cases}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit("source manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SystemExit("source manifest must be an object")
    return value


def _case_payload(case: Mapping[str, Any]) -> Mapping[str, Any]:
    proposal_case = case.get("experiment_case")
    if proposal_case is None:
        return case
    if not isinstance(proposal_case, Mapping):
        raise SystemExit("proposal record has an invalid experiment_case")
    return proposal_case


def _multi_panel_cases(
    source: Mapping[str, Any],
    *,
    min_panels: int,
) -> List[Dict[str, Any]]:
    cases = source.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("source manifest has no 'cases' list")
    selected: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        payload = _case_payload(case)
        panels = payload.get("panels")
        if isinstance(panels, list) and len(panels) >= min_panels:
            selected.append(dict(payload))
    if not selected:
        raise SystemExit(
            f"no cases with at least {min_panels} panels found in source manifest"
        )
    return selected


def _unique_sources(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen: Dict[str, str] = {}
    for case in cases:
        for panel in case.get("panels", []):
            if not isinstance(panel, Mapping):
                continue
            data_path = panel.get("data_path")
            sha = panel.get("data_sha256")
            if isinstance(data_path, str) and isinstance(sha, str):
                if data_path in seen and seen[data_path] != sha:
                    raise SystemExit(
                        f"conflicting sha256 for {data_path}: "
                        f"{seen[data_path]} vs {sha}"
                    )
                seen[data_path] = sha
    return [{"data_path": p, "data_sha256": s} for p, s in sorted(seen.items())]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bind_panel_data_hashes(cases: List[Dict[str, Any]]) -> None:
    hashes: Dict[str, str] = {}
    for case in cases:
        panels = case.get("panels")
        if not isinstance(panels, list):
            raise SystemExit(f"case {case.get('case_id')!r} has no panels list")
        for index, panel in enumerate(panels):
            if not isinstance(panel, dict):
                raise SystemExit(
                    f"case {case.get('case_id')!r} panel {index} is not an object"
                )
            data_path = panel.get("data_path")
            if not isinstance(data_path, str) or not data_path:
                raise SystemExit(
                    f"case {case.get('case_id')!r} panel {index} has no data_path"
                )
            path = Path(data_path)
            if path.is_symlink() or not path.is_file():
                raise SystemExit(
                    f"case {case.get('case_id')!r} panel {index} data file is missing"
                )
            actual = hashes.get(data_path)
            if actual is None:
                actual = _sha256_file(path)
                hashes[data_path] = actual
            expected = panel.get("data_sha256")
            if expected is not None and (
                not isinstance(expected, str) or expected != actual
            ):
                raise SystemExit(
                    f"case {case.get('case_id')!r} panel {index} data SHA-256 changed"
                )
            panel["data_sha256"] = actual


def build(
    source_path: Path,
    *,
    min_panels: int = 2,
) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    source = _load(source_path)
    cases = _multi_panel_cases(source, min_panels=min_panels)
    _bind_panel_data_hashes(cases)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    derived = {
        "schema_version": source.get("schema_version", "1.0"),
        "derived_from": {
            "source_manifest": str(source_path),
            "source_manifest_sha256": source_hash,
            "source_kind": (
                "sealed_benchmark"
                if source.get("provenance") is not None
                else "proposal_or_legacy"
            ),
            "selection": f"cases with len(panels) >= {min_panels}",
            "dataset_mode": "legacy",
            "note": (
                "Provenance-free legacy view of selected multi-panel cases; "
                "panel data_sha256 integrity is still enforced on load."
            ),
        },
        "cases": cases,
    }
    sources = _unique_sources(cases)
    return derived, sources


def _rewrite_roots(
    derived: Dict[str, Any],
    sources: List[Dict[str, str]],
    *,
    source_root: str,
    target_root: str,
) -> None:
    """Rewrite every panel ``data_path`` prefix from ``source_root`` to
    ``target_root`` in place (data bytes -- and thus ``data_sha256`` -- are
    unchanged). This lets the derived manifest reference cluster-absolute paths
    so no runtime remapping is required."""
    src = source_root.rstrip("/")
    dst = target_root.rstrip("/")

    def remap(path: str) -> str:
        if path == src:
            return dst
        if path.startswith(src + "/"):
            return dst + path[len(src):]
        raise SystemExit(
            f"data_path {path!r} is not under source root {src!r}"
        )

    for case in derived["cases"]:
        for panel in case.get("panels", []):
            if isinstance(panel, Mapping) and isinstance(
                panel.get("data_path"), str
            ):
                panel["data_path"] = remap(panel["data_path"])
    for source in sources:
        source["data_path"] = remap(source["data_path"])
    derived["derived_from"]["data_root_rewrite"] = {
        "source_root": src,
        "target_root": dst,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sealed",
        "--source",
        dest="source",
        required=True,
        type=Path,
        help="Sealed JSON benchmark manifest or proposal JSONL source.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sources-out", type=Path, default=None)
    parser.add_argument(
        "--min-panels",
        type=int,
        default=2,
        help="Keep only cases with at least this many panels (default: 2).",
    )
    parser.add_argument(
        "--rewrite-source-root",
        default=None,
        help="Absolute data_path prefix to replace (e.g. the Mac repo root).",
    )
    parser.add_argument(
        "--rewrite-target-root",
        default=None,
        help="Replacement prefix (e.g. the cluster repo root).",
    )
    args = parser.parse_args(argv)

    if bool(args.rewrite_source_root) != bool(args.rewrite_target_root):
        raise SystemExit(
            "--rewrite-source-root and --rewrite-target-root must be given "
            "together"
        )
    if args.min_panels < 2:
        raise SystemExit("--min-panels must be at least 2")

    derived, sources = build(args.source, min_panels=args.min_panels)
    if args.rewrite_source_root:
        _rewrite_roots(
            derived,
            sources,
            source_root=args.rewrite_source_root,
            target_root=args.rewrite_target_root,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.sources_out is not None:
        args.sources_out.parent.mkdir(parents=True, exist_ok=True)
        args.sources_out.write_text(
            json.dumps(sources, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"[built] {args.out}")
    print(f"  multi-panel cases: {len(derived['cases'])}")
    for case in derived["cases"]:
        print(
            f"    {case['case_id']}  panels={len(case.get('panels', []))}"
            f"  split={case.get('split')}"
        )
    print(f"  unique source files: {len(sources)}")
    for source in sources:
        print(f"    {source['data_sha256'][:12]}  {source['data_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

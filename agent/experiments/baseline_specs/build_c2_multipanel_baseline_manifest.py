"""Derive a legacy multi-panel comparison manifest from the sealed C2 benchmark.

The sealed C2 testbed (``final_benchmark_renderable_v1_seed0/benchmark_manifest.json``)
carries a top-level ``provenance`` block that requires the full corpus
source-binding footprint (candidates / corpus manifests / evidence bundles) to be
present and hash-matched before it will load under ``dataset_mode="sealed_benchmark"``.
For the external-baseline head-to-head we only need the sealed *multi-panel* cases
themselves -- the panels, their per-panel ``data_sha256`` integrity, and the full
``evaluation_expectation`` -- because the comparison is scored by the same
``app.evaluation`` evaluator regardless of how the corpus was assembled.

This builder extracts the multi-panel cases verbatim and writes them as a
provenance-free manifest that loads under ``dataset_mode="legacy"``. Under legacy
mode ``verify_case_data_files`` STILL hash-verifies every panel's ``data_path``
against its sealed ``data_sha256``, so the data the baselines and PheroViz see is
byte-identical to the sealed testbed. Only the corpus-construction audit binding
(validated separately at seal time) is omitted -- it is irrelevant to the
apples-to-apples model comparison.

Usage:
    python -m experiments.baseline_specs.build_c2_multipanel_baseline_manifest \
        --sealed <benchmark_manifest.json> \
        --out <derived_manifest.json>

The command also prints (and optionally writes with ``--sources-out``) the unique
source files that must be synced to the runtime repo root, each with its sealed
sha256, so the caller can materialize exactly those bytes on the cluster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _multi_panel_cases(sealed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cases = sealed.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("sealed manifest has no 'cases' list")
    selected: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        panels = case.get("panels")
        if isinstance(panels, list) and len(panels) > 1:
            selected.append(dict(case))
    if not selected:
        raise SystemExit("no multi-panel cases found in sealed manifest")
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


def build(sealed_path: Path) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    sealed = _load(sealed_path)
    cases = _multi_panel_cases(sealed)
    derived = {
        "schema_version": sealed.get("schema_version", "1.0"),
        "derived_from": {
            "sealed_manifest": str(sealed_path),
            "sealed_manifest_hash": sealed.get("manifest_hash"),
            "selection": "cases with len(panels) > 1",
            "dataset_mode": "legacy",
            "note": (
                "Provenance-free legacy view of the sealed multi-panel cases; "
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
    parser.add_argument("--sealed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sources-out", type=Path, default=None)
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

    derived, sources = build(args.sealed)
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

"""Synthesize a combined article-record JSONL over a multi-publisher content root.

The C2 corpus now spans three independent publisher clusters (Nature
Communications, eLife, EMBO family). Their per-article provenance manifests are
written by the harvesters in two on-disk layouts:

  * Nature:      ``<content_root>/_provenance/<article_id>.json``
  * eLife/EMBO:  ``<content_root>/<article_id>/meta/provenance.json``

``build-manifest`` consumes a single JSONL of article records, so this tool walks
BOTH layouts and emits one record per DOI (deduplicated). Each emitted record is
the already-verified provenance manifest, which ``build_article_manifest`` /
``evaluate_record`` accept directly (they read ``doi``, ``article_url``,
``journal`` and ``license``). No fields are invented; records are only read from
disk and passed through unchanged, so the fail-closed manifest build re-validates
every one against the content root.

Run from ``nature_download/``::

    python3 -m corpus.build_combined_records \
        --content-root outputs/extreme_merged \
        --out outputs/combined_articles.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .provenance import normalize_doi
except ImportError:  # pragma: no cover - direct-script fallback
    from corpus.provenance import normalize_doi


def iter_provenance_records(content_root: Path) -> "list[dict[str, Any]]":
    """Return provenance records from both on-disk layouts, deduped by DOI."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(rec, dict):
            return
        doi = normalize_doi(rec.get("doi") or rec.get("DOI"))
        key = doi or f"nokey:{path}"
        if key in seen:
            return
        seen.add(key)
        records.append(rec)

    # eLife/EMBO (and any Nature dir that carries an in-dir manifest)
    for meta in sorted(content_root.glob("*/meta/provenance.json")):
        _add(meta)
    # Nature top-level provenance for articles without an in-dir manifest
    prov_dir = content_root / "_provenance"
    if prov_dir.is_dir():
        for prov in sorted(prov_dir.glob("*.json")):
            _add(prov)
    return records


def main(argv: "list[str]") -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--content-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    content_root = Path(args.content_root).expanduser()
    if not content_root.is_dir():
        raise SystemExit(f"content-root not a directory: {content_root}")

    records = iter_provenance_records(content_root)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    journals: dict[str, int] = {}
    for rec in records:
        journals[str(rec.get("journal"))] = journals.get(str(rec.get("journal")), 0) + 1
    print(
        f"[done] combined records={len(records)} out={out_path} "
        f"journals={dict(sorted(journals.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

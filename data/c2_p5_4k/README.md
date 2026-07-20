# C2 dataset (`c2_p5_4k`) — FROZEN

Frozen: `2026-07-20T00:42:37.232952+00:00`  ·  commit `5d8b05881635`  ·  **4000 articles**

Real published multi-panel scientific figures with per-figure **Source Data**
(CC-BY, >=5 panels), harvested from Nature Communications, eLife, and the EMBO
family. Each article dir holds `figures/` (PNG + caption `.txt`), `source_data/`
(XLSX/CSV/ZIP), and `meta/` (provenance + figure/source-data manifests).

## What is version-controlled
To keep the git repo small, **only the dataset definition is committed**:
`_manifest.jsonl`, `_frozen.json`, `_provenance/*.json`, this `README.md`. The
heavy per-article content is git-ignored (see `.gitignore`) and lives on local
disk. Every source-data file is pinned by `bytes` + `sha256` and every figure /
source-data file has its origin URL in `_manifest.jsonl`, so the content is fully
reproducible.

## Per-journal breakdown
| Journal | Articles |
|---|---|
| eLife | 2176 |
| The EMBO Journal | 467 |
| Nature Communications | 437 |
| EMBO Reports | 395 |
| EMBO Molecular Medicine | 339 |
| Molecular Systems Biology | 123 |
| Life Science Alliance | 63 |

## Rebuilding the heavy content
Re-run the harvesters (`nature_download/`), or re-download each file from the
`article_url` / `source_data[].url` recorded in `_manifest.jsonl` and verify the
`sha256`.

Regenerate this frozen snapshot with:
```
python3 finalize_c2_4k.py --threshold 4000 --dest data/c2_p5_4k
```

# C2 Case-Level Dataset (`c2_cases_v1`)

Panel-level freeze of the C2 source-fidelity benchmark. **Unit = one panel case**
`(doi, figure, panel)` whose published source table is *honestly composable*
under the pipeline's own simple-2D contract
(`corpus.c2_p5plus_doi_clusters._panel_composable`).

This complements the article-level freeze `data/c2_p5_4k/` (4000 CC-BY articles):
that is the **corpus**; this is the **machine-usable case set** extracted from it.

## Headline numbers

| Metric | Value |
| --- | ---: |
| Panel cases | **2544** |
| Unique DOIs | **588** |
| Unique figures | **680** |
| Published original on disk (byte-pinned) | **2544 / 2544** |
| **P5+ compositional stratum** (figures with ≥5 composable panels) | **224 figures / 204 DOIs / 1611 cases** |

Panel-count strata (by composable panels per figure): `1:193 · 2:116 · 3:80 · 4:67 · 5+:224`.

## Provenance tiers (honest scope)

Each case carries a `provenance_tier`:

| Tier | Cases | Meaning | Status |
| --- | ---: | --- | --- |
| `A_raw` | **141** | Published sheet is already clean 2D; **zero reshaping**. | Sealed-eligible |
| `B_normalizer` | **2403** | Ragged/multi-header/merged published sheet reshaped by the **deterministic** `source_table_normalizer.normalize_source_sheet`. | **Exploratory** — requires approving the normalization step as part of the benchmark contract |

**Tier B is not fabrication.** Every B case pins the *published original* xlsx
`sha256`, the exact reshaping recipe (`header_row_index`, `dropped_side_blocks`,
`aggregated_replicates`, `orientation`, implementation function) **and** the
normalized-output `sha256` — fully reproducible against the source of record.

Of the 204 P5+ DOIs, only **4** are fully tier-A (raw-clean); the rest depend on
the normalizer. No case is human-review-verified yet (`eligible_for_experiment=0`
upstream).

## Files

- `_manifest.jsonl` — 2544 rows (one per panel case). Fields: `case_id`, `doi`,
  `journal`, `year`, `figure_no`, `panel_id`, `chart_family`, `provenance_tier`,
  `original_source_table` (byte-pinned published original: `relative_path`,
  `sha256`, `sheet_name`, `source_url`, `on_disk`), `working_source_table` (what
  the model reads; for tier B the normalized CSV), `source_normalization` (recipe,
  tier B only), `intent`, `evaluation_expectation` (the task ground truth),
  `figure_composable_panels`, `is_p5plus`.
- `_frozen.json` — freeze metadata + `manifest_sha256` + full stratification.

`content_materialized: false` — the manifest pins the heavy source tables by
`sha256`; the raw xlsx/csv live under `nature_download/outputs/…` (gitignored,
local only), identical to the `c2_p5_4k` convention.

## Reproduce

```bash
python finalize_c2_cases.py \
  --proposed nature_download/outputs/c2_full_casecount_exploratory_20260720/corrected_sheet_binding/proposals/proposed.jsonl \
  --dest data/c2_cases_v1
```

Source proposals `sha256`: `f84fbb64db2d563bbe77e485e5ce0a241c40a5dc76217f668106704b54b23f4a`
(code commit `5d8b0588`).

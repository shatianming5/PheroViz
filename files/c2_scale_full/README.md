# C2-extreme full-corpus scale-up — raw P≥5 ceiling & pipeline fix

Honest attempt to move C2-extreme from *bounded* (committed 0673c4a) to *powered* by
mining the **entire** harvested corpus, not the historical 862-DOI case-build subset.
All numbers Python-verified; RAW (non-normalizer) tier only; no exploratory normalizer.

## Root cause (verified, corrects "N small = data exhausted")
Harvest was **not** the bottleneck. Two downstream stages were:
1. **Case-build coverage** — only **862 distinct DOI** had ever been case-built
   (union of all prior candidate files), out of ~3,800 complete DOI. The 15-DOI raw-P5
   ceiling came from that subset.
2. **Figure→source binding ambiguity** — the case-builder discards panels whose
   source-table binding is uncertain. Full-scale build over 3,421 DOI produced
   **5,079 candidates but 21,025 ambiguous** (~80% discarded).

## What was done (4 parallel GPT-5.6-Terra agents)
- **Coverage (L1):** rebuilt corpus manifest **532 → 3,421 DOI** (single-root
  `extreme_content`; two-root strict-complete union = 3,796). Ran build-cases at full
  scale: candidates=5,079 / **941 DOI with candidates** (+79 over 862), ambiguous=21,025.
  All candidates direct/hash-bound raw tables; **0 normalizer**.
- **Ambiguous recovery (L2):** taxonomy of 10,834 ambiguous = 7,076 name-parse fail,
  2,515 figure/annotation asset, 980 supplementary-figure, 213 multi-panel conflict.
  Evidence-based binding improvement recovered **+230 candidates** (892→1,122 on the
  532-article probe; structural P5 DOI 55→69) — but **strict V4 raw-P5 multi DOI 3→3
  (+0)**: recovered bindings are single-panel / fail the strict table-structure gate.
  Patch (`cases_binding_improvement.patch`) held out of the sealed pipeline because it
  does not move the multi-panel headline; 31 tests + 10 spot-checks pass.
- **Harvest + integrity gate:** corpus DOI 3,829 → ~4,127; LSA net +12 (unique 77 —
  qualifying LSA is scarce, far below the 1,760 ceiling). Added a fail-closed
  figure+source+meta download gate (`corpus/completeness.py`) across all 4 harvesters +
  consolidation; 20 incomplete dirs quarantined to `_rejected_no_source/`; 36 tests pass.
- **Raw-P5 census:** baseline reproduced **15** (frozen SHA `d7c655…`). Full conservative
  RAW-coverage **union: P≥3/5/6 = 52 / 23 / 17**. Frozen pool
  `frozen_rawp5_v4_coverage_union.proposed.jsonl` SHA `cac3f955…`, 208 records
  (23 multi parents + 135 singles), **normalizer_count=0 (asserted)**.

## K-gate verdict (proposed-only; curation still required)
| threshold | value | vs raw P≥5 = 23 |
|---|---|---|
| conservative Holm K | 62 | **NOT met** (23 ≪ 62) |
| optimistic Holm K | 15 | met (23 ≥ 15) |
| optimistic single K | 12 | met (23 ≥ 12) |

**Honest conclusion.** Casing the full corpus (862 → 3,421 DOI) + ambiguous recovery +
LSA completion raises the raw P≥5 multi-DOI universe only **15 → 23**. This is the true
full-corpus ceiling, not a subset artifact — the 1.7%/DOI extrapolation to ~65 is
**falsified** (the 862 were the easy subset; the extra ~2,500 cased DOI die in ambiguous
binding or the strict no-normalizer red line). Data scale-up alone **cannot** reach the
conservative K=62. The committed *bounded* C2-extreme verdict stands and is strengthened.

## FINAL verdict — jury-verified + integrity-corroborated (bounded on every honest axis)
The bounded conclusion is confirmed by **four** independent measurements, the strongest of
which is the **pristine-sealed** (`code_dirty=false`) dual-VLM curation of the full 23-DOI
coverage-union pool: **N′ = 9/23**, still far below every K gate. Every honest axis agrees.

| axis | measurement | source (python-verified) | vs K |
|---|---|---|---|
| **pristine-sealed N′ (23-DOI coverage-union curation)** | **N′ = 9 verified P≥5 multi DOI of 23** (`elife.{101990,38114,49574,76541,78294}`, `s44319-025-{00373-0,00484-8}`, `s44321-025-{00216-4,00233-3}`); 14 rejected (8 both-reject + 6 judge-disagreement, detailed failure reasons); gateway_error 3 | `curation_review_23pool/summary.json` + `reviews.jsonl` (23 parent reviews), SHA `cac3f955…` pinned, **`code_dirty=false`** (commit `87bcce2`) → sealed-eligible, 0 data-normalizer (only license-URL normalization), independent recount 0 discrepancies (3-way: summary/crosscheck/re-count), judges claude-sonnet-4.6 + gemini-3.5-flash | **FAIL** — 9 < single K=12 < optimistic Holm K=15 ≪ conservative Holm K=[41,62] |
| jury-verified N′ (reject-conversion path) | **N′ = 8** (5 baseline + 3 newly-converted multi DOI) — ⚠ NOTE: the 5-DOI baseline is **normalizer-inflated** (only `s44319-025-00373-0` is genuinely raw-P5; the other 4 relied on `exploratory_normalizer` to reach P5) | `../c2_reclassify_review/v4_lift_result.json`, SHA-bound to frozen V4 `58f1d3…`/`0d38f5…`, cross-validation 0 discrepancies | **FAIL** < optimistic K=12 ≪ conservative K=62 |
| clean-raw N′ (direct-raw pool, dual-VLM jury) | **3 verified P≥5 multi DOI** of 15 proposed (`elife.49574`, `s44319-025-00484-8`, `s44321-025-00216-4`); singles 57/124; 12 rejected (panel reasons: chart_family 48, semantic/visual 19, binding 15, schema 8) | `../c2_reclassify_review/raw15_sealed_result.json`, SHA `d7c655…` pinned, independent recount 0 discrepancies. ⚠ **diagnostic-grade**: `code_dirty=true` → `eligible_as_sealed_evidence=false` (this stricter direct-raw run corroborates the pristine-sealed 23-pool above) | **FAIL** — 3 ≪ optimistic K=12 ≪ Holm K=[41,62] |
| raw P≥5 proposed ceiling | 23 (coverage-union `cac3f955…`) / 15 (direct-raw `d7c655…`) | this README + `../c2_reclassify_raw_p5/…direct_raw_p5_v4.proposed.jsonl` (139 rec = 124 singles + 15 parents) | ≪ K=62 |
| only pool reaching K=62 | 204-DOI pool is **BLOCKED_EXPLORATORY_NORMALIZER**: 1538/1611 singles + 223/224 P≥5 multis (203/204 DOI) carry ≥1 normalizer component | `../c2_reclassify_integrity/K_final_reclassified.json` | **forbidden** by no-normalizer red line |
| K itself | **not firmly estimable**: P5+ DOI-level paired-gap SD `not_estimable` (only 1 renderable P5+ DOI, `10.1038/s41467-021-25210-5`); proxy SD from 6 mixed-P DOI → K∈[15 Holm @SD=0.11, 62 Holm @SD=0.25] | `../c2_reclassify_integrity/K_final_reclassified.json` (matches `c2_power_K_final.json` `7e9e60be…`) | K is a bracket, not a point |

The 3 newly-converted multi DOI (V4 reject re-review): `10.1038/s41467-022-30409-1`,
`10.1038/s44318-023-00005-0`, `10.1038/s44318-025-00395-3`; the 5 baseline verified:
`10.7554/elife.95867`, `10.7554/elife.97860`, `10.1038/s44318-025-00510-4`,
`10.1038/s44318-025-00634-7`, `10.1038/s44319-025-00373-0`.

**Bottom line.** C2-extreme is **BOUNDED**. The strongest honest evidence is the
**pristine-sealed** (`code_dirty=false`, SHA `cac3f955`) dual-VLM curation of the full 23-DOI
coverage-union raw P≥5 pool: only **9 of 23** parents pass both judges — below the single
K=12, the optimistic Holm K=15, and far below the conservative Holm K=[41,62]. The stricter
direct-raw pool (15 DOI, diagnostic-grade) corroborates with just **3** verified, and the
reject-conversion path adds no real lift (its "baseline 5" was normalizer-inflated; genuine
raw baseline = 1). The only pool large enough to reach K=62 (204 DOI) is disqualified by the
exploratory-normalizer red line, and K itself is unbounded above (P5+ variance not estimable
from ≥2 clusters). No amount of honest data scale-up powers C2-extreme at the integrity
level; this is a rigorous, information-rich NULL/bounded result, not a pipeline gap.

## Files
- `frozen_rawp5_v4_coverage_union.proposed.jsonl` (+ `freeze_manifest.{json,sha256}`) — the sealed 23-DOI raw P≥5 pool.
- `rawp5_report.json` — baseline reproduction + full-scale census + K-gate.
- `fullscale_raw_candidate_filter_audit.json` — raw-only filter audit (normalizer excluded=0).
- `ambiguous_taxonomy.json`, `v4_recovery_measurement.json` — L2 diagnosis + recovery delta.
- `manifest_summary.json` — 3,421-DOI corpus manifest (SHA `c1162c48…`).
- `download_gate_description.json` — fail-closed figure+source+meta gate (36 tests).

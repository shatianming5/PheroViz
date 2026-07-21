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
The 23-DOI pool meets the optimistic K only; whether that suffices is decided by the
**empirical paired-gap variance**, which requires a fresh sealed dual-VLM curation review
of these 23 proposed DOI (next step → verified N′ + measured SD → powered/bounded).

## Files
- `frozen_rawp5_v4_coverage_union.proposed.jsonl` (+ `freeze_manifest.{json,sha256}`) — the sealed 23-DOI raw P≥5 pool.
- `rawp5_report.json` — baseline reproduction + full-scale census + K-gate.
- `fullscale_raw_candidate_filter_audit.json` — raw-only filter audit (normalizer excluded=0).
- `ambiguous_taxonomy.json`, `v4_recovery_measurement.json` — L2 diagnosis + recovery delta.
- `manifest_summary.json` — 3,421-DOI corpus manifest (SHA `c1162c48…`).
- `download_gate_description.json` — fail-closed figure+source+meta gate (36 tests).

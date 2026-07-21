# Sealed C2 readiness

## Verdict

**BLOCKED; do not execute.** The sealed code path now enforces `K=62`
independent verified P>=5 DOI clusters for the two-contrast Holm plan, but no
current reclassified pool is eligible for sealing. The prior A2 checkpoint/evidence
is stale. Both the named immutable V3 transport and the separately hash-bound
c3ea/737c controlled-alias transport remain historical-judge-selected
**diagnostic-only** artifacts; the current dual-transport clarification requires
one complete hash-bound transport for any fresh V3 evidence. The apparent 204-DOI V4
pool is also prohibited: 223/224 P>=5 multis use
`exploratory_normalizer` source components. The frozen direct-raw, non-normalized V4
replacement pool is structurally clean but has only **15 unreviewed P>=5 DOI
clusters**, not the required 62. Moreover, its 139-record freeze/script are
untracked and its 5,377-candidate raw input is git-ignored; it is not yet a
durably reproducible review input.

Required before launch:

1. Preserve the exact raw candidate input, generated proposal pool, freeze
   manifest, and generator script in a hash-addressed immutable store (or
   force-add them in the supervisor's clean integration commit). The current
   direct-raw artifacts are local/untracked and must not be reviewed as sealed input.
2. Expand the demonstrated 15-DOI direct-raw V4 P>=5 pool into one non-overlapping final
   pool with at least 62 DOI clusters from raw hash-bound, **non-normalized**
   source tables. Do not repair the normalizer-tainted 204-DOI pool in place.
3. Freeze and re-review that raw **full** batch with both judges, then rebuild
   candidates using only its new evidence.
4. Confirm the report rule/version, proposed hash, review input hash, review
   evidence, rebuilt candidate seals, and `benchmark_manifest.json` chain.
5. Run from a clean committed worktree. Both build and run refuse dirty trees.
6. Let the wrapper verify `qualified_multi_dois >= 62`; panels/seeds/renders
   never count as independent DOI clusters.

## Exact future command sequence

```bash
cd /Users/tommy/Downloads/mayi/PheroViz-c2-closure-integration-terra
PY=/opt/homebrew/Caskroom/miniforge/base/bin/python3
REPO="$PWD"

# Must be clean only after the supervisor freezes/integrates V4.
test -z "$(git status --porcelain)"

# 1. A3 must first produce and hash-freeze one raw, non-normalized V4 full pool.
#    NEVER use reproposed_strict_rejects*, the overlapping strict aggregates, or
#    files/c2_reclassify_pool/reproposed_pool_multi.jsonl (normalizer-tainted).
FINAL_PROPOSED="$REPO/PATH/TO/FROZEN_RAW_NON_NORMALIZED_V4_FULL_POOL.jsonl"
test -f "$FINAL_PROPOSED"

# The sealed builder repeats this check and refuses a normalizer path.
"$PY" - "$FINAL_PROPOSED" <<'PY'
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    source = record.get("source_table") or {}
    values = [source.get("path"), source.get("relative_path"), source.get("path_root")]
    if any("exploratory_normalizer" in str(value).casefold() for value in values):
        raise SystemExit("refuse exploratory-normalizer source in sealed input")
PY

# 2. A2's fresh live two-judge review of exactly that frozen full hash.
REVIEW_OUT="$REPO/nature_download/outputs/c2_v4_final_review"
"$PY" nature_download/nature_all_in_one.py review-proposals \
  --proposed "$FINAL_PROPOSED" \
  --out "$REVIEW_OUT" \
  --judge-model claude-sonnet-4.6 \
  --judge-model gemini-3.5-flash

# 3. Rebind fresh review evidence. These are the frozen corpus/pool paths
#    selected by A3; do not substitute an exploratory normalizer output.
CORPUS_MANIFEST="$REPO/PATH/TO/FROZEN_FINAL/corpus_manifest.jsonl"
CONTENT_ROOT="$REPO/PATH/TO/FROZEN_FINAL"
CASES_OUT="$REPO/nature_download/outputs/c2_v4_final_cases"
"$PY" nature_download/nature_all_in_one.py build-cases \
  --corpus-manifest "$CORPUS_MANIFEST" \
  --content-root "$CONTENT_ROOT" \
  --evidence "$REVIEW_OUT/evidence.json" \
  --out "$CASES_OUT"

# 4. Assemble and expand a gateway-free sealed dry-run. This refuses N'<62.
BENCHMARK_OUT="$REPO/nature_download/outputs/c2_v4_p5_benchmark"
DRY_RUN_OUT="$REPO/nature_download/outputs/c2_v4_p5_dry_run"
bash files/run_c2_when_ready.sh \
  --repo-root "$REPO" --python "$PY" \
  --benchmark-out "$BENCHMARK_OUT" --run-out "$DRY_RUN_OUT" \
  --seed 20260721 --min-panels 5 --min-dois 62 \
  --candidate "$CASES_OUT/candidates.jsonl" \
  --proposed "$FINAL_PROPOSED" \
  --reviews "$REVIEW_OUT/reviews.jsonl" \
  --evidence "$REVIEW_OUT/evidence.json" \
  --dry-run

# 5. Only after reviewing the sealed dry-run artifacts, launch explicitly.
EXECUTE_OUT="$REPO/nature_download/outputs/c2_v4_p5_execute"
bash files/run_c2_when_ready.sh \
  --repo-root "$REPO" --python "$PY" \
  --dataset-manifest "$BENCHMARK_OUT/benchmark_manifest.json" \
  --manifest-data-root "$REPO" --run-out "$EXECUTE_OUT" \
  --min-panels 5 --min-dois 62 --execute
```

`build_c2_benchmark.py --minimum-qualified-dois` and
`run_c2_v3.py --min-dois` both enforce the DOI gate; the wrapper passes 62 to
both. `build_c2_benchmark.py`, canonical benchmark assembly, and runtime
manifest validation reject `exploratory_normalizer` sources. The execution
wrapper exposes neither a legacy manifest nor an exploratory-normalizer path.
At audit time the V4 full-pool freeze wrapper/directory are uncommitted; their
proposal bytes are HEAD-identical, but a supervisor clean integration commit is
still required before any sealed invocation. That commit cannot make the
normalizer-tainted 204-DOI diagnostic freeze eligible.
The current untracked freeze registry independently verifies the five freezes,
records both V3 diagnostic transports in its atomic readonly generation
(`e6a40…`; file `fc842…`), and declares `sealed_eligible_freezes=[]`. It is an
audit inventory, never an authorization to review or execute.

## Fresh offline validation

A disposable clean worktree assembled a synthetic 5-panel, 1-DOI sealed bundle
with fake local judge clients and ran `run_c2_v3.py --dataset-manifest
... --dry-run`: one P5 multi case, one matrix, nine expanded specs, return code
0, and **zero gateway calls**. The fixture/worktree were deleted afterwards.
It intentionally used `--minimum-qualified-dois 1`, so it validates mechanics
only and is not a K=62 scientific result.

Current V4 wide-melt runtime validation additionally passed all three targeted
tests: deterministic materialization, categorical fidelity, and virtual-series
palette cohesion. A full V4 sealed dry-run is intentionally deferred until the
current dirty integration is frozen and A2 produces the required fresh full
review evidence; bypassing the clean-worktree gate would invalidate the check.
The current targeted C2 guard suite also passed: `12 passed` for
`agent/tests/test_run_c2_v3.py` and `nature_download/tests/test_benchmark.py`.

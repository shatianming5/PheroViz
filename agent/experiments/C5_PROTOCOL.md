# C5 sealed visual-form protocol

C5 is supporting visual-form evidence. Programmatic fidelity and cohesion remain
authoritative for correctness.

## Frozen judges

`configs/model_registry.yml` defines exactly two independent judges:

- `visual-form-primary-v1`: `claude-sonnet-4.6`
- `visual-form-secondary-v1`: `gemini-3.5-flash`

Both use `anthropic_messages`, the declared compatibility-gateway endpoint
class, and exactly 1024 output tokens. The returned served identity must exactly
match the registry. Registry, prompt, rubric, input-summary, input-manifest,
record, selected-candidate, and selected-render hashes are sealed into batches
and sidecars.

Both judges freeze `validation_attempts=3`. Per render, C5 makes at most three
total calls and retries only response-level failures that leave no valid strict
schema score: empty/invalid JSON, extra or missing keys, invalid score or
diagnostics, and `max_tokens` truncation. It stops on the first valid response
and never selects among valid scores. Served-identity mismatch,
authentication/outage, provenance failure, and input mutation are not retried.

Each v3 sidecar binds the ordered attempt ledger and actual call count. Only
the final successful attempt may contain `valid_score`; three invalid responses
remain a failure. The v3 batch binds per-run attempt counts and their exact
total. All earlier C5 batches are incompatible and `NEVER_MERGE`.

The request contains one selected image and the fixed visual-form prompt only.
Source data, method metadata, prior scores, generation feedback, and editing
feedback are excluded. Dirty code, run-root input, stale summaries or renders,
unregistered/mismatched identities, failed batches, and missing/duplicate/extra
sidecars fail closed.

## Execution order

Only start after each common-commit tier root has exactly 171 terminal rows,
non-method failures are resolved, C4 selection is final, and the code worktree is
clean:

```bash
cd agent
export PYTHONPATH=.

python -m experiments.cli rejudge TIER/summary.json \
  --judge-model claude-sonnet-4.6 --out TIER/c5/primary
python -m experiments.cli rejudge TIER/summary.json \
  --judge-model gemini-3.5-flash --out TIER/c5/secondary

python -m experiments.cli merge-rejudge TIER/summary.json TIER/c5/primary \
  --out TIER/c5/first_rejudged_summary.json
python -m experiments.cli merge-rejudge \
  TIER/c5/first_rejudged_summary.json TIER/c5/secondary \
  --out TIER/c5/final_rejudged_summary.json

python -m experiments.cli analyze TIER/c5/final_rejudged_summary.json \
  --reference flat_iterative --methods best_of_n pheroviz_full \
  --metric metric.visual_form.claude-sonnet-4.6 \
  --second-judge-metric metric.visual_form.gemini-3.5-flash \
  --seed 17029 --bootstrap-resamples 10000 --panel-scope all \
  --out TIER/c5/kendall
```

Both judge batches must originate from the same original strict summary, not
from sequentially modified summaries. Merge schema 2.0 preserves each batch
hash, sidecar map, identity, and immutable original-summary lineage.

Kendall tau-b ranks the three methods independently within each backbone after
seed, task, and DOI-equal aggregation. The DOI bootstrap uses 10,000 resamples,
seed 17029, and a percentile 95% interval. The frozen decision passes only when
every backbone has estimable tau-b at least 0.5 and interval lower bound strictly
above zero. Otherwise report the rankings separately and make no visual-judge
concordance or correctness claim.

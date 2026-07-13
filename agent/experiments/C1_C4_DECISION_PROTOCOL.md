# C1/C4 Decision-Report Protocol

`decision-report` is a deterministic, outcome-independent interpretation layer.
It consumes only strict `analysis.json` artifacts and one declared cross-analysis
Holm-family artifact; it never runs experiments, metrics, or models.

## Frozen contract

- Budget: exactly six renders (`B_R=6`), never wall-clock.
- Reference: `flat_iterative`.
- Reported C1 methods: `pheroviz_full` and `best_of_n`.
- C4 confirmatory method: `pheroviz_full` only.
- Metrics/scopes: `metric.data_fidelity` over `all` panels and
  `metric.series_cohesion` over `multi_panel`.
- Practical margins: `delta_F=0.02`, `delta_C=0.25`.
- Family: exactly three backbones × two metrics, adjusted together by Holm.
- Inputs: exactly six analysis artifacts, one metric artifact per backbone.
- Summaries: exactly three distinct summaries; F and C must share the same
  summary within a backbone and no summary may span two backbones.
- Experiment commit: every tier must be
  `d90d655b96bfb3b95bfdc37665692957969d8968`.
- Alpha: `0.05`.

C4 passes only when every one of the six cells has an effect strictly above its
metric margin, a 95% confidence interval with lower bound strictly above zero,
and cross-family Holm-adjusted `p < 0.05`. Otherwise the all-tier claim is
blocked and the report emits tier-scoped decisions and failed criteria.
Programmatic correctness remains privileged; visual-form scores are not inputs.

C1 is descriptive under the frozen render budget. It reports both PheroViz-full
versus flat and best-of-N versus flat for each tier and metric. Wall-clock is
always marked unavailable because no equal-hardware/equal-concurrency timing
artifact is accepted by this protocol.

## Provenance and failure behavior

The command reloads and recomputes each strict analysis through the production
statistics loader, recomputes the Holm family from its manifest, and requires
exact input coverage. It fails closed on dirty code, stale/forged hashes,
missing or duplicate inputs, mixed summaries/commits/manifests, selectors
outside the six-cell family, wrong methods/scopes, or non-render budgets.

The report binds experiment, analysis, Holm-family, and decision-code commits;
summary, analysis, file, family, family-config, family-manifest, dataset
manifest, and metric-config hashes; tier-to-summary and
tier-to-experiment-commit maps; panel scopes; methods; and budget. The entire
report is sealed by `decision_report_hash`.

```bash
python -m experiments.cli decision-report \
  holm_family.json \
  frontier-fidelity-analysis.json frontier-cohesion-analysis.json \
  mid-fidelity-analysis.json mid-cohesion-analysis.json \
  open-fidelity-analysis.json open-cohesion-analysis.json \
  --out c1_c4_decision_report.json
```

Output must validate against
`schemas/c1_c4_decision_report.schema.json`.

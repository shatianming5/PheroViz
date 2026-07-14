# Final provenance stage

`phero-experiments provenance-stage` builds immutable, hash-bound indexes
without opening outcomes or calling models.

The generator stage accepts exactly the frontier, mid, and open completion
reports. A `COMPLETE` index requires 513 unique clean-`d90d655` records,
57 rows per tier/method, exact summary and record hashes, a recorded served
identity or snapshot/revision, and explicit endpoint class, hosting engine,
precision, and access timestamp. Missing historical fields are `null` and listed
in `needed`; registry aliases are never substituted.

The C5 stage accepts six independent valid batch paths plus one final merged
summary and analysis per tier. It verifies batch and artifact hashes,
requested/served identities, endpoint class, rubric/prompt/input hashes,
selected-render bindings, and attempt totals against the final report.

If a sealed source is unavailable, the command writes
`INCOMPLETE_SOURCE_ARTIFACTS`, lists every required path/hash, and exits 3.
If all 513 records are hash-verified but historical identity fields were never
recorded, it writes `PARTIAL_DISCLOSURE`, preserves those fields as `null`,
lists each exact disclosure gap, and exits 3. It never fills those fields from
the registry or a request alias.
Hash mismatches, duplicates, dirty source records, or forged `COMPLETE` status
raise an error and produce no accepted complete index.

```bash
python -m experiments.cli provenance-stage \
  experiments/preflight/final_provenance_stage_inputs.json \
  --out experiments/preflight/FINAL_INDEX.json
```

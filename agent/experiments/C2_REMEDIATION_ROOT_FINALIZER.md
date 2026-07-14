# C2 fresh-remediation root finalizer

`c2-remediation-root-finalize` seals one new root from a separately acquired
raw-evidence root. It does not acquire articles, invoke models, alter the raw
root, or overwrite an existing target.

The target leaf name is fixed to
`ccby_sr_npj_chunk<NNN>_rerun3_clean_ca98442`; a different name fails before
any target is created.

```bash
cd agent
python -m experiments c2-remediation-root-finalize 013 \
  --raw-root /absolute/raw-attempt-evidence \
  --target-root /absolute/absent-remediation-root \
  --source-chunk /absolute/chunks/chunk_013.jsonl \
  --frozen-universe /absolute/universe.jsonl \
  --freeze-summary /absolute/freeze_summary.json \
  --worktree /absolute/clean-ca98442-worktree
```

The finalizer statically binds chunks `001`–`012` to 200 records and `013` to
63 records, their exact source hashes, the frozen universe/summary hashes, and
the clean acquisition commit. It rejects source-byte, order, count, status,
receipt, event, provenance, symlink, unsafe-parent, or target-collision
mismatches before sealing.

Only the policy-designated replacement chunks `001`–`008`, `011`, and `013`
are accepted. The full 13-chunk table remains compiled for binding; retained
`009`/`010`/`012` candidates are never accepted as fresh-finalizer targets.

Raw controls must already contain the three fixed passes, pre-download binding,
config, per-pass receipts/events, processed-success and skipped-status maps,
logs, zero exits, final provenance, and a hash-sealed
`control/execution_evidence.json` with clean secret-scan and zero-exit test
records. The config must pin the established strict CC-BY/input-sort/postfetch
limits and one positive worker count for all three passes. The execution
evidence must attest the one-fresh-root cap and no active retained pipeline.
`downloaded` is derived only from a processed-success ID absent from
`_skipped.txt`; it is forbidden in skipped maps. The output recomputes
raw-attestation entries and retry2 terminal rows, then emits relative
provenance links, downstream empty-case evidence, inventory,
artifact manifest, report, and validation.

The generated preservation ledger records protected old-root contracts and
retained `009`/`010`/`012` exclusions. For a chunk with a protected-root
contract, the finalizer recomputes its compact inventory and sealed-report hash
before target creation, before report publication, and after publication in a
postseal verification record; any mismatch fails closed.
This finalizer intentionally does not implement the Final3 production P-policy
map or admission aggregation.

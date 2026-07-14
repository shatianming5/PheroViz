# C2 full-replacement V2 — Stage A only

This is an isolated successor to `c2-terminal-finalize`; it does not alter V1.
It exists to exercise the reviewed V2 verifier with synthetic data only.

## Production gate

`phero-experiments c2-full-replacement-finalize MANIFEST --out REPORT` has no
policy, mapping, digest, root, evidence-skip, environment, or path-selector
option. During Stage A its fixed production resolver always fails before
reading the manifest or creating output, because no package-internal,
Git-reviewed policy resource and compiled resource-byte SHA-256 exist yet.

The only injectable policy API is explicitly test-named:
`compile_synthetic_policy_for_testing` together with
`prepare_full_replacement_finalization_for_testing`. It accepts an in-memory
synthetic object and is not a security boundary inside a hostile Python
process. No production CLI route calls it.

Do not add a production policy, root ID, digest, manifest, or evidence bundle
until the separate Stage B forensic approval.

## Fixed Stage-A contract

The synthetic compiler requires the literal ordered roster `001` through
`013`, all thirteen replacement/retirement plan entries, and the frozen
partition `12 × 200 + 63 = 2,463`. Chunk `013` has global ordinals
`2401..2463` and local ordinals `1..63` on **every** initial/retry1/retry2 raw
stream. It cannot be padded, reordered, or treated as a 200-record chunk.

Policy mapping rows have exactly:

```json
{
  "global_ordinal": 1,
  "doi_id": "10.xxxx/normalized-doi",
  "p_disposition": "P2",
  "independent_cluster_id": "policy-literal"
}
```

Every full synthetic policy contains 2,463 ordered rows. Mapping aggregation is
shared by validation and reporting. No evidence, manifest, report, CLI
argument, or hash claim can supply or alter P/cluster semantics.

The closed P enum is `P1`, `P2`, `P3_4`, `P5PLUS`. Every P1 row has exactly
`P1_NONINFERENTIAL`; it has zero countable independent clusters and is never
inference eligible. A present P1 row is deficient. An absent P1 stratum is
`NOT_PRESENT` rather than deficient. P2/P3_4/P5PLUS are deficient only when
present with fewer than two distinct policy-literal clusters. A zero-row
non-P1 stratum is also `NOT_PRESENT`, has zero clusters, is nondeficient, and
is not inference eligible. P5PLUS deficiency has blocked-status precedence.
Every deficiency produces `UNSUPPORTED`, `NOT_RUN` trend status, and `NOT_RUN`
equivalence status.

## Raw evidence layout and recomputation

The V2 manifest lives at the trusted evidence-root leaf. Every artifact path is
strictly manifest-relative, has a full SHA-256 byte binding, cannot contain
`.`/`..`, absolute paths, backslashes, duplicate paths, symlink traversal, or
hard-link aliases.

For each chunk, the manifest binds:

1. one canonical mapping JSONL artifact;
2. an ordered `initial`, `retry1`, `retry2` set;
3. for each attempt, a raw JSONL stream, processed-success JSON object, and
   skipped-status JSON object.

Each raw row contains `attempt_id`, `global_ordinal`, `local_ordinal`, DOI,
and `raw_disposition` (`PROCESSED` or `SKIPPED`). The processed-success and
skipped-status records must be the exact ordered disjoint partition of that
raw stream. `DOWNLOADED` is derived only from a verified processed-success row
with no skipped entry. It is forbidden in skipped-status data; each skipped
entry must contain a closed non-download terminal status. Retry2 is the
terminal outcome source.

The verifier opens the trusted evidence directory chain with no-follow
directory descriptors. It opens each artifact once with `O_NOFOLLOW`, hashes
those exact bytes while reading, parses the captured bytes, retains the
device/inode identity, and never reopens the path to validate it. The canonical
mapping is compared row-for-row and ordinal-for-ordinal with the synthetic
compiled map. The final report retains the complete canonical
DOI/ordinal/attempt ledger linked to every raw artifact digest; summaries alone
never establish attempt facts.

## Output protocol

`--out` must name an absent leaf under a pre-existing trusted parent chain. The
parent chain must be root/current-euid owned, non-group/world-writable, and
free of mutating ACLs. The output parent and evidence root must be distinct and
non-containing; the output must be outside the evidence root and cannot alias
any validated input by lexical path, device/inode, or hard link.

V2 never uses rename/replace publication. It creates a mode-0600
descriptor-relative staging file in the output directory, fsyncs it, then
publishes only by descriptor-relative hard-link creation to the final leaf.
`EEXIST` fails without overwrite. The writer validates final/staging inode
identity, fsyncs the directory, identity-checks staging before unlinking it,
and fsyncs again. Unsupported link-at-style primitives, a leaf/parent swap,
or a cleanup/durability failure produce no success result. A failed link leaves
the restrictive staging name orphaned rather than risk unlinking a concurrently
reused pathname.

## Stage B remains mandatory

Stage B must add one package-internal production policy resource containing the
reviewed full 2,463-row map, partition, all replacement/retirement bindings,
raw-source manifest, and evidence commitments. Its exact resource bytes must
be checked against a compiled-in SHA-256 before parsing. It must preserve all
of the Stage-A no-follow, one-read, raw-ledger, P1, collision, trusted-parent,
and no-replace rules. Until then this implementation is not an admission route
for any real C2 evidence.

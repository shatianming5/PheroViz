# C2 fresh-remediation root finalizer

`c2-remediation-root-finalize` seals one new root from a separately acquired
raw-evidence root. It does not acquire articles, invoke models, alter the raw
root, or overwrite an existing target.

The target leaf name is fixed to
`ccby_sr_npj_chunk<NNN>_rerun3_clean_ca98442`; a different name fails before
any target is created.
Its direct parent must already exist and satisfy the trusted private staging
parent checks; the finalizer never creates that parent.

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
provenance links, downstream evidence, inventory, artifact manifest, report,
and validation.

A `downloaded` result in any fixed pass is source-bearing, even if retry2 later
reports a source-less terminal status; therefore neither case can authorize an
empty chain. Its provenance must state exactly `download_status: "downloaded"`
and reference
`content/_source_evidence/<article_id>.json`. That descriptor is hash- and
byte-bound, links back to the same DOI/article/provenance path, and enumerates
only regular, descriptor-relative `content/_sources/<article_id>/...` files
with their DOI, byte counts, and SHA-256 digests. Raw `content/` must contain
exactly the terminal provenance files and those referenced descriptors/source
files; unreferenced source artifacts fail before target creation.

The finalizer emits the empty canonical/P chain only when every terminal row is
source-less. If any source-bearing row exists, it emits exhaustive acquisition
evidence and `control/source_classification_blocked.json`, then fails with
`NOT_SEALABLE_SOURCE_CLASSIFICATION_BUILDER_REQUIRED`; it never fabricates an
empty canonical/P or sealed-report chain. A separately approved deterministic
canonical/P builder is required before such a root can be sealed.

The canonical target leaf is never created directly. Its pre-existing direct
parent is the private staging parent: it is opened by a retained no-follow FD
with the full ancestor chain descriptor-validated, must be owned by root or
the current EUID, must not be group/world writable, and must have no mutating
ACL. The finalizer never creates that parent. It then creates a random,
mode-0700 private staging directory descriptor-relatively beneath that FD,
stamps and retains its inode/FD, and performs every generated directory
operation, read, write, hash, inventory traversal, and secret scan through
no-follow descriptors rooted there. At publication it uses only an atomic
descriptor-relative no-replace directory rename: Darwin
`renameatx_np(RENAME_EXCL)` or Linux `renameat2(RENAME_NOREPLACE)`. It never
uses a precheck plus an overwrite-capable rename; unavailable native support
fails closed and retains the private staging root for forensic inspection. The
canonical lexical parent/leaf must map to the original staging inode
immediately after publication and again before return; parent/leaf swaps fail
closed.

The local race boundary is arbitrary competing users, not a malicious process
with the staging-parent owner's EUID (which has unrestricted filesystem
control and is out of scope). There is no portable native `mkdir` operation
that returns a directory FD atomically; the pre-existing private parent
prevents an in-scope actor from replacing the random staging name before its
first no-follow FD open.

The generated preservation ledger records protected old-root contracts and
retained `009`/`010`/`012` exclusions. For a chunk with a protected-root
contract, the finalizer recomputes its compact inventory and sealed-report hash
before target creation, before report publication, and after publication in a
postseal verification record; any mismatch fails closed.
This finalizer intentionally does not implement the Final3 production P-policy
map or admission aggregation.

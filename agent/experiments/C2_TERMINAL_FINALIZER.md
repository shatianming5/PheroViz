# C2 terminal-admission finalizer

## M1 production trust boundary

Every production library and CLI entry point first fails with
`M1_EXTERNAL_TRUST_LOCK_UNAVAILABLE`. This repository intentionally contains no
independently signed, deployment-pinned external M1 artifact or adapter, so no
manifest, output path, environment value, flag, or local resource can authorize
finalization. The denial occurs before caller-path resolution or evidence/output
access. Private `*_for_testing` helpers are not CLI-selectable production routes.

Run only against an explicit `c2_terminal_admission_manifest.schema.json`
manifest:

```bash
cd agent
python -m experiments c2-terminal-finalize admission-manifest.json --out final-report.json
```

The manifest names exactly thirteen sealed JSON reports (`001` through `013`),
binds their file and semantic hashes, and carries the exact frozen-universe
hash, DOI-list hash, per-chunk totals, and clean full commit. It never accepts
a live output root. Report paths are relative to the manifest, non-symlinked,
and fail on missing or stale files. The report and manifest file digests in the
final output are computed from the exact bytes parsed and validated in one read.
The output path must not resolve to the manifest or any sealed chunk report;
relative paths and symlink aliases are rejected before writing.
Writes use a fresh exclusive staging file in the normalized output directory,
so legacy predictable staging names cannot alias sealed evidence.
Leaf output symlinks and platforms without descriptor-relative atomic writes
fail closed.
After publication, the visible parent and leaf are revalidated against the
anchored descriptor and captured published-output inode; a detected parent/leaf
swap fails without reporting output success.
For finalizer success, the output parent must already exist as an absolute
no-symlink directory chain owned by root or the current effective uid, with no
group/world-writable component. Every opened ancestor is also inspected for
ACL metadata: on macOS this uses the descriptor-based native extended-ACL API,
not `ls` output; any mutating (or unknown) allow permission is rejected.
Deny-only ACL entries do not grant mutation and may remain. Linux rejects either
POSIX ACL metadata xattr conservatively; unsupported or uninspectable ACL
metadata fails closed. The threat model permits arbitrary *other* local users
to race names and rejects their writable/symlinked/ACL-granted chains; full
same-euid filesystem control is out of scope. The finalizer never creates an
output parent after this trust check. A rejected output path produces no
reported final-report path.
If the descriptor-relative rename fails, one random mode-0600 staging file can
remain in the output directory; it is deliberately left for verified
operational cleanup rather than risking pathname-based deletion of reused data.

Chunks `009`–`012` must be replacement roots and must explicitly exclude their
superseded root IDs. Every DOI needs three terminal attempts and complete,
non-selective source evidence. The final report hashes the manifest bindings,
per-stratum independent DOI-cluster counts, and the complete source DOI-ID
list hash. It never runs or reports trend, equivalence, or metric analyses:
both analysis statuses remain `NOT_RUN`.

Terminal outcome statuses use the closed schema allow-list; a truthy
`terminal` flag cannot admit an unknown, queued, pending, or running status.

If a stratum has fewer than two independent DOI clusters, it emits a blocked
status. A deficient `P=5+` stratum is always
`BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS`; blocked reports set
`claim_status=UNSUPPORTED`.

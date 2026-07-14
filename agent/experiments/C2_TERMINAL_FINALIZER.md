# C2 terminal-admission finalizer

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

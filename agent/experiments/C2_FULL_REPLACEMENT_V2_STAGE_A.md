# C2 full-replacement V2.1 — source-classification Stage A

This isolated V2.1 path supersedes the rejected Stage-A full DOI→P-map design.
It does not alter V1 terminal admission and it does not admit production C2
evidence.

## Production boundary

`phero-experiments c2-full-replacement-finalize MANIFEST --out REPORT` accepts
only the manifest and `--out`. It has no policy, map, digest, root, evidence
skip, environment, or path selector. During Stage A the fixed production policy
resolver fails before reading either path or creating output: no Git-reviewed
package resource and compiled resource-byte SHA-256 exist.

The explicitly test-named in-process APIs accept synthetic ordered acquisition
policies only. They are structural APIs, not a hostile-process security
boundary. No CLI path can select them.

## Two populations; no fabricated P labels

V2.1 validates all 2,463 ordered acquisition DOI in the fixed
`001..013`/`12×200+63` partition. The test policy pins only the frozen ordered
DOI, chunk bindings, replacement plan, code/rule hashes, and source evidence
rules. It **never** contains a P label or cluster for every acquisition DOI.

Every DOI has one ordered `acquisition_dispositions` record derived from sealed
retry2 terminal evidence:

- non-download terminal states become the matching `NON_STRATIFIED_*` result;
- a downloaded DOI with a verified empty source inventory becomes
  `NON_STRATIFIED_DOWNLOADED_NO_VERIFIED_SOURCE`;
- a downloaded DOI with no qualifying canonical case becomes
  `NON_STRATIFIED_SOURCE_NO_CANONICAL_CASE`;
- only a downloaded DOI with complete verified source/canonical evidence and
  one all-case stratum becomes `STRATIFIED_SOURCE_CANONICAL`.

No acquisition-only or non-stratified record has a P label or cluster. P1,
P2, P3_4, and P5PLUS exist only in `stratified_source_classifications`.
Cluster identity is exactly normalized `doi_id`, never a synthetic alias.

## Descriptor-rooted evidence

The manifest is at the trusted evidence-root leaf. All dynamic paths are safe
manifest-relative paths; symlink traversal, hard-link aliases, dot traversal,
duplicate input paths, stale hashes, and parent replacement are rejected.
Every artifact is opened once with descriptor-relative `O_NOFOLLOW`, hashed
while its exact bytes are read, parsed from those captured bytes, and retained
by device/inode identity. The root ancestry, every artifact intermediate
directory, and every leaf are descriptor-validated for root/current-EUID
ownership, non-group/world-writable modes, inspectable non-mutating ACL state,
and (for leaves) regular-file/single-hard-link identity. Before and after
publication, every retained artifact and traversed directory is re-opened from
the root descriptor and must retain its identity, trust metadata, and bytes.

For every chunk, V2.1 verifies initial/retry1/retry2 raw streams,
processed-success, skipped-status, a terminal-outcome ledger, and a sealed
terminal report. The closed raw-status adapter is:

`downloaded`, `no-source-data`, `no-figures`, `no-usable-content`,
`policy-rejected`, `fetch-error`, `download-failed`, `retry-exhausted`.

`DOWNLOADED` derives only from processed-success evidence. It is forbidden in
skipped-status records. Chunk 013 requires exactly global ordinals `2401..2463`
and local ordinals `1..63` on every attempt.

## Source/canonical derivation

Every downloaded DOI must have exactly one verified source inventory, one
raw-source-evidence inventory, and one complete, clean canonical-builder
output. The raw-source inventory binds every candidate ID to descriptor-read
raw bytes and its SHA-256; each candidate, panel, and source table must retain
that same raw-byte binding. Candidate, proposal, review, and canonical parent
bindings are separate no-follow hashed artifacts tied to the same normalized
parent DOI. Builder input/output, code/rule hashes, full parent DOI hash, and
no-model-selection flags are checked.

Each canonical case recomputes verified panel membership from raw source-table
artifacts. Panel/source candidate parent DOI must equal the case parent DOI;
cross-DOI candidates, duplicate panels/candidates/fingerprints, missing source
tables, or asserted count/P disagreement reject. A multi-panel case requires
distinct verified same-DOI source cases. A canonical source-case record is
provenance support for a panel; it is not an experiment case selected into the
DOI's all-case set.

All eligible canonical cases for a DOI are retained, sorted by Unicode
`case_id`, and hashed. P derives solely from recomputed panel count:

| qualified panels | derived stratum |
| --- | --- |
| 1 | P1 |
| 2 | P2 |
| 3–4 | P3_4 |
| ≥5 | P5PLUS |

Multiple eligible cases are permitted only when all derive the same stratum;
each has equal case weight under `DOI_CASE_AGGREGATION_V1`. Mixed strata reject
instead of selecting a favorable case.

Coverage counts only DOI-level stratified classifications. P1 is reported but
outside the coverage gate. P2, P3_4, and P5PLUS each require at least two
independent DOI clusters. P5PLUS failure takes blocked-status precedence. All
outputs retain `NOT_RUN` trend/equivalence fields and never generate an
analysis.

## Output protocol

The output parent and evidence root must be trusted, distinct, and
non-containing. `--out` must be absent and outside all validated evidence;
lexical, device/inode, and hard-link aliases reject.

Publication uses only descriptor-relative same-directory staging plus
hard-link no-replace publication. It fsyncs the staging file and parent,
verifies inode identity, then identity-checks staging before unlink. There is
no rename/overwrite/copy fallback. Unsupported primitives, output races,
symlinks, cleanup uncertainty, or parent/leaf replacement return no success.
A failed link leaves a restrictive staging file rather than deleting a
potentially reused pathname.

The validated admission retains immutable evidence snapshots, not a mutable
authoritative report. Any diagnostic report projection is read-only. The writer
revalidates the complete evidence snapshot and rebuilds/validates the report
from it immediately before publication; it accepts no caller-supplied report
object or self-hash as a publication authority.

## Stage B gate

Stage B has a closed schema and a package-internal, no-selector resolver
foundation, but it has no production resource, compiled resource-byte digest,
or approved evidence commitments. The resolver therefore still fails before it
can read a manifest or evidence. A future resource needs forensic approval and
must pin frozen universe bindings, replacement plan, source/canonical rule
hashes, and raw-source commitments. It must not add a full acquisition DOI→P
map. Only verified source/canonical case evidence may derive a P stratum. Until
then, no production manifest, root, report, scientific claim, or analysis is
authorized.

The separate Stage-B source-extension code-attestation registry is also
non-admissive. When a source extension is independently reviewed, its
manifest-only attestation must provide opaque implementation/attestation commit
identities, manifest/blob-set digests, and the exact ordered runtime path/role
roster required by that registry. The registry does not read Git, a worktree,
the manifest, or evidence, and does not authorize the extension by itself.
Its self-omitting canonical SHA-256 is its only registry identity; it accepts
no declared free-form identifier.

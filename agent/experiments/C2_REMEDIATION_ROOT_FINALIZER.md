# C2 fresh-remediation root finalizer

## Owner execution and M1 admission boundaries

The fixed `OWNER_AUTHORIZED_NON_INDEPENDENT` capability permits remediation,
source build/replay, and non-admissive sealing. It does not supply independent
M1 verification and cannot authorize admission, publication, a scientific
outcome, or an outcome-dependent input choice. The finalizer therefore produces
evidence only; the independent admission boundary remains deny-only.

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
  --worktree /absolute/clean-ca98442-worktree \
  --source-bearing-v2
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
The raw acquisition/downloader binding always remains the frozen acquisition
commit `ca98442b9e805110089b03083cb240e19b58d4a2`; no source-extension
authorization can replace it or claim that extension code acquired raw data.
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
source-less. The default remains fail-closed for any source-bearing row:
it emits exhaustive acquisition evidence and
`control/source_classification_blocked.json`, then fails with
`NOT_SEALABLE_SOURCE_CLASSIFICATION_BUILDER_REQUIRED`; it never fabricates an
empty canonical/P or sealed-report chain.

`--source-bearing-v2` is an explicit opt-in for the deterministic V2 extension.
Every source-bearing descriptor must then be the closed
`c2-source-evidence-v2` form with typed assets, exact declared format tuples,
and descriptor-bound candidate hints. The extension classifies bytes from
retained FDs before checking declared role/format; it fully accounts every ZIP
(including XLSX and nested containers), writes each derived member, proves the
member-to-consumption/candidate bijection, runs only
`C2_V2_STRUCTURAL_REVIEW_V1` (no model), recomputes panels/P from all retained
canonical cases, and independently replays the chain during staging and again
immediately before atomic publication. Missing,
legacy, malformed, or ambiguous evidence fails closed. A byte-valid table with
no deterministic source mapping receives an explicit, hash-bound source-only
exclusion; it never falls back to an empty chain.

The package-internal Stage-B source-extension registry is compile-pinned and
runtime-verified. It binds the reviewed implementation/attestation commits and
the closed runtime path/blob roster but carries no P labels, clusters, outcomes,
or admission authority. Runtime bytes, the registry resource, or its compiled
pin changing independently causes source-bearing V2 sealing to fail closed.

## Fixed chunk-001 source pilot

`c2_m5_source_pilot` is the only supported adapter from the frozen downloader's
operational files to a strict chunk-001 raw root. A stdlib-only caller first
hashes the bootstrap bytes against an externally distributed release pin. The
authenticated bootstrap then verifies the externally pinned manifest digest
and manifest-only commit before importing any checkout helper. It binds the
frozen chunk, universe, summary, downloader commit, and protected legacy root,
and takes an exclusive parent lease. Each of `initial`, `retry1`, and `retry2`
runs in a separate credential-free workspace behind the
HTTPS/public-IP/size network guard. The adapter preserves the raw postfetch log
while deriving disjoint downloaded-only `processed.txt` and source-less
`_skipped.txt` files; assets from any successful pass are retained even when
retry2 is source-less.

```bash
cd agent
export LANG=C LC_ALL=C
export C2_M5_EXPECTED_ATTESTATION_COMMIT=__M5_ATTESTATION_COMMIT__
export C2_M5_EXPECTED_MANIFEST_SHA256=__M5_MANIFEST_SHA256__
export C2_M5_EXPECTED_BOOTSTRAP_SHA256=__M5_BOOTSTRAP_SHA256__
export C2_M5_EXPECTED_RELEASE_RUNNER_SHA256=__M5_RELEASE_RUNNER_SHA256__
export C2_M5_EXPECTED_PYTHON_SHA256=4b42b1a117605cafc8607b67b0892a609c2cd125012dd56288abeed8c89cdfb1
export C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256=0432398c0d1b2ff35a741b2758dccfb08d9f1aad39abac2c4da9cfcc84e6d225
PYTHON=/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9
PYTHON_LIBRARY=/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Python3
PYTHON_STDLIB_ZIP_PARENT=/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib
PYTHON_STDLIB_ZIP="$PYTHON_STDLIB_ZIP_PARENT/python39.zip"
PYTHON_STDLIB="$PYTHON_STDLIB_ZIP_PARENT/python3.9"
require_root_safe_path() {
  local metadata owner remainder mode kind acl_output
  metadata=$(/usr/bin/stat -f '%u:%Lp:%HT' "$1") || exit 1
  owner=${metadata%%:*}
  remainder=${metadata#*:}
  mode=${remainder%%:*}
  kind=${metadata##*:}
  [ "$owner" = 0 ] && [ "$kind" = "$2" ] || exit 1
  [ $((8#$mode & 8#022)) -eq 0 ] || exit 1
  acl_output=$(/bin/ls -lde "$1") || exit 1
  case "$acl_output" in *" allow "*) exit 1 ;; esac
}
for path in \
  / \
  /Library \
  /Library/Developer \
  /Library/Developer/CommandLineTools \
  /Library/Developer/CommandLineTools/Library \
  /Library/Developer/CommandLineTools/Library/Frameworks \
  /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework \
  /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions \
  /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9 \
  /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin \
  "$PYTHON_STDLIB_ZIP_PARENT" \
  "$PYTHON_STDLIB"; do
  require_root_safe_path "$path" Directory
done
require_root_safe_path "$PYTHON" "Regular File"
require_root_safe_path "$PYTHON_LIBRARY" "Regular File"
STDLIB_METADATA_UNSAFE=$(
  /usr/bin/find -x "$PYTHON_STDLIB" \
    \( -path "$PYTHON_STDLIB/site-packages" -o \
       -path "$PYTHON_STDLIB/config-3.9-darwin" \) -prune -o \
    \( -type l -o ! -user root -o -perm +022 -o \
       \( ! -type f ! -type d \) \) -print -quit
) || exit 1
[ -z "$STDLIB_METADATA_UNSAFE" ] || exit 1
STDLIB_ACLS=$(
  /usr/bin/find -x "$PYTHON_STDLIB" \
    \( -path "$PYTHON_STDLIB/site-packages" -o \
       -path "$PYTHON_STDLIB/config-3.9-darwin" \) -prune -o \
    -exec /bin/ls -lde {} \;
) || exit 1
case "$STDLIB_ACLS" in *" allow "*) exit 1 ;; esac
[ "$(/usr/bin/shasum -a 256 "$PYTHON" | /usr/bin/cut -d " " -f 1)" = \
  "$C2_M5_EXPECTED_PYTHON_SHA256" ] || exit 1
[ "$(/usr/bin/shasum -a 256 "$PYTHON_LIBRARY" | /usr/bin/cut -d " " -f 1)" = \
  "$C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256" ] || exit 1
[ "$(/usr/bin/stat -f '%u:%Lp:%HT' "$PYTHON_STDLIB_ZIP_PARENT")" = \
  "0:755:Directory" ] || exit 1
[ ! -e "$PYTHON_STDLIB_ZIP" ] && [ ! -L "$PYTHON_STDLIB_ZIP" ] || exit 1
BOOTSTRAP="$PWD/c2_m5_source_pilot_bootstrap.py"
m5() {
  /usr/bin/env -i \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  HOME=/var/empty LANG=C LC_ALL=C \
  C2_M5_EXPECTED_ATTESTATION_COMMIT="$C2_M5_EXPECTED_ATTESTATION_COMMIT" \
  C2_M5_EXPECTED_MANIFEST_SHA256="$C2_M5_EXPECTED_MANIFEST_SHA256" \
  C2_M5_EXPECTED_BOOTSTRAP_SHA256="$C2_M5_EXPECTED_BOOTSTRAP_SHA256" \
  C2_M5_EXPECTED_PYTHON_SHA256="$C2_M5_EXPECTED_PYTHON_SHA256" \
  C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256="$C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256" \
  "$PYTHON" -I -S -B -c '
import hashlib, os, stat, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
metadata = os.fstat(fd)
payload = bytearray()
try:
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        payload.extend(block)
        if len(payload) > 2 * 1024 * 1024:
            raise SystemExit("M5 bootstrap is oversized")
finally:
    os.close(fd)
if (
    not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != os.geteuid()
    or metadata.st_nlink != 1
    or metadata.st_mode & 0o022
    or hashlib.sha256(payload).hexdigest()
       != os.environ["C2_M5_EXPECTED_BOOTSTRAP_SHA256"]
):
    raise SystemExit("M5 external bootstrap pin mismatch")
sys.argv = sys.argv[1:]
exec(
    compile(bytes(payload), path, "exec"),
    {"__name__": "__main__", "__file__": path, "__package__": None},
)
' "$BOOTSTRAP" "$@"
}
```

Execution refuses to publish a raw root unless at least one descriptor-bound
`c2-source-evidence-v2` asset exists. The `finalize` subcommand repeats that
read-only source-bearing check before invoking the existing V2 finalizer with
the fixed target name. It still returns only non-independent, non-admissive
evidence.

All three commands require the reviewed external release pins and an exact
clean repository tree. Direct execution of the checkout bootstrap without
those pins is unsupported and refuses. After pre-authentication, the bootstrap
compares the Git index to the pinned HEAD tree and directly hashes every
worktree file with Git blob framing; it does not invoke checkout-controlled
clean/smudge filters. It rejects `__pycache__`, `.pyc`, `.pyo`, `.pytest_cache`,
or any other tracked, untracked, or ignored extra. Run tests with
`PYTHONDONTWRITEBYTECODE=1` and remove test/cache artifacts before production;
never place the raw or sealed roots inside this checkout. Replace the four
remaining `__M5_*__` placeholders only with values published by the reviewed
release; the externally checked CPython digest must also match that release.

The release topology is part of the trust contract:

1. remove the superseded source-extension test/runtime manifests and commit the
   complete implementation, parser, dependency-manifest, and test bytes;
2. add only the regenerated test attestation in its direct child commit;
3. add only the regenerated production runtime manifest in the next commit;
4. rotate the Stage-B registry, compile pin, and hard-coded registry test;
5. commit the final reviewed M5 runtime anchor (an empty anchor is permitted when
   review requires no byte changes);
6. add only `c2_m5_source_pilot_attestation_v1.json` in its direct child commit.

Verify each one-file attestation commit with `git diff-tree --no-commit-id
--name-status -r HEAD`, then run:

```bash
/usr/bin/env -i \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  HOME=/var/empty LANG=C LC_ALL=C \
  C2_M5_EXPECTED_RELEASE_RUNNER_SHA256="$C2_M5_EXPECTED_RELEASE_RUNNER_SHA256" \
  C2_M5_EXPECTED_ATTESTATION_COMMIT="$C2_M5_EXPECTED_ATTESTATION_COMMIT" \
  C2_M5_EXPECTED_MANIFEST_SHA256="$C2_M5_EXPECTED_MANIFEST_SHA256" \
  C2_M5_EXPECTED_BOOTSTRAP_SHA256="$C2_M5_EXPECTED_BOOTSTRAP_SHA256" \
  C2_M5_EXPECTED_PYTHON_SHA256="$C2_M5_EXPECTED_PYTHON_SHA256" \
  C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256="$C2_M5_EXPECTED_PYTHON_LIBRARY_SHA256" \
  "$PYTHON" -I -S -B -c '
import hashlib, os, stat, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
metadata = os.fstat(fd)
payload = bytearray()
try:
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        payload.extend(block)
        if len(payload) > 2 * 1024 * 1024:
            raise SystemExit("M5 release runner is oversized")
finally:
    os.close(fd)
if (
    not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != os.geteuid()
    or metadata.st_nlink != 1
    or metadata.st_mode & 0o022
    or hashlib.sha256(payload).hexdigest()
       != os.environ["C2_M5_EXPECTED_RELEASE_RUNNER_SHA256"]
):
    raise SystemExit("M5 external release-runner pin mismatch")
sys.argv = [path]
exec(
    compile(bytes(payload), path, "exec"),
    {"__name__": "__main__", "__file__": path, "__package__": None},
)
' "$PWD/c2_m5_release_test_runner.py"
```

Only after that release runner exits zero and all four independent freeze
reviews report no BLOCKER/HIGH may the same shell start acquisition:

```bash
m5 execute \
  --raw-root /absolute/output/ccby_sr_npj_chunk001_rerun3_raw_ca98442 \
  --source-chunk /absolute/chunks/chunk_001.jsonl \
  --frozen-universe /absolute/universe.jsonl \
  --freeze-summary /absolute/freeze_summary.json \
  --worktree /absolute/clean-ca98442-worktree \
  --workers 4

m5 validate \
  --raw-root /absolute/output/ccby_sr_npj_chunk001_rerun3_raw_ca98442 \
  --source-chunk /absolute/chunks/chunk_001.jsonl \
  --frozen-universe /absolute/universe.jsonl \
  --freeze-summary /absolute/freeze_summary.json \
  --worktree /absolute/clean-ca98442-worktree

m5 finalize \
  --raw-root /absolute/output/ccby_sr_npj_chunk001_rerun3_raw_ca98442 \
  --target-root /absolute/output/ccby_sr_npj_chunk001_rerun3_clean_ca98442 \
  --source-chunk /absolute/chunks/chunk_001.jsonl \
  --frozen-universe /absolute/universe.jsonl \
  --freeze-summary /absolute/freeze_summary.json \
  --worktree /absolute/clean-ca98442-worktree
```

The release runner verifies the native Python/stdlib binding, the externally
pinned live M5 manifest, and both hash-bound dependency archives before
creating its own ACL-checked mode-0700 temporary root. It materializes all test
and application sources from the authenticated manifest into a private source
snapshot; the checkout is never on the child import path. It disables ambient
pytest plugins and conftests, fixes the config/rootdir, uses only the fixed test
roster, and rejects unreviewed session-control topology before execution. A
trusted in-session supervisor, which never imports pytest or test code, alone
owns the completion writer. It waits for the exact worker, emits one bounded
status frame, closes the writer, and remains the unreaped SID leader; the outer
runner accepts status only after EOF and leader-identity verification. Every
approved nested spawn uses the registry binding pinned when containment is
installed and remains in the leader group until its parent appends a pending
registration and explicitly releases it to create the new PGID; the trampoline
then adds a live anchor before target execution.
Root-session mode is process-nonreentrant after containment installation.
Cleanup freezes two identical stopped-state censuses, rereads the pinned
descriptor, attempts every independently validated anchored group, verifies
only the trusted supervisor remains, and then kills and reaps that exact child.
Any registry, census, signal, cleanup, or survivor error makes release fail.

This is deterministic cleanup for the hash-attested process topology, not a
sandbox for hostile same-UID code. Unapproved session creation, daemonization,
native process control, or registry bypass is an attestation violation and
requires a new implementation anchor and review. The contract does not claim
containment across hostile same-UID interference, outer `SIGKILL` or fatal
runtime crash, reboot, or kernel failure; those require a VM or a separately
provisioned UID. A record-only HUP/INT/TERM supervisor remains active across
attestation, setup, tests, post-test verification, and mode-0700 temporary-root
cleanup. The acquisition adapter applies the same record-only lifecycle before
lease creation and before each downloader spawn, and restores the caller's
handlers and exact signal mask on every exit. It blocks TERM while assigning the
spawned worker, then handles cleanup requests record-only. A trusted live group
leader waits the exact downloader worker and reports status through a
non-inherited pipe;
cleanup first asks that leader to reap the worker, kills the still-pinned group,
drains captured output through EOF under the fixed byte cap, and only then
reaps the leader. Unpublished staging is discarded and closed while the
exclusive acquisition lease is still held; a failed or unverifiable rollback
or process cleanup leaves that lease in place and blocks a later writer.
Target-root construction
also removes any staging directory created before its constructor can return.
Publication blocks lifecycle signals around one commit point: a signal observed
before that point aborts and rolls back, while one arriving after the point
commits and returns the complete non-admissive root. Evidence records the
verified retained PID set rather than assuming it is empty. Before Python
starts, the shell gate rejects unsafe
ownership, write bits, symlinks, or mutating Darwin ACLs across the fixed
executable, runtime library, stdlib, and their ancestry. Bare `pytest`, ambient
`PYTHONPATH`, user/site packages, and unpinned plugins are not release evidence.

The M5 bootstrap rechecks the manifest-only topology, all Git blobs, the exact
root-owned Apple CPython 3.9.6 executable/runtime-library/micro/SOABI/native
arm64 process, the required absence of the higher-priority `python39.zip`
import root, and the 775-file root-owned stdlib inventory, plus a vendored
282-file dependency archive before execution. The wrapper invokes the resolved
framework executable under an empty environment, so ambient `DEVELOPER_DIR`,
Python, and dynamic-loader variables cannot select code before authentication.
Every runtime ancestor and stdlib entry must be root-owned and
non-group/world-writable.

Each acquisition attempt additionally has a 10,000-request, 8-GiB cumulative
response budget, a 16-MiB bounded subprocess log, and one three-hour deadline
covering subprocess execution plus source/archive harvesting. The finalizer
replays the request/byte budget against the downloaded-status harvest and
applies the no-candidate-hints/real-source-asset gate to the same retained raw
snapshot that it copies and seals. The resulting qualification hash is part of
the sealed report trust chain. After every report and validation write, the
finalizer emits `sealed_report_v1/terminal_manifest.json`; its returned SHA-256
is the external anchor for the complete published root.

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
before target creation and again during prepublication sealing; any mismatch
fails closed. All reports, inventories, sidecars, and validation records are
written and descriptor-revalidated in private staging before the atomic rename.
After publication it performs only read-only inode/identity checks and never
writes into the published root.
This finalizer intentionally does not implement the Final3 production P-policy
map or admission aggregation.

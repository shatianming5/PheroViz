#!/usr/bin/env bash
# Idempotent, no-delete union of all PheroViz DOI directories.
#
# Consolidates the qualifying corpus from the local crawl output plus the
# remote crawl nodes into a single directory (outputs/extreme_merged).
# rsync -a is additive and idempotent, so re-running only copies new/changed
# DOI directories; nothing is ever deleted from the merged corpus. A final
# fail-closed gate moves figure-only/source-only payloads to
# ``_rejected_no_source/`` rather than retaining or deleting them.
#
# All cluster-specific access details are read from environment variables so
# that no host, user, internal IP, or SSH port is hard-coded in this file.
# Required (the script fails fast if any is unset):
#   PV_CLUSTER_HOST  public SSH host/DNS of the crawl cluster gateway
#   PV_CLUSTER_USER  SSH login user on the cluster nodes
#   PV_N67_IP        internal IP of the ProxyJump-only node (reached via N9)
# Optional (sensible defaults shown):
#   PV_PORT_N9 / PV_PORT_N134 / PV_PORT_N138 / PV_PORT_N183 / PV_PORT_N136
#                    per-node SSH ports (default: 22)
#   PV_BASE          local repo's nature_download dir (auto-detected from this
#                    script's location: documentation/ is a sibling of it)
#   PV_LOG           consolidation log path (default: $BASE/outputs/...)
#   PV_REMOTE_CONTENT   default remote content path used by most nodes
#   PV_N134_CONTENT / PV_N138_CONTENT / PV_N183_CONTENT / PV_N136_CONTENT
#                    per-node content-path overrides (e.g. scratch disks / tmp)
#   PV_N138_LEGACY / PV_N136_LEGACY   optional legacy SSD shard paths
set -u

: "${PV_CLUSTER_HOST:?set PV_CLUSTER_HOST to the crawl cluster SSH host}"
: "${PV_CLUSTER_USER:?set PV_CLUSTER_USER to the crawl cluster SSH user}"
: "${PV_N67_IP:?set PV_N67_IP to the internal IP of the proxy-only node}"

PV_PORT_N9="${PV_PORT_N9:-22}"
PV_PORT_N134="${PV_PORT_N134:-22}"
PV_PORT_N138="${PV_PORT_N138:-22}"
PV_PORT_N183="${PV_PORT_N183:-22}"
PV_PORT_N136="${PV_PORT_N136:-22}"

# Auto-detect the local repo's nature_download dir from this script's location
# (documentation/ is a sibling of nature_download/). Override with PV_BASE.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${PV_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)/nature_download}"
MERGED="$BASE/outputs/extreme_merged"
LOG="${PV_LOG:-$BASE/outputs/pv_consolidate_v2.log}"

# Remote crawl layout. Defaults assume ~USER/pheroviz_crawl; override per node.
PV_REMOTE_CONTENT="${PV_REMOTE_CONTENT:-/home/$PV_CLUSTER_USER/pheroviz_crawl/outputs/extreme_content}"
PV_N134_CONTENT="${PV_N134_CONTENT:-/mnt/scratch/$PV_CLUSTER_USER/pheroviz_crawl/outputs/extreme_content}"
PV_N138_CONTENT="${PV_N138_CONTENT:-/tmp/pheroviz_crawl/outputs/extreme_content}"
PV_N183_CONTENT="${PV_N183_CONTENT:-$PV_REMOTE_CONTENT}"
PV_N136_CONTENT="${PV_N136_CONTENT:-/tmp/pheroviz_crawl/outputs/extreme_content}"

mkdir -p "$MERGED" "$(dirname "$LOG")"

say() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

doi_count() {
  (
    cd "$MERGED" &&
      ls -1 | grep -v '^_' | wc -l | tr -d ' '
  )
}

xlsx_count() {
  find "$MERGED" -type f -iname '*.xlsx' | wc -l | tr -d ' '
}

quarantine_incomplete_articles() {
  PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}" python3 - "$MERGED" <<'PY'
import sys
from corpus.completeness import quarantine_incomplete_article_dirs

moved = quarantine_incomplete_article_dirs(sys.argv[1])
print(f"quarantined_incomplete_articles={len(moved)}")
for path in moved:
    print(f"  {path}")
PY
}

report_result() {
  local label=$1 before=$2 rc=$3 after
  after=$(doi_count)
  say "$label rc=$rc delta=$((after - before)) total=$after"
}

sync_local() {
  local label=$1 source=$2 before rc
  before=$(doi_count)
  if [ ! -d "$source" ]; then
    say "$label MISSING source=$source delta=0 total=$before"
    return
  fi
  rsync -a "$source/" "$MERGED/"
  rc=$?
  report_result "$label" "$before" "$rc"
}

sync_direct() {
  local label=$1 port=$2 source=$3 before rc
  before=$(doi_count)
  rsync -a --timeout=180 \
    -e "ssh -p $port -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=30" \
    "$PV_CLUSTER_USER@$PV_CLUSTER_HOST:${source%/}/" "$MERGED/"
  rc=$?
  report_result "$label" "$before" "$rc"
}

sync_proxy67() {
  local source=$1 before rc
  local ssh_cmd
  ssh_cmd="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=30 -o ProxyCommand='ssh -p $PV_PORT_N9 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=30 -W $PV_N67_IP:22 $PV_CLUSTER_USER@$PV_CLUSTER_HOST'"
  before=$(doi_count)
  rsync -a --timeout=180 -e "$ssh_cmd" \
    "$PV_CLUSTER_USER@$PV_N67_IP:${source%/}/" "$MERGED/"
  rc=$?
  report_result "n67-current" "$before" "$rc"
}

say "START merged=$MERGED current=$(doi_count)"

sync_local "local-extreme_content" "$BASE/outputs/extreme_content"
sync_local "local-elife_2020_2026" "$BASE/outputs/elife_2020_2026"

sync_direct "n9-current" "$PV_PORT_N9" "$PV_REMOTE_CONTENT"
sync_direct "n134-current-scratch" "$PV_PORT_N134" "$PV_N134_CONTENT"
sync_proxy67 "$PV_REMOTE_CONTENT"
sync_direct "n138-current" "$PV_PORT_N138" "$PV_N138_CONTENT"
sync_direct "n183-current" "$PV_PORT_N183" "$PV_N183_CONTENT"
sync_direct "n136-current" "$PV_PORT_N136" "$PV_N136_CONTENT"

# Optional legacy SSD shards (only synced if the override vars are set).
if [ -n "${PV_N138_LEGACY:-}" ]; then
  sync_direct "n138-legacy-ssd" "$PV_PORT_N138" "$PV_N138_LEGACY"
fi
if [ -n "${PV_N136_LEGACY:-}" ]; then
  sync_direct "n136-legacy-ssd" "$PV_PORT_N136" "$PV_N136_LEGACY"
fi

say "Applying figure+source+metadata completeness gate"
quarantine_incomplete_articles | while IFS= read -r line; do
  say "$line"
done
say "DONE doi_dirs=$(doi_count) xlsx=$(xlsx_count) size=$(du -sh "$MERGED" | awk '{print $1}')"

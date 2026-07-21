#!/usr/bin/env bash
# Build (or consume) a sealed C2 manifest, then run the P5+ C2 harness.
#
# Default mode is gateway-free --dry-run.  --execute is deliberately explicit
# for the eventual serial owner launch; it is never inferred from environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/opt/homebrew/Caskroom/miniforge/base/bin/python3"

die() {
  echo "run_c2_when_ready.sh: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage (assemble then gateway-free dry-run):
  bash files/run_c2_when_ready.sh \
    --benchmark-out nature_download/outputs/c2_verified_p5_benchmark \
    --run-out nature_download/outputs/c2_verified_p5_dry_run \
    --seed 20260721 \
    --candidate <rebuilt_verified/candidates.jsonl> \
    --proposed <review_batch/proposed.jsonl> \
    --reviews <review_batch/reviews.jsonl> \
    --evidence <review_batch/evidence.json>

Repeat --candidate for each rebuilt candidate batch. Repeat the complete
--proposed/--reviews/--evidence triple for every independent review batch.
Candidates must have been rebuilt with `build-cases --evidence`; this wrapper
never changes curation_status or eligibility fields.

For a derived multi-panel proposal batch, review that derived batch again and
use its resulting evidence to rebuild candidates. Do not also pass the older
source-single review bundle when it contains the same candidate IDs: sealed
assembly rejects cross-bundle duplicates.

Usage (reuse an already sealed manifest):
  bash files/run_c2_when_ready.sh \
    --dataset-manifest <benchmark_manifest.json> \
    --manifest-data-root <absolute root encoded in that manifest> \
    --run-out nature_download/outputs/c2_verified_p5_dry_run

Common options:
  --repo-root <checkout>       Default: checkout containing this script
  --python <python>            Default: Miniforge Python 3
  --min-panels <N>             Default: 5; values below 5 are rejected
  --model <name>               Default: gpt-5.6-sol
  --dry-run                    Default; expands specs only, no gateway calls
  --execute                    Explicit future production launch only

The harness always uses:
  --dataset-manifest, --profile benchmark, --case-kind multi,
  --rounds-per-case 1, and the requested P5+ threshold.
It never exposes --input, --legacy-manifest, --normalizer-exploratory, or a
bypass path, so those unsealed modes cannot be selected through this wrapper.
Relative paths are interpreted relative to --repo-root.
EOF
}

REPO_ROOT="$DEFAULT_REPO_ROOT"
PYTHON_BIN="$DEFAULT_PYTHON"
BENCHMARK_OUT=""
DATASET_MANIFEST=""
MANIFEST_DATA_ROOT=""
RUN_OUT=""
SEED=""
MIN_PANELS=5
MODEL="gpt-5.6-sol"
MODE="dry-run"
declare -a CANDIDATES=()
declare -a PROPOSED=()
declare -a REVIEWS=()
declare -a EVIDENCE=()

while (($#)); do
  case "$1" in
    --repo-root)
      (($# >= 2)) || die "--repo-root requires a value"
      REPO_ROOT="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --benchmark-out)
      (($# >= 2)) || die "--benchmark-out requires a value"
      BENCHMARK_OUT="$2"
      shift 2
      ;;
    --dataset-manifest)
      (($# >= 2)) || die "--dataset-manifest requires a value"
      DATASET_MANIFEST="$2"
      shift 2
      ;;
    --manifest-data-root)
      (($# >= 2)) || die "--manifest-data-root requires a value"
      MANIFEST_DATA_ROOT="$2"
      shift 2
      ;;
    --run-out)
      (($# >= 2)) || die "--run-out requires a value"
      RUN_OUT="$2"
      shift 2
      ;;
    --seed)
      (($# >= 2)) || die "--seed requires a value"
      SEED="$2"
      shift 2
      ;;
    --min-panels)
      (($# >= 2)) || die "--min-panels requires a value"
      MIN_PANELS="$2"
      shift 2
      ;;
    --model)
      (($# >= 2)) || die "--model requires a value"
      MODEL="$2"
      shift 2
      ;;
    --candidate)
      (($# >= 2)) || die "--candidate requires a value"
      CANDIDATES+=("$2")
      shift 2
      ;;
    --proposed)
      (($# >= 2)) || die "--proposed requires a value"
      PROPOSED+=("$2")
      shift 2
      ;;
    --reviews)
      (($# >= 2)) || die "--reviews requires a value"
      REVIEWS+=("$2")
      shift 2
      ;;
    --evidence)
      (($# >= 2)) || die "--evidence requires a value"
      EVIDENCE+=("$2")
      shift 2
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --execute)
      MODE="execute"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
[[ -d "$REPO_ROOT/.git" ]] || die "--repo-root is not a Git checkout: $REPO_ROOT"
cd "$REPO_ROOT"
[[ "$MIN_PANELS" =~ ^[0-9]+$ && "$MIN_PANELS" -ge 5 ]] ||
  die "--min-panels must be an integer >= 5 for C2-extreme"
[[ -n "$RUN_OUT" ]] || die "--run-out is required"
[[ ! -e "$RUN_OUT" ]] || die "--run-out must not exist: $RUN_OUT"

if [[ -n "$DATASET_MANIFEST" ]]; then
  [[ -z "$BENCHMARK_OUT" && -z "$SEED" ]] ||
    die "--dataset-manifest cannot be combined with --benchmark-out or --seed"
  [[ ${#CANDIDATES[@]} -eq 0 && ${#PROPOSED[@]} -eq 0 &&
     ${#REVIEWS[@]} -eq 0 && ${#EVIDENCE[@]} -eq 0 ]] ||
    die "assembly inputs cannot be combined with --dataset-manifest"
  [[ -f "$DATASET_MANIFEST" ]] || die "manifest does not exist: $DATASET_MANIFEST"
else
  [[ -n "$BENCHMARK_OUT" && -n "$SEED" ]] ||
    die "assembly mode requires --benchmark-out and --seed"
  [[ ${#CANDIDATES[@]} -gt 0 ]] || die "assembly mode requires --candidate"
  [[ ${#PROPOSED[@]} -gt 0 &&
     ${#PROPOSED[@]} -eq ${#REVIEWS[@]} &&
     ${#PROPOSED[@]} -eq ${#EVIDENCE[@]} ]] ||
    die "assembly mode requires equal non-zero proposed/reviews/evidence triples"

  BUILD_ARGS=(
    --repo-root "$REPO_ROOT"
    --python "$PYTHON_BIN"
    --out "$BENCHMARK_OUT"
    --seed "$SEED"
    --minimum-multi-panels "$MIN_PANELS"
  )
  for item in "${CANDIDATES[@]}"; do BUILD_ARGS+=(--candidate "$item"); done
  for index in "${!PROPOSED[@]}"; do
    BUILD_ARGS+=(
      --proposed "${PROPOSED[$index]}"
      --reviews "${REVIEWS[$index]}"
      --evidence "${EVIDENCE[$index]}"
    )
  done
  "$PYTHON_BIN" "$REPO_ROOT/files/build_c2_benchmark.py" "${BUILD_ARGS[@]}"
  DATASET_MANIFEST="$BENCHMARK_OUT/benchmark_manifest.json"
fi

[[ -n "$MANIFEST_DATA_ROOT" ]] || MANIFEST_DATA_ROOT="$REPO_ROOT"
[[ "$MANIFEST_DATA_ROOT" = /* ]] ||
  die "--manifest-data-root must be an absolute path"

RUN_ARGS=(
  --dataset-manifest "$DATASET_MANIFEST"
  --manifest-data-root "$MANIFEST_DATA_ROOT"
  --output "$RUN_OUT"
  --profile benchmark
  --rounds-per-case 1
  --model "$MODEL"
  --min-panels "$MIN_PANELS"
  --case-kind multi
  --python "$PYTHON_BIN"
)
if [[ "$MODE" == "dry-run" ]]; then
  echo "C2 sealed dry-run: no --execute and no gateway call." >&2
  RUN_ARGS+=(--dry-run)
else
  echo "C2 sealed execution explicitly requested; serial owner controls gateway use." >&2
  RUN_ARGS+=(--execute)
fi

exec "$PYTHON_BIN" "$REPO_ROOT/agent/run_c2_v3.py" "${RUN_ARGS[@]}"

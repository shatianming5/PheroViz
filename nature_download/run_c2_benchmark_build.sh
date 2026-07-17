#!/bin/bash
set -e
cd /Users/tommy/Downloads/mayi/PheroViz-c2-closure-integration-terra/nature_download

# Stop the miner
pkill -f run_miner_loop.sh || true
pkill -f nature_all_in_one.py || true

# Step 1: Build the corpus manifest from the downloaded content
echo "Building manifest..."
python3 nature_all_in_one.py build-manifest \
    --jsonl outputs/search_auto/articles.jsonl \
    --content-root outputs/nature_content \
    --out outputs/c2_extreme_manifest \
    --require-cc-by \
    --source-data-origin supplementary_information

# Step 2: Build the cases
echo "Building cases..."
python3 nature_all_in_one.py build-cases \
    --corpus-manifest outputs/c2_extreme_manifest/corpus_manifest.jsonl \
    --content-root outputs/nature_content \
    --out outputs/c2_extreme_cases

# Step 3: Propose singles
echo "Proposing cases..."
python3 nature_all_in_one.py propose-cases \
    --candidates outputs/c2_extreme_cases/candidates.jsonl \
    --out outputs/c2_extreme_proposals

echo "Done building pipeline for P>=5 dataset."

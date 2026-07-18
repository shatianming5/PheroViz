#!/bin/bash
export ANTHROPIC_BASE_URL="http://1.14.177.180:4142"
export ANTHROPIC_AUTH_TOKEN="sk-intern"
export VLM_MODEL="claude-sonnet-5"
export VLM_REQUIRED="1"
export LLM_MODEL="gpt-5.6-sol"

python3 -m experiments run experiments/matrices/c2_extreme_final_benchmark_v3_repaired.yaml

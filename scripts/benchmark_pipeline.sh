#!/bin/bash
set -e

# Default Values
TYPE="cod"       # cot or cod
SCENARIO="medium"  # easy, medium, hard
BASE_TARGET="unsloth/Qwen2.5-14B-Instruct"
BASE_DRAFT="Qwen/Qwen2.5-0.5B-Instruct"

# Parse Flags
while getopts "t:s:" opt; do
  case $opt in
    t) TYPE="$OPTARG" ;;
    s) SCENARIO="$OPTARG" ;;
    *) echo "Usage: $0 -t [cot|cod] -s [easy|medium|hard]"; exit 1 ;;
  esac
done

echo "========================================================"
echo "🚀 Starting Benchmark Pipeline | Type: $TYPE | Scenario: $SCENARIO"
echo "========================================================"

# Define Adapter Paths
ADAPTER_TARGET="models/target_${TYPE}_${SCENARIO}"
ADAPTER_DRAFT="models/draft_${TYPE}_${SCENARIO}"

# Define Data Path
DATA_PATH="data/processed/${TYPE}_${SCENARIO}.jsonl"

# Construct Run Name
RUN_NAME="${TYPE}_${SCENARIO}_spec_benchmark"

# Check if adapters exist
if [ ! -d "$ADAPTER_TARGET" ]; then
    echo "⚠️ Warning: Target adapter $ADAPTER_TARGET not found. Ensure training is complete."
fi

if [ ! -d "$ADAPTER_DRAFT" ]; then
    echo "⚠️ Warning: Draft adapter $ADAPTER_DRAFT not found."
fi

echo "👉 Running Benchmark..."
python tests/benchmark.py \
    --scenario "$SCENARIO" \
    --target-base-model "$BASE_TARGET" \
    --target-adapter "$ADAPTER_TARGET" \
    --draft-base-model "$BASE_DRAFT" \
    --draft-adapter "$ADAPTER_DRAFT" \
    --data-path "$DATA_PATH" \
    --use-speculative \
    --run-name "$RUN_NAME"

echo "✅ Benchmark Completed for $RUN_NAME"

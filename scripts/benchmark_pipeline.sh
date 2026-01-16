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
echo "Starting Benchmark Pipeline | Type: $TYPE | Scenario: $SCENARIO"
echo "========================================================"

# Define Paths
ADAPTER_TARGET="models/target_${TYPE}_${SCENARIO}"
MERGED_TARGET="models/target_${TYPE}_${SCENARIO}_merged"

ADAPTER_DRAFT="models/draft_${TYPE}_${SCENARIO}"
MERGED_DRAFT="models/draft_${TYPE}_${SCENARIO}_merged"

# Define Data Path
DATA_PATH="data/processed/${TYPE}_${SCENARIO}.jsonl"

# Construct Run Name
RUN_NAME="${TYPE}_${SCENARIO}_spec_benchmark"

# -------------------------------------------------
# 1. Determine Target Model Configuration
# -------------------------------------------------
if [ -f "$MERGED_TARGET/config.json" ]; then
    echo "✅ Found Merged Target Model: $MERGED_TARGET"
    TARGET_BASE_ARG="$MERGED_TARGET"
    TARGET_ADAPTER_ARG=""
else
    echo "⚠️  Merged Target Model not found. Falling back to Base + Adapter."
    TARGET_BASE_ARG="$BASE_TARGET"
    TARGET_ADAPTER_ARG="$ADAPTER_TARGET"
    
    if [ ! -d "$ADAPTER_TARGET" ]; then
        echo "❌ ERROR: Target adapter not found at $ADAPTER_TARGET"
        exit 1
    fi
fi

# -------------------------------------------------
# 2. Determine Draft Model Configuration
# -------------------------------------------------
if [ -f "$MERGED_DRAFT/config.json" ]; then
    echo "✅ Found Merged Draft Model: $MERGED_DRAFT"
    DRAFT_MODEL_ARG="$MERGED_DRAFT"
    # Note: Benchmark script expects merged draft model path
else
    echo "⚠️  Merged Draft Model not found. Draft adapters must be merged for speculative decoding."
    # Check if adapter exists, maybe warn user they need to merge
    if [ -d "$ADAPTER_DRAFT" ]; then
        echo "ℹ️  Found draft adapter at $ADAPTER_DRAFT but not merged."
        echo "    Run merge_adapter.py manually if you want to use it."
    fi
    # Use base draft? Or fail? The script defaults to base if adapter not passed.
    DRAFT_MODEL_ARG=""
    echo "Using Base Draft Model (no adapter/merge)..."
fi

echo "Running Benchmark..."
PYTHONPATH=. python -m tests.benchmark \
    --scenario "$SCENARIO" \
    --target-base-model "$TARGET_BASE_ARG" \
    --target-adapter "$TARGET_ADAPTER_ARG" \
    --draft-base-model "$BASE_DRAFT" \
    --merged-draft-model "$DRAFT_MODEL_ARG" \
    --data-path "$DATA_PATH" \
    --use-speculative \
    --run-name "$RUN_NAME"

echo "Benchmark Completed for $RUN_NAME"

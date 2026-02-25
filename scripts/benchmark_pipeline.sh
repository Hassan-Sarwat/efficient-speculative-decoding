#!/bin/bash
# ==========================================================================
# Benchmark Pipeline for Speculative Decoding Experiments
# ==========================================================================
# Usage: bash scripts/benchmark_pipeline.sh -t [cot|cod] -s [easy|medium|hard]
# 
# This script:
# 1. Merges LoRA adapters into temporary full models
# 2. Runs BOTH baseline and speculative decoding benchmarks
# 3. Generates comparison metrics
# 4. Cleans up temporary files
# ==========================================================================

set -e  # Exit on error

# -------------------------------------------------
# Parse Arguments
# -------------------------------------------------
TRAIN_TYPE=""
SCENARIO=""
MODE="both"  # default

while getopts "t:s:m:" opt; do
  case $opt in
    t) TRAIN_TYPE=$OPTARG ;;
    s) SCENARIO=$OPTARG ;;
    m) MODE=$OPTARG ;;
    *) echo "Usage: $0 -t [cot|cod] -s [easy|medium|hard] [-m baseline|speculative|both]" >&2
       exit 1 ;;
  esac
done

if [ -z "$TRAIN_TYPE" ] || [ -z "$SCENARIO" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 -t [cot|cod] -s [easy|medium|hard] [-m baseline|speculative|both]"
    exit 1
fi

if [[ "$MODE" != "baseline" && "$MODE" != "speculative" && "$MODE" != "both" ]]; then
    echo "Error: -m must be one of: baseline, speculative, both"
    exit 1
fi

# -------------------------------------------------
# Configuration
# -------------------------------------------------
BASE_TARGET="Qwen/Qwen3-14B"
BASE_DRAFT="Qwen/Qwen3-0.6B"

TARGET_ADAPTER="models/target_${TRAIN_TYPE}_${SCENARIO}"
DRAFT_ADAPTER="models/draft_${TRAIN_TYPE}_${SCENARIO}"

DATA_PATH="data/tests/${SCENARIO}_test.jsonl"
RUN_NAME="${TRAIN_TYPE}_${SCENARIO}_benchmark"

TEMP_MERGED_TARGET="models/temp_merged_target_${TRAIN_TYPE}_${SCENARIO}"
TEMP_MERGED_DRAFT="models/temp_merged_draft_${TRAIN_TYPE}_${SCENARIO}"

echo "========================================================"
echo "Starting Benchmark Pipeline | Type: $TRAIN_TYPE | Scenario: $SCENARIO"
echo "========================================================"

# -------------------------------------------------
# 1. Merge Target Adapter (if exists)
# -------------------------------------------------
echo ""
echo "Checking for Target Adapter..."
if [ -d "$TARGET_ADAPTER" ]; then
    echo "Found Target Adapter at $TARGET_ADAPTER"
    echo "Merging Target Adapter into Temporary Model..."
    
    python src/merge_adapter.py \
        --base_model "$BASE_TARGET" \
        --adapter_path "$TARGET_ADAPTER" \
        --output_path "$TEMP_MERGED_TARGET"
    
    echo "Target merged to $TEMP_MERGED_TARGET"
    TARGET_BASE_ARG="$TEMP_MERGED_TARGET"
    TARGET_ADAPTER_ARG=""  # Don't use adapter since we merged
else
    echo "No Target Adapter found at $TARGET_ADAPTER."
    echo "Using Base Target Model."
    TARGET_BASE_ARG="$BASE_TARGET"
    TARGET_ADAPTER_ARG=""
fi

# -------------------------------------------------
# 2. Merge Draft Adapter (if exists)
# -------------------------------------------------
echo ""
echo "Checking for Draft Adapter..."
if [ -d "$DRAFT_ADAPTER" ]; then
    echo "Found Draft Adapter at $DRAFT_ADAPTER"
    echo "Merging Draft Adapter into Temporary Model..."
    
    python src/merge_adapter.py \
        --base_model "$BASE_DRAFT" \
        --adapter_path "$DRAFT_ADAPTER" \
        --output_path "$TEMP_MERGED_DRAFT"
    
    echo "Draft merged to $TEMP_MERGED_DRAFT"
    DRAFT_MODEL_ARG="$TEMP_MERGED_DRAFT"
else
    echo "No Draft Adapter found at $DRAFT_ADAPTER."
    echo "Using Base Draft Model."
    DRAFT_MODEL_ARG="$BASE_DRAFT"
fi

# -------------------------------------------------
# 3. Run Benchmark (BOTH Baseline + Speculative)
# -------------------------------------------------
echo ""
echo "Running Benchmark (mode: $MODE)..."

# Build mode flag
if [ "$MODE" == "both" ]; then
    MODE_FLAG="--run-both"
elif [ "$MODE" == "speculative" ]; then
    MODE_FLAG="--use-speculative"
else
    MODE_FLAG=""  # baseline is the default when neither flag is passed
fi

python tests/benchmark.py \
    --scenario "$SCENARIO" \
    --target-base-model "$TARGET_BASE_ARG" \
    --target-adapter "$TARGET_ADAPTER_ARG" \
    --draft-base-model "$BASE_DRAFT" \
    --merged-draft-model "$DRAFT_MODEL_ARG" \
    --data-path "$DATA_PATH" \
    $MODE_FLAG \
    --run-name "$RUN_NAME"

# -------------------------------------------------
# 4. Cleanup (Delete Temporary Merged Models)
# -------------------------------------------------
echo ""
echo "Cleaning up temporary merged models..."

if [ -d "$TEMP_MERGED_TARGET" ]; then
    echo "   Removing $TEMP_MERGED_TARGET..."
    rm -rf "$TEMP_MERGED_TARGET"
fi

if [ -d "$TEMP_MERGED_DRAFT" ]; then
    echo "   Removing $TEMP_MERGED_DRAFT..."
    rm -rf "$TEMP_MERGED_DRAFT"
fi

echo "Cleanup Complete"
echo "========================================================"
echo "Results:"
echo "  - Detailed CSV: outputs/${RUN_NAME}_${SCENARIO}.csv"
echo "  - Unified Metrics: outputs/metrics_${RUN_NAME}.json"
echo "  - Comparison: outputs/comparison_${RUN_NAME}.txt"
echo "========================================================"
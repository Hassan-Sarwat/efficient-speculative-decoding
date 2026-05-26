#!/bin/bash
# ==========================================================================
# Benchmark Pipeline for Speculative Decoding Experiments
# ==========================================================================
# Usage: bash scripts/benchmark_pipeline.sh -t [cot|cod|base] -s [easy|medium|hard] [-p cot|cod]
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
MODE="both"
NUM_SAMPLES=""
PROMPT_TYPE=""

KEEP_MODELS=0

while getopts "t:s:m:n:kp:" opt; do
  case $opt in
    t) TRAIN_TYPE=$OPTARG ;;
    s) SCENARIO=$OPTARG ;;
    m) MODE=$OPTARG ;;
    n) NUM_SAMPLES=$OPTARG ;;
    k) KEEP_MODELS=1 ;;
    p) PROMPT_TYPE=$OPTARG ;;
    *) echo "Usage: $0 -t [cot|cod|base] -s [easy|medium|hard] [-m baseline|speculative|both] [-n num_samples] [-p cot|cod] [-k]" >&2
       exit 1 ;;
  esac
done

if [ -z "$TRAIN_TYPE" ] || [ -z "$SCENARIO" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 -t [cot|cod|base] -s [easy|medium|hard] [-m baseline|speculative|both] [-n num_samples]"
    exit 1
fi

if [[ "$TRAIN_TYPE" != "cot" && "$TRAIN_TYPE" != "cod" && "$TRAIN_TYPE" != "base" ]]; then
    echo "Error: -t must be one of: cot, cod, base"
    exit 1
fi

if [[ "$MODE" != "baseline" && "$MODE" != "speculative" && "$MODE" != "both" ]]; then
    echo "Error: -m must be one of: baseline, speculative, both"
    exit 1
fi

if [ -n "$PROMPT_TYPE" ] && [[ "$PROMPT_TYPE" != "cot" && "$PROMPT_TYPE" != "cod" ]]; then
    echo "Error: -p must be one of: cot, cod"
    exit 1
fi

if [ -n "$NUM_SAMPLES" ] && ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Error: -n must be a positive integer"
    exit 1
fi

# -------------------------------------------------
# Configuration
# -------------------------------------------------
export PYTHONPATH=$PYTHONPATH:.

BASE_TARGET="Qwen/Qwen3-14B"
BASE_DRAFT="Qwen/Qwen3-0.6B"

TARGET_ADAPTER="models/target_${TRAIN_TYPE}_${SCENARIO}"

# 'base' runs use both untrained base models. Prompt type defaults to cod (the
# primary research condition) but can be overridden with -p cot|cod.
if [ "$TRAIN_TYPE" == "base" ]; then
    DRAFT_ADAPTER="models/draft_untrained_${SCENARIO}"
    BENCHMARK_TYPE="${PROMPT_TYPE:-cod}"
    TEMP_MERGED_DRAFT="models/temp_merged_draft_untrained_${SCENARIO}"
else
    DRAFT_ADAPTER="models/draft_${TRAIN_TYPE}_${SCENARIO}"
    BENCHMARK_TYPE="$TRAIN_TYPE"
    TEMP_MERGED_DRAFT="models/temp_merged_draft_${TRAIN_TYPE}_${SCENARIO}"
fi

DATA_PATH="data/tests/${SCENARIO}_test.jsonl"
# For base runs, include prompt type in run name to avoid collisions between
# -p cot and -p cod runs.
if [ "$TRAIN_TYPE" == "base" ]; then
    RUN_NAME="base_${BENCHMARK_TYPE}_${SCENARIO}_benchmark"
else
    RUN_NAME="${TRAIN_TYPE}_${SCENARIO}_benchmark"
fi

TEMP_MERGED_TARGET="models/temp_merged_target_${TRAIN_TYPE}_${SCENARIO}"

echo "========================================================"
echo "Starting Benchmark Pipeline | Type: $TRAIN_TYPE | Scenario: $SCENARIO | Mode: $MODE | Samples: ${NUM_SAMPLES:-all}"
echo "========================================================"

# -------------------------------------------------
# 1. Merge Target Adapter (if exists)
# -------------------------------------------------
echo ""
echo "Checking for Target Adapter..."
if [ -d "$TARGET_ADAPTER" ]; then
    echo "Found Target Adapter at $TARGET_ADAPTER"
    
    if [ -d "$TEMP_MERGED_TARGET" ]; then
        echo "Found existing merged Target Model at $TEMP_MERGED_TARGET. Skipping merge."
        TARGET_BASE_ARG="$TEMP_MERGED_TARGET"
        TARGET_ADAPTER_ARG=""
    else
        echo "Merging Target Adapter into Temporary Model..."
        
        python src/merge_adapter.py \
            --base_model "$BASE_TARGET" \
            --adapter_path "$TARGET_ADAPTER" \
            --output_path "$TEMP_MERGED_TARGET"
        
        echo "Target merged to $TEMP_MERGED_TARGET"
        TARGET_BASE_ARG="$TEMP_MERGED_TARGET"
        TARGET_ADAPTER_ARG=""  # Don't use adapter since we merged
    fi
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
    
    if [ -d "$TEMP_MERGED_DRAFT" ]; then
        echo "Found existing merged Draft Model at $TEMP_MERGED_DRAFT. Skipping merge."
        DRAFT_MODEL_ARG="$TEMP_MERGED_DRAFT"
    else
        echo "Merging Draft Adapter into Temporary Model..."
        
        python src/merge_adapter.py \
            --base_model "$BASE_DRAFT" \
            --adapter_path "$DRAFT_ADAPTER" \
            --output_path "$TEMP_MERGED_DRAFT"
        
        echo "Draft merged to $TEMP_MERGED_DRAFT"
        DRAFT_MODEL_ARG="$TEMP_MERGED_DRAFT"
    fi
else
    echo "No Draft Adapter found at $DRAFT_ADAPTER."
    echo "Using Base Draft Model."
    DRAFT_MODEL_ARG="$BASE_DRAFT"
fi

# -------------------------------------------------
# 3. Run Benchmark
# -------------------------------------------------
echo ""
echo "Running Benchmark (mode: $MODE, samples: ${NUM_SAMPLES:-all})..."

# Build mode flag
if [ "$MODE" == "both" ]; then
    MODE_FLAG="--run-both"
elif [ "$MODE" == "speculative" ]; then
    MODE_FLAG="--use-speculative"
else
    MODE_FLAG=""
fi

# Build num-samples flag
NUM_SAMPLES_FLAG=""
if [ -n "$NUM_SAMPLES" ]; then
    NUM_SAMPLES_FLAG="--num-samples $NUM_SAMPLES"
fi

python tests/benchmark.py \
    --scenario "$SCENARIO" \
    --type "$BENCHMARK_TYPE" \
    --target-base-model "$TARGET_BASE_ARG" \
    --target-adapter "$TARGET_ADAPTER_ARG" \
    --draft-base-model "$BASE_DRAFT" \
    --merged-draft-model "$DRAFT_MODEL_ARG" \
    --data-path "$DATA_PATH" \
    --run-name "$RUN_NAME" \
    $MODE_FLAG \
    $NUM_SAMPLES_FLAG

# -------------------------------------------------
# 4. Cleanup (Delete Temporary Merged Models)
# -------------------------------------------------
echo ""
if [ "$KEEP_MODELS" == "1" ]; then
    echo "Keeping temporary merged models on disk per -k flag."
else
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
fi
echo "========================================================"
echo "Results:"
echo "  - Detailed CSV: outputs/${RUN_NAME}_${SCENARIO}.csv"
echo "  - Unified Metrics: outputs/metrics_${RUN_NAME}.json"
echo "  - Comparison: outputs/comparison_${RUN_NAME}.txt"
echo "========================================================"
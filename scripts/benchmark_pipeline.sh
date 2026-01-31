#!/bin/bash
set -e

# Default Values
TYPE="cod"       # cot or cod
SCENARIO="medium"  # easy, medium, hard
BASE_TARGET="Qwen/Qwen3-14B"
BASE_DRAFT="Qwen/Qwen3-0.6B"

# Parse Flags
while getopts "t:s:" opt; do
  case $opt in
    t) TYPE="$OPTARG" ;;
    s) SCENARIO="$OPTARG" ;;
    *) echo "Usage: $0 -t [cot|cod] -s [easy|medium|hard]"; exit 1 ;;
  esac
done

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

echo "========================================================"
echo "Starting Benchmark Pipeline | Type: $TYPE | Scenario: $SCENARIO"
echo "========================================================"

# Define Paths
ADAPTER_TARGET="models/target_${TYPE}_${SCENARIO}"
ADAPTER_DRAFT="models/draft_${TYPE}_${SCENARIO}"

# Define Data Path
DATA_PATH="data/processed/${TYPE}_${SCENARIO}.jsonl"

# Construct Run Name
RUN_NAME="${TYPE}_${SCENARIO}_spec_benchmark"

# Temporary Merged Model Paths
TEMP_MERGED_TARGET="models/temp_merged_target_${TYPE}_${SCENARIO}"
TEMP_MERGED_DRAFT="models/temp_merged_draft_${TYPE}_${SCENARIO}"

# -------------------------------------------------
# 1. Merge Target Model (Ephemeral)
# -------------------------------------------------
echo ""
echo "Checking for Target Adapter..."
if [ -d "$ADAPTER_TARGET" ] && [ -f "$ADAPTER_TARGET/adapter_config.json" ]; then
    echo "Found Target Adapter at $ADAPTER_TARGET"
    echo "Merging Target Adapter into Temporary Model..."
    
    python src/merge_adapter.py \
        --base_model "$BASE_TARGET" \
        --adapter_path "$ADAPTER_TARGET" \
        --output_path "$TEMP_MERGED_TARGET" \
        --force
        
    TARGET_BASE_ARG="$TEMP_MERGED_TARGET"
    TARGET_ADAPTER_ARG=""
    
    echo "Target merged to $TEMP_MERGED_TARGET"
else
    echo "No Target Adapter found or invalid. Using Base Model + Runtime Adapter if available."
    TARGET_BASE_ARG="$BASE_TARGET"
    TARGET_ADAPTER_ARG="$ADAPTER_TARGET"
fi

# -------------------------------------------------
# 2. Merge Draft Model (Ephemeral)
# -------------------------------------------------
echo ""
echo "Checking for Draft Adapter..."
if [ -d "$ADAPTER_DRAFT" ] && [ -f "$ADAPTER_DRAFT/adapter_config.json" ]; then
    echo "Found Draft Adapter at $ADAPTER_DRAFT"
    echo "Merging Draft Adapter into Temporary Model..."
    
    python src/merge_adapter.py \
        --base_model "$BASE_DRAFT" \
        --adapter_path "$ADAPTER_DRAFT" \
        --output_path "$TEMP_MERGED_DRAFT" \
        --force
        
    DRAFT_MODEL_ARG="$TEMP_MERGED_DRAFT"
    
    echo "Draft merged to $TEMP_MERGED_DRAFT"
else
    echo "No Draft Adapter found. Using Base Draft Model."
    DRAFT_MODEL_ARG="$BASE_DRAFT"
fi

# -------------------------------------------------
# 3. Run Benchmark
# -------------------------------------------------
echo ""
echo "Running Benchmark..."
python -m tests.benchmark \
    --scenario "$SCENARIO" \
    --target-base-model "$TARGET_BASE_ARG" \
    --target-adapter "$TARGET_ADAPTER_ARG" \
    --draft-base-model "$BASE_DRAFT" \
    --merged-draft-model "$DRAFT_MODEL_ARG" \
    --data-path "$DATA_PATH" \
    --use-speculative \
    --run-name "$RUN_NAME"

echo "Benchmark Completed for $RUN_NAME"

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

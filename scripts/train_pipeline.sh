#!/bin/bash
set -e  # Exit on any error

# Default Values
TYPE="cot"       # cot or cod
SCENARIO="easy"  # easy, medium, hard
BASE_TARGET="Qwen/Qwen2.5-14B-Instruct"
WANDB_PROJECT="peft_cob"

# Parse Flags
while getopts "t:s:" opt; do
  case $opt in
    t) TYPE="$OPTARG" ;;
    s) SCENARIO="$OPTARG" ;;
    *) echo "Usage: $0 -t [cot|cod] -s [easy|medium|hard]"; exit 1 ;;
  esac
done

echo "========================================================"
echo "🚀 Starting Pipeline | Type: $TYPE | Scenario: $SCENARIO"
echo "========================================================"

# Define Paths
DATA_TRAIN="data/processed/${TYPE}_${SCENARIO}.jsonl"
ADAPTER_TARGET="models/target_${TYPE}_${SCENARIO}"
DATA_DISTILLED="data/distilled/${TYPE}_${SCENARIO}.jsonl"
ADAPTER_DRAFT="models/draft_${TYPE}_${SCENARIO}"
TARGET_OUTPUT_DIR="models/checkpoints/target_${TYPE}_${SCENARIO}"
DRAFT_OUTPUT_DIR="models/checkpoints/draft_${TYPE}_${SCENARIO}"

# Configs
CFG_TARGET="configs/target_14b.yaml"
CFG_DRAFT="configs/draft_0-5b.yaml"

# Environments
ENV_TRAIN="env_train/bin/activate"
ENV_SERVE="env_serve/bin/activate"

# ✅ Validation: Check if input data exists
if [ ! -f "$DATA_TRAIN" ]; then
    echo "❌ ERROR: Training data not found at $DATA_TRAIN"
    exit 1
fi

echo "✅ Found training data: $DATA_TRAIN ($(wc -l < "$DATA_TRAIN") samples)"

# ---------------------------------------------------------
# Step 1: Train Target Model (14B)
# ---------------------------------------------------------
echo ""
echo "=========================================="
echo "📚 [STEP 1/3] Training Target Model (14B)"
echo "=========================================="
START_TIME=$(date +%s)

source $ENV_TRAIN

python src/train.py $CFG_TARGET \
    --data_file "$DATA_TRAIN" \
    --final_save_path "$ADAPTER_TARGET" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "target_${TYPE}_${SCENARIO}" \
    --output_dir "$TARGET_OUTPUT_DIR"

deactivate

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "⏱️  Step 1 completed in ${DURATION}s"

# ✅ Verify adapter was created
if [ ! -f "$ADAPTER_TARGET/adapter_config.json" ]; then
    echo "❌ ERROR: Target adapter not found at $ADAPTER_TARGET"
    exit 1
fi
echo "✅ Target adapter saved successfully"

# ---------------------------------------------------------
# Step 2: Distill Data (Teacher: 14B -> Student Data)
# ---------------------------------------------------------
echo ""
echo "=========================================="
echo "🔬 [STEP 2/3] Distilling Data (14B -> 0.5B)"
echo "=========================================="
echo "Strategy: FP16 inference with aggressive memory optimization"
echo "Expected VRAM: ~22GB (fits in 24GB)"
START_TIME=$(date +%s)

source $ENV_SERVE

# ✅ Key Changes for 24GB VRAM:
# - No quantization (FP16 only)
# - Reduced batch size (--batch_size 8)
# - Lower GPU utilization (--gpu_memory_utilization 0.90)
# - Validation enabled (--validation_threshold 0.85)

python src/distill_data.py \
    --base_model "$BASE_TARGET" \
    --adapter_path "$ADAPTER_TARGET" \
    --input_file "$DATA_TRAIN" \
    --output_file "$DATA_DISTILLED" \
    --batch_size 8 \
    --gpu_memory_utilization 0.90 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 1024 \
    --validation_threshold 0.85

deactivate

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "⏱️  Step 2 completed in ${DURATION}s"

# ✅ Verify distilled data was created
if [ ! -f "$DATA_DISTILLED" ]; then
    echo "❌ ERROR: Distilled data not found at $DATA_DISTILLED"
    exit 1
fi

DISTILLED_COUNT=$(wc -l < "$DATA_DISTILLED")
echo "✅ Distilled dataset created: ${DISTILLED_COUNT} samples"

if [ "$DISTILLED_COUNT" -lt 100 ]; then
    echo "⚠️  WARNING: Very few distilled samples (${DISTILLED_COUNT}). Check for errors."
fi

# ---------------------------------------------------------
# Step 3: Train Draft Model (0.5B) on Distilled Data
# ---------------------------------------------------------
echo ""
echo "=========================================="
echo "🎯 [STEP 3/3] Training Draft Model (0.5B)"
echo "=========================================="
START_TIME=$(date +%s)

source $ENV_TRAIN

python src/train.py $CFG_DRAFT \
    --data_file "$DATA_DISTILLED" \
    --final_save_path "$ADAPTER_DRAFT" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "draft_${TYPE}_${SCENARIO}" \
    --output_dir "$DRAFT_OUTPUT_DIR"

deactivate

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "⏱️  Step 3 completed in ${DURATION}s"

# ✅ Verify draft adapter was created
if [ ! -f "$ADAPTER_DRAFT/adapter_config.json" ]; then
    echo "❌ ERROR: Draft adapter not found at $ADAPTER_DRAFT"
    exit 1
fi
echo "✅ Draft adapter saved successfully"

# ---------------------------------------------------------
# Final Summary
# ---------------------------------------------------------
echo ""
echo "========================================================"
echo "🎉 PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================"
echo "Type: $TYPE | Scenario: $SCENARIO"
echo ""
echo "📂 Output Files:"
echo "  - Target Adapter: $ADAPTER_TARGET"
echo "  - Distilled Data: $DATA_DISTILLED (${DISTILLED_COUNT} samples)"
echo "  - Draft Adapter:  $ADAPTER_DRAFT"
echo ""
echo "Next Steps:"
echo "  1. Run benchmark: bash scripts/benchmark_pipeline.sh -t $TYPE -s $SCENARIO"
echo "  2. Compare with baseline"
echo "  3. Analyze results in outputs/"
echo "========================================================"
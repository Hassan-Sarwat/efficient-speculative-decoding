#!/bin/bash
set -e  # Exit on any error

# Default Values
TYPE="cot"       # cot or cod
SCENARIO="easy"  # easy, medium, hard
BASE_TARGET="Qwen/Qwen2.5-14B-Instruct"
BASE_DRAFT="Qwen/Qwen2.5-0.5B-Instruct"
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

# Merged model paths (for inference)
MERGED_TARGET="models/target_${TYPE}_${SCENARIO}_merged"
MERGED_DRAFT="models/draft_${TYPE}_${SCENARIO}_merged"

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
echo "📚 [STEP 1/4] Training Target Model (14B)"
echo "=========================================="
START_TIME=$(date +%s)

source $ENV_TRAIN

python src/train.py $CFG_TARGET \
    --data_file "$DATA_TRAIN" \
    --final_save_path "$ADAPTER_TARGET" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "target_${TYPE}_${SCENARIO}" \
    --load_in_4bit True \
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
# Step 2: Distill Data (Generate Training Data for Draft)
# ---------------------------------------------------------
echo ""
echo "=========================================="
echo "🔄 [STEP 2/4] Distilling Data from Target"
echo "=========================================="
START_TIME=$(date +%s)

source $ENV_SERVE

python src/distill_data.py \
    --base_model "$BASE_TARGET" \
    --adapter_path "$ADAPTER_TARGET" \
    --input_file "$DATA_TRAIN" \
    --output_file "$DATA_DISTILLED" \
    --validation_threshold 0.80

deactivate

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "⏱️  Step 2 completed in ${DURATION}s"

# ✅ Count distilled samples
DISTILLED_COUNT=$(wc -l < "$DATA_DISTILLED")
echo "✅ Generated $DISTILLED_COUNT distilled samples"

if [ "$DISTILLED_COUNT" -lt 10 ]; then
    echo "❌ ERROR: Too few distilled samples ($DISTILLED_COUNT)"
    exit 1
fi

# ---------------------------------------------------------
# Step 3: Train Draft Model (0.5B)
# ---------------------------------------------------------
echo ""
echo "=========================================="
echo "📖 [STEP 3/4] Training Draft Model (0.5B)"
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

# ✅ Verify adapter was created
if [ ! -f "$ADAPTER_DRAFT/adapter_config.json" ]; then
    echo "❌ ERROR: Draft adapter not found at $ADAPTER_DRAFT"
    exit 1
fi
echo "✅ Draft adapter saved successfully"

# ---------------------------------------------------------
# Step 4: Merge LoRA Adapters (For Inference)
# ---------------------------------------------------------
echo ""
echo "=========================================="
echo "🔗 [STEP 4/4] Merging LoRA Adapters"
echo "=========================================="
START_TIME=$(date +%s)

source $ENV_TRAIN


# Merge Target Model
echo "Merging Target Model..."
python src/merge_adapter.py \
    --base_model "$BASE_TARGET" \
    --adapter_path "$ADAPTER_TARGET" \
    --output_path "$MERGED_TARGET"

# ✅ Verify merged models exist
if [ ! -f "$MERGED_TARGET/config.json" ]; then
    echo "❌ ERROR: Merged target model not found at $MERGED_TARGET"
    exit 1
fi

echo "✅ Merged target model saved successfully"


# Merge Draft Model
echo "Merging Draft Model..."
python src/merge_adapter.py \
    --base_model "$BASE_DRAFT" \
    --adapter_path "$ADAPTER_DRAFT" \
    --output_path "$MERGED_DRAFT"

deactivate

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "⏱️  Step 4 completed in ${DURATION}s"

# ✅ Verify merged models exist
if [ ! -f "$MERGED_DRAFT/config.json" ]; then
    echo "❌ ERROR: Merged draft model not found at $MERGED_DRAFT"
    exit 1
fi

echo "✅ Merged models saved successfully"

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
echo "  - Target Adapter:      $ADAPTER_TARGET"
echo "  - Target Merged:       $MERGED_TARGET"
echo "  - Distilled Data:      $DATA_DISTILLED (${DISTILLED_COUNT} samples)"
echo "  - Draft Adapter:       $ADAPTER_DRAFT"
echo "  - Draft Merged:        $MERGED_DRAFT"
echo ""
echo "Next Steps:"
echo "  1. Run benchmark: bash scripts/benchmark_pipeline.sh -t $TYPE -s $SCENARIO"
echo "  2. Use merged models for speculative decoding inference"
echo "  3. Compare results in outputs/"
echo "========================================================"
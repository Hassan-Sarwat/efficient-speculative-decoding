#!/bin/bash
set -e 

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
# We use the Processed Training data as the source for everything
DATA_TRAIN="data/training/${TYPE}_${SCENARIO}.jsonl"

# Output Paths
ADAPTER_TARGET="models/target_${TYPE}_${SCENARIO}_14b"
DATA_DISTILLED="data/distilled/${TYPE}_${SCENARIO}.jsonl"
ADAPTER_DRAFT="models/draft_${TYPE}_${SCENARIO}_0.5b"

# Configs
CFG_TARGET="configs/target_14b.yaml"
CFG_DRAFT="configs/draft_0-5b.yaml"

# Environments
ENV_TRAIN="env_train/bin/activate"
ENV_SERVE="env_serve/bin/activate"

# ---------------------------------------------------------
# Step 1: Train Target Model (14B)
# ---------------------------------------------------------
echo "--- [1/3] Training Target Model ($TYPE) ---"
source $ENV_TRAIN
python train.py $CFG_TARGET \
    --data_file "$DATA_TRAIN" \
    --final_save_path "$ADAPTER_TARGET" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "target_${TYPE}_${SCENARIO}"
deactivate

# ---------------------------------------------------------
# Step 2: Distill Data (Teacher: 14B -> Student Data)
# ---------------------------------------------------------
# We use DATA_TRAIN as input to ensure we only distill the clean, valid samples.
echo "--- [2/3] Distilling Data (Using clean training prompts) ---"
source $ENV_SERVE
python distill_data.py \
    --base_model "$BASE_TARGET" \
    --adapter_path "$ADAPTER_TARGET" \
    --input_file "$DATA_TRAIN" \
    --output_file "$DATA_DISTILLED"
deactivate

# ---------------------------------------------------------
# Step 3: Train Draft Model (0.5B) on Distilled Data
# ---------------------------------------------------------
echo "--- [3/3] Training Draft Model ($TYPE 0.5B) ---"
source $ENV_TRAIN
python train.py $CFG_DRAFT \
    --data_file "$DATA_DISTILLED" \
    --final_save_path "$ADAPTER_DRAFT" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "draft_${TYPE}_${SCENARIO}"
deactivate

echo "✅ Pipeline Complete for $TYPE - $SCENARIO"
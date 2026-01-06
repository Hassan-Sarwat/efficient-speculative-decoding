#!/bin/bash
set -e 

# Default Values
TYPE="cot"       # cot or cod
SCENARIO="easy"  # easy, medium, hard
BASE_TARGET="Qwen/Qwen2.5-14B-Instruct"

# Parse Flags
while getopts "t:s:" opt; do
  case $opt in
    t) TYPE="$OPTARG" ;;
    s) SCENARIO="$OPTARG" ;;
    *) echo "Usage: $0 -t [cot|cod] -s [easy|medium|hard]"; exit 1 ;;
  esac
done

echo "========================================================"
echo "Starting Pipeline | Type: $TYPE | Scenario: $SCENARIO"
echo "========================================================"

# Define Paths
DATA_TRAIN="data/training/${TYPE}_${SCENARIO}.jsonl"
DATA_RAW="data/raw/${SCENARIO}.jsonl" # For distillation input

# Output Paths
ADAPTER_TARGET="models/target_${TYPE}_${SCENARIO}_14b"
DATA_DISTILLED="data/distilled/${TYPE}_${SCENARIO}_distill.jsonl"
ADAPTER_DRAFT="models/draft_${TYPE}_${SCENARIO}_0.5b"

# Configs (We reuse the same YAML for structure, override data/output via CLI)
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
# We override data_file and final_save_path from the CLI
python train.py $CFG_TARGET \
    --data_file "$DATA_TRAIN" \
    --final_save_path "$ADAPTER_TARGET"
deactivate

# ---------------------------------------------------------
# Step 2: Distill Data (Generate predictions)
# ---------------------------------------------------------
echo "--- [2/3] Distilling Data (Teacher: 14B -> Student Data) ---"
source $ENV_SERVE
python distill_data.py \
    --base_model "$BASE_TARGET" \
    --adapter_path "$ADAPTER_TARGET" \
    --input_file "$DATA_RAW" \
    --output_file "$DATA_DISTILLED"
deactivate

# ---------------------------------------------------------
# Step 3: Train Draft Model (0.5B) on Distilled Data
# ---------------------------------------------------------
echo "--- [3/3] Training Draft Model ($TYPE 0.5B) ---"
source $ENV_TRAIN
python train.py $CFG_DRAFT \
    --data_file "$DATA_DISTILLED" \
    --final_save_path "$ADAPTER_DRAFT"
deactivate

echo "Pipeline Complete for $TYPE - $SCENARIO"
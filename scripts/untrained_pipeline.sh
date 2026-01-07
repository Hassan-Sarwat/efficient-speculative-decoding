#!/bin/bash
set -e 

# Usage: bash scripts/untrained_pipeline.sh <scenario>
# Example: bash scripts/untrained_pipeline.sh easy

SCENARIO=$1

if [ -z "$SCENARIO" ]; then
    echo "❌ Error: No scenario provided."
    echo "Usage: $0 <easy|medium|hard>"
    exit 1
fi

echo "--- 🚀 Untrained Pipeline Started for Scenario: $SCENARIO ---"

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Define Paths
# We use the processed files just to get the instructions
INPUT_DATA="data/processed/cot_${SCENARIO}.jsonl" 
OUTPUT_DATA="data/untrained_${SCENARIO}.jsonl"
DRAFT_SAVE_PATH="models/draft_untrained_${SCENARIO}"
CHECKPOINT_DIR="models/checkpoints/draft_untrained_${SCENARIO}"

# Environments
ENV_TRAIN="env_train/bin/activate"
ENV_SERVE="env_serve/bin/activate"

# Validate Input
if [ ! -f "$INPUT_DATA" ]; then
    echo "❌ Error: Input data '$INPUT_DATA' not found."
    exit 1
fi

# Step 1: Generate Distilled Data from Untrained Target (using vLLM)
echo "[1/2] Generating Distilled Data from Untrained Target..."
source $ENV_SERVE
python distill_untrained.py \
    --input_file "$INPUT_DATA" \
    --output_file "$OUTPUT_DATA" \
    --base_model "unsloth/Qwen2.5-14B-Instruct"
deactivate

# Step 2: Train the Draft Model
echo "[2/2] Training Draft Model (0.5B) on Untrained Data..."
source $ENV_TRAIN
python train.py \
    --model_name "unsloth/Qwen2.5-0.5B-Instruct" \
    --data_file "$OUTPUT_DATA" \
    --final_save_path "$DRAFT_SAVE_PATH" \
    --output_dir "$CHECKPOINT_DIR" \
    --load_in_4bit True \
    --max_seq_length 2048 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2.0e-4 \
    --num_train_epochs 3 \
    --optim "adamw_8bit" \
    --logging_steps 1 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0
deactivate

echo "--- ✅ Pipeline Success! Untrained Draft Model ready in $DRAFT_SAVE_PATH ---"
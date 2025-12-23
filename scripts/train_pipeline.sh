#!/bin/bash
set -e 

echo "--- 🚀 Pipeline Started ---"

# Ensure we are in the project root (relative to this script)
cd "$(dirname "$0")/.."

# Step 1: Train the Big Model
echo "[1/2] Training Target Model (14B)..."
python train.py configs/target_14b.yaml

# Step 2: Distill Data (Generate predictions from Target)
echo "[2/3] Generating Distilled Dataset..."
python distill_data.py

# Step 3: Train the Small Model
echo "[3/3] Training Draft Model (0.5B)..."
python train.py configs/draft_0-5b.yaml

echo "--- ✅ Pipeline Success! Models ready in /app/models ---"
#!/bin/bash
set -e 

echo "--- 🚀 Pipeline Started ---"

# Step 1: Train the Big Model
echo "[1/2] Training Target Model (14B)..."
python train.py configs/target_14b.yaml

# Step 2: Train the Small Model
echo "[2/2] Training Draft Model (0.5B)..."
python train.py configs/draft_0-5b.yaml

echo "--- ✅ Pipeline Success! Models ready in /app/models ---"
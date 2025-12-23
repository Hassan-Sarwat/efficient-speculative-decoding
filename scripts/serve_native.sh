#!/bin/bash
set -e

ENV_NAME="env_serve"

# Source conda
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "❌ Error: Could not find conda.sh"
    exit 1
fi

echo "🚀 Activating '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "▶️  Starting Benchmark Server..."
cd "$(dirname "$0")/.."

python benchmark.py

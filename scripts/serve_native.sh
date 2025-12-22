#!/bin/bash
set -e

ENV_NAME="env_serve"

# Source conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    # Fallback: try to find conda
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

echo "🚀 Activating '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "▶️  Starting Benchmark Server..."
cd "$(dirname "$0")/.."

python benchmark.py

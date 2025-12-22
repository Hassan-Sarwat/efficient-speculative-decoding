#!/bin/bash
set -e

ENV_NAME="env_train"

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

echo "Displaying GPU Info..."
nvidia-smi

echo "▶️  Starting Training Pipeline..."
# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Execute the existing pipeline script
# Note: The original train_pipeline.sh might need minor tweaks if it assumed absolute paths in Docker,
# but it mainly calls python scripts.
./scripts/train_pipeline.sh

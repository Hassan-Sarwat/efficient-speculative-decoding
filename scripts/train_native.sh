#!/bin/bash
set -e

ENV_NAME="env_train"

ENV_NAME="env_train"

# Activation Logic
if [ -f "../$ENV_NAME/Scripts/activate" ]; then
    source "../$ENV_NAME/Scripts/activate"
elif [ -f "../$ENV_NAME/bin/activate" ]; then
    source "../$ENV_NAME/bin/activate"
elif [ -f "$ENV_NAME/Scripts/activate" ]; then
    source "$ENV_NAME/Scripts/activate"
elif [ -f "$ENV_NAME/bin/activate" ]; then
    source "$ENV_NAME/bin/activate"
else
    echo "❌ Error: Could not find activate script for '$ENV_NAME'"
    echo "   Did you run 'uv_setup_envs.sh'?"
    exit 1
fi

echo "🚀 Activated '$ENV_NAME'"

echo "Displaying GPU Info..."
nvidia-smi

echo "▶️  Starting Training Pipeline..."
# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Execute the existing pipeline script
# Note: The original train_pipeline.sh might need minor tweaks if it assumed absolute paths in Docker,
# but it mainly calls python scripts.
bash scripts/train_pipeline.sh

#!/bin/bash
set -e

ENV_NAME="env_serve"

ENV_NAME="env_serve"

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

echo "▶️  Starting Benchmark Server..."
cd "$(dirname "$0")/.."

python benchmark.py

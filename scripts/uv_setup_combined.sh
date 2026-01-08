#!/bin/bash
set -e

# ==========================================
# Unified Environment Setup (UV Edition)
# ==========================================

# 1. Install uv (if not found)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is on the PATH for this session
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
elif [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
else
    export PATH="$HOME/.local/bin:$PATH"
fi

# ==========================================
# Helper Functions
# ==========================================
get_activate_script() {
    local env_name=$1
    if [ -f "$env_name/Scripts/activate" ]; then
        echo "$env_name/Scripts/activate"
    elif [ -f "$env_name/bin/activate" ]; then
        echo "$env_name/bin/activate"
    else
        echo ""
    fi
}

# ==========================================
# Unified Environment Creation
# ==========================================
echo "--- Setting up Unified Environment ---"
if [ ! -d "env_unified" ]; then
    echo "📦 Creating 'env_unified'..."
    uv venv env_unified --python 3.11
fi

ACTIVATE_SCRIPT=$(get_activate_script "env_unified")
if [ -z "$ACTIVATE_SCRIPT" ]; then
    echo "❌ Could not find activation script for env_unified"
    exit 1
fi
source "$ACTIVATE_SCRIPT"

echo "⬇️  Installing dependencies..."
# Added setuptools explicitly for Triton/Unsloth compilation
uv pip install setuptools wheel

# Force correct CUDA version for vLLM (Torch 2.5.1 + cu124)
echo "⬇️  Installing PyTorch 2.5.1 (cu124)..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install combined requirements
echo "⬇️  Installing combined requirements..."
uv pip install -r requirements-combined.txt

deactivate

echo ""
echo "=== 🚀 Setup Unified Environment Complete! ==="
echo "To activate: source $ACTIVATE_SCRIPT"

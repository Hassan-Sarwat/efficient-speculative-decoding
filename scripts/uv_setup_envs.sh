#!/bin/bash
set -e

# ==========================================
# Production ML Environment Setup
# Training: Unsloth for LoRA fine-tuning
# Serving: vLLM 0.9.1+ for speculative decoding
# ==========================================

echo "🚀 Starting environment setup..."
echo ""

# 1. Install uv (if not found)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is on PATH
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
elif [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
else
    export PATH="$HOME/.local/bin:$PATH"
fi

# Helper function to find activation script
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
# Environment 1: Training (Unsloth)
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Setting up Training Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "env_train" ]; then
    echo "Creating virtual environment..."
    uv venv env_train --python 3.11
fi

ACTIVATE_TRAIN=$(get_activate_script "env_train")
source "$ACTIVATE_TRAIN"

echo "Installing dependencies..."
uv pip install setuptools wheel
uv pip install -r requirements-train.txt

# Fix known torchao conflict with unsloth
uv pip uninstall torchao 2>/dev/null || true

echo "✅ Training environment ready"
deactivate

# ==========================================
# Environment 2: Serving (vLLM 0.9.1+)
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ Setting up Serving Environment  "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "env_serve" ]; then
    echo "Creating virtual environment..."
    uv venv env_serve --python 3.11
fi

ACTIVATE_SERVE=$(get_activate_script "env_serve")
source "$ACTIVATE_SERVE"

echo "Installing PyTorch for CUDA 12.4..."
# UV auto-detects CUDA version with --torch-backend=auto
# Using explicit cu124 for reproducibility
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "Installing vLLM and dependencies..."
# vLLM will auto-install: transformers, accelerate, numpy, ray, etc.
uv pip install -r requirements-serve.txt

echo "Verifying installation..."
python -c "from vllm import LLM; from vllm.config import SpeculativeConfig; print('✓ vLLM with SpeculativeConfig ready')" 2>&1 | grep -q "ready" && echo "✅ vLLM verified" || echo "⚠️  Verification failed - check manually"

deactivate

# ==========================================
# Summary
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
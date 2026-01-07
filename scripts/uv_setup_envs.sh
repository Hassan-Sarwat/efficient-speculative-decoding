#!/bin/bash
set -e

# ==========================================
# ML Environment Setup (UV Edition)
# ==========================================

# 1. Install uv (if not found)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# --- FIX: Ensure uv is on the PATH for this session ---
# Check standard install locations
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
elif [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
else
    # Manual fallback
    export PATH="$HOME/.local/bin:$PATH"
fi

# ==========================================
# 0. Helper Functions
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
# 2. Training Environment (Unsloth)
# ==========================================
echo "--- Setting up Training Environment ---"
if [ ! -d "env_train" ]; then
    echo "📦 Creating 'env_train'..."
    uv venv env_train --python 3.11
fi

ACTIVATE_TRAIN=$(get_activate_script "env_train")
source "$ACTIVATE_TRAIN"

echo "⬇️  Installing training dependencies..."
# Added setuptools explicitly for Triton/Unsloth compilation
uv pip install setuptools wheel
uv pip install unsloth
uv pip install -r requirements-train.txt

# Fix torchao conflict
uv pip uninstall torchao 2>/dev/null || true
deactivate

# ==========================================
# 3. Serving Environment (vLLM)
# ==========================================
echo "--- Setting up Serving Environment ---"
if [ ! -d "env_serve" ]; then
    echo "📦 Creating 'env_serve'..."
    uv venv env_serve --python 3.11
fi

ACTIVATE_SERVE=$(get_activate_script "env_serve")
source "$ACTIVATE_SERVE"

echo "⬇️  Installing serving dependencies..."
# Added setuptools explicitly for Triton compilation
uv pip install setuptools wheel
uv pip install "numpy<2.0.0"

# Force correct CUDA version for vLLM
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

uv pip install -r requirements-serve.txt
deactivate

echo ""
echo "=== 🚀 Setup Complete! ==="
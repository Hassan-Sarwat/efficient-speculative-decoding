#!/bin/bash
set -e

# ==========================================
# ML Environment Setup (Architect Edition)
# Switch: Miniconda -> Miniforge (Avoids TOS errors)
# ==========================================

TRAIN_ENV="env_train"
SERVE_ENV="env_serve"
PYTHON_VER="3.11"

echo "=== 🛠️  ML Environment Setup (Miniforge) ==="
echo ""

# ==========================================
# 0. Ensure Conda (via Miniforge)
# ==========================================
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniforge3 (Community Edition)..."
    # Using Miniforge avoids the 'CondaToSNonInteractiveError'
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O miniforge.sh
    
    # Install silently to home directory
    bash miniforge.sh -b -p "$HOME/miniforge3"
    rm miniforge.sh
    
    # Initialize conda for the shell
    "$HOME/miniforge3/bin/conda" init bash
    
    # Source immediately for this script's execution
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    # If conda exists, ensure we can use it
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ==========================================
# 1. Training Environment (Unsloth)
# ==========================================
echo ""
echo "=== Training Environment ($TRAIN_ENV) ==="

if conda info --envs | grep -q "^$TRAIN_ENV "; then
    echo "✅ '$TRAIN_ENV' already exists. Skipping."
else
    echo "📦 Creating '$TRAIN_ENV'..."
    conda create --name "$TRAIN_ENV" python=$PYTHON_VER -y
    conda activate "$TRAIN_ENV"
    
    pip install --upgrade pip
    
    # Unsloth
    echo "⬇️  Installing Unsloth..."
    pip install unsloth

    # Deps
    echo "⬇️  Installing training stack..."
    pip install datasets scipy pandas wandb pyyaml "numpy<2.0.0"

    # Fix torchao conflict if present
    pip uninstall torchao -y 2>/dev/null || true
    
    conda deactivate
    echo "✅ '$TRAIN_ENV' setup complete."
fi

# ==========================================
# 2. Serving Environment (vLLM)
# ==========================================
echo ""
echo "=== Serving Environment ($SERVE_ENV) ==="

if conda info --envs | grep -q "^$SERVE_ENV "; then
    echo "✅ '$SERVE_ENV' already exists. Skipping."
else
    echo "📦 Creating '$SERVE_ENV'..."
    conda create -n "$SERVE_ENV" python=$PYTHON_VER -y
    conda activate "$SERVE_ENV"

    pip install --upgrade pip

    # vLLM Stack (Torch 2.5.1 + CUDA 12.4)
    echo "⬇️  Installing vLLM stack..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
    
    pip install vllm==0.6.4.post1 transformers==4.46.0 accelerate==0.34.0
    pip install xformers ray pandas datasets scipy "numpy<2.0.0"

    conda deactivate
    echo "✅ '$SERVE_ENV' setup complete."
fi

echo ""
echo "=== 🚀 Setup Complete ==="
echo "⚠️  IMPORTANT: Run the command below to refresh your shell:"
echo "    source ~/.bashrc"
echo ""
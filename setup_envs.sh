#!/bin/bash
set -e

# ==========================================
# ML Environment Setup
# Creates two environments:
#   - env_train: Fine-tuning with Unsloth
#   - env_serve: Inference with vLLM
# ==========================================

TRAIN_ENV="env_train"
SERVE_ENV="env_serve"
PYTHON_VER="3.11"

echo "=== 🛠️  ML Environment Setup ==="
echo ""

# ==========================================
# OS Checks
# ==========================================
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "❌ Error: Run this inside WSL 2, not native Windows."
    exit 1
fi

if uname -r | grep -qi "microsoft" && ! uname -r | grep -qi "WSL2"; then
    echo "⚠️  WARNING: WSL 1 detected. GPU won't work. Upgrade to WSL 2."
    sleep 3
fi

# ==========================================
# Ensure Conda
# ==========================================
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p ~/miniconda3
    rm miniconda.sh
    source ~/miniconda3/bin/activate
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# ==========================================
# 1. Training Environment (Unsloth)
# ==========================================
echo ""
echo "=== Training Environment ($TRAIN_ENV) ==="

if conda info --envs | grep -q "^$TRAIN_ENV "; then
    echo "✅ '$TRAIN_ENV' already exists. Skipping."
else
    echo "📦 Creating '$TRAIN_ENV'..."
    
    # Step 1: Create minimal env with just Python
    conda create --name "$TRAIN_ENV" python=$PYTHON_VER -y
    conda activate "$TRAIN_ENV"
    
    pip install --upgrade pip

    # Step 2: Install unsloth from PyPI (stable release, not git HEAD)
    # Let unsloth handle PyTorch and its dependencies
    echo "⬇️  Installing Unsloth (stable PyPI release)..."
    pip install unsloth

    # Step 3: Install additional training dependencies
    echo "⬇️  Installing additional dependencies..."
    pip install \
        datasets \
        scipy \
        pandas \
        wandb \
        pyyaml \
        "numpy<2.0.0"

    # Step 4: Remove torchao if present (known to cause conflicts)
    pip uninstall torchao -y 2>/dev/null || true

    # Verify
    echo "🔍 Verifying..."
    python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.version.cuda}')
import unsloth
print('   Unsloth: OK ✅')
"
    
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

    # PyTorch 2.5.1 + CUDA 12.4 (required by vLLM 0.6.4.post1)
    echo "⬇️  Installing PyTorch 2.5.1..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
        --no-cache-dir

    # Pinned versions for inference
    echo "⬇️  Installing vLLM and dependencies..."
    pip install \
        vllm==0.6.4.post1 \
        transformers==4.46.0 \
        accelerate==0.34.0

    # Additional dependencies
    pip install \
        xformers \
        ray \
        pandas \
        datasets \
        scipy \
        "numpy<2.0.0"

    # Verify
    echo "🔍 Verifying..."
    python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.version.cuda}')
import vllm
print(f'   vLLM: {vllm.__version__}')
import transformers
print(f'   Transformers: {transformers.__version__}')
print('   All imports OK ✅')
"
    
    conda deactivate
    echo "✅ '$SERVE_ENV' setup complete."
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=== 🚀 Setup Complete ==="
echo ""
echo "Training:  conda activate $TRAIN_ENV"
echo "           python finetune.py configs/target_14b.yaml"
echo ""
echo "Serving:   conda activate $SERVE_ENV"
echo "           python benchmark.py"
echo ""
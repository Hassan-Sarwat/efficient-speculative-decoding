#!/bin/bash
set -e

# ==========================================
# Unified ML Environment Setup
# Single env: Unsloth (training) + vLLM (speculative decoding inference)
# ==========================================

echo "Starting environment setup..."
echo ""

# 1. Install uv (if not found)
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
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
# Single Environment: env/
# ==========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setting up Unified Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    uv venv env --python 3.11
fi

ACTIVATE=$(get_activate_script "env")
source "$ACTIVATE"

echo "Installing dependencies..."
# unsafe-best-match lets uv reach across PyPI + PyTorch index when needed.
# vLLM 0.20 pulls torch==2.11.0+cu130 and matching nvidia-*-cu13 wheels itself —
# do NOT preinstall torch from a CUDA-mismatched index.
uv pip install -r requirements.txt --index-strategy unsafe-best-match

echo "Verifying installation..."
python - <<'PY'
import torch, transformers, vllm, unsloth, bitsandbytes, peft, trl
print(f"  torch={torch.__version__}  cuda_built={torch.version.cuda}  cuda_avail={torch.cuda.is_available()}")
print(f"  transformers={transformers.__version__}")
print(f"  vllm={vllm.__version__}")
print(f"  unsloth={unsloth.__version__}")
print(f"  bitsandbytes={bitsandbytes.__version__}  peft={peft.__version__}  trl={trl.__version__}")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
assert hasattr(LLM, "get_metrics"), "LLM.get_metrics missing"
from unsloth import FastLanguageModel
print("  all critical imports OK")
PY

deactivate

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Activate with: source env/bin/activate  (or  env\\Scripts\\activate  on Windows)"

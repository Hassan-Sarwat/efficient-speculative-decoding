# Install uv (one time)
set -e

# ==========================================
# ML Environment Setup (UV Edition)
# ==========================================

# Install uv (one time)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv already installed"
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
# 1. Training Environment
# ==========================================
if [ -d "env_train" ]; then
    echo "✅ 'env_train' already exists. Skipping creation."
else
    echo "📦 Creating 'env_train'..."
    uv venv env_train --python 3.11
fi

ACTIVATE_TRAIN=$(get_activate_script "env_train")
if [ -z "$ACTIVATE_TRAIN" ]; then
    echo "❌ Error: Could not find activate script for env_train"
    exit 1
fi
source "$ACTIVATE_TRAIN"

# Install Dependencies
echo "⬇️  Installing training dependencies..."
uv pip install unsloth
uv pip install -r requirements-train.txt

# Fix torchao conflict (Parity with setup_envs.sh)
echo "🔧 Fixing torchao conflict..."
uv pip uninstall torchao 2>/dev/null || true

# Deactivate
deactivate 2>/dev/null || true

# ==========================================
# 2. Serving Environment
# ==========================================
if [ -d "env_serve" ]; then
    echo "✅ 'env_serve' already exists. Skipping creation."
else
    echo "📦 Creating 'env_serve'..."
    uv venv env_serve --python 3.11
fi

ACTIVATE_SERVE=$(get_activate_script "env_serve")
if [ -z "$ACTIVATE_SERVE" ]; then
    echo "❌ Error: Could not find activate script for env_serve"
    exit 1
fi
source "$ACTIVATE_SERVE"

# Install vLLM Stack
echo "⬇️  Installing serving dependencies..."
uv pip install "numpy<2.0.0"
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements-serve.txt

echo ""
echo "=== 🚀 Setup Complete ==="
echo "To activate manually:"
echo "  source env_train/bin/activate  (Linux/Mac)"
echo "  source env_train/Scripts/activate (Windows)"
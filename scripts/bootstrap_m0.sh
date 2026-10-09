#!/usr/bin/env bash
# ==============================================================================
# Milestone 0 Bootstrap: Environment & Sidecar Service Initializer
# Designed for Linux (Codespaces, Cloud Shell) and macOS
# ==============================================================================
set -e

echo "🚀 [M0 Bootstrap] Starting environment initialization..."

# 1. Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 [M0 Bootstrap] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✅ [M0 Bootstrap] uv version: $(uv --version)"

# 2. Sync Python virtual environment with uv
echo "📦 [M0 Bootstrap] Synchronizing Python environment from pyproject.toml..."
uv sync --extra dev

# 3. Check / Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "🤖 [M0 Bootstrap] Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Please install Ollama on macOS via 'brew install ollama' or downloading from ollama.com"
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

# 4. Start Ollama daemon if not running
mkdir -p logs
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚡ [M0 Bootstrap] Starting background Ollama daemon..."
    ollama serve > logs/ollama.log 2>&1 &
    sleep 3
fi

# Verify Ollama reachability
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ [M0 Bootstrap] Ollama daemon is active at http://localhost:11434."
    
    # 5. Pull default lightweight model if not already present
    DEFAULT_MODEL="qwen2.5:0.5b"
    if ! ollama list | grep -q "$DEFAULT_MODEL"; then
        echo "📥 [M0 Bootstrap] Pulling baseline model: $DEFAULT_MODEL..."
        ollama pull "$DEFAULT_MODEL"
    else
        echo "✅ [M0 Bootstrap] Model $DEFAULT_MODEL is already cached."
    fi
else
    echo "⚠️ [M0 Bootstrap] Ollama daemon not reachable at localhost:11434 (will proceed in offline/mock mode)."
fi

# 6. Check Cloudflared
if command -v cloudflared &> /dev/null; then
    echo "✅ [M0 Bootstrap] cloudflared is ready: $(cloudflared --version | head -n 1)"
else
    echo "ℹ️ [M0 Bootstrap] cloudflared not found. Install via 'scripts/install_cloudflared.sh' or apt/brew when ready."
fi

# 7. Run M0 Test Suite
echo "🧪 [M0 Bootstrap] Running verification test suite..."
uv run pytest tests/test_m0.py -v

echo "🎉 [M0 Bootstrap] Milestone 0 is fully verified and ready!"

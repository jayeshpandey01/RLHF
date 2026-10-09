#!/usr/bin/env bash
# ==============================================================================
# Cloudflare Tunnel Runner: Exposes FastAPI backend (Port 8000)
# ==============================================================================
set -e

PORT=${1:-8000}

if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared is not installed."
    echo "Install on Linux/Codespaces: curl -LO https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared-linux-amd64.deb"
    echo "Install on macOS: brew install cloudflared"
    exit 1
fi

echo "🌐 Starting Cloudflare Quick Tunnel to http://localhost:$PORT..."
cloudflared tunnel --url "http://localhost:$PORT"

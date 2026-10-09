# Autonomous Continual Learning LLM

An autonomous, self-updating post-training pipeline that learns from user search queries, scrapes and cleans web content, updates a model with LoRA & Experience Replay, gates releases with automated benchmarks, and hot-swaps active models in Ollama with zero downtime.

Designed to run seamlessly in **GitHub Codespaces**, **Google Cloud Shell**, and local environments with **Cloudflare Tunnels**.

---

## Quick Start (Milestone 0)

### 1. In GitHub Codespaces / DevContainer
When you open this repository in a GitHub Codespace, the `.devcontainer` configuration automatically executes `scripts/bootstrap_m0.sh`, installs dependencies with `uv`, launches the background Ollama daemon, and caches the baseline model.

### 2. Local Setup
```bash
# Make bootstrap executable and run
chmod +x scripts/*.sh
bash scripts/bootstrap_m0.sh
```

### 3. Exposing via Cloudflare Tunnel
To get an instant, public HTTPS URL for your Codespace or Cloud Shell session:
```bash
bash scripts/run_tunnel.sh 8000
```

### 4. Running the Test Suite
```bash
uv run pytest tests/test_m0.py -v
```

---

## Project Structure

```
autonomous-learning-llm/
├── .devcontainer/              # Codespaces / VS Code Docker parity
├── core/                       # Core contracts, hardware strategy, config
│   ├── config.py               # Pydantic Settings
│   ├── hardware.py             # CUDA / MPS / CPU adaptive strategy
│   ├── schemas.py              # Pydantic V2 data contracts
│   └── ollama_client.py        # Ollama API client
├── ingestion/                  # Web scrapers (Trafilatura & DuckDuckGo)
├── datasets/                   # Versioned JSONL datasets & anchor replay buffers
├── training/                   # SFT (LoRA), DPO, and GRPO training pipelines
├── evaluation/                 # Benchmark suites & citation verification gates
├── orchestration/              # Background job queues & Ollama deployment
├── apps/                       # FastAPI chat backend & Open WebUI integrations
├── scripts/                    # Bootstrap & tunnel scripts
└── tests/                      # Pytest verification suites
```
# RLHF

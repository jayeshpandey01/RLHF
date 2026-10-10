# Autonomous Continual Learning LLM Prototype

Research and prototype code for a continual-learning LLM pipeline: web ingestion, curation, retrieval, replay, MiniGPT/LoRA training, evaluation, and serving components.

**Implementation status:** the repository contains connected component paths and tests, but it is not yet a configured, autonomous production system. The default FastAPI app has no LLM backend configured; the Ollama client is separate from the MiniGPT training path, and the continual-learning API requires an injected training callback. See [the research document's implementation status](docs/research.md#pipeline-implementation-status) before treating it as an end-to-end deployment.

---

## Setup and validation

### Run the verification pipeline

Requires Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/). If uv is not installed, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then run the same verification command in GitHub Codespaces or Google Cloud Shell:

```bash
bash scripts/verify.sh
```

It synchronizes dependencies from `pyproject.toml`, runs Ruff's fatal-error checks, and runs the full pytest suite. GitHub Actions runs this pipeline on pushes, pull requests, and manual dispatches.

To install dependencies without running checks:

```bash
uv sync --extra dev
```

The milestone-zero bootstrap script is available for setting up Ollama separately:
```bash
bash scripts/bootstrap_m0.sh
```

Do not expose the API or its model promotion/training endpoints through a public tunnel without adding authentication and access controls.

This CI pipeline validates the repository; it does not deploy to Google Cloud or configure a production model backend.

---

## Project Structure

```
autonomous-learning-llm/
├── core/                       # Core contracts, hardware strategy, config
│   ├── config.py               # Pydantic Settings
│   ├── hardware.py             # CUDA / MPS / CPU adaptive strategy
│   ├── schemas.py              # Pydantic V2 data contracts
│   └── ollama_client.py        # Ollama API client
├── ingestion/                  # Multi-source web and research ingestion
├── curation/                   # Quality filters, deduplication, replay buffer
├── retrieval/                  # BM25, dense, and hybrid retrieval
├── training/                   # MiniGPT, SFT/LoRA, and preference trainers
├── evaluation/                 # Benchmark suites & citation verification gates
├── orchestration/              # Query, training-cycle, and event orchestration
├── apps/                       # FastAPI gateway and model serving manager
├── data/                       # Local corpora and provenance artifacts
├── scripts/                    # Bootstrap & tunnel scripts
└── tests/                      # Pytest verification suites
```
# RLHF

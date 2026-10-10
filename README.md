# Autonomous Continual Learning LLM Prototype

<p align="center">
  <img src="assets/bunny.jpg" alt="Autonomous Continual Learning LLM Architecture" width="100%" style="border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Ollama-Qwen2.5--0.5B-000000?style=flat-square&logo=ollama&logoColor=white" alt="Ollama Backend" />
  <img src="https://img.shields.io/badge/Tests-212%20Passed-brightgreen?style=flat-square&logo=pytest&logoColor=white" alt="Pytest Suite" />
  <img src="https://img.shields.io/badge/Code%20Style-Ruff-black?style=flat-square" alt="Ruff" />
</p>

An end-to-end, closed-loop system prototype for **autonomous continual post-training in Large Language Models**. The system integrates live web curation, hybrid retrieval, factual citation grounding, experience replay, parameter-efficient fine-tuning (LoRA), preference alignment (DPO/PPO/GRPO), and non-parametric statistical hypothesis release gating with zero-downtime atomic rollbacks.

> [!NOTE]
> **Implementation Scope**: The default FastAPI application performs live web collection, curation, persistent hybrid indexing, and Ollama-backed grounded inference. Curated documents feed a separate Hugging Face Qwen LoRA post-training worker; candidate checkpoints are promoted into Ollama only when the statistical benchmark release gate approves them. For complete theoretical foundations, mathematical derivations, and architecture specifications, consult the **[11-Chapter Academic Research Monograph](docs/research.md)**.

---

## Key System Capabilities

<p align="center">
  <img src="assets/study_calm.jpg" alt="Continual Learning Workspace" width="100%" style="border-radius: 10px; max-height: 380px; object-fit: cover;" />
</p>

| Pipeline Stage | Module | Engineering Specification |
| :--- | :--- | :--- |
| **1. Web Ingestion** | `ingestion/` | Multi-source scraping via Trafilatura DOM parser, HTML text density ratios, SHA-256 provenance tracking, and rate-limited async fetching. |
| **2. Curation & LSH** | `curation/` | MinHash Locality-Sensitive Hashing (LSH) for near-duplicate elimination ($s = 0.80$, 32 bands $	imes$ 4 rows), 13-gram benchmark contamination filtering, and lexical quality scoring. |
| **3. Hybrid Retrieval** | `retrieval/` | Sparse Okapi BM25 + dense sentence embeddings fused via Reciprocal Rank Fusion (RRF, $k=60$) with bounded in-memory caching. |
| **4. Citation Grounding** | `evaluation/` | Sentence-level ALCE citation recall & precision, RAG Triad faithfulness verification ($> 85\%$ threshold), and fallback citation extraction. |
| **5. Experience Replay** | `curation/` | Stratified replay buffer maintaining a strict **70% Golden Anchor / 30% Streaming Web** mixture ratio to mathematically guard against catastrophic forgetting. |
| **6. Continual SFT & LoRA** | `training/` | Completion-only token masking (zero gradient on user prompts), PyTorch LoRA adapter injection ($r=8, lpha=16$), and adaptive cosine learning-rate schedules. |
| **7. Preference Alignment** | `training/` | Modular alignment engines supporting Direct Preference Optimization (**DPO**, **cDPO**, **IPO**), Actor-Critic **PPO** with GAE, and group-relative policy optimization (**DeepSeek-R1 GRPO**). |
| **8. Statistical Gating** | `evaluation/` | Four-bucket release evaluation using Paired Bootstrap Resampling ($B=1000, 	ext{CI}_{0.95}$), Wilcoxon Signed-Rank tests, and Edwards continuity-corrected McNemar discordance tests. |
| **9. Autonomous Rollback** | `evaluation/` | Atomic state pointer management; automatically promotes validated checkpoints or hot-swaps serving back to the golden anchor upon detected degradation. |

---

## Quickstart & Installation

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended)
- **Local LLM Backend**: [Ollama](https://ollama.com/) (running `qwen2.5:0.5b`)

### 1. Install `uv` & Dependencies

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchronize all runtime and development dependencies
uv sync --extra dev
```

### 2. Run Verification Pipeline

Execute the automated test suite (212 unit tests across all 6 milestone subsystems):

```bash
bash scripts/verify.sh
```

Or run pytest directly:

```bash
uv run pytest
```

### 3. Bootstrap Ollama (Optional)

If setting up Ollama on macOS or Linux, download the base `qwen2.5:0.5b` model:

```bash
bash scripts/bootstrap_m0.sh
```

---

## Running the Live Query-to-Learning Service

### Step 1: Inspect Detected Hardware

The engine automatically selects the optimal device strategy (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback):

```bash
uv run python -c "from core.hardware import get_device_strategy; s = get_device_strategy(); print(f'Device: {s.device_name} | Precision: {s.precision} | Accumulation Steps: {s.gradient_accumulation_steps}')"
```

### Step 2: Start the FastAPI Gateway

Launch the background service:

```bash
mkdir -p logs
nohup uv run uvicorn apps.server:create_app --factory --host 0.0.0.0 --port 8000 > logs/server.log 2>&1 &
```

Confirm health status:

```bash
curl -fsS --retry 10 --retry-connrefused --retry-delay 1 http://localhost:8000/health | python -m json.tool
```

### Step 3: Execute an Augmented Query

Submit a query. The engine researches the topic, curates and indexes accepted source documents, retrieves supporting passages, and generates a grounded response with passage citations:

```bash
curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"How does Direct Preference Optimization eliminate the reward model?","top_k":3}'   | python -m json.tool
```

> [!TIP]
> If the small model hallucinates or ignores citation formatting, the **CitationVerifier** intercepts the response and returns a directly grounded citation extract from the top-scoring passage.

### Step 4: Trigger Autonomous Continual Post-Training

Accepted documents persist in `data/knowledge_corpus.jsonl`. When 5 or more new reviewed records accumulate, the background LoRA post-training worker automatically queues. You can also trigger learning manually:

```bash
curl -sS -X POST http://localhost:8000/trigger_learning   -H "Content-Type: application/json"   -d '{"force":true}'   | python -m json.tool
```

### Step 5: Monitor Learning Job & Gate Evaluation

Poll the job status using the returned `job_id`:

```bash
curl -sS http://localhost:8000/learning/jobs/<JOB_ID> | python -m json.tool
```

- **`PROMOTED`**: Checkpoint cleared all 4 release gate criteria and was hot-swapped into Ollama serving.
- **`BLOCKED`**: Candidate exhibited regression or failed statistical confidence; serving remains safely anchored to the previous checkpoint.
- **`FAILED`**: Training crashed or data was unreviewed.

To view the live replay buffer distribution and active jobs:

```bash
curl -sS http://localhost:8000/status | python -m json.tool
```

### Step 6: Atomic Manual Rollback

If required, instantly roll back the active serving model to the initial golden anchor:

```bash
curl -sS -X POST http://localhost:8000/models/rollback | python -m json.tool
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System readiness, active model checkpoint, and device diagnostics. |
| `POST` | `/query` | Grounded RAG query with live research, citation verification, and auto-fallback. |
| `POST` | `/trigger_learning` | Queues bounded LoRA SFT worker on the stratified replay buffer. |
| `GET` | `/learning/jobs/{job_id}` | Polls training metrics, statistical gate report, and promotion status. |
| `GET` | `/status` | Reports anchor vs. web replay counts, index size, and active jobs. |
| `POST` | `/models/rollback` | Atomically reverts the serving pointer to the baseline golden anchor. |

---

## Project Structure

```
autonomous-learning-llm/
├── core/                       # Core abstractions, hardware detection, Pydantic schemas
│   ├── config.py               # Global environment settings & thresholds
│   ├── hardware.py             # CUDA / Apple Silicon MPS / CPU adaptive strategy
│   ├── schemas.py              # Strict Pydantic V2 data contracts
│   └── ollama_client.py        # Robust asynchronous Ollama client
├── ingestion/                  # Multi-source web crawling & DOM extraction
│   ├── scraper.py              # Trafilatura HTML cleaner & density parser
│   └── provenance.py           # SHA-256 content-addressable manifest logger
├── curation/                   # Text deduplication, quality scoring, replay buffer
│   ├── minhash.py              # MinHash LSH near-duplicate index
│   ├── quality_filter.py       # Heuristic language & stopword density filters
│   └── replay_buffer.py        # 70% Anchor / 30% Web stratified experience replay
├── retrieval/                  # Hybrid evidence retrieval subsystem
│   ├── bm25.py                 # Okapi BM25 sparse keyword index
│   ├── embeddings.py           # Dense semantic similarity encoder
│   └── hybrid_search.py        # Reciprocal Rank Fusion (RRF) rank combiner
├── training/                   # Model architectures & training algorithms
│   ├── m1_autograd/            # Reverse-mode automatic differentiation DAG engine
│   ├── m1_transformer/         # MiniGPT causal autoregressive decoder
│   ├── m3_sft/                 # LoRA adapter injection & completion loss trainer
│   └── m5_preference/          # Bradley-Terry RM, DPO, cDPO, IPO, PPO, and GRPO
├── evaluation/                 # Benchmark suites, citation auditing, release gates
│   ├── benchmark_suite.py      # 30-case four-bucket stratified golden benchmark
│   ├── citation_verifier.py    # ALCE proposition entailment & RAG Triad faithfulness
│   ├── llm_judge.py            # Position-debiased pairwise LLM evaluator
│   ├── statistical_gating.py   # Bootstrap CI (B=1000), Wilcoxon, and McNemar tests
│   └── rollback_orchestrator.py# Atomic deployment state manager & hot-swap pointer
├── orchestration/              # Closed-loop query, training, and lifecycle pipelines
│   └── loop_orchestrator.py    # End-to-end continuous learning state machine
├── apps/                       # Production serving interfaces
│   └── server.py               # FastAPI gateway with background worker tasks
├── scripts/                    # Bootstrap, verification, and end-to-end smoke test utilities
├── docs/                       # Comprehensive academic monograph (docs/research.md)
└── tests/                      # 212 comprehensive pytest unit and integration tests
```

---

## Security & Operational Notes

- **Network Security**: The API endpoints (especially `/trigger_learning` and `/models/rollback`) are intended for internal service orchestration. Do not expose this service through public tunnels (e.g. ngrok, Cloudflare Tunnel) without adding mutual TLS, reverse proxy authentication, or API keys.
- **Stateless Post-Training Worker**: The Hugging Face LoRA post-training worker runs as an isolated asynchronous task to preserve serving gateway responsiveness.
- **Audit Logging**: Every web document, training job, evaluation report, and model rollback is immutably logged under `data/provenance_manifest.jsonl` and `logs/`.



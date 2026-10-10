# Autonomous Continual Learning LLM Prototype


<p align="center">
  <img src="assets/bunny.jpg" alt="Calm Koala Resting" width="48%" style="border-radius: 10px;" />
</p>

Research and prototype code for a continual-learning LLM pipeline: web ingestion, curation, retrieval, replay, MiniGPT/LoRA training, evaluation, and serving components.

**Implementation status:** the default FastAPI app now performs live web collection, curation, persistent indexing, and Ollama-backed grounded answers. Curated documents feed a separate Hugging Face Qwen LoRA worker; candidates are promoted into Ollama only when the statistical benchmark gate approves them. This is a single-process prototype, not a durable or authenticated production service. See [the research document's implementation status](docs/research.md#pipeline-implementation-status).

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

### Check hardware and start the API

After synchronizing dependencies, inspect the detected compute strategy with the
attributes provided by `DeviceStrategy`:

```bash
uv run python -c "from core.hardware import get_device_strategy; s = get_device_strategy(); print(f'Detected: {s.device_name} | Dtype: {s.precision} | Grad Accum: {s.gradient_accumulation_steps}')"
```

Start the FastAPI gateway and allow a few seconds for Uvicorn to become ready
before checking its health endpoint:

```bash
mkdir -p logs
nohup uv run uvicorn apps.server:create_app --factory --host 0.0.0.0 --port 8000 > logs/server.log 2>&1 &
curl -fsS --retry 10 --retry-connrefused --retry-delay 1 http://localhost:8000/health | python -m json.tool
```

If the health check still fails, inspect `logs/server.log` for the startup error.

### Run the query-to-learning loop

Keep Ollama running with `qwen2.5:0.5b` installed and start the API as above.
In a second terminal, submit a query; the API researches the topic, curates and
persists accepted source documents, retrieves evidence, then asks Ollama for a
cited response. If the small model ignores citation instructions or produces
an answer the verifier cannot ground, the API returns a directly cited
extract from the most relevant retrieved abstract instead of presenting the
unsupported generation as fact. The response's `research` field reports the
fetch, curation, and indexing counts:

```bash
curl -sS -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"How does Direct Preference Optimization eliminate the reward model?","top_k":3}' \
  | python -m json.tool
```

Accepted documents are stored under `data/knowledge_corpus.jsonl` and indexed
again at server startup. At five or more accepted web documents, a query
automatically queues the bounded CPU LoRA worker. To start training before that
threshold, submit an explicit learning request:

```bash
curl -sS -X POST http://localhost:8000/trigger_learning \
  -H "Content-Type: application/json" -d '{"force":true}' | python -m json.tool
```

The response includes a `job_id`; poll `GET /learning/jobs/<job_id>` until the
job is `PROMOTED`, `BLOCKED`, or `FAILED`. Training uses
`Qwen/Qwen2.5-0.5B-Instruct`, 20 CPU steps by default, and a 70/30 anchor/web
sample mix. Set `HF_BASE_MODEL` or `HF_SFT_MAX_STEPS` before starting the API to
override those defaults. Model downloads, CPU training, full benchmark
inference, and Ollama packaging can take substantial time and disk space.
Candidates that fail the statistical gate are retained as adapters with a
gate report but are not created or promoted in Ollama. `/status` reports
replay counts and the active learning job. `/models/rollback` returns serving
to the original Ollama anchor.

The standalone `scripts/run_ollama_e2e.py` remains a temporary-artifact
ingestion/retrieval/inference smoke test; it does not fine-tune. Live quality
depends on retrieved sources and model output, so inspect citations and
`data/provenance_manifest.jsonl`. Do not expose the API or its learning/model
management endpoints through a public tunnel without authentication and access
controls.

Do not expose the API or its model promotion/training endpoints through a public tunnel without adding authentication and access controls.

This CI pipeline validates the repository; it does not deploy to Google Cloud or configure a production model backend.



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

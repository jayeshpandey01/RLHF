import os
import re
from functools import partial
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from apps.learning_jobs import LearningJobManager
from apps.live_knowledge import LiveKnowledgeService
from apps.serving_manager import ModelServingManager
from core.config import settings
from orchestration.loop_orchestrator import ContinualLearningOrchestrator
from training.m3_sft.hf_worker import HuggingFaceLoRAWorker


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language user query")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of evidence documents to retrieve")
    stream: bool = Field(
        default=False,
        description="Streaming is not implemented; true returns HTTP 501.",
    )


class QueryResponse(BaseModel):
    query: str
    response: str
    cached: bool
    similarity: float
    evidence: list[dict[str, Any]]
    model_version: int
    learning_job_id: str | None = None
    research: dict[str, Any] | None = None


class TriggerLearningRequest(BaseModel):
    force: bool = Field(default=True, description="Force continual learning cycle regardless of threshold")


class PromoteModelRequest(BaseModel):
    version: int | None = Field(
        default=None,
        ge=1,
        description="Optional positive target version number; version zero is reserved for rollback",
    )


def create_app(
    orchestrator: ContinualLearningOrchestrator | None = None,
    serving_manager: ModelServingManager | None = None,
    live_knowledge: LiveKnowledgeService | None = None,
    training_worker: HuggingFaceLoRAWorker | None = None,
    learning_jobs: LearningJobManager | None = None,
) -> FastAPI:
    """Application factory constructing FastAPI gateway with dependency injection."""
    app = FastAPI(
        title="Continual RAG & Alignment LLM Serving Gateway",
        description="High-concurrency FastAPI gateway with L1 semantic cache, hybrid retrieval, and zero-downtime hot-swapping.",
        version="1.0.0",
    )

    is_default_app = orchestrator is None
    orch = orchestrator or ContinualLearningOrchestrator(active_model=settings.DEFAULT_MODEL)
    mgr = serving_manager or ModelServingManager(
        active_model=orch.active_model,
        anchor_model=orch.anchor_model,
        model_version=orch.model_version,
    )
    knowledge = live_knowledge
    if knowledge is None and is_default_app:
        knowledge = LiveKnowledgeService(orch.retriever)
    worker = training_worker
    if worker is None and is_default_app:
        worker = HuggingFaceLoRAWorker(
            corpus_path=(knowledge.corpus_path if knowledge else settings.DATA_DIR / "knowledge_corpus.jsonl")
        )
    jobs = learning_jobs
    if jobs is None and worker is not None:
        jobs = LearningJobManager()
    use_ollama_backend = is_default_app

    def promote_ollama_candidate(model_tag: str) -> int:
        version = mgr.promote_candidate(candidate_model=model_tag)
        orch.active_model = model_tag
        orch.model_version = version
        orch.semantic_cache.clear()
        return version

    def submit_training_job() -> tuple[str, bool]:
        if worker is None or jobs is None:
            raise RuntimeError("The Hugging Face training worker is not configured.")
        active_model = mgr.get_active_model()
        if isinstance(active_model, str):
            train = partial(
                worker.train_evaluate_and_package,
                baseline_model=active_model,
            )
        elif isinstance(worker, HuggingFaceLoRAWorker):
            raise RuntimeError(
                "Hugging Face candidate evaluation requires the active Ollama model tag."
            )
        else:
            train = worker.train_evaluate_and_package
        return jobs.submit(train, promote_ollama_candidate)

    def generate_with_ollama(query: str, evidence: list, model: str) -> str:
        context = "\n\n".join(
            f"[{index}] Source: {item.url or item.doc_id}\n{item.text}"
            for index, item in enumerate(evidence, start=1)
        )
        prompt = (
            f"Question: {query}\n\nEvidence:\n{context or 'No web evidence was retrieved.'}\n\n"
            "Answer only with claims supported by the evidence. Add inline citations such as [1] "
            "for every factual claim. If the evidence is insufficient, say so."
        )
        response = httpx.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer accurately, briefly, and only from supplied evidence. "
                            "Use inline numbered citations like [1]; do not repeat source URLs."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "")
        answer = re.sub(
            r"[ \t]+Source:\s*https?://\S+",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer.")
        return answer

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_version": mgr.model_version,
            "service": "continual-rag-llm",
        }

    @app.get("/status")
    async def get_system_status():
        orch_status = orch.get_status()
        serving_stats = mgr.get_stats()
        return {
            "orchestrator": orch_status,
            "serving_manager": serving_stats,
            "model_version": mgr.model_version,
            "knowledge": knowledge.replay_buffer.get_stats() if knowledge is not None else None,
            "learning_job_id": jobs.active_job_id() if jobs is not None else None,
            "active_model": mgr.get_active_model(),
        }

    @app.post("/query", response_model=QueryResponse)
    async def process_user_query(req: QueryRequest):
        if req.stream:
            raise HTTPException(
                status_code=501,
                detail="Streaming responses are not implemented yet.",
            )
        try:
            learning_job_id = None
            collection = None
            should_train = False
            if knowledge is not None:
                collection = await knowledge.collect_and_index(req.query)
                minimum_documents = int(os.getenv("LEARNING_TRIGGER_THRESHOLD", "5"))
                web_count = collection.get("replay_stats", {}).get("web_count", 0)
                should_train = (
                    collection.get("indexed", 0) > 0
                    and web_count >= minimum_documents
                    and jobs is not None
                    and jobs.active_job_id() is None
                )
            with mgr.read_context() as active_model:
                model_version = mgr.model_version
                result = await run_in_threadpool(
                    orch.process_query,
                    query=req.query,
                    top_k=req.top_k,
                    active_model=active_model,
                    generator_fn=(
                        generate_with_ollama
                        if (
                            isinstance(active_model, str)
                            and (
                                use_ollama_backend
                                or active_model
                                in {settings.DEFAULT_MODEL, settings.FALLBACK_MODEL}
                                or active_model.startswith("rlhf-candidate-")
                            )
                        )
                        else None
                    ),
                )
            if should_train:
                learning_job_id, _ = submit_training_job()
            result["model_version"] = model_version
            result["learning_job_id"] = learning_job_id
            result["research"] = collection
            return QueryResponse(**result)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Ollama inference failed: {e}") from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/trigger_learning")
    async def trigger_learning_cycle(req: TriggerLearningRequest):
        if worker is not None and jobs is not None:
            if knowledge is None or knowledge.replay_buffer.get_stats()["web_count"] == 0:
                raise HTTPException(
                    status_code=409,
                    detail="No curated web documents are available; submit a query before training.",
                )
            web_count = knowledge.replay_buffer.get_stats()["web_count"]
            minimum_documents = int(os.getenv("LEARNING_TRIGGER_THRESHOLD", "5"))
            if not req.force and web_count < minimum_documents:
                return {
                    "status": "SKIPPED",
                    "reason": "DATA_THRESHOLD_NOT_MET",
                    "web_documents": web_count,
                    "required_web_documents": minimum_documents,
                    "model_version": mgr.model_version,
                }
            try:
                job_id, created = submit_training_job()
                return {
                    "status": "QUEUED" if created else "ALREADY_RUNNING",
                    "job_id": job_id,
                    "model_version": mgr.model_version,
                    "forced": req.force,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e
        try:
            report = await run_in_threadpool(
                orch.execute_continual_learning_cycle,
                force=req.force,
            )
            if report.get("status") == "PROMOTED":
                mgr.promote_candidate(
                    candidate_model=orch.active_model,
                    new_version=orch.model_version,
                )
            elif report.get("status") == "ROLLED_BACK":
                mgr.rollback_to_anchor()
            return report
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/learning/jobs/{job_id}")
    async def get_learning_job(job_id: str):
        if jobs is None:
            raise HTTPException(status_code=503, detail="Background learning is not configured.")
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Learning job not found.")
        return job

    @app.post("/models/promote")
    async def promote_model(req: PromoteModelRequest):
        cand = orch.candidate_model
        if cand is None:
            raise HTTPException(
                status_code=409,
                detail="No trained candidate model is available to promote.",
            )
        try:
            new_v = mgr.promote_candidate(candidate_model=cand, new_version=req.version)
            orch.active_model = cand
            orch.model_version = new_v
            orch.semantic_cache.clear()
            return {
                "status": "PROMOTED",
                "new_version": new_v,
                "active_readers": mgr.get_stats()["active_readers"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/models/rollback")
    async def rollback_model():
        try:
            new_v = mgr.rollback_to_anchor()
            orch.active_model = mgr.anchor_model
            orch.model_version = new_v
            orch.semantic_cache.clear()
            return {
                "status": "ROLLED_BACK",
                "model_version": new_v,
                "anchor_restored": True,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app

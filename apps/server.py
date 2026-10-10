from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from apps.serving_manager import ModelServingManager
from orchestration.loop_orchestrator import ContinualLearningOrchestrator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language user query")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of evidence documents to retrieve")
    stream: bool = Field(default=False, description="Enable Server-Sent Events (SSE) streaming")


class QueryResponse(BaseModel):
    query: str
    response: str
    cached: bool
    similarity: float
    evidence: List[Dict[str, Any]]
    model_version: int


class TriggerLearningRequest(BaseModel):
    force: bool = Field(default=True, description="Force continual learning cycle regardless of threshold")


class PromoteModelRequest(BaseModel):
    version: Optional[int] = Field(default=None, description="Optional target version number")


def create_app(
    orchestrator: Optional[ContinualLearningOrchestrator] = None,
    serving_manager: Optional[ModelServingManager] = None,
) -> FastAPI:
    """Application factory constructing FastAPI gateway with dependency injection."""
    app = FastAPI(
        title="Continual RAG & Alignment LLM Serving Gateway",
        description="High-concurrency FastAPI gateway with L1 semantic cache, hybrid retrieval, and zero-downtime hot-swapping.",
        version="1.0.0",
    )

    orch = orchestrator or ContinualLearningOrchestrator()
    mgr = serving_manager or ModelServingManager(
        active_model=orch.active_model,
        anchor_model=orch.anchor_model,
        model_version=orch.model_version,
    )

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
        }

    @app.post("/query", response_model=QueryResponse)
    async def process_user_query(req: QueryRequest):
        if req.stream:
            raise HTTPException(
                status_code=501,
                detail="Streaming responses are not implemented yet.",
            )
        try:
            with mgr.read_context():
                result = await run_in_threadpool(
                    orch.process_query,
                    query=req.query,
                    top_k=req.top_k,
                )
            result["model_version"] = mgr.model_version
            return QueryResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/trigger_learning")
    async def trigger_learning_cycle(req: TriggerLearningRequest):
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
            raise HTTPException(status_code=500, detail=str(e))

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
            raise HTTPException(status_code=500, detail=str(e))

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
            raise HTTPException(status_code=500, detail=str(e))

    return app

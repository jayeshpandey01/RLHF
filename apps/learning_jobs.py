"""Single-worker background job registry for CPU-heavy candidate training."""

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from uuid import uuid4


class LearningJobManager:
    """Run one training job at a time and expose durable-for-process status snapshots."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlhf-training")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None
        self._future: Future | None = None

    def submit(
        self,
        worker: Callable[[], dict[str, Any]],
        on_approved: Callable[[str], int],
    ) -> tuple[str, bool]:
        with self._lock:
            if self._active_job_id is not None:
                return self._active_job_id, False
            job_id = uuid4().hex
            self._active_job_id = job_id
            self._jobs[job_id] = {"job_id": job_id, "status": "QUEUED"}
            self._future = self._executor.submit(self._run, job_id, worker, on_approved)
            return job_id, True

    def _run(
        self,
        job_id: str,
        worker: Callable[[], dict[str, Any]],
        on_approved: Callable[[str], int],
    ) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "RUNNING"
        try:
            result = worker()
            if result.get("status") == "APPROVED":
                model_tag = result.get("ollama_model")
                if not model_tag:
                    raise RuntimeError("Approved training result did not include an Ollama model tag.")
                result["model_version"] = on_approved(model_tag)
                status = "PROMOTED"
            elif result.get("status") == "BLOCKED":
                status = "BLOCKED"
            else:
                raise RuntimeError(f"Unexpected training result status: {result.get('status')!r}")
            with self._lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "status": status,
                    "result": result,
                }
        except Exception as exc:
            with self._lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "status": "FAILED",
                    "error": str(exc),
                }
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id

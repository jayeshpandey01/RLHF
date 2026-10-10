from contextlib import contextmanager
from collections import defaultdict
import threading
import time
from typing import Any, Dict, Generator, Optional


class ModelServingManager:
    """Thread-safe model serving manager with zero-downtime atomic pointer swapping.

    Implements a fair Reader-Writer lock pattern using Condition variables (Courtois et al., 1971; Chen et al., 2021):
      - In-flight inference requests acquire a shared reader context.
      - Candidate model promotion waits for active in-flight readers to finish cleanly.
      - Writer priority prevents writer starvation by holding off new readers while promotion is queued.
      - Re-entrant read safety avoids self-deadlock within the same thread.
    """

    def __init__(
        self,
        active_model: Optional[Any] = None,
        anchor_model: Optional[Any] = None,
        model_version: int = 1,
    ):
        self.active_model = active_model
        self.anchor_model = anchor_model or active_model
        self.candidate_model: Optional[Any] = None
        self.model_version = int(model_version)

        self._lock = threading.Lock()
        self._readers_ok = threading.Condition(self._lock)
        self._writers_ok = threading.Condition(self._lock)
        self._active_readers = 0
        self._waiting_writers = 0
        self._writing = False
        self._reader_threads: Dict[int, int] = defaultdict(int)

        self.total_promotions = 0
        self.total_rollbacks = 0

    @contextmanager
    def read_context(self) -> Generator[Any, None, None]:
        """Shared reader context manager allowing concurrent model inference passes."""
        tid = threading.get_ident()
        with self._lock:
            # Re-entrant reads on the same thread never block
            if self._reader_threads[tid] == 0:
                while self._writing or self._waiting_writers > 0:
                    self._readers_ok.wait()
            self._active_readers += 1
            self._reader_threads[tid] += 1
            model_ref = self.active_model

        try:
            yield model_ref
        finally:
            with self._lock:
                self._active_readers -= 1
                self._reader_threads[tid] -= 1
                if self._reader_threads[tid] == 0:
                    del self._reader_threads[tid]
                if self._active_readers == 0:
                    self._writers_ok.notify_all()

    def get_active_model(self) -> Any:
        """Return the current active model reference."""
        with self._lock:
            return self.active_model

    def promote_candidate(
        self,
        candidate_model: Any,
        new_version: Optional[int] = None,
    ) -> int:
        """Atomically hot-swap active model with candidate model under exclusive write lock.

        Waits for all in-flight readers to complete before performing the reference swap.
        """
        with self._lock:
            self._waiting_writers += 1
            while self._writing or self._active_readers > 0:
                self._writers_ok.wait()
            self._waiting_writers -= 1
            self._writing = True

        try:
            self.candidate_model = candidate_model
            self.active_model = candidate_model
            if new_version is not None:
                self.model_version = int(new_version)
            else:
                self.model_version += 1
            self.total_promotions += 1
            return self.model_version
        finally:
            with self._lock:
                self._writing = False
                self._readers_ok.notify_all()
                self._writers_ok.notify_all()

    def rollback_to_anchor(self) -> int:
        """Atomically revert active serving model to golden baseline anchor under exclusive write lock."""
        with self._lock:
            self._waiting_writers += 1
            while self._writing or self._active_readers > 0:
                self._writers_ok.wait()
            self._waiting_writers -= 1
            self._writing = True

        try:
            self.active_model = self.anchor_model
            self.model_version = 0
            self.total_rollbacks += 1
            return self.model_version
        finally:
            with self._lock:
                self._writing = False
                self._readers_ok.notify_all()
                self._writers_ok.notify_all()

    def get_stats(self) -> Dict[str, Any]:
        """Return serving manager diagnostics."""
        with self._lock:
            return {
                "model_version": self.model_version,
                "is_anchor": (self.active_model is self.anchor_model),
                "active_readers": self._active_readers,
                "waiting_writers": self._waiting_writers,
                "total_promotions": self.total_promotions,
                "total_rollbacks": self.total_rollbacks,
            }


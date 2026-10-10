from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryEvent:
    """Telemetry and knowledge trigger event from the serving gateway."""
    event_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class TokenBucketRateLimiter:
    """Token Bucket rate limiter for bursty high-concurrency event ingestion (Turner, 1986)."""

    def __init__(self, rate_per_second: float = 100.0, burst_limit: int = 200):
        self.rate = float(rate_per_second)
        self.capacity = float(burst_limit)
        self.tokens = float(burst_limit)
        self.last_update = time.time()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


class AsyncEventQueue:
    """Thread-safe bounded asynchronous event queue with rate limiting and backpressure."""

    def __init__(
        self,
        max_size: int = 10000,
        rate_limit_per_second: float = 100.0,
        burst_limit: int = 200,
    ):
        self.max_size = int(max_size)
        self.rate_limiter = TokenBucketRateLimiter(
            rate_per_second=rate_limit_per_second, burst_limit=burst_limit
        )
        self._queue: deque[TelemetryEvent] = deque()
        self._lock = threading.Lock()

        self.total_enqueued: int = 0
        self.total_dropped_rate_limit: int = 0
        self.total_dropped_queue_full: int = 0
        self.total_dequeued: int = 0

    def enqueue(self, event: TelemetryEvent) -> bool:
        """Enqueue an event with non-blocking backpressure checks.

        Returns:
            True if successfully enqueued; False if dropped due to rate limit or capacity.
        """
        if not self.rate_limiter.allow():
            with self._lock:
                self.total_dropped_rate_limit += 1
            return False

        with self._lock:
            if len(self._queue) >= self.max_size:
                self.total_dropped_queue_full += 1
                return False

            self._queue.append(event)
            self.total_enqueued += 1
            return True

    def dequeue_batch(self, max_batch_size: int = 50) -> List[TelemetryEvent]:
        """Atomically dequeue up to max_batch_size events."""
        with self._lock:
            batch: List[TelemetryEvent] = []
            while self._queue and len(batch) < max_batch_size:
                batch.append(self._queue.popleft())
            self.total_dequeued += len(batch)
            return batch

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_size": len(self._queue),
                "max_size": self.max_size,
                "total_enqueued": self.total_enqueued,
                "total_dequeued": self.total_dequeued,
                "dropped_rate_limit": self.total_dropped_rate_limit,
                "dropped_queue_full": self.total_dropped_queue_full,
            }

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()


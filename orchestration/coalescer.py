from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class PendingQuery:
    query: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestCoalescer:
    """Sliding-window request coalescer and debouncer.

    Suppresses redundant scraping tasks when multiple concurrent users submit
    identical or near-duplicate queries within a short time window (Turner, 1986).
    Thread-safe under concurrent reader/writer execution.
    """

    def __init__(
        self,
        debounce_window_seconds: float = 1.0,
        similarity_threshold: float = 0.85,
    ):
        self.debounce_window = float(debounce_window_seconds)
        self.similarity_threshold = float(similarity_threshold)
        self.pending_queries: List[PendingQuery] = []
        self._lock = threading.Lock()

        self.total_received: int = 0
        self.total_coalesced: int = 0
        self.total_dispatched: int = 0

    def add_query(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register an incoming query into the pending coalescer buffer."""
        clean = query.strip()
        if not clean:
            return False

        with self._lock:
            self.total_received += 1
            self.pending_queries.append(
                PendingQuery(
                    query=clean,
                    timestamp=time.time(),
                    metadata=metadata or {},
                )
            )
        return True

    NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "none", "cannot", "hardly", "barely", "without"}

    def _tokenize(self, text: str) -> Set[str]:
        return set(text.lower().split())

    def _jaccard_similarity(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        if not tokens_a and not tokens_b:
            return 1.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return float(intersection / union) if union > 0 else 0.0

    def drain_coalesced_queries(self, force: bool = False) -> List[str]:
        """Drain ready queries whose debounce window has expired (or all if force=True).

        Returns:
            List of unique canonical query strings for ingestion.
        """
        now = time.time()
        with self._lock:
            if not self.pending_queries:
                return []

            ready: List[PendingQuery] = []
            remaining: List[PendingQuery] = []

            for item in self.pending_queries:
                if force or (now - item.timestamp) >= self.debounce_window:
                    ready.append(item)
                else:
                    remaining.append(item)

            self.pending_queries = remaining

            if not ready:
                return []

            # Deduplicate & cluster ready queries
            canonical_queries: List[str] = []
            canonical_tokens: List[Set[str]] = []

            for item in ready:
                tokens = self._tokenize(item.query)
                has_neg = bool(tokens & self.NEGATION_WORDS)
                is_duplicate = False
                for c_tokens in canonical_tokens:
                    has_c_neg = bool(c_tokens & self.NEGATION_WORDS)
                    # Do not coalesce queries with opposite negation polarity
                    if has_neg != has_c_neg:
                        continue
                    if self._jaccard_similarity(tokens, c_tokens) >= self.similarity_threshold:
                        is_duplicate = True
                        self.total_coalesced += 1
                        break

                if not is_duplicate:
                    canonical_queries.append(item.query)
                    canonical_tokens.append(tokens)
                    self.total_dispatched += 1

            return canonical_queries

    def get_stats(self) -> Dict[str, Any]:
        """Return coalescer throughput and suppression metrics."""
        with self._lock:
            suppression_rate = (
                (self.total_coalesced / self.total_received) if self.total_received > 0 else 0.0
            )
            return {
                "total_received": self.total_received,
                "total_coalesced": self.total_coalesced,
                "total_dispatched": self.total_dispatched,
                "suppression_rate": suppression_rate,
                "pending_count": len(self.pending_queries),
                "debounce_window_seconds": self.debounce_window,
            }

    def is_empty(self) -> bool:
        with self._lock:
            return len(self.pending_queries) == 0

    def clear(self) -> None:
        with self._lock:
            self.pending_queries.clear()


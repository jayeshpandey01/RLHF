from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from retrieval.dense import SimpleHashEmbedder


@dataclass
class SemanticCacheEntry:
    """Cached response entry with dense embedding and metadata."""
    query: str
    embedding: torch.Tensor
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """Thread-safe L1 response cache using cosine similarity over query vectors."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
        ttl_seconds: Optional[float] = None,
        embedder: Optional[Callable[[str], torch.Tensor]] = None,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds) if ttl_seconds is not None else None
        self.embedder = embedder or SimpleHashEmbedder(dim=128)

        self._lock = threading.RLock()
        self.entries: List[SemanticCacheEntry] = []
        self.total_lookups: int = 0
        self.total_hits: int = 0
        self.total_misses: int = 0

    def _get_embedding(self, query: str) -> torch.Tensor:
        """Produce normalized 1D float32 tensor."""
        vec = self.embedder(query)
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        else:
            vec = vec.detach().to(dtype=torch.float32, device="cpu").squeeze()
        norm = torch.norm(vec, p=2)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    def lookup(self, query: str) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """Perform semantic cache lookup.

        Returns:
            (response, cosine_similarity, metadata) if max similarity >= threshold, else None.
        """
        with self._lock:
            self.total_lookups += 1
            if not self.entries:
                self.total_misses += 1
                return None

            current_time = time.time()
            if self.ttl_seconds is not None:
                self.entries = [
                    e for e in self.entries if (current_time - e.timestamp) <= self.ttl_seconds
                ]
                if not self.entries:
                    self.total_misses += 1
                    return None

            q_vec = self._get_embedding(query)
            cached_matrix = torch.stack([e.embedding for e in self.entries])
            similarities = torch.matmul(cached_matrix, q_vec)

            best_idx = int(torch.argmax(similarities).item())
            best_sim = float(similarities[best_idx].item())

            if best_sim >= self.similarity_threshold:
                self.total_hits += 1
                entry = self.entries[best_idx]
                entry.hit_count += 1
                return entry.response, best_sim, dict(entry.metadata)

            self.total_misses += 1
            return None

    def put(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store query and response into semantic cache."""
        if not query.strip() or not response.strip():
            return

        q_vec = self._get_embedding(query)

        with self._lock:
            if len(self.entries) >= self.max_entries:
                self.entries.pop(0)

            self.entries.append(
                SemanticCacheEntry(
                    query=query,
                    embedding=q_vec,
                    response=response,
                    metadata=metadata or {},
                    timestamp=time.time(),
                    hit_count=0,
                )
            )

    def clear(self) -> None:
        """Clear cache completely."""
        with self._lock:
            self.entries.clear()
            self.total_lookups = 0
            self.total_hits = 0
            self.total_misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return cache hit rate and volume statistics."""
        with self._lock:
            hit_rate = (self.total_hits / self.total_lookups) if self.total_lookups > 0 else 0.0
            return {
                "entries_count": len(self.entries),
                "total_lookups": self.total_lookups,
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "hit_rate": hit_rate,
                "similarity_threshold": self.similarity_threshold,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self.entries)

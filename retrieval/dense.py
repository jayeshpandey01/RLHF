import hashlib
import math
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from core.hardware import get_device_strategy
from retrieval.schemas import RetrievalDocument, SearchResult


class SimpleHashEmbedder:
    """Deterministic character n-gram hashing baseline.

    Produces dense vectors without a pretrained model; similarity is lexical/subword-based,
    not learned semantic similarity.
    """

    def __init__(self, dim: int = 128, ngram_range: tuple = (2, 4)):
        self.dim = dim
        self.ngram_range = ngram_range

    def encode(self, text: str) -> torch.Tensor:
        clean = text.lower().strip()
        vec = torch.zeros(self.dim, dtype=torch.float32)
        if not clean:
            return vec

        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(clean) >= n:
                for i in range(len(clean) - n + 1):
                    ngrams.append(clean[i : i + n])
            else:
                ngrams.append(clean)

        for gram in ngrams:
            # Deterministic MD5 hash bucket
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[idx] += sign

        norm = torch.norm(vec, p=2)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    def __call__(self, text: str) -> torch.Tensor:
        return self.encode(text)


class DenseEmbeddingIndex:
    """Vector index for dense semantic retrieval with cosine similarity."""

    def __init__(
        self,
        dim: int = 128,
        embedder: Optional[Callable[[str], Union[torch.Tensor, np.ndarray, Sequence[float]]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.dim = dim
        self.device = device or get_device_strategy().device
        self.embedder = embedder or SimpleHashEmbedder(dim=dim)

        self.documents: Dict[str, RetrievalDocument] = {}
        self.doc_ids: List[str] = []
        self.embedding_matrix: Optional[torch.Tensor] = None

    def _to_unit_tensor(self, vector: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> torch.Tensor:
        """Convert input vector to a normalized 1D float32 PyTorch tensor."""
        if isinstance(vector, torch.Tensor):
            t = vector.detach().to(dtype=torch.float32, device="cpu").squeeze()
        elif isinstance(vector, np.ndarray):
            t = torch.from_numpy(vector.astype(np.float32)).squeeze()
        else:
            t = torch.tensor(vector, dtype=torch.float32)

        if t.ndim != 1 or t.shape[0] != self.dim:
            raise ValueError(f"Expected embedding vector of dimension ({self.dim},), got shape {tuple(t.shape)}")

        norm = torch.norm(t, p=2)
        if norm > 1e-12:
            t = t / norm
        return t

    def add_document(
        self,
        document: RetrievalDocument,
        embedding: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
    ) -> None:
        """Add a document and its embedding vector to the index."""
        if document.doc_id in self.documents:
            self.remove_document(document.doc_id)

        if embedding is None:
            embedding = self.embedder(document.text)

        vec = self._to_unit_tensor(embedding).unsqueeze(0)  # (1, dim)

        self.documents[document.doc_id] = document
        self.doc_ids.append(document.doc_id)

        if self.embedding_matrix is None:
            self.embedding_matrix = vec
        else:
            self.embedding_matrix = torch.cat([self.embedding_matrix, vec], dim=0)

    def add_documents(
        self,
        documents: List[RetrievalDocument],
        embeddings: Optional[Sequence[Union[torch.Tensor, np.ndarray, Sequence[float]]]] = None,
    ) -> None:
        """Batch add documents to the dense index."""
        for i, doc in enumerate(documents):
            emb = embeddings[i] if embeddings is not None else None
            self.add_document(doc, embedding=emb)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index by ID."""
        if doc_id not in self.documents:
            return

        idx = self.doc_ids.index(doc_id)
        self.doc_ids.pop(idx)
        del self.documents[doc_id]

        if self.embedding_matrix is not None:
            if len(self.doc_ids) == 0:
                self.embedding_matrix = None
            else:
                self.embedding_matrix = torch.cat(
                    [self.embedding_matrix[:idx], self.embedding_matrix[idx + 1 :]], dim=0
                )

    def search(
        self,
        query: Union[str, torch.Tensor, np.ndarray, Sequence[float]],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Perform dense cosine similarity search."""
        if len(self.doc_ids) == 0 or self.embedding_matrix is None:
            return []

        if isinstance(query, str):
            q_vec = self._to_unit_tensor(self.embedder(query))
        else:
            q_vec = self._to_unit_tensor(query)

        # Cosine similarity via dot product (both matrix and query are L2 normalized)
        scores = torch.matmul(self.embedding_matrix, q_vec)  # (N,)

        k = min(top_k, len(self.doc_ids))
        top_scores, top_indices = torch.topk(scores, k=k, largest=True)

        results = []
        for rank, (score_val, idx_val) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            doc_id = self.doc_ids[idx_val]
            doc = self.documents[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score=float(score_val),
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    url=doc.url,
                    rank=rank,
                    sub_scores={"dense": float(score_val)},
                )
            )

        return results

    def clear(self) -> None:
        """Clear the entire index."""
        self.documents.clear()
        self.doc_ids.clear()
        self.embedding_matrix = None

    def __len__(self) -> int:
        return len(self.doc_ids)

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from retrieval.bm25 import BM25Index
from retrieval.dense import DenseEmbeddingIndex
from retrieval.schemas import RetrievalDocument, SearchResult


class HybridRetriever:
    """Hybrid Retrieval Engine combining Okapi BM25 and Dense Cosine Search.

    Supports:
      1. Reciprocal Rank Fusion (RRF):
         RRF(d) = sum_{m in {bm25, dense}} w_m / (k_rrf + rank_m(d))
      2. Convex Score Blending:
         Score_blend(d) = alpha * S_norm_bm25(d) + (1 - alpha) * S_norm_dense(d)
    """

    def __init__(
        self,
        bm25_index: Optional[BM25Index] = None,
        dense_index: Optional[DenseEmbeddingIndex] = None,
        k_rrf: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        self.bm25 = bm25_index if bm25_index is not None else BM25Index()
        self.dense = dense_index if dense_index is not None else DenseEmbeddingIndex()
        self.k_rrf = int(k_rrf)
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)

        self.documents: Dict[str, RetrievalDocument] = {}

    def add_document(
        self,
        document: RetrievalDocument,
        embedding: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
    ) -> None:
        """Index a document into both BM25 and Dense indices."""
        self.documents[document.doc_id] = document
        self.bm25.add_document(document)
        self.dense.add_document(document, embedding=embedding)

    def add_documents(
        self,
        documents: List[RetrievalDocument],
        embeddings: Optional[Sequence[Union[torch.Tensor, np.ndarray, Sequence[float]]]] = None,
    ) -> None:
        """Batch add documents into both indices."""
        for i, doc in enumerate(documents):
            emb = embeddings[i] if embeddings is not None else None
            self.add_document(doc, embedding=emb)

    def remove_document(self, doc_id: str) -> None:
        """Remove document from both indices."""
        if doc_id in self.documents:
            del self.documents[doc_id]
        self.bm25.remove_document(doc_id)
        self.dense.remove_document(doc_id)

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        top_k: int = 5,
        fusion_mode: str = "rrf",
        k_rrf: Optional[int] = None,
        alpha: float = 0.5,
        candidate_pool_multiplier: int = 4,
    ) -> List[SearchResult]:
        """Perform hybrid retrieval blending lexical BM25 and dense semantic search.

        Args:
            query: Natural language query string.
            query_embedding: Optional precomputed vector for the query.
            top_k: Number of final ranked documents to return.
            fusion_mode: 'rrf' (Reciprocal Rank Fusion) or 'blend' (Convex Score Blending).
            k_rrf: Smoothing constant for RRF (defaults to self.k_rrf = 60).
            alpha: Weight for BM25 when fusion_mode='blend' (range [0, 1]).
            candidate_pool_multiplier: Multiplier to expand candidate search pool from each index.

        Returns:
            List of fused SearchResult objects ranked descending.
        """
        if len(self.documents) == 0:
            return []

        pool_size = max(top_k * candidate_pool_multiplier, 20)
        bm25_results = self.bm25.search(query, top_k=pool_size)
        dense_results = self.dense.search(query_embedding if query_embedding is not None else query, top_k=pool_size)

        if fusion_mode == "rrf":
            return self._fuse_rrf(
                bm25_results=bm25_results,
                dense_results=dense_results,
                top_k=top_k,
                k_rrf=k_rrf or self.k_rrf,
            )
        elif fusion_mode == "blend":
            return self._fuse_convex(
                bm25_results=bm25_results,
                dense_results=dense_results,
                top_k=top_k,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unknown fusion_mode '{fusion_mode}'. Must be 'rrf' or 'blend'.")

    def _fuse_rrf(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        top_k: int,
        k_rrf: int,
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion: RRF(d) = sum_m w_m / (k + rank_m(d))."""
        rrf_scores: Dict[str, float] = defaultdict(float)
        meta_tracker: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for res in bm25_results:
            rank = res.rank
            rrf_scores[res.doc_id] += self.bm25_weight / (k_rrf + rank)
            meta_tracker[res.doc_id]["bm25_score"] = res.score
            meta_tracker[res.doc_id]["bm25_rank"] = rank

        for res in dense_results:
            rank = res.rank
            rrf_scores[res.doc_id] += self.dense_weight / (k_rrf + rank)
            meta_tracker[res.doc_id]["dense_score"] = res.score
            meta_tracker[res.doc_id]["dense_rank"] = rank

        sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        final_results = []
        for rank, (doc_id, fused_score) in enumerate(sorted_items, start=1):
            doc = self.documents[doc_id]
            final_results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score=float(fused_score),
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    url=doc.url,
                    rank=rank,
                    sub_scores=dict(meta_tracker[doc_id]),
                )
            )

        return final_results

    def _fuse_convex(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        top_k: int,
        alpha: float,
    ) -> List[SearchResult]:
        """Convex Score Blending with min-max normalization."""
        # Collect all candidate doc_ids
        candidate_ids = set([r.doc_id for r in bm25_results] + [r.doc_id for r in dense_results])
        if not candidate_ids:
            return []

        # Map scores
        bm25_map = {r.doc_id: r.score for r in bm25_results}
        dense_map = {r.doc_id: r.score for r in dense_results}

        # Normalize BM25
        if bm25_map:
            min_b = min(bm25_map.values())
            max_b = max(bm25_map.values())
            b_range = max_b - min_b
            norm_bm25 = {
                d: ((s - min_b) / b_range if b_range > 1e-12 else 1.0) for d, s in bm25_map.items()
            }
        else:
            norm_bm25 = {}

        # Normalize Dense (cosine sim is typically in [-1, 1])
        if dense_map:
            min_d = min(dense_map.values())
            max_d = max(dense_map.values())
            d_range = max_d - min_d
            norm_dense = {
                d: ((s - min_d) / d_range if d_range > 1e-12 else 1.0) for d, s in dense_map.items()
            }
        else:
            norm_dense = {}

        blended_scores: Dict[str, float] = {}
        sub_scores_map: Dict[str, Dict[str, float]] = defaultdict(dict)

        for d in candidate_ids:
            b_score = norm_bm25.get(d, 0.0)
            d_score = norm_dense.get(d, 0.0)
            score = alpha * b_score + (1.0 - alpha) * d_score
            blended_scores[d] = score
            sub_scores_map[d] = {
                "bm25_raw": bm25_map.get(d, 0.0),
                "bm25_norm": b_score,
                "dense_raw": dense_map.get(d, 0.0),
                "dense_norm": d_score,
            }

        sorted_items = sorted(blended_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        final_results = []
        for rank, (doc_id, score) in enumerate(sorted_items, start=1):
            doc = self.documents[doc_id]
            final_results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score=float(score),
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    url=doc.url,
                    rank=rank,
                    sub_scores=dict(sub_scores_map[doc_id]),
                )
            )

        return final_results

    def clear(self) -> None:
        """Clear all indexed documents."""
        self.documents.clear()
        self.bm25.clear()
        self.dense.clear()

    def __len__(self) -> int:
        return len(self.documents)


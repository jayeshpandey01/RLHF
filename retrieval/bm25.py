import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

from retrieval.schemas import RetrievalDocument, SearchResult


class BM25Index:
    """Okapi BM25 Lexical Search Index with Robertson smoothed IDF.

    Implements:
        BM25(q, d) = sum_{t in q} IDF(t) * [f(t, d) * (k_1 + 1)] / [f(t, d) + k_1 * (1 - b + b * (|d| / avgdl))]
        where IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords: Optional[Set[str]] = None):
        self.k1 = float(k1)
        self.b = float(b)
        self.stopwords = set(stopwords) if stopwords else set()

        self.documents: Dict[str, RetrievalDocument] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_term_freqs: Dict[str, Counter] = {}
        # Inverted index: term -> {doc_id: frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.total_tokens: int = 0
        self.avgdl: float = 0.0

    def tokenize(self, text: str) -> List[str]:
        """Extract alphanumeric lowercase tokens excluding stopwords."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def add_document(self, document: RetrievalDocument) -> None:
        """Add a single document to the BM25 index."""
        if document.doc_id in self.documents:
            self.remove_document(document.doc_id)

        tokens = self.tokenize(document.text)
        doc_len = len(tokens)
        term_freqs = Counter(tokens)

        self.documents[document.doc_id] = document
        self.doc_lengths[document.doc_id] = doc_len
        self.doc_term_freqs[document.doc_id] = term_freqs

        for term, freq in term_freqs.items():
            self.inverted_index[term][document.doc_id] = freq

        self.total_tokens += doc_len
        self._recompute_avgdl()

    def add_documents(self, documents: List[RetrievalDocument]) -> None:
        """Batch add multiple documents to the BM25 index."""
        for doc in documents:
            self.add_document(doc)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index by its ID."""
        if doc_id not in self.documents:
            return

        doc_len = self.doc_lengths.pop(doc_id, 0)
        self.total_tokens -= doc_len

        term_freqs = self.doc_term_freqs.pop(doc_id, Counter())
        for term in term_freqs:
            if term in self.inverted_index and doc_id in self.inverted_index[term]:
                del self.inverted_index[term][doc_id]
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

        del self.documents[doc_id]
        self._recompute_avgdl()

    def _recompute_avgdl(self) -> None:
        num_docs = len(self.documents)
        self.avgdl = (self.total_tokens / num_docs) if num_docs > 0 else 0.0

    def get_idf(self, term: str) -> float:
        """Compute Robertson smoothed Inverse Document Frequency (IDF).

        Formula:
            IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        Guaranteed non-negative for all 1 <= n(t) <= N.
        """
        N = len(self.documents)
        if N == 0:
            return 0.0

        n_t = len(self.inverted_index.get(term, {}))
        if n_t == 0:
            return 0.0

        return math.log(((N - n_t + 0.5) / (n_t + 0.5)) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform BM25 search ranking across indexed documents.

        Args:
            query: User search text query.
            top_k: Maximum number of top documents to return.

        Returns:
            List of SearchResult objects sorted descending by score.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens or len(self.documents) == 0:
            return []

        # Find candidate documents containing at least one query token
        scores: Dict[str, float] = defaultdict(float)
        query_token_counts = Counter(query_tokens)

        for term, q_weight in query_token_counts.items():
            idf = self.get_idf(term)
            if idf <= 0.0:
                continue

            posting = self.inverted_index.get(term, {})
            for doc_id, tf in posting.items():
                d_len = self.doc_lengths.get(doc_id, 0)
                len_norm = (1.0 - self.b) + self.b * (d_len / self.avgdl if self.avgdl > 0 else 1.0)
                term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm))
                scores[doc_id] += q_weight * term_score

        if not scores:
            return []

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            doc = self.documents[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score=float(score),
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    url=doc.url,
                    rank=rank,
                    sub_scores={"bm25": float(score)},
                )
            )

        return results

    def clear(self) -> None:
        """Clear the entire index."""
        self.documents.clear()
        self.doc_lengths.clear()
        self.doc_term_freqs.clear()
        self.inverted_index.clear()
        self.total_tokens = 0
        self.avgdl = 0.0

    def __len__(self) -> int:
        return len(self.documents)


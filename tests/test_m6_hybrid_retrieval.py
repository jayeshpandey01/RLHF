import math
import unittest

import numpy as np
import torch

from retrieval import (
    BM25Index,
    DenseEmbeddingIndex,
    HybridRetriever,
    RetrievalDocument,
    SearchResult,
    SimpleHashEmbedder,
)


class TestM6HybridRetrieval(unittest.TestCase):
    """Milestone 6 - Phase 6.1: Hybrid Retrieval Engine Test Suite."""

    def setUp(self):
        self.sample_docs = [
            RetrievalDocument(
                doc_id="doc_quantum",
                text="Quantum computing utilizes quantum superposition and quantum entanglement to perform complex state calculations.",
                url="https://arxiv.org/abs/quantum_overview",
                metadata={"domain": "physics", "author": "Feynman"},
            ),
            RetrievalDocument(
                doc_id="doc_rlhf",
                text="Direct Preference Optimization and Reinforcement Learning from Human Feedback align autoregressive language models with human preferences.",
                url="https://arxiv.org/abs/rlhf_alignment",
                metadata={"domain": "ai", "author": "Rafailov"},
            ),
            RetrievalDocument(
                doc_id="doc_biology",
                text="CRISPR Cas9 gene editing allows molecular biologists to modify DNA sequences and alter cellular genetic functions.",
                url="https://nature.com/articles/crispr_cas9",
                metadata={"domain": "biology", "author": "Doudna"},
            ),
        ]

    def test_bm25_smoothed_idf_non_negativity(self):
        """Verify Robertson smoothed IDF is strictly positive even when term appears in all N documents."""
        bm25 = BM25Index()
        # Add 3 documents all containing the word 'information'
        docs = [
            RetrievalDocument(doc_id="1", text="Universal information technology systems."),
            RetrievalDocument(doc_id="2", text="Information entropy and mutual information theory."),
            RetrievalDocument(doc_id="3", text="Information retrieval algorithms."),
        ]
        bm25.add_documents(docs)

        # n(t) = 3, N = 3
        # Robertson IDF: ln((3 - 3 + 0.5) / (3 + 0.5) + 1) = ln(0.5 / 3.5 + 1) = ln(1.142857) > 0
        idf = bm25.get_idf("information")
        self.assertGreater(idf, 0.0)
        expected_idf = math.log((0.5 / 3.5) + 1.0)
        self.assertAlmostEqual(idf, expected_idf, places=5)

    def test_bm25_term_frequency_saturation_and_length_penalty(self):
        """Verify term frequency saturation and document length normalization."""
        bm25 = BM25Index(k1=1.5, b=0.75)
        # Doc 1: short, contains 'neural' 1 time
        # Doc 2: short, contains 'neural' 5 times
        # Doc 3: extremely long padding, contains 'neural' 1 time
        docs = [
            RetrievalDocument(doc_id="short_1", text="neural network architecture"),
            RetrievalDocument(doc_id="short_5", text="neural neural neural neural neural network architecture"),
            RetrievalDocument(
                doc_id="long_1",
                text="neural network architecture " + " ".join(["irrelevant"] * 100),
            ),
        ]
        bm25.add_documents(docs)

        results = bm25.search("neural", top_k=3)
        self.assertEqual(len(results), 3)

        doc_ranks = {r.doc_id: r.rank for r in results}
        scores = {r.doc_id: r.score for r in results}

        # short_5 should have higher score than short_1
        self.assertGreater(scores["short_5"], scores["short_1"])
        # short_1 should have higher score than long_1 because of length normalization penalty
        self.assertGreater(scores["short_1"], scores["long_1"])
        self.assertEqual(doc_ranks["short_5"], 1)
        self.assertEqual(doc_ranks["short_1"], 2)
        self.assertEqual(doc_ranks["long_1"], 3)

    def test_dense_cosine_similarity_exactness(self):
        """Verify DenseEmbeddingIndex computes exact cosine similarities and top-K ranking."""
        dim = 4
        dense_index = DenseEmbeddingIndex(dim=dim)

        # Create known synthetic vectors
        # Doc 1: [1, 0, 0, 0]
        # Doc 2: [0.7071, 0.7071, 0, 0]
        # Doc 3: [0, 1, 0, 0]
        # Doc 4: [-1, 0, 0, 0]
        docs = [
            (RetrievalDocument(doc_id="d1", text="d1"), [1.0, 0.0, 0.0, 0.0]),
            (RetrievalDocument(doc_id="d2", text="d2"), [1.0, 1.0, 0.0, 0.0]),
            (RetrievalDocument(doc_id="d3", text="d3"), [0.0, 1.0, 0.0, 0.0]),
            (RetrievalDocument(doc_id="d4", text="d4"), [-1.0, 0.0, 0.0, 0.0]),
        ]
        for doc, emb in docs:
            dense_index.add_document(doc, embedding=emb)

        # Query vector: [1, 0, 0, 0]
        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = dense_index.search(query_vec, top_k=4)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].doc_id, "d1")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

        self.assertEqual(results[1].doc_id, "d2")
        self.assertAlmostEqual(results[1].score, 1.0 / math.sqrt(2), places=4)

        self.assertEqual(results[2].doc_id, "d3")
        self.assertAlmostEqual(results[2].score, 0.0, places=4)

        self.assertEqual(results[3].doc_id, "d4")
        self.assertAlmostEqual(results[3].score, -1.0, places=4)

    def test_dense_simple_hash_embedder_semantic_match(self):
        """Verify SimpleHashEmbedder produces valid normalized vectors and semantic relevance."""
        embedder = SimpleHashEmbedder(dim=128)
        v1 = embedder("reinforcement learning with human feedback")
        v2 = embedder("reinforcement learning alignment")
        v3 = embedder("photosynthesis in plant chloroplasts")

        # Vectors must be unit length
        self.assertAlmostEqual(float(torch.norm(v1, p=2)), 1.0, places=5)
        self.assertAlmostEqual(float(torch.norm(v2, p=2)), 1.0, places=5)
        self.assertAlmostEqual(float(torch.norm(v3, p=2)), 1.0, places=5)

        # Cosine similarity between related phrases must be higher than unrelated
        sim_related = float(torch.dot(v1, v2))
        sim_unrelated = float(torch.dot(v1, v3))
        self.assertGreater(sim_related, sim_unrelated)

    def test_hybrid_rrf_scale_invariance_and_rank_blending(self):
        """Verify Reciprocal Rank Fusion (RRF) preserves rank scale invariance."""
        retriever = HybridRetriever(k_rrf=60)
        retriever.add_documents(self.sample_docs)

        # Retrieve using natural language query
        results = retriever.retrieve(
            query="quantum superposition entanglement",
            top_k=3,
            fusion_mode="rrf",
        )

        self.assertEqual(len(results), 3)
        # Quantum doc must rank #1
        self.assertEqual(results[0].doc_id, "doc_quantum")
        self.assertEqual(results[0].rank, 1)

        # Check that RRF score formula was strictly followed
        # RRF = sum w / (60 + r)
        expected_score = (1.0 / (60 + 1)) + (1.0 / (60 + results[0].sub_scores.get("dense_rank", 1)))
        self.assertAlmostEqual(results[0].score, expected_score, places=4)

    def test_hybrid_convex_blending_alpha_extremes(self):
        """Verify convex score blending matches BM25 when alpha=1.0 and Dense when alpha=0.0."""
        retriever = HybridRetriever()
        retriever.add_documents(self.sample_docs)

        # Alpha = 1.0 (pure normalized BM25)
        results_bm25_pure = retriever.retrieve(
            query="human preferences alignment",
            top_k=2,
            fusion_mode="blend",
            alpha=1.0,
        )
        self.assertEqual(results_bm25_pure[0].doc_id, "doc_rlhf")
        self.assertAlmostEqual(results_bm25_pure[0].score, 1.0, places=4)

        # Alpha = 0.0 (pure normalized Dense)
        results_dense_pure = retriever.retrieve(
            query="quantum state calculation",
            top_k=2,
            fusion_mode="blend",
            alpha=0.0,
        )
        self.assertEqual(results_dense_pure[0].doc_id, "doc_quantum")
        self.assertAlmostEqual(results_dense_pure[0].score, 1.0, places=4)

    def test_hybrid_retrieval_provenance_and_metadata(self):
        """Verify top-K search results preserve URL provenance, metadata, and sub-score metrics."""
        retriever = HybridRetriever()
        retriever.add_documents(self.sample_docs)

        results = retriever.retrieve(query="CRISPR Cas9 gene editing DNA", top_k=1)
        self.assertEqual(len(results), 1)

        top_res = results[0]
        self.assertEqual(top_res.doc_id, "doc_biology")
        self.assertEqual(top_res.url, "https://nature.com/articles/crispr_cas9")
        self.assertEqual(top_res.metadata["domain"], "biology")
        self.assertEqual(top_res.metadata["author"], "Doudna")
        self.assertIn("bm25_rank", top_res.sub_scores)
        self.assertIn("dense_rank", top_res.sub_scores)

    def test_index_lifecycle_add_remove_clear(self):
        """Verify document lifecycle management: add, remove, and clear."""
        retriever = HybridRetriever()
        self.assertEqual(len(retriever), 0)

        retriever.add_documents(self.sample_docs)
        self.assertEqual(len(retriever), 3)

        retriever.remove_document("doc_rlhf")
        self.assertEqual(len(retriever), 2)
        res = retriever.retrieve("Direct Preference Optimization", top_k=2)
        self.assertNotIn("doc_rlhf", [r.doc_id for r in res])

        retriever.clear()
        self.assertEqual(len(retriever), 0)
        self.assertEqual(len(retriever.bm25), 0)
        self.assertEqual(len(retriever.dense), 0)


if __name__ == "__main__":
    unittest.main()


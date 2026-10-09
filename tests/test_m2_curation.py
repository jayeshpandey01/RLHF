"""Unit and integration test suite for Milestone 2 / Phase 2.2: Data Cleaning & Deduplication."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import BenchmarkCaseSchema, ScrapedDocumentSchema
from curation.contamination import BenchmarkContaminationGuard
from curation.filters import C4QualityFilter
from curation.minhash_lsh import MinHashLSH
from curation.pipeline import CurationPipeline
from ingestion.router import QueryIntent


class TestM2Curation(unittest.TestCase):
    """Rigorous test suite for C4 quality filters, MinHash LSH, and benchmark decontamination."""

    def setUp(self):
        self.clean_sample = (
            "Reinforcement learning from human feedback has emerged as a cornerstone for aligning "
            "large language models with human intent and societal values. "
            "In this paradigm, an initial policy is trained on large corpora of text using next-token "
            "prediction objectives to capture broad world knowledge and syntactic structures. "
            "Subsequently, pairwise preference comparisons are collected to fit a reward model. "
            "Direct Preference Optimization simplifies this optimization by implicitly parameterizing "
            "the reward function directly through the language model policy. "
            "By doing so, it avoids the complexities and computational overhead of training an actor-critic "
            "reinforcement learning pipeline, ensuring stable and reliable convergence across distributed nodes. "
            "Our experiments demonstrate significant improvements in downstream benchmark evaluations."
        )

    def test_c4_quality_filter_valid_text(self):
        """Verify that high-quality, grammatical text passes all C4 heuristic gates."""
        filt = C4QualityFilter(min_words=30)
        is_valid, reason, metrics = filt.evaluate(self.clean_sample)
        self.assertTrue(is_valid, f"Expected pass, got: {reason}")
        self.assertEqual(reason, "PASSED")
        self.assertGreater(metrics["word_count"], 30)
        self.assertGreater(metrics["stopword_density"], 0.15)
        self.assertLess(metrics["line_repetition_rate"], 0.20)

    def test_c4_quality_filter_word_count_bounds(self):
        """Verify that short fragments and oversized dumps are strictly rejected."""
        filt = C4QualityFilter(min_words=50, max_words=200)

        # Underflow (< 50 words)
        short_text = "This is a very short text snippet. It only has a few words."
        is_valid, reason, _ = filt.evaluate(short_text)
        self.assertFalse(is_valid)
        self.assertIn("WORD_COUNT_TOO_LOW", reason)

        # Overflow (> 200 words)
        long_text = "word " * 250 + "."
        is_valid, reason, _ = filt.evaluate(long_text)
        self.assertFalse(is_valid)
        self.assertIn("WORD_COUNT_TOO_HIGH", reason)

    def test_c4_quality_filter_line_repetition(self):
        """Verify that repetitive boilerplate, navigation bars, and copy-paste loops are rejected."""
        filt = C4QualityFilter(min_words=20, max_line_repetition_rate=0.20)

        repetitive_lines = "\n".join([
            "This is a legitimate article describing reinforcement learning.",
            "Click here to subscribe to our newsletter for daily updates.",
            "Click here to subscribe to our newsletter for daily updates.",
            "Click here to subscribe to our newsletter for daily updates.",
            "Click here to subscribe to our newsletter for daily updates.",
            "Click here to subscribe to our newsletter for daily updates.",
            "This is the end of the informative article about machine learning.",
        ])

        is_valid, reason, _ = filt.evaluate(repetitive_lines)
        self.assertFalse(is_valid)
        self.assertIn("EXCESSIVE_LINE_REPETITION", reason)

    def test_c4_quality_filter_stopword_density(self):
        """Verify that SEO keyword stuffing and word salad without natural grammar are rejected."""
        filt = C4QualityFilter(min_words=20, min_stopword_density=0.10)

        # Keyword stuffing with 0% stopwords
        seo_stuffing = (
            "buy cheap running shoes discount best athletic footwear sale online "
            "marathon sprinting sneakers affordable price top brand catalog review "
            "store delivery warranty performance footwear discount."
        )

        is_valid, reason, _ = filt.evaluate(seo_stuffing)
        self.assertFalse(is_valid)
        self.assertIn("LOW_STOPWORD_DENSITY", reason)

    def test_c4_quality_filter_character_repetition(self):
        """Verify that generator infinite loops and corrupted byte streams are rejected."""
        filt = C4QualityFilter(min_words=5, max_char_repetition_rate=0.15)

        # Corrupted looping string
        loop_text = (
            "The model output entered an unrecoverable failure loop: "
            "asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf "
            "asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf."
        )

        is_valid, reason, _ = filt.evaluate(loop_text)
        self.assertFalse(is_valid)
        self.assertIn("EXCESSIVE_CHAR_REPETITION", reason)

    def test_minhash_lsh_identical_and_near_duplicates(self):
        """Verify that MinHash LSH detects identical and near-duplicate documents (s >= 0.80)."""
        lsh = MinHashLSH(num_perm=128, num_bands=32, shingle_size=4, threshold=0.80)

        doc_a = self.clean_sample
        # Near-duplicate: changing 2 words in 100-word text (>85% Jaccard similarity)
        doc_b = self.clean_sample.replace("downstream benchmark evaluations", "empirical benchmark evaluations")
        # Completely different document
        doc_c = (
            "To make authentic Italian carbonara, combine fresh eggs with grated pecorino cheese, "
            "crispy guanciale, and freshly ground black pepper over al dente spaghetti pasta."
        )

        sig_a = lsh.compute_signature(doc_a)
        sig_b = lsh.compute_signature(doc_b)
        sig_c = lsh.compute_signature(doc_c)

        # 1. Self similarity must be 1.0
        self.assertAlmostEqual(lsh.jaccard_similarity(sig_a, sig_a), 1.0, places=4)

        # 2. Near-duplicate similarity must be high (>= 0.80)
        sim_ab = lsh.jaccard_similarity(sig_a, sig_b)
        self.assertGreaterEqual(sim_ab, 0.80)

        # 3. Dissimilar documents must have low similarity (< 0.20)
        sim_ac = lsh.jaccard_similarity(sig_a, sig_c)
        self.assertLess(sim_ac, 0.20)

        # Index doc_a
        lsh.index("doc_a", sig_a)
        self.assertEqual(lsh.size(), 1)

        # Query doc_b against index with threshold 0.70
        is_dup, max_sim, match_id = lsh.is_duplicate(sig_b, threshold=0.70)
        self.assertTrue(is_dup)
        self.assertEqual(match_id, "doc_a")

        # Query doc_c against index
        is_dup_c, max_sim_c, _ = lsh.is_duplicate(sig_c, threshold=0.70)
        self.assertFalse(is_dup_c)

    def test_benchmark_contamination_guard(self):
        """Verify that 13-gram benchmark contamination guard identifies test set leakage."""
        guard = BenchmarkContaminationGuard(ngram_size=13)

        golden_case = BenchmarkCaseSchema(
            case_id="case_001",
            question="What is the exact gradient scaling factor used in Pre-LayerNorm Transformers?",
            ground_truth_context=(
                "In Pre-LayerNorm Transformer architectures, residual branch scaling factor "
                "one over square root of two times L ensures unit activation variance at initialization."
            ),
            golden_keywords=["Pre-LayerNorm", "gradient scaling"],
        )
        guard.add_benchmark_case(golden_case)
        self.assertGreater(guard.size(), 0)

        # Contaminated document containing the exact 13-gram
        contaminated_doc = (
            "Recent discussions highlight that in Pre-LayerNorm Transformer architectures, "
            "residual branch scaling factor one over square root of two times L ensures unit activation "
            "variance at initialization across all deep transformer blocks."
        )

        is_contam, leaked = guard.is_contaminated(contaminated_doc)
        self.assertTrue(is_contam)
        self.assertIsNotNone(leaked)

        # Clean document
        clean_doc = (
            "Gradient descent optimizes loss surfaces by propagating backpropagation signals "
            "through computational graphs according to the multivariate chain rule."
        )
        is_contam_clean, _ = guard.is_contaminated(clean_doc)
        self.assertFalse(is_contam_clean)

    def test_curation_pipeline_e2e_batch(self):
        """Verify that the end-to-end CurationPipeline filters duplicates, spam, and contamination."""
        pipeline = CurationPipeline()

        # Set lenient word count for tests
        pipeline.quality_filter.min_words = 20

        # Register benchmark case
        pipeline.contamination_guard.add_benchmark_text(
            "the quick brown fox jumps over the lazy dog and runs across the wide open green meadow"
        )

        doc_valid = ScrapedDocumentSchema(
            url="https://example.com/valid",
            raw_html="<p>Valid content</p>",
            clean_markdown=(
                "Deep learning models require carefully curated datasets to prevent functional collapse. "
                "By systematically pruning duplicate documents and low-quality noisy sequences, "
                "we preserve the natural entropy of the language model during continual learning. "
                "Our experiments confirm that deduplicated data achieves superior convergence."
            ),
        )

        doc_exact_dup = ScrapedDocumentSchema(
            url="https://example.com/exact_dup",
            raw_html="<p>Valid content</p>",
            clean_markdown=doc_valid.clean_markdown,
        )

        doc_short_spam = ScrapedDocumentSchema(
            url="https://example.com/spam",
            raw_html="<p>Spam</p>",
            clean_markdown="Click now. Cheap deals.",
        )

        doc_contaminated = ScrapedDocumentSchema(
            url="https://example.com/leak",
            raw_html="<p>Leak</p>",
            clean_markdown=(
                "This document accidentally contains test set data where "
                "the quick brown fox jumps over the lazy dog and runs across the wide open green meadow "
                "in front of the audience during evaluation testing."
            ),
        )

        batch = [doc_valid, doc_exact_dup, doc_short_spam, doc_contaminated]
        accepted, stats = pipeline.process_batch(batch)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].url, "https://example.com/valid")
        self.assertTrue(accepted[0].is_valid)
        self.assertEqual(len(accepted[0].minhash_signature), 128)

        # Audit stats verification
        self.assertEqual(stats["total_input"], 4)
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["rejected_exact_duplicate"], 1)
        self.assertEqual(stats["rejected_quality"], 1)
        self.assertEqual(stats["rejected_contamination"], 1)

    def test_disjoint_set_union_find(self):
        """Verify DisjointSet path compression, rank union, and component grouping (Phase 2.3)."""
        from curation.cluster import DisjointSet

        dsu = DisjointSet()
        # Initial singletons
        dsu.find("doc_1")
        dsu.find("doc_2")
        dsu.find("doc_3")
        dsu.find("doc_4")

        # Union 1-2 and 2-3 (transitive component 1, 2, 3)
        self.assertTrue(dsu.union("doc_1", "doc_2"))
        self.assertTrue(dsu.union("doc_2", "doc_3"))
        self.assertFalse(dsu.union("doc_1", "doc_3"))  # Already connected

        comps = dsu.get_components()
        self.assertEqual(len(comps), 2)  # One cluster of {1, 2, 3}, one singleton {4}
        root_1 = dsu.find("doc_1")
        self.assertEqual(sorted(comps[root_1]), ["doc_1", "doc_2", "doc_3"])

    def test_cluster_deduplicator(self):
        """Verify connected components cluster resolution and canonical document election (Phase 2.3)."""
        from core.schemas import CleanedDocumentSchema
        from curation.cluster import ClusterDeduplicator

        # 3 near-duplicate versions of the same research article
        doc_short = CleanedDocumentSchema(
            url="https://news.com/brief",
            clean_text="Transformers use multi-head attention to model sequences in deep learning.",
            word_count=10,
            line_repetition_rate=0.05,
            sha256_hash="hash_short",
        )
        doc_comprehensive = CleanedDocumentSchema(
            url="https://arxiv.org/abs/transformer_paper",
            clean_text=(
                "Transformers rely on multi-head attention mechanisms to project representations into "
                "query, key, and value subspaces. By applying variance scaling factor one over square root "
                "of d_k, attention score variance is preserved, preventing softmax gradients from vanishing. "
                "Extensive empirical evaluations confirm superior convergence across translation benchmarks."
            ),
            word_count=45,
            line_repetition_rate=0.02,
            sha256_hash="hash_comp",
        )
        doc_syndicated = CleanedDocumentSchema(
            url="https://blog.com/repost",
            clean_text="Transformers rely on multi-head attention mechanisms to project representations.",
            word_count=10,
            line_repetition_rate=0.10,
            sha256_hash="hash_syn",
        )

        dedup = ClusterDeduplicator()
        # Candidate pairs matching Jaccard >= 0.80
        pairs = [
            (doc_short.url, doc_comprehensive.url),
            (doc_comprehensive.url, doc_syndicated.url),
        ]

        canonicals, metrics = dedup.deduplicate([doc_short, doc_comprehensive, doc_syndicated], pairs)

        self.assertEqual(len(canonicals), 1)
        self.assertEqual(canonicals[0].url, doc_comprehensive.url)  # Comprehensive document elected!
        self.assertEqual(metrics["pruned_duplicates"], 2)
        self.assertEqual(metrics["multi_doc_clusters"], 1)

    def test_experience_replay_buffer_and_memmap(self):
        """Verify 70:30 convex combination sampling and zero-RAM memmap binary export (Phase 2.4)."""
        import tempfile
        from curation.replay_buffer import ExperienceReplayBuffer
        from training.m1_bigram.tokenizer import CharTokenizer
        from training.m1_transformer.dataset import MemoryMappedTokenDataset

        buffer = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)

        # Populate anchor pool (70%)
        anchor_data = [
            f"Anchor baseline sample {i}: Language models learn grammar and syntax from human conversation."
            for i in range(20)
        ]
        buffer.add_anchor_data(anchor_data)

        # Populate web pool (30%)
        web_data = [
            f"Web scraped domain sample {i}: Reinforcement learning from human feedback aligns policies."
            for i in range(20)
        ]
        buffer.add_web_data(web_data)

        # Test batch sampling mixture ratio
        batch = buffer.sample_batch(batch_size=100)
        self.assertEqual(len(batch), 100)

        anchor_count = sum(1 for item in batch if "Anchor baseline sample" in item)
        self.assertEqual(anchor_count, 70)  # Exactly 70% anchor samples!

        # Test binary export & memmap streaming
        with tempfile.TemporaryDirectory() as tmpdir:
            train_bin, val_bin = buffer.export_sharded_binaries(
                output_dir=tmpdir, target_tokens=1_000, train_val_split=0.90
            )
            self.assertTrue(train_bin.exists())
            self.assertTrue(val_bin.exists())

            # Load into MemoryMappedTokenDataset for O(1) streaming
            ds = MemoryMappedTokenDataset(train_bin)
            self.assertGreater(len(ds), 500)

            # Sample random batch
            x, y = ds.get_batch(batch_size=4, block_size=16)
            self.assertEqual(x.shape, (4, 16))
            self.assertEqual(y.shape, (4, 16))

    def test_continual_data_orchestrator_e2e_to_training(self):
        """Verify full end-to-end flow from Ingestion (2.1) -> Curation (2.2) -> Clustering (2.3) -> Replay (2.4) -> MiniGPT Training (1.4)."""
        import asyncio
        import tempfile
        from unittest.mock import AsyncMock
        from curation.orchestrator import ContinualDataOrchestrator
        from ingestion.provenance import ProvenanceLogger
        from training.m1_bigram.tokenizer import CharTokenizer
        from training.m1_transformer.dataset import MemoryMappedTokenDataset
        from training.m1_transformer.train import train

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "test_manifest.jsonl"
            prov_logger = ProvenanceLogger(manifest_path=str(manifest_file), auto_load=False)
            
            orchestrator = ContinualDataOrchestrator()
            orchestrator.router.provenance = prov_logger
            orchestrator.curation.quality_filter.min_words = 15

            # Mock 2 scraped web documents from Phase 2.1
            scraped_1 = ScrapedDocumentSchema(
                url="https://arxiv.org/abs/transformer_ai",
                raw_html="<p>Full article</p>",
                clean_markdown=(
                    "Transformers utilize multi-head attention to map representations into query, key, "
                    "and value spaces with scaled dot-product attention preventing gradient vanishing. "
                    "Empirical evaluations prove high perplexity gains across benchmarks."
                ),
            )
            scraped_2 = ScrapedDocumentSchema(
                url="https://arxiv.org/abs/transformer_ai_copy",
                raw_html="<p>Duplicate</p>",
                clean_markdown=scraped_1.clean_markdown,  # Exact/near duplicate
            )

            orchestrator.router.route_and_fetch = AsyncMock(
                return_value=(QueryIntent.ACADEMIC_RESEARCH, [scraped_1, scraped_2])
            )

            # Run E2E Ingestion -> Curation -> Cluster Dedup -> Replay Buffer -> Binary Export
            train_bin, val_bin, metrics = asyncio.run(
                orchestrator.ingest_curate_and_export(
                    query="Transformer attention mechanism",
                    max_scraped=2,
                    output_dir=tmpdir,
                    target_tokens=600,
                )
            )

            self.assertTrue(train_bin.exists())
            self.assertTrue(val_bin.exists())
            self.assertEqual(metrics["scraped_count"], 2)
            # The duplicate was pruned during curation or cluster deduplication
            self.assertGreaterEqual(metrics["canonical_count"], 1)

            # Load exported binaries with MemoryMappedTokenDataset (1.4)
            train_ds = MemoryMappedTokenDataset(train_bin)
            val_ds = MemoryMappedTokenDataset(val_bin)

            # Use aligned tokenizer directly from orchestrator export
            tokenizer = metrics["tokenizer"]

            # Execute 5 steps of MiniGPT training on the newly ingested & curated replay data
            model = train(
                max_steps=5,
                batch_size=2,
                block_size=16,
                eval_interval=5,
                save_checkpoint=False,
                custom_data=(train_ds, val_ds, tokenizer),
            )
            self.assertIsNotNone(model)
            self.assertGreater(model.get_num_params(), 0)


if __name__ == "__main__":
    unittest.main()

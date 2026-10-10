import math
from pathlib import Path
import tempfile
import unittest

from curation import ExperienceReplayBuffer, ModelCollapseGuard, ReplaySample
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.dataset import MemoryMappedTokenDataset


class TestM6ReplayBuffer(unittest.TestCase):
    """Milestone 6 - Phase 6.2: Experience Replay Buffer & Model Collapse Guard Test Suite."""

    def setUp(self):
        self.guard = ModelCollapseGuard(
            min_shannon_entropy=2.5,
            min_vocab_diversity=0.20,
            max_ngram_repetition_rate=0.25,
            min_tokens=15,
        )

        self.anchor_texts = [
            f"Anchor baseline sample {i}: Human conversations adhere to linguistic grammar and syntax rules."
            for i in range(25)
        ]
        self.web_texts = [
            f"Web domain scraped passage {i}: Reinforcement learning with verifiable reward signals guides policy optimization."
            for i in range(25)
        ]

    def test_shannon_entropy_calculation_exactness(self):
        """Verify Shannon token entropy matches analytical closed form for uniform distribution."""
        # For a uniform distribution of K distinct tokens, H(T) = log2(K)
        # K = 8 -> H(T) = 3.0 bits
        tokens_uniform = ["a", "b", "c", "d", "e", "f", "g", "h"] * 5
        entropy = self.guard.calculate_entropy(tokens_uniform)
        self.assertAlmostEqual(entropy, 3.0, places=5)

        # For a single repeated token, H(T) = 0.0
        tokens_constant = ["repeat"] * 20
        entropy_const = self.guard.calculate_entropy(tokens_constant)
        self.assertAlmostEqual(entropy_const, 0.0, places=5)

    def test_vocabulary_diversity_and_ngram_repetition_exactness(self):
        """Verify vocabulary diversity ratio and n-gram repetition metrics."""
        # 4 unique tokens out of 20 total -> diversity = 4 / 20 = 0.20
        tokens = ["word1", "word2", "word3", "word4"] * 5
        div = self.guard.calculate_vocab_diversity(tokens)
        self.assertAlmostEqual(div, 0.20, places=5)

        # 4-gram repetition: ("word1", "word2", "word3", "word4") repeats 5 times in 17 total 4-grams
        rep = self.guard.calculate_max_ngram_repetition(tokens, n=4)
        expected_rep = 5.0 / 17.0
        self.assertAlmostEqual(rep, expected_rep, places=4)

    def test_model_collapse_guard_rejects_synthetic_degenerate_loops(self):
        """Verify ModelCollapseGuard detects repetitive degenerate hallucination loops."""
        # 1. Low vocabulary diversity loop
        low_div_text = (
            "This model is definitely the best model ever. "
            "This model is definitely the best model ever. "
            "This model is definitely the best model ever. "
            "This model is definitely the best model ever. "
            "This model is definitely the best model ever."
        )
        is_safe_1, reason_1, metrics_1 = self.guard.evaluate_text(low_div_text)
        self.assertFalse(is_safe_1)
        self.assertTrue("COLLAPSE_LOW_VOCAB_DIVERSITY" in reason_1)
        self.assertLess(metrics_1["vocab_diversity"], 0.20)

        # 2. High n-gram repetition loop (short cyclic sequence)
        ngram_loop_text = "system failure error occurred " * 10
        is_safe_2, reason_2, metrics_2 = self.guard.evaluate_text(ngram_loop_text)
        self.assertFalse(is_safe_2)
        self.assertTrue("COLLAPSE" in reason_2)
        self.assertGreater(metrics_2["ngram_repetition_rate"], 0.25)

    def test_model_collapse_guard_accepts_natural_corpus(self):
        """Verify ModelCollapseGuard passes rich natural text with high entropy and diversity."""
        natural_text = (
            "Direct Preference Optimization formulates the preference alignment problem "
            "as a closed-form implicit reward function over policy log-probabilities, "
            "completely bypassing the instability of training an explicit token-level critic network."
        )
        is_safe, reason, metrics = self.guard.evaluate_text(natural_text)
        self.assertTrue(is_safe)
        self.assertEqual(reason, "PASSED_QUALITY_GATE")
        self.assertGreaterEqual(metrics["shannon_entropy"], 2.5)
        self.assertGreaterEqual(metrics["vocab_diversity"], 0.20)

    def test_experience_replay_buffer_quarantines_degraded_web_samples(self):
        """Verify ExperienceReplayBuffer filters and quarantines degenerate web samples."""
        buffer = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)
        buffer.add_anchor_data(self.anchor_texts)

        # 2 valid web samples and 1 degenerate repetitive loop
        valid_web_1 = "Quantum computing relies on quantum bits or qubits that can exist in superposition states."
        valid_web_2 = "Photonic integrated circuits enable low-latency optical computing on semiconductor chips."
        degenerate_web = "error error error error error error error error error error error error error error error error error"

        res1 = buffer.add_web_sample(ReplaySample(text=valid_web_1))
        res2 = buffer.add_web_sample(ReplaySample(text=valid_web_2))
        res3 = buffer.add_web_sample(ReplaySample(text=degenerate_web))

        self.assertTrue(res1)
        self.assertTrue(res2)
        self.assertFalse(res3)  # Quarantined!

        self.assertEqual(len(buffer.web_pool), 2)
        self.assertEqual(len(buffer.quarantine_pool), 1)
        guard_stats = buffer.get_collapse_guard_stats()
        self.assertEqual(guard_stats["quarantined_total"], 1)

    def test_stratified_sampling_exact_mixture_ratio(self):
        """Verify that sample_replay_batch and sample_batch strictly preserve the 70/30 mixture."""
        buffer = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)
        buffer.add_anchor_data(self.anchor_texts)
        buffer.add_web_data(self.web_texts)

        # Batch size 100 -> exactly 70 anchor, 30 web
        batch_100 = buffer.sample_replay_batch(batch_size=100)
        self.assertEqual(len(batch_100), 100)
        n_anchor_100 = sum(1 for s in batch_100 if s.source == "anchor")
        n_web_100 = sum(1 for s in batch_100 if s.source == "web")
        self.assertEqual(n_anchor_100, 70)
        self.assertEqual(n_web_100, 30)

        # Batch size 10 -> exactly 7 anchor, 3 web
        batch_10 = buffer.sample_replay_batch(batch_size=10)
        self.assertEqual(len(batch_10), 10)
        n_anchor_10 = sum(1 for s in batch_10 if s.source == "anchor")
        n_web_10 = sum(1 for s in batch_10 if s.source == "web")
        self.assertEqual(n_anchor_10, 7)
        self.assertEqual(n_web_10, 3)

    def test_web_ingestion_returns_newly_accepted_count(self):
        """Verify ingestion counts accepted samples, not the buffer's cumulative size."""
        buffer = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)
        buffer.add_web_data([self.web_texts[0]])

        accepted = buffer.add_web_data(
            [
                "Quantum processors use superconducting circuits to manipulate qubits and implement gate-based algorithms.",
                "error error error error error error error error error error error error error error error",
            ]
        )

        self.assertEqual(accepted, 1)
        self.assertEqual(len(buffer.web_pool), 2)
        self.assertEqual(len(buffer.quarantine_pool), 1)

    def test_sampling_returns_full_batch_when_only_one_pool_is_populated(self):
        """Use the available replay pool for the full batch when its counterpart is empty."""
        anchor_only = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)
        anchor_only.add_anchor_data(self.anchor_texts)
        anchor_batch = anchor_only.sample_replay_batch(batch_size=10)
        self.assertEqual(len(anchor_batch), 10)
        self.assertTrue(all(sample.source == "anchor" for sample in anchor_batch))

        web_only = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)
        web_only.add_web_data(self.web_texts)
        web_batch = web_only.sample_replay_batch(batch_size=10)
        self.assertEqual(len(web_batch), 10)
        self.assertTrue(all(sample.source == "web" for sample in web_batch))

    def test_sliding_window_fifo_capacity_eviction(self):
        """Verify FIFO sliding window capacity eviction on streaming web pool while anchor remains immutable."""
        max_capacity = 5
        buffer = ExperienceReplayBuffer(anchor_ratio=0.70, max_web_capacity=max_capacity, seed=42)
        buffer.add_anchor_data(self.anchor_texts)
        self.assertEqual(len(buffer.anchor_pool), len(self.anchor_texts))

        # Add 10 web samples sequentially with distinct doc_ids
        for i in range(10):
            buffer.add_web_sample(
                ReplaySample(
                    text=f"Unique valid streaming document index {i} detailing neural network architectures.",
                    doc_id=f"doc_{i}",
                )
            )

        # Web pool must be capped at max_capacity = 5
        self.assertEqual(len(buffer.web_pool), 5)
        # Oldest samples (0..4) should have been evicted; newest samples (5..9) must remain
        retained_ids = [s.doc_id for s in buffer.web_pool]
        self.assertEqual(retained_ids, [f"doc_{i}" for i in range(5, 10)])

        # Anchor pool must be completely untouched
        self.assertEqual(len(buffer.anchor_pool), len(self.anchor_texts))

    def test_zero_ram_binary_token_export_and_memmap(self):
        """Verify replay corpus export to disk binary and MemoryMappedTokenDataset O(1) batch streaming."""
        buffer = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)
        buffer.add_anchor_data(self.anchor_texts)
        buffer.add_web_data(self.web_texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_bin, val_bin = buffer.export_sharded_binaries(
                output_dir=tmpdir,
                target_tokens=1_200,
                train_val_split=0.90,
            )
            self.assertTrue(train_bin.exists())
            self.assertTrue(val_bin.exists())

            # Load via MemoryMappedTokenDataset
            ds = MemoryMappedTokenDataset(train_bin)
            self.assertGreater(len(ds), 500)

            # Sample batch of tokens
            x, y = ds.get_batch(batch_size=4, block_size=16)
            self.assertEqual(x.shape, (4, 16))
            self.assertEqual(y.shape, (4, 16))

    def test_document_level_partition_zero_leakage(self):
        """Verify train and validation exported binaries partition documents without overlap."""
        buffer = ExperienceReplayBuffer(anchor_ratio=0.70, seed=42)
        # Distinct recognizable markers in anchor and web docs
        anchor_docs = [f"AnchorDocUniqueMarker_{i} detailed reasoning content" for i in range(10)]
        web_docs = [f"WebDocUniqueMarker_{i} fresh domain facts" for i in range(10)]
        buffer.add_anchor_data(anchor_docs)
        buffer.add_web_data(web_docs)

        all_text = " ".join(anchor_docs + web_docs)
        tokenizer = CharTokenizer(all_text)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_bin, val_bin = buffer.export_sharded_binaries(
                tokenizer=tokenizer,
                output_dir=tmpdir,
                target_tokens=500,
                train_val_split=0.80,
            )
            train_ds = MemoryMappedTokenDataset(train_bin)
            val_ds = MemoryMappedTokenDataset(val_bin)

            train_text = tokenizer.decode(train_ds.data.tolist())
            val_text = tokenizer.decode(val_ds.data.tolist())

            # Find all unique markers present in train and in val
            train_markers = {m for m in [f"AnchorDocUniqueMarker_{i}" for i in range(10)] if m in train_text}
            val_markers = {m for m in [f"AnchorDocUniqueMarker_{i}" for i in range(10)] if m in val_text}

            # Guarantee ZERO overlap between train markers and val markers
            overlap = train_markers.intersection(val_markers)
            self.assertEqual(len(overlap), 0, f"Found leaked document markers across splits: {overlap}")
            self.assertGreater(len(train_markers), 0)
            self.assertGreater(len(val_markers), 0)


if __name__ == "__main__":
    unittest.main()

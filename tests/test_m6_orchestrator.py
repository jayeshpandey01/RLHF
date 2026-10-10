import unittest
from concurrent.futures import ThreadPoolExecutor

from evaluation.statistical_gating import GateDecision
from orchestration import (
    AsyncEventQueue,
    ContinualLearningOrchestrator,
    RequestCoalescer,
    SemanticCache,
    TelemetryEvent,
)
from retrieval import HybridRetriever, RetrievalDocument
from training.m5_preference.benchmark_suite import AlignmentBenchmarkSuite, AlignmentTestCase


class TestM6Orchestrator(unittest.TestCase):
    """Milestone 6 - Phase 6.3: Autonomous Continual Learning Loop Orchestrator Test Suite."""

    def test_semantic_cache_hit_and_miss_lifecycle(self):
        """Verify L1 Semantic Cache detects semantically equivalent queries and returns cached response."""
        cache = SemanticCache(similarity_threshold=0.90)
        cache.put(
            query="What is Direct Preference Optimization in machine learning?",
            response="DPO is a closed-form preference alignment method.",
            metadata={"source": "llm"},
        )

        # 1. Exact query match -> Cache Hit
        res_exact = cache.lookup("What is Direct Preference Optimization in machine learning?")
        self.assertIsNotNone(res_exact)
        resp, sim, meta = res_exact
        self.assertEqual(resp, "DPO is a closed-form preference alignment method.")
        self.assertGreaterEqual(sim, 0.99)

        # 2. Semantically similar query -> Cache Hit
        res_similar = cache.lookup("What is Direct Preference Optimization for machine learning?")
        self.assertIsNotNone(res_similar)
        self.assertGreaterEqual(res_similar[1], 0.90)

        # 3. Unrelated query -> Cache Miss
        res_unrelated = cache.lookup("How do photosynthesis enzymes function in marine algae?")
        self.assertIsNone(res_unrelated)

        stats = cache.get_stats()
        self.assertEqual(stats["total_lookups"], 3)
        self.assertEqual(stats["total_hits"], 2)
        self.assertEqual(stats["total_misses"], 1)

    def test_semantic_cache_is_safe_for_concurrent_queries(self):
        cache = SemanticCache()
        cache.put("same request", "cached response")

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(cache.lookup, ["same request"] * 100))

        self.assertTrue(all(result is not None for result in results))
        self.assertEqual(cache.get_stats()["total_lookups"], 100)
        self.assertEqual(cache.get_stats()["total_hits"], 100)

    def test_request_coalescer_debounce_and_deduplication(self):
        """Verify RequestCoalescer suppresses redundant concurrent queries within debounce window."""
        coalescer = RequestCoalescer(debounce_window_seconds=0.5, similarity_threshold=0.80)

        # 10 bursty incoming queries regarding the exact same breaking topic
        for i in range(10):
            coalescer.add_query("Quantum computing entanglement breakthrough 2026")

        # Drain before window expires (not forced) -> should be empty
        drained_early = coalescer.drain_coalesced_queries(force=False)
        self.assertEqual(len(drained_early), 0)

        # Force drain -> exactly 1 unique canonical query dispatched, 9 suppressed!
        drained = coalescer.drain_coalesced_queries(force=True)
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0], "Quantum computing entanglement breakthrough 2026")

        stats = coalescer.get_stats()
        self.assertEqual(stats["total_received"], 10)
        self.assertEqual(stats["total_coalesced"], 9)
        self.assertEqual(stats["total_dispatched"], 1)
        self.assertAlmostEqual(stats["suppression_rate"], 0.90, places=4)

    def test_request_coalescer_negation_polarity_separation(self):
        """Verify queries with opposite negation polarity are not conflated as duplicates."""
        coalescer = RequestCoalescer(debounce_window_seconds=0.1, similarity_threshold=0.70)
        coalescer.add_query("Is model collapse preventable with anchor replay data?")
        coalescer.add_query("Is model collapse not preventable with anchor replay data?")

        drained = coalescer.drain_coalesced_queries(force=True)
        # Must retain both queries because of polarity difference
        self.assertEqual(len(drained), 2)
        self.assertIn("Is model collapse preventable with anchor replay data?", drained)
        self.assertIn("Is model collapse not preventable with anchor replay data?", drained)

    def test_async_event_queue_rate_limiting_and_backpressure(self):
        """Verify AsyncEventQueue enforces token bucket rate limiting and non-blocking backpressure."""
        # Rate limit: 10 per sec, burst capacity: 5
        queue = AsyncEventQueue(max_size=100, rate_limit_per_second=10.0, burst_limit=5)

        accepted = 0
        dropped = 0
        # Rapidly attempt to push 20 events
        for i in range(20):
            event = TelemetryEvent(event_type="test", payload={"idx": i})
            if queue.enqueue(event):
                accepted += 1
            else:
                dropped += 1

        self.assertEqual(accepted, 5)  # Burst limit
        self.assertEqual(dropped, 15)  # Dropped due to rate limit
        self.assertEqual(queue.size(), 5)

        # Dequeue batch
        batch = queue.dequeue_batch(max_batch_size=10)
        self.assertEqual(len(batch), 5)
        self.assertEqual(queue.size(), 0)

    def test_orchestrator_query_fast_path_cache_bypass(self):
        """Verify process_query utilizes Semantic Cache on second turn, bypassing retrieval."""
        orchestrator = ContinualLearningOrchestrator()

        # Seed hybrid retriever with an evidence passage
        doc = RetrievalDocument(
            doc_id="doc_bio",
            text="Ribosomes synthesize cellular proteins by translating messenger RNA transcripts.",
            url="https://nature.com/ribosomes",
        )
        orchestrator.retriever.add_document(doc)

        # Turn 1: Cache Miss -> Retrieves and caches
        res1 = orchestrator.process_query("What cellular organelle synthesizes proteins?")
        self.assertFalse(res1["cached"])
        self.assertEqual(len(res1["evidence"]), 1)
        self.assertIn("Ribosomes", res1["response"])

        # Turn 2: Exact same query -> Cache Hit (< 5ms fast path)
        res2 = orchestrator.process_query("What cellular organelle synthesizes proteins?")
        self.assertTrue(res2["cached"])
        self.assertGreaterEqual(res2["similarity"], 0.99)
        self.assertEqual(res2["response"], res1["response"])

    def test_orchestrator_coalesced_knowledge_ingestion(self):
        """Verify coalesced queries are ingested, filtered into replay buffer, and indexed in retriever."""
        orchestrator = ContinualLearningOrchestrator()

        # Submit queries that need knowledge
        orchestrator.process_query("Tell me about topological quantum qubits.")
        orchestrator.process_query("Topological quantum qubits explanation.")

        # Ingest coalesced queries
        num_ingested = orchestrator.ingest_coalesced_knowledge(force_drain=True)
        self.assertGreaterEqual(num_ingested, 1)

        # Verify added to replay buffer
        self.assertGreaterEqual(len(orchestrator.replay_buffer.web_pool), 1)
        # Verify added to hybrid retriever
        self.assertGreaterEqual(len(orchestrator.retriever), 1)

    def test_orchestrator_does_not_index_quarantined_samples(self):
        orchestrator = ContinualLearningOrchestrator()
        orchestrator.coalescer.add_query("repetitive error sample")

        accepted = orchestrator.ingest_coalesced_knowledge(
            scraper_fn=lambda queries: ["error " * 30],
            force_drain=True,
        )

        self.assertEqual(accepted, 0)
        self.assertEqual(len(orchestrator.replay_buffer.web_pool), 0)
        self.assertEqual(len(orchestrator.retriever), 0)

    def test_orchestrator_continual_learning_cycle_proceed_deploy(self):
        """Verify end-to-end continual training cycle issues PROCEED_DEPLOY and atomically updates model."""
        custom_suite = AlignmentBenchmarkSuite(
            test_cases=[
                AlignmentTestCase(
                    prompt="What is 20 + 22?",
                    ground_truth="42",
                    domain="reasoning",
                ),
            ]
        )
        orchestrator = ContinualLearningOrchestrator(
            training_trigger_threshold=1,
            benchmark_suite=custom_suite,
        )
        orchestrator.replay_buffer.add_anchor_data(["Baseline grammar and dialogue anchor sample."])
        orchestrator.coalescer.add_query("Advanced neural network optimizations")
        orchestrator.ingest_coalesced_knowledge(force_drain=True)

        self.assertEqual(orchestrator.model_version, 1)

        # Candidate producing correct reasoning completions
        cand_fn = lambda p: "<think>Sum 20 and 22 gives 42.</think><answer>42</answer>"
        base_fn = lambda p: "Incomplete baseline output without answer."

        # Execute training cycle
        report = orchestrator.execute_continual_learning_cycle(
            force=True,
            candidate_generate_fn=cand_fn,
            baseline_generate_fn=base_fn,
            train_fn=lambda model, batch: object(),
        )

        self.assertEqual(report["status"], "PROMOTED")
        self.assertEqual(report["decision"], GateDecision.PROCEED_DEPLOY.value)
        self.assertEqual(orchestrator.model_version, 2)  # Atomically hot-swapped!
        self.assertEqual(orchestrator.training_cycles_completed, 1)
        self.assertIsNotNone(orchestrator.active_model)

    def test_orchestrator_continual_learning_cycle_candidate_rejection_and_active_rollback(self):
        """Verify continual training cycle rejects degraded candidate without reverting active model, and test active rollback."""
        orchestrator = ContinualLearningOrchestrator()
        orchestrator.model_version = 2
        orchestrator.replay_buffer.add_anchor_data(["A stable anchor example for the baseline model."])

        # Degraded candidate with broken output vs compliant baseline
        degraded_cand_fn = lambda p: "malformed text with zero reasoning or answers"
        compliant_base_fn = lambda p: "<think>Reasoning</think><answer>5</answer>"

        report = orchestrator.execute_continual_learning_cycle(
            force=True,
            candidate_generate_fn=degraded_cand_fn,
            baseline_generate_fn=compliant_base_fn,
            train_fn=lambda model, batch: object(),
        )

        # Candidate rejected before deployment; active model remains version 2!
        self.assertEqual(report["status"], "REJECTED")
        self.assertEqual(report["decision"], GateDecision.TRIGGER_ROLLBACK.value)
        self.assertEqual(orchestrator.model_version, 2)

        # Now test active model rollback
        rollback_res = orchestrator.rollback_to_anchor(reason="Post-deployment degradation")
        self.assertEqual(rollback_res["status"], "ROLLED_BACK")
        self.assertEqual(orchestrator.model_version, 0)

    def test_orchestrator_does_not_promote_without_training_callback(self):
        orchestrator = ContinualLearningOrchestrator(training_trigger_threshold=1)
        orchestrator.replay_buffer.add_anchor_data(["A stable anchor example for the baseline model."])

        report = orchestrator.execute_continual_learning_cycle(
            force=True,
            candidate_generate_fn=lambda prompt: "<think>Reasoning</think><answer>Accurate answer</answer>",
            baseline_generate_fn=lambda prompt: "Baseline response",
        )

        self.assertEqual(report["status"], "SKIPPED")
        self.assertEqual(report["reason"], "TRAIN_FN_NOT_CONFIGURED")
        self.assertEqual(orchestrator.model_version, 1)
        self.assertIsNone(orchestrator.active_model)

    def test_orchestrator_status_diagnostics(self):
        """Verify get_status returns complete telemetry across all decoupled subsystems."""
        orchestrator = ContinualLearningOrchestrator()
        orchestrator.process_query("What is machine learning?")

        status = orchestrator.get_status()
        self.assertEqual(status["total_queries_served"], 1)
        self.assertEqual(status["model_version"], 1)
        self.assertIn("semantic_cache", status)
        self.assertIn("coalescer", status)
        self.assertIn("event_queue", status)
        self.assertIn("replay_buffer", status)


if __name__ == "__main__":
    unittest.main()

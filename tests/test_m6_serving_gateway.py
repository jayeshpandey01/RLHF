import unittest

from fastapi.testclient import TestClient

from apps import ModelServingManager, create_app
from orchestration import ContinualLearningOrchestrator
from retrieval import RetrievalDocument


class TestM6ServingGateway(unittest.TestCase):
    """Milestone 6 - Phase 6.4: High-Throughput FastAPI Serving Gateway & Zero-Downtime Hot-Swap Test Suite."""

    def setUp(self):
        self.orchestrator = ContinualLearningOrchestrator(training_trigger_threshold=5)
        # Pre-seed hybrid retriever with sample document
        doc = RetrievalDocument(
            doc_id="doc_rag_1",
            text="Direct Preference Optimization trains policy weights without needing a separate critic network.",
            url="https://arxiv.org/abs/dpo_overview",
            metadata={"domain": "ai"},
        )
        self.orchestrator.retriever.add_document(doc)

        self.serving_manager = ModelServingManager(
            active_model="active_mock_model_v1",
            anchor_model="golden_anchor_baseline",
            model_version=1,
        )

        self.app = create_app(
            orchestrator=self.orchestrator,
            serving_manager=self.serving_manager,
        )
        self.client = TestClient(self.app)

    def test_serving_manager_rwlock_and_active_readers(self):
        """Verify ModelServingManager tracks concurrent readers under read_context."""
        self.assertEqual(self.serving_manager.get_stats()["active_readers"], 0)

        with self.serving_manager.read_context() as m1:
            self.assertEqual(m1, "active_mock_model_v1")
            self.assertEqual(self.serving_manager.get_stats()["active_readers"], 1)

            with self.serving_manager.read_context() as m2:
                self.assertEqual(m2, "active_mock_model_v1")
                self.assertEqual(self.serving_manager.get_stats()["active_readers"], 2)

        self.assertEqual(self.serving_manager.get_stats()["active_readers"], 0)

    def test_serving_manager_atomic_promotion(self):
        """Verify promote_candidate updates model version and active reference atomically."""
        new_v = self.serving_manager.promote_candidate("candidate_mock_model_v2")
        self.assertEqual(new_v, 2)
        self.assertEqual(self.serving_manager.model_version, 2)
        self.assertEqual(self.serving_manager.get_active_model(), "candidate_mock_model_v2")

        stats = self.serving_manager.get_stats()
        self.assertEqual(stats["total_promotions"], 1)
        self.assertFalse(stats["is_anchor"])

    def test_serving_manager_rollback_to_anchor(self):
        """Verify rollback_to_anchor restores golden baseline model atomically."""
        self.serving_manager.promote_candidate("candidate_model_v2")
        self.assertEqual(self.serving_manager.model_version, 2)

        v_rollback = self.serving_manager.rollback_to_anchor()
        self.assertEqual(v_rollback, 0)
        self.assertEqual(self.serving_manager.model_version, 0)
        self.assertEqual(self.serving_manager.get_active_model(), "golden_anchor_baseline")
        self.assertTrue(self.serving_manager.get_stats()["is_anchor"])

    def test_gateway_health_endpoint(self):
        """Verify GET /health returns service status and current model version."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model_version"], 1)
        self.assertEqual(data["service"], "continual-rag-llm")

    def test_gateway_status_endpoint(self):
        """Verify GET /status aggregates telemetry across orchestrator and serving manager."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("orchestrator", data)
        self.assertIn("serving_manager", data)
        self.assertEqual(data["model_version"], 1)
        self.assertIn("semantic_cache", data["orchestrator"])

    def test_gateway_query_uncached_then_cached(self):
        """Verify POST /query serves grounded answer on turn 1 and cached fast path on turn 2."""
        payload = {"query": "How does Direct Preference Optimization eliminate the critic?", "top_k": 2}

        # Turn 1: Cache Miss
        r1 = self.client.post("/query", json=payload)
        self.assertEqual(r1.status_code, 200)
        d1 = r1.json()
        self.assertFalse(d1["cached"])
        self.assertEqual(d1["model_version"], 1)
        self.assertGreaterEqual(len(d1["evidence"]), 1)

        # Turn 2: Cache Hit (< 5ms)
        r2 = self.client.post("/query", json=payload)
        self.assertEqual(r2.status_code, 200)
        d2 = r2.json()
        self.assertTrue(d2["cached"])
        self.assertGreaterEqual(d2["similarity"], 0.99)
        self.assertEqual(d2["response"], d1["response"])

    def test_gateway_rejects_unimplemented_streaming(self):
        response = self.client.post(
            "/query",
            json={"query": "Explain DPO.", "stream": True},
        )
        self.assertEqual(response.status_code, 501)
        self.assertIn("not implemented", response.json()["detail"])

    def test_gateway_trigger_learning_endpoint(self):
        """Verify POST /trigger_learning executes continual learning cycle."""
        response = self.client.post("/trigger_learning", json={"force": True})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_version", data)

    def test_gateway_model_promotion_and_rollback_endpoints(self):
        """Verify POST /models/promote updates version and /models/rollback restores anchor."""
        self.orchestrator.candidate_model = "trained_candidate_v2"

        # 1. Promote to target version 4
        res_promote = self.client.post("/models/promote", json={"version": 4})
        self.assertEqual(res_promote.status_code, 200)
        self.assertEqual(res_promote.json()["new_version"], 4)

        # Verify /health reflects promoted version
        res_health1 = self.client.get("/health")
        self.assertEqual(res_health1.json()["model_version"], 4)

        # 2. Rollback to anchor baseline
        res_rollback = self.client.post("/models/rollback")
        self.assertEqual(res_rollback.status_code, 200)
        self.assertEqual(res_rollback.json()["model_version"], 0)

        # Verify /health reflects rollback version 0
        res_health2 = self.client.get("/health")
        self.assertEqual(res_health2.json()["model_version"], 0)

    def test_gateway_model_promotion_requires_candidate(self):
        response = self.client.post("/models/promote", json={"version": 4})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.serving_manager.model_version, 1)

    def test_concurrent_reader_writer_synchronization(self):
        """Verify fair reader-writer synchronization under concurrent reader and writer threads."""
        import threading
        import time

        mgr = ModelServingManager(
            active_model="model_v1",
            anchor_model="golden_anchor",
            model_version=1,
        )
        read_results = []
        errors = []

        def reader_task():
            try:
                with mgr.read_context() as m:
                    time.sleep(0.02)
                    read_results.append(m)
            except Exception as e:
                errors.append(e)

        def writer_task():
            try:
                time.sleep(0.01)
                mgr.promote_candidate("model_v2", new_version=2)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader_task) for _ in range(5)]
        writer = threading.Thread(target=writer_task)

        for t in threads:
            t.start()
        writer.start()

        for t in threads:
            t.join()
        writer.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(read_results), 5)
        self.assertEqual(mgr.model_version, 2)
        self.assertEqual(mgr.get_active_model(), "model_v2")


if __name__ == "__main__":
    unittest.main()

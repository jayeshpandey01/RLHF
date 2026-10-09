"""Standalone test runner for Milestone 0 using standard unittest."""

import asyncio
from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from core.hardware import get_device_strategy
from core.schemas import (
    UserQuerySchema,
    ScrapedDocumentSchema,
    CleanedDocumentSchema,
    ChatMessageSchema,
    TrainingSampleSchema,
    BenchmarkCaseSchema,
    EvaluationReportSchema,
)
from core.ollama_client import OllamaClient


class TestMilestone0(unittest.TestCase):
    """Test suite verifying Milestone 0 foundation."""

    def test_01_hardware_detection(self):
        """Verify hardware detection accurately identifies host hardware."""
        strategy = get_device_strategy()
        print(f"\n[Hardware Detected] {strategy.description}")
        self.assertIn(strategy.device_name, ["cuda", "mps", "cpu"])
        self.assertIsNotNone(strategy.torch_device)
        self.assertIsNotNone(strategy.dtype)
        self.assertGreaterEqual(strategy.micro_batch_size, 1)

    def test_02_settings_and_paths(self):
        """Verify settings load and directory paths exist."""
        self.assertTrue(settings.OLLAMA_BASE_URL.startswith("http"))
        self.assertEqual(settings.DEFAULT_MODEL, "qwen2.5:0.5b")
        self.assertEqual(settings.ANCHOR_DATA_RATIO, 0.70)
        self.assertTrue(settings.DATASETS_DIR.exists())
        self.assertTrue(settings.CHECKPOINTS_DIR.exists())

    def test_03_schemas_and_validation(self):
        """Verify Pydantic V2 data contracts."""
        # Query
        query = UserQuerySchema(query="What is RLHF?")
        self.assertEqual(query.query, "What is RLHF?")

        # Scraped doc & hash
        doc = ScrapedDocumentSchema(
            url="https://example.com/rlhf",
            raw_html="<p>RLHF content</p>",
            clean_markdown="RLHF content",
        )
        h = doc.compute_hash()
        self.assertEqual(len(h), 64)

        # Training sample
        sample = TrainingSampleSchema(
            messages=[
                ChatMessageSchema(role="user", content="Hello"),
                ChatMessageSchema(role="assistant", content="Hi! How can I help?"),
            ],
            origin="anchor_golden",
        )
        self.assertEqual(sample.messages[-1].role, "assistant")

        # Invalid sample (does not end with assistant)
        with self.assertRaises(ValueError):
            TrainingSampleSchema(
                messages=[
                    ChatMessageSchema(role="assistant", content="Hi"),
                    ChatMessageSchema(role="user", content="Hello"),
                ]
            )

        # Evaluation report
        report = EvaluationReportSchema(
            candidate_name="cand_01",
            perplexity=10.5,
            citation_faithfulness=0.95,
            base_accuracy=0.80,
            candidate_accuracy=0.88,
            regression_delta=0.08,
            decision="APPROVED",
        )
        self.assertEqual(report.decision, "APPROVED")

    def test_04_ollama_client_resilience(self):
        """Verify Ollama client health check does not throw unhandled exceptions."""
        client = OllamaClient(base_url="http://localhost:11434", timeout=1.0)
        is_alive = asyncio.run(client.check_health())
        self.assertIsInstance(is_alive, bool)
        print(f"\n[Ollama Server Status] Reachable: {is_alive}")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMilestone0)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


"""Test suite for Milestone 0: Production Foundation & Engineering Harness."""

import pytest
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


def test_hardware_detection():
    """Verify hardware detection returns valid strategy."""
    strategy = get_device_strategy()
    assert strategy.device_name in ["cuda", "mps", "cpu"]
    assert strategy.torch_device is not None
    assert strategy.dtype is not None
    assert strategy.micro_batch_size >= 1
    assert strategy.gradient_accumulation_steps >= 1
    assert len(strategy.description) > 0


def test_settings_configuration():
    """Verify application settings and directory paths."""
    assert settings.OLLAMA_BASE_URL.startswith("http")
    assert settings.DEFAULT_MODEL == "qwen2.5:0.5b"
    assert settings.ANCHOR_DATA_RATIO == 0.70
    assert settings.NEW_WEB_DATA_RATIO == 0.30
    assert settings.PROJECT_ROOT.exists()
    assert settings.DATASETS_DIR.exists()


def test_schema_user_query():
    """Verify user query validation."""
    query = UserQuerySchema(query="What is reinforcement learning?")
    assert query.query == "What is reinforcement learning?"
    assert query.session_id == "default-session"
    assert query.timestamp is not None


def test_schema_scraped_document_hash():
    """Verify SHA-256 computation on scraped documents."""
    doc = ScrapedDocumentSchema(
        url="https://example.com/rlhf",
        raw_html="<html><body>RLHF is Reinforcement Learning from Human Feedback</body></html>",
        clean_markdown="RLHF is Reinforcement Learning from Human Feedback",
    )
    doc_hash = doc.compute_hash()
    assert len(doc_hash) == 64
    assert doc.sha256_hash == doc_hash


def test_schema_training_sample_validation():
    """Verify ChatML conversation validation."""
    valid_sample = TrainingSampleSchema(
        messages=[
            ChatMessageSchema(role="user", content="Explain PPO."),
            ChatMessageSchema(role="assistant", content="PPO is Proximal Policy Optimization."),
        ],
        origin="web_scraped",
    )
    assert valid_sample.messages[-1].role == "assistant"

    # Must raise error if sample does not end with assistant
    with pytest.raises(ValueError):
        TrainingSampleSchema(
            messages=[
                ChatMessageSchema(role="assistant", content="Hello"),
                ChatMessageSchema(role="user", content="Explain PPO."),
            ]
        )


def test_schema_evaluation_report():
    """Verify evaluation gate report schema."""
    report = EvaluationReportSchema(
        candidate_name="candidate_v1",
        perplexity=12.4,
        citation_faithfulness=0.92,
        base_accuracy=0.80,
        candidate_accuracy=0.88,
        regression_delta=0.08,
        decision="APPROVED",
    )
    assert report.decision == "APPROVED"
    assert report.citation_faithfulness >= 0.85


@pytest.mark.asyncio
async def test_ollama_client_resilience():
    """Verify Ollama client initializes cleanly and handles offline state gracefully."""
    client = OllamaClient(base_url="http://localhost:11434", timeout=1.0)
    # Regardless of whether daemon is running or not, health check must not crash
    is_healthy = await client.check_health()
    assert isinstance(is_healthy, bool)

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from apps.learning_jobs import LearningJobManager
from apps.live_knowledge import LiveKnowledgeService
from apps.server import create_app
from apps.serving_manager import ModelServingManager
from core.config import settings
from core.schemas import ScrapedDocumentSchema
from evaluation.citation_verifier import CitationVerifier
from ingestion.router import QueryIntent
from orchestration.loop_orchestrator import ContinualLearningOrchestrator
from retrieval import HybridRetriever, RetrievalDocument, SearchResult


class FakeKnowledge:
    def __init__(self, indexed=0, web_count=0):
        self.replay_buffer = SimpleNamespace(
            get_stats=lambda: {"web_count": web_count, "anchor_count": 3}
        )
        self.indexed = indexed
        self.calls = []

    async def collect_and_index(self, query):
        self.calls.append(query)
        return {
            "indexed": self.indexed,
            "replay_stats": self.replay_buffer.get_stats(),
        }


class FakeTrainingWorker:
    def train_evaluate_and_package(self, baseline_model=None):
        return {
            "status": "BLOCKED",
            "decision": "BLOCK_RELEASE",
            "baseline_model": baseline_model,
        }


class TestLiveLearningAPI(unittest.TestCase):
    def _orchestrator(self, model):
        orchestrator = ContinualLearningOrchestrator(active_model=model)
        orchestrator.retriever.add_document(
            RetrievalDocument(
                doc_id="dpo-source",
                text="DPO trains a policy on preferred and dispreferred responses [1].",
                url="https://example.test/dpo",
            )
        )
        return orchestrator

    def test_query_collects_then_uses_ollama_with_citation_prompt(self):
        knowledge = FakeKnowledge()
        model = settings.DEFAULT_MODEL
        orchestrator = self._orchestrator(model)
        manager = ModelServingManager(active_model=model, anchor_model=model)
        app = create_app(
            orchestrator=orchestrator,
            serving_manager=manager,
            live_knowledge=knowledge,
        )
        mock_response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {
                    "content": (
                        "DPO trains a policy on preferred and dispreferred responses [1]. "
                        "Source: https://example.test/dpo"
                    )
                }
            },
        )

        with patch("apps.server.httpx.post", return_value=mock_response) as post:
            response = TestClient(app).post(
                "/query",
                json={"query": "How does DPO work?", "top_k": 1},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(knowledge.calls, ["How does DPO work?"])
        self.assertIn("[1] Source: https://example.test/dpo", post.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertIn("inline citations", post.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertNotIn("Source: https://", response.json()["response"])
        self.assertNotIn("could not be verified", response.json()["response"])

    def test_query_queues_training_and_job_endpoint_reports_gate_block(self):
        knowledge = FakeKnowledge(indexed=1, web_count=5)
        jobs = LearningJobManager()
        model = FakeModel()
        orchestrator = self._orchestrator(model)
        manager = ModelServingManager(active_model=model, anchor_model=model)
        app = create_app(
            orchestrator=orchestrator,
            serving_manager=manager,
            live_knowledge=knowledge,
            training_worker=FakeTrainingWorker(),
            learning_jobs=jobs,
        )
        client = TestClient(app)

        response = client.post("/query", json={"query": "How does DPO work?", "top_k": 1})
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["learning_job_id"]
        self.assertTrue(job_id)

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job_response = client.get(f"/learning/jobs/{job_id}")
            if job_response.json()["status"] not in {"QUEUED", "RUNNING"}:
                break
            time.sleep(0.01)

        self.assertEqual(job_response.status_code, 200)
        self.assertEqual(job_response.json()["status"], "BLOCKED")
        self.assertEqual(manager.model_version, 1)

    def test_training_job_evaluates_against_current_active_ollama_model(self):
        knowledge = FakeKnowledge(indexed=1, web_count=5)
        jobs = LearningJobManager()
        worker = FakeTrainingWorker()
        model = "rlhf-active-model"
        orchestrator = self._orchestrator(model)
        manager = ModelServingManager(active_model=model, anchor_model=settings.DEFAULT_MODEL)
        client = TestClient(
            create_app(
                orchestrator=orchestrator,
                serving_manager=manager,
                live_knowledge=knowledge,
                training_worker=worker,
                learning_jobs=jobs,
            )
        )

        response = client.post("/query", json={"query": "How does DPO work?", "top_k": 1})
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["learning_job_id"]

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job_response = client.get(f"/learning/jobs/{job_id}")
            if job_response.json()["status"] not in {"QUEUED", "RUNNING"}:
                break
            time.sleep(0.01)

        self.assertEqual(job_response.json()["result"]["baseline_model"], model)

    def test_live_knowledge_curates_persists_indexes_and_reloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = HybridRetriever()
            service = LiveKnowledgeService(
                retriever,
                corpus_path=root / "corpus.jsonl",
                provenance_path=root / "provenance.jsonl",
            )
            text = " ".join(
                [
                    "Direct Preference Optimization trains language models using comparative human feedback.",
                    "Researchers provide preferred and rejected responses for the same prompt.",
                    "The policy learns to increase the likelihood assigned to preferred outputs.",
                    "Unlike traditional reinforcement learning, DPO does not fit a standalone reward model.",
                    "Its objective is derived from the KL-regularized optimal policy.",
                    "A reference model supplies the probability baseline during optimization.",
                    "The training corpus should preserve the source URL for traceable evidence.",
                    "Careful evaluation checks factual accuracy and behavior on held-out tasks.",
                ]
            )
            document = ScrapedDocumentSchema(
                url="https://example.test/dpo",
                raw_html=text,
                clean_markdown=text,
            )
            service.router.route_and_fetch = AsyncMock(
                return_value=(QueryIntent.ACADEMIC_RESEARCH, [document])
            )

            result = asyncio.run(service.collect_and_index("Direct Preference Optimization"))

            self.assertEqual(result["indexed"], 1)
            self.assertEqual(service.replay_buffer.get_stats()["web_count"], 1)
            self.assertEqual(len(retriever.retrieve("Direct Preference Optimization", top_k=1)), 1)
            saved = json.loads((root / "corpus.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["url"], "https://example.test/dpo")

            reloaded_retriever = HybridRetriever()
            reloaded = LiveKnowledgeService(
                reloaded_retriever,
                corpus_path=root / "corpus.jsonl",
                provenance_path=root / "provenance.jsonl",
            )
            self.assertEqual(reloaded.replay_buffer.get_stats()["web_count"], 1)
            self.assertEqual(
                len(reloaded_retriever.retrieve("Direct Preference Optimization", top_k=1)),
                1,
            )

    def test_uncited_or_unsupported_answer_falls_back_to_cited_evidence(self):
        orchestrator = self._orchestrator(FakeModel())
        response = orchestrator.process_query(
            query="How does DPO work?",
            top_k=1,
            active_model=FakeModel(),
            generator_fn=lambda *_: "An unsupported claim about unrelated science.",
        )

        self.assertEqual(
            response["response"],
            "DPO trains a policy on preferred and dispreferred responses. [1]",
        )

    def test_extract_fallback_selects_relevant_sentence_from_lower_ranked_source(self):
        evidence = [
            SearchResult(
                doc_id="related-paper",
                score=0.9,
                text="# Abstract\nThis paper discusses a related preference learning approach.",
            ),
            SearchResult(
                doc_id="foundational-dpo-paper",
                score=0.7,
                text=(
                    "# Abstract\nDirect Preference Optimization eliminates the need "
                    "for a separate reward model by directly optimizing the policy on "
                    "preference data."
                ),
            ),
        ]

        response = ContinualLearningOrchestrator._extractive_fallback(
            "How does Direct Preference Optimization eliminate the reward model?",
            evidence,
        )

        self.assertEqual(
            response,
            "Direct Preference Optimization eliminates the need for a separate reward model by directly optimizing the policy on preference data. [2]",
        )
        metrics = CitationVerifier().verify_response(
            response=response,
            ground_truth_context=" ".join(result.text for result in evidence),
            reference_documents={
                str(index): result.text
                for index, result in enumerate(evidence, start=1)
            },
        )
        self.assertEqual(metrics.valid_citations, 1)
        self.assertEqual(metrics.total_citations, 1)
        self.assertGreaterEqual(metrics.faithfulness_score, 0.5)

    def test_ollama_is_not_called_when_retrieval_has_no_evidence(self):
        orchestrator = ContinualLearningOrchestrator(active_model=FakeModel())
        calls = []
        response = orchestrator.process_query(
            query="A question with no indexed evidence",
            generator_fn=lambda *_: calls.append("called") or "Unsupported response.",
        )

        self.assertEqual(response["response"], "I couldn't find source evidence to answer this query.")
        self.assertEqual(calls, [])


class FakeModel:
    def generate(self, prompt):
        return "DPO trains a policy from preferences [1]."

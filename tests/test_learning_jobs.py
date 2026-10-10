import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apps.learning_jobs import LearningJobManager
from evaluation.benchmark_suite import GoldenBenchmarkSuite, GoldenTestCase
from training.m3_sft.hf_worker import HuggingFaceLoRAWorker, _anchor_examples, _read_web_examples


class TestLearningJobs(unittest.TestCase):
    def test_learning_job_records_gate_block_without_promoting(self):
        jobs = LearningJobManager()
        promoted = []
        job_id, created = jobs.submit(
            lambda: {"status": "BLOCKED", "decision": "BLOCK_RELEASE"},
            lambda model: promoted.append(model) or 2,
        )

        self.assertTrue(created)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = jobs.get(job_id)
            if job and job["status"] not in {"QUEUED", "RUNNING"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "BLOCKED")
        self.assertEqual(promoted, [])

    def test_learning_job_promotes_only_approved_ollama_model(self):
        jobs = LearningJobManager()
        promoted = []
        job_id, _ = jobs.submit(
            lambda: {"status": "APPROVED", "ollama_model": "rlhf-candidate-test"},
            lambda model: promoted.append(model) or 2,
        )

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = jobs.get(job_id)
            if job and job["status"] not in {"QUEUED", "RUNNING"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "PROMOTED")
        self.assertEqual(promoted, ["rlhf-candidate-test"])
        self.assertEqual(job["result"]["model_version"], 2)

    def test_learning_job_surfaces_worker_failure(self):
        jobs = LearningJobManager()

        def fail():
            raise RuntimeError("training failed")

        job_id, _ = jobs.submit(fail, lambda _: 2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = jobs.get(job_id)
            if job and job["status"] == "FAILED":
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "FAILED")
        self.assertEqual(job["error"], "training failed")

    def test_training_examples_preserve_anchor_and_web_source(self):
        self.assertEqual(len(_anchor_examples()), 3)
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            long_text = "Evidence text with a verified source. Important factual conclusion here. Additional background context for the domain."
            corpus_path.write_text(
                json.dumps(
                    {
                        "url": "https://example.test/source",
                        "text": long_text,
                        "instruction": "What factual conclusion does the source provide?",
                        "answer": "It provides an important factual conclusion.",
                        "supervision_status": "reviewed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = _read_web_examples(corpus_path)

        self.assertEqual(len(examples), 1)
        prompt, answer = examples[0]
        self.assertIn("https://example.test/source", prompt)
        self.assertIn("https://example.test/source", answer)
        # Verify answer is grounded but not a verbatim copy of the full prompt
        self.assertNotEqual(prompt, answer)
        self.assertNotIn(long_text, answer)
        self.assertTrue(answer.endswith("Source: https://example.test/source"))

    def test_training_examples_truncation_safety(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            huge_text = "Sentence one is clear. Sentence two has key facts. " + ("Extra noise text. " * 100)
            corpus_path.write_text(
                json.dumps(
                    {
                        "url": "https://example.test/long-source",
                        "text": huge_text,
                        "instruction": "What are the key facts?",
                        "answer": "The source presents two clear facts.",
                        "supervision_status": "reviewed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = _read_web_examples(corpus_path)
            self.assertEqual(len(examples), 1)
            prompt, answer = examples[0]
            # Prompt must be bounded and answer must be concise
            self.assertLessEqual(len(answer), 400)
            self.assertIn("Source: https://example.test/long-source", answer)

    def test_unlabeled_corpus_is_not_used_as_sft_supervision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            corpus_path.write_text(
                json.dumps(
                    {
                        "url": "https://example.test/source",
                        "text": "Unlabeled crawled text remains retrieval evidence.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(_read_web_examples(corpus_path), [])
            result = HuggingFaceLoRAWorker(corpus_path=corpus_path).train_evaluate_and_package()
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["reason"], "NO_VALIDATED_WEB_SUPERVISION")

    def test_sft_rejects_unreviewed_web_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "corpus.jsonl"
            corpus_path.write_text(
                json.dumps(
                    {
                        "url": "https://example.test/source",
                        "text": "Source evidence for this record.",
                        "instruction": "What does this source say?",
                        "answer": "It provides some evidence.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "supervision_status"):
                _read_web_examples(corpus_path)

    def test_baseline_generation_uses_active_model_and_candidate_token_budget(self):
        case = GoldenTestCase(
            case_id="test_case",
            category="ml_theory",
            bucket="core_domain",
            question="What does AdamW decouple?",
            ground_truth_context="AdamW decouples weight decay from gradient updates.",
            reference_answer="AdamW decouples weight decay from gradient updates.",
            required_citations=["Loshchilov & Hutter, 2019"],
            golden_keywords=["AdamW", "weight decay"],
        )
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "AdamW decouples weight decay [Loshchilov & Hutter, 2019]."}},
        )
        worker = HuggingFaceLoRAWorker(corpus_path=Path("unused.jsonl"))

        with patch("training.m3_sft.hf_worker.httpx.post", return_value=response) as post:
            worker._score_ollama_baseline(
                GoldenBenchmarkSuite([case]),
                baseline_model="current-active-model",
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "current-active-model")
        self.assertEqual(payload["options"], {"temperature": 0, "num_predict": 120})

    def test_case_score_penalizes_unsupported_keyword_stuffing(self):
        case = GoldenTestCase(
            case_id="test_case",
            category="ml_theory",
            bucket="core_domain",
            question="What does AdamW decouple?",
            ground_truth_context="AdamW decouples weight decay from gradient updates.",
            reference_answer="AdamW decouples weight decay from gradient updates.",
            required_citations=["Loshchilov & Hutter, 2019"],
            golden_keywords=["AdamW", "weight decay"],
        )
        supported = (
            "AdamW decouples weight decay from gradient updates "
            "[Loshchilov & Hutter, 2019]."
        )
        stuffed = (
            "AdamW weight decay AdamW weight decay nonsense unsupported unrelated "
            "word salad [Loshchilov & Hutter, 2019]."
        )

        self.assertGreater(
            HuggingFaceLoRAWorker._case_score(case, supported),
            HuggingFaceLoRAWorker._case_score(case, stuffed),
        )

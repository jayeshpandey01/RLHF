import json
import tempfile
import time
import unittest
from pathlib import Path

from apps.learning_jobs import LearningJobManager
from training.m3_sft.hf_worker import _anchor_examples, _read_web_examples


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
            corpus_path.write_text(
                json.dumps(
                    {
                        "url": "https://example.test/source",
                        "text": "Evidence text with a verified source.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = _read_web_examples(corpus_path)

        self.assertEqual(len(examples), 1)
        self.assertIn("https://example.test/source", examples[0][0])
        self.assertIn("https://example.test/source", examples[0][1])

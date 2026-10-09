"""Unit and integration tests for Milestone 4 Phase 4.1: Golden Domain Benchmark Suite."""

import tempfile
import unittest
from pathlib import Path

from core.schemas import BenchmarkCaseSchema
from curation.contamination import BenchmarkContaminationGuard
from evaluation.benchmark_suite import (
    GoldenTestCase,
    GoldenBenchmarkSuite,
    create_default_golden_suite_cases,
)
from training.m3_sft.prompt import ChatMLFormatter


class TestM4BenchmarkSuite(unittest.TestCase):
    """Test suite validating Phase 4.1 Golden Benchmark Suite and schema contracts."""

    def setUp(self):
        self.suite = GoldenBenchmarkSuite()

    def test_default_suite_completeness(self):
        """Verify the suite contains exactly 30 test cases with unique IDs and non-empty fields."""
        self.assertEqual(len(self.suite), 30)
        seen_ids = set()

        for case in self.suite:
            self.assertNotIn(case.case_id, seen_ids, f"Duplicate case ID found: {case.case_id}")
            seen_ids.add(case.case_id)

            # Integrity checks
            self.assertGreater(len(case.question.strip()), 10)
            self.assertGreater(len(case.ground_truth_context.strip()), 20)
            self.assertGreater(len(case.reference_answer.strip()), 15)
            self.assertGreater(len(case.golden_keywords), 0)
            self.assertIn(case.category, ["ml_theory", "transformer_internals", "peft_sft", "systems_curation"])
            self.assertIn(case.bucket, ["core_domain", "edge_case", "adversarial_bait", "regression_replay"])
            self.assertIn(case.difficulty, ["easy", "medium", "hard"])

    def test_four_bucket_distribution(self):
        """Verify the four-bucket stratification matches the 40/20/20/20 design specification."""
        summary = self.suite.get_summary()
        buckets = summary["buckets"]

        self.assertEqual(buckets.get("core_domain", 0), 12)
        self.assertEqual(buckets.get("edge_case", 0), 6)
        self.assertEqual(buckets.get("adversarial_bait", 0), 6)
        self.assertEqual(buckets.get("regression_replay", 0), 6)

        # Check all 4 categories are covered
        categories = summary["categories"]
        self.assertGreaterEqual(len(categories), 4)

    def test_schema_roundtrip(self):
        """Verify bidirectional conversion between GoldenTestCase and BenchmarkCaseSchema."""
        case = self.suite[0]
        schema = case.to_schema()

        self.assertIsInstance(schema, BenchmarkCaseSchema)
        self.assertEqual(schema.case_id, case.case_id)
        self.assertEqual(schema.question, case.question)
        self.assertEqual(schema.ground_truth_context, case.ground_truth_context)
        self.assertEqual(schema.required_citations, case.required_citations)
        self.assertEqual(schema.golden_keywords, case.golden_keywords)

        reconstructed = GoldenTestCase.from_schema(
            schema,
            category=case.category,
            bucket=case.bucket,
            reference_answer=case.reference_answer,
            difficulty=case.difficulty,
        )
        self.assertEqual(reconstructed.case_id, case.case_id)
        self.assertEqual(reconstructed.reference_answer, case.reference_answer)

        # Batch schema conversion
        schemas = self.suite.to_schemas()
        self.assertEqual(len(schemas), 30)
        suite_from_schemas = GoldenBenchmarkSuite.from_schemas(schemas)
        self.assertEqual(len(suite_from_schemas), 30)

    def test_chatml_prompt_formatting(self):
        """Verify test cases format properly into canonical ChatML prompt strings."""
        case = self.suite[0]
        messages = case.to_chat_messages()

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[2].role, "assistant")

        prompt_str = case.format_prompt()
        self.assertIn(ChatMLFormatter.START_TAG, prompt_str)
        self.assertIn(ChatMLFormatter.ASSISTANT_HEADER, prompt_str)
        self.assertIn(case.question, prompt_str)
        self.assertIn(case.ground_truth_context, prompt_str)

    def test_keyword_coverage_calculation(self):
        """Verify keyword recall calculation on partial, full, and zero matches."""
        case = self.suite.get_by_id("core_ml_001")
        self.assertIsNotNone(case)

        # Full match
        full_text = "The Baur-Strassen theorem proves that 5C operations suffice for reverse-mode computational complexity."
        self.assertEqual(case.evaluate_keyword_coverage(full_text), 1.0)

        # Partial match (2 of 4 keywords)
        partial_text = "Baur-Strassen theorem discusses reverse-mode automatic differentiation."
        score = case.evaluate_keyword_coverage(partial_text)
        self.assertAlmostEqual(score, 0.5)

        # Zero match
        zero_text = "This text mentions unrelated concepts like databases."
        self.assertEqual(case.evaluate_keyword_coverage(zero_text), 0.0)

    def test_citation_presence_calculation(self):
        """Verify required citation presence detection."""
        case = self.suite.get_by_id("core_ml_001")
        self.assertIsNotNone(case)

        valid_response = "According to Baur & Strassen, 1983, reverse-mode complexity is bounded."
        self.assertEqual(case.evaluate_citation_presence(valid_response), 1.0)

        invalid_response = "Backpropagation is efficient according to someone else."
        self.assertEqual(case.evaluate_citation_presence(invalid_response), 0.0)

    def test_suite_filtering_and_sampling(self):
        """Verify filtering by category, bucket, difficulty, and deterministic sampling."""
        # Category filtering
        ml_suite = self.suite.filter_by_category("ml_theory")
        for c in ml_suite:
            self.assertEqual(c.category, "ml_theory")
        self.assertGreater(len(ml_suite), 0)

        # Bucket filtering
        edge_suite = self.suite.filter_by_bucket("edge_case")
        self.assertEqual(len(edge_suite), 6)
        for c in edge_suite:
            self.assertEqual(c.bucket, "edge_case")

        # Difficulty filtering
        hard_suite = self.suite.filter_by_difficulty("hard")
        for c in hard_suite:
            self.assertEqual(c.difficulty, "hard")

        # Deterministic sampling
        sample_1 = self.suite.sample(5, seed=123)
        sample_2 = self.suite.sample(5, seed=123)
        self.assertEqual(len(sample_1), 5)
        self.assertEqual([c.case_id for c in sample_1], [c.case_id for c in sample_2])

    def test_contamination_guard_integration(self):
        """Verify benchmark registration with BenchmarkContaminationGuard flags leaking documents."""
        guard = BenchmarkContaminationGuard(ngram_size=13)
        added_ngrams = self.suite.register_with_contamination_guard(guard)

        self.assertGreater(added_ngrams, 50)
        self.assertGreater(guard.size(), 50)

        # Test leak detection: text containing a verbatim benchmark snippet
        case = self.suite[0]
        leaked_text = f"We note that {case.ground_truth_context} Furthermore, this is essential."
        is_contam, matched = guard.is_contaminated(leaked_text)
        self.assertTrue(is_contam)
        self.assertIsNotNone(matched)

        # Clean document must pass
        clean_text = "The quick brown fox jumps over the lazy dog in an unrelated outdoor setting with no math."
        is_contam_clean, _ = guard.is_contaminated(clean_text)
        self.assertFalse(is_contam_clean)

    def test_json_serialization_roundtrip(self):
        """Verify JSON export and import preserve all fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / "benchmarks" / "golden_suite.json"
            exported_path = self.suite.export_to_json(json_file)
            self.assertTrue(exported_path.exists())

            loaded_suite = GoldenBenchmarkSuite.load_from_json(exported_path)
            self.assertEqual(len(loaded_suite), len(self.suite))

            for orig, loaded in zip(self.suite, loaded_suite):
                self.assertEqual(orig.case_id, loaded.case_id)
                self.assertEqual(orig.question, loaded.question)
                self.assertEqual(orig.reference_answer, loaded.reference_answer)
                self.assertEqual(orig.category, loaded.category)
                self.assertEqual(orig.bucket, loaded.bucket)
                self.assertEqual(orig.golden_keywords, loaded.golden_keywords)
                self.assertEqual(orig.required_citations, loaded.required_citations)


if __name__ == "__main__":
    unittest.main()


"""Unit and integration tests for Milestone 4 Phase 4.2: Citation Verifier & RAG Triad."""

import unittest

from evaluation.benchmark_suite import GoldenBenchmarkSuite
from evaluation.citation_verifier import (
    CitationTag,
    SentenceClaim,
    CitationMetrics,
    CitationVerifier,
)


class TestM4CitationVerifier(unittest.TestCase):
    """Test suite validating Phase 4.2 Citation Extraction, Precision, Recall, and Faithfulness."""

    def setUp(self):
        self.verifier = CitationVerifier(entailment_threshold=0.50, strict_mode=True)
        self.suite = GoldenBenchmarkSuite()

    def test_citation_extraction_formats(self):
        """Verify extraction across numeric, range, author-year, and markdown formats."""
        text = (
            "Transformers rely on self-attention [1]. Multiple heads are used [2, 3] "
            "and scale across layers [4-6]. Furthermore, Vaswani et al. (2017) demonstrated "
            "convergence, while [Hu et al., 2021] introduced LoRA. "
            "See details at [Attention Docs](https://arxiv.org/abs/1706.03762) and [Source A]."
        )
        tags = self.verifier.extract_citations(text)
        self.assertGreaterEqual(len(tags), 6)

        tag_types = {t.tag_type for t in tags}
        self.assertIn("numeric", tag_types)
        self.assertIn("numeric_range", tag_types)
        self.assertIn("author_year", tag_types)
        self.assertIn("markdown_url", tag_types)
        self.assertIn("named_source", tag_types)

        # Range expansion check: [4-6] -> ["4", "5", "6"]
        range_tag = next(t for t in tags if t.tag_type == "numeric_range")
        self.assertEqual(range_tag.normalized_indices, ["4", "5", "6"])

    def test_sentence_splitting_and_abbreviation_safety(self):
        """Verify abbreviations like et al. and i.e. do not prematurely split sentences."""
        text = (
            "According to Vaswani et al. (2017), self-attention computes dot products. "
            "This mechanism is efficient, i.e., it enables constant path routing."
        )
        claims = self.verifier.segment_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertIn("Vaswani et al. (2017)", claims[0].raw_text)
        self.assertIn("i.e.", claims[1].clean_claim)

    def test_boilerplate_filtering(self):
        """Verify conversational boilerplate is tagged as non-factual and excluded from penalties."""
        text = (
            "Certainly, here is the summary of the paper. "
            "Transformers replace recurrence with multi-head self-attention [1]. "
            "I hope this helps!"
        )
        claims = self.verifier.segment_claims(text)
        self.assertEqual(len(claims), 3)

        self.assertFalse(claims[0].is_factual)  # Boilerplate
        self.assertTrue(claims[1].is_factual)   # Factual statement
        self.assertFalse(claims[2].is_factual)  # Boilerplate pleasantry

    def test_entailment_grounding_calculation(self):
        """Verify hybrid lexical containment and Jaccard grounding scoring."""
        context = (
            "Baur and Strassen (1983) proved that any scalar rational function f computable "
            "in C elementary operations has its gradient vector computable in at most 5C operations."
        )
        supported_claim = "Baur and Strassen proved that gradient vectors require at most 5C operations."
        unsupported_claim = "Quantum computers simulate biochemical molecular dynamics in polynomial time."

        is_supp, score_supp = self.verifier.check_entailment(supported_claim, context)
        self.assertTrue(is_supp)
        self.assertGreaterEqual(score_supp, 0.50)

        is_unsupp, score_unsupp = self.verifier.check_entailment(unsupported_claim, context)
        self.assertFalse(is_unsupp)
        self.assertLess(score_unsupp, 0.30)

    def test_perfect_citation_verification(self):
        """Verify that grounded statements with valid citations achieve 100% precision, recall, and faithfulness."""
        context = (
            "Vaswani et al. (2017) introduced the Transformer architecture based entirely on attention mechanisms. "
            "Self-attention connects all pairs of input tokens in O(1) sequential operations."
        )
        response = (
            "Vaswani et al. (2017) proposed the Transformer using attention mechanisms [1]. "
            "Self-attention connects all token positions with O(1) sequential operations [1]."
        )
        doc_map = {"1": context}

        metrics = self.verifier.verify_response(
            response=response,
            ground_truth_context=context,
            reference_documents=doc_map,
        )

        self.assertEqual(metrics.citation_precision, 1.0)
        self.assertEqual(metrics.citation_recall, 1.0)
        self.assertEqual(metrics.citation_f1, 1.0)
        self.assertEqual(metrics.faithfulness_score, 1.0)
        self.assertEqual(len(metrics.unsupported_claims), 0)
        self.assertEqual(len(metrics.hallucinated_citations), 0)

    def test_hallucinated_citation_detection(self):
        """Verify that spurious citations and hallucinated claims are penalized."""
        context = (
            "Hu et al. (2021) introduced LoRA to parameterize weight updates via low-rank decomposition."
        )
        # Statement claims quantum computing but cites [1]
        response = (
            "Quantum entanglement enables instant teleportation across light years [1]. "
            "LoRA decomposes updates into low-rank matrices [1]."
        )
        doc_map = {"1": context}

        metrics = self.verifier.verify_response(
            response=response,
            ground_truth_context=context,
            reference_documents=doc_map,
        )

        # Precision must be strictly less than 1.0 due to the false claim citing [1]
        self.assertEqual(metrics.valid_citations, 1)
        self.assertEqual(metrics.total_citations, 2)
        self.assertAlmostEqual(metrics.citation_precision, 0.50)
        self.assertAlmostEqual(metrics.faithfulness_score, 0.50)
        self.assertEqual(len(metrics.unsupported_claims), 1)
        self.assertIn("1", metrics.hallucinated_citations)

    def test_negation_contradiction_detection(self):
        """Verify that direct negation inversion ('not improve' vs 'improves') is rejected."""
        context = "The clinical trial found that the treatment improves survival in the patient cohort."
        contradictory_claim = "The clinical trial found that the treatment does not improve survival in the patient cohort."
        is_entailed, score = self.verifier.check_entailment(contradictory_claim, context)
        self.assertFalse(is_entailed)
        self.assertEqual(score, 0.0)

    def test_nonexistent_citation_id_rejection(self):
        """Verify that citing non-existent doc ID [999] fails citation verification."""
        context = "The study demonstrated significant efficiency gains with attention."
        response = "The study demonstrated significant efficiency gains with attention [999]."
        doc_map = {"1": context}
        metrics = self.verifier.verify_response(response, context, doc_map)
        self.assertEqual(metrics.valid_citations, 0)
        self.assertIn("999", metrics.hallucinated_citations)
        self.assertEqual(metrics.citation_precision, 0.0)

    def test_missing_citation_recall_penalty(self):
        """Verify that factual claims lacking citations lower Citation Recall."""
        context = (
            "AdamW decouples weight decay directly from gradient updates to avoid scaling penalties."
        )
        # Factual statement without any citation
        response = (
            "AdamW decouples weight decay from gradient updates."
        )
        metrics = self.verifier.verify_response(
            response=response,
            ground_truth_context=context,
        )

        # Faithfulness is 1.0 because the statement is true to context, but citation recall is 0.0
        self.assertEqual(metrics.faithfulness_score, 1.0)
        self.assertEqual(metrics.citation_recall, 0.0)
        self.assertEqual(metrics.citation_precision, 0.0)

    def test_benchmark_case_integration(self):
        """Verify direct evaluation against a GoldenTestCase from Phase 4.1."""
        case = self.suite.get_by_id("core_ml_001")
        self.assertIsNotNone(case)

        # 1. Test verification with the reference answer
        metrics_ref = self.verifier.verify_benchmark_case(case, case.reference_answer)
        self.assertIsInstance(metrics_ref, CitationMetrics)
        self.assertGreaterEqual(metrics_ref.faithfulness_score, 0.50)
        self.assertGreater(metrics_ref.total_sentences, 0)

        # 2. Test fully grounded response matching case context
        grounded_resp = (
            "Baur and Strassen (1983) proved that any scalar rational function computable in C operations "
            "has its gradient computable in at most 5C operations [Baur & Strassen, 1983]."
        )
        metrics_grounded = self.verifier.verify_benchmark_case(case, grounded_resp)
        self.assertEqual(metrics_grounded.faithfulness_score, 1.0)
        self.assertEqual(metrics_grounded.citation_precision, 1.0)
        self.assertEqual(metrics_grounded.citation_recall, 1.0)



if __name__ == "__main__":
    unittest.main()

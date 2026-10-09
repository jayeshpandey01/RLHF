"""Unit and Integration Tests for Milestone 4, Phase 4.3 (LLM-as-a-Judge Engine)."""

import json
import unittest

from evaluation.benchmark_suite import create_default_golden_suite_cases
from evaluation.llm_judge import (
    JudgeScore,
    LLMJudge,
    PairwiseJudgeResult,
)


class TestM4LLMJudge(unittest.TestCase):
    """Test suite for LLM-as-a-Judge engine, rubric calibration, and bias mitigation."""

    def setUp(self):
        self.cases = create_default_golden_suite_cases()
        self.case_map = {c.case_id: c for c in self.cases}
        self.heuristic_judge = LLMJudge()

    def test_single_evaluation_heuristic_quality(self):
        """Verify heuristic evaluation assigns high scores to grounded, accurate responses."""
        case = self.case_map["core_peft_007"]
        good_response = (
            "LoRA adapts parameters via low-rank decomposition Delta W = (alpha / r) * B A [Hu et al., 2021]. "
            "Matrix B uses zero initialization to guarantee base model identity preservation at step zero."
        )
        score = self.heuristic_judge.evaluate_benchmark_case(case, good_response)
        self.assertIsInstance(score, JudgeScore)
        self.assertEqual(score.mode, "heuristic")
        self.assertGreaterEqual(score.accuracy, 4.0)
        self.assertGreaterEqual(score.faithfulness, 4.0)
        self.assertGreaterEqual(score.clarity, 3.5)
        self.assertGreaterEqual(score.conciseness, 3.5)
        self.assertGreaterEqual(score.calibrated_score, 4.0)
        self.assertAlmostEqual(score.length_penalty, 1.0, places=2)

    def test_empty_and_irrelevant_responses(self):
        """Verify empty responses receive minimal calibrated scores."""
        empty_score = self.heuristic_judge.evaluate_single(
            prompt="Explain LoRA.",
            response="",
            golden_reference="LoRA uses low-rank matrices.",
        )
        self.assertEqual(empty_score.calibrated_score, 1.0)
        self.assertEqual(empty_score.accuracy, 1.0)

        gibberish = "The quick brown fox jumps over the lazy dog repeatedly again and again."
        gib_score = self.heuristic_judge.evaluate_single(
            prompt="Explain LoRA rank decomposition.",
            response=gibberish,
            golden_reference="LoRA decomposes weight updates into matrices A and B with rank r.",
            required_keywords=["rank", "decomposition", "matrices"],
        )
        self.assertLess(gib_score.accuracy, 2.5)

    def test_length_penalty_discounting(self):
        """Verify verbose padded responses are penalized by exponential length decay."""
        gold = "LoRA decomposes weight updates W = W_0 + (alpha/r)*B*A."
        concise_resp = "LoRA decomposes parameter updates into low-rank matrices B and A scaled by alpha / r."
        # Overly verbose response padded with fluffy filler
        verbose_resp = (
            "In regards to the question about LoRA, it is truly essential to understand that LoRA "
            "decomposes parameter updates into low-rank matrices B and A scaled by alpha / r. "
            "Furthermore, one should always remember that deep neural networks have many layers, "
            "and machine learning models require careful tuning, optimization, data collection, "
            "monitoring, validation, verification, logging, analysis, infrastructure management, "
            "and continuous alignment across every possible post-training setting imaginable."
        )

        score_concise = self.heuristic_judge.evaluate_single(
            prompt="Explain LoRA.",
            response=concise_resp,
            golden_reference=gold,
        )
        score_verbose = self.heuristic_judge.evaluate_single(
            prompt="Explain LoRA.",
            response=verbose_resp,
            golden_reference=gold,
        )

        self.assertEqual(score_concise.length_penalty, 1.0)
        self.assertLess(score_verbose.length_penalty, 0.90)
        self.assertLess(score_verbose.calibrated_score, score_verbose.raw_composite)

    def test_pairwise_position_swap_debiasing(self):
        """Verify symmetric two-pass position swapping eliminates position bias."""
        # Simulate a biased judge that always favors the first candidate presented
        def biased_first_position_model(prompt: str) -> str:
            # If Candidate 1 is evaluated first, give it 5.0 and candidate 2 2.0
            data = {
                "rationale": "I prefer the first candidate I saw.",
                "candidate_1": {
                    "accuracy": 5.0,
                    "faithfulness": 5.0,
                    "clarity": 5.0,
                    "conciseness": 5.0,
                },
                "candidate_2": {
                    "accuracy": 2.0,
                    "faithfulness": 2.0,
                    "clarity": 2.0,
                    "conciseness": 2.0,
                },
            }
            return json.dumps(data)

        biased_judge = LLMJudge(model_fn=biased_first_position_model)
        result = biased_judge.evaluate_pair(
            prompt="Explain LoRA.",
            response_a="LoRA injects low-rank trainable matrices.",
            response_b="LoRA injects low-rank trainable matrices.",
            golden_reference="LoRA injects low-rank trainable matrices.",
        )

        # In Pass 1 (A, B): Candidate 1 is A -> A wins (5.0 vs 2.0)
        self.assertEqual(result.pass1_winner, "A")
        # In Pass 2 (B, A): Candidate 1 is B -> B wins (5.0 vs 2.0)
        self.assertEqual(result.pass2_winner, "B")
        # Debiased average: Score A = (5.0 + 2.0) / 2 = 3.5, Score B = (2.0 + 5.0) / 2 = 3.5
        self.assertAlmostEqual(result.score_a, result.score_b, places=3)
        self.assertEqual(result.winner, "TIE")
        # Position inconsistency coefficient kappa_pos should be high (> 0.25)
        self.assertGreater(result.position_inconsistency, 0.25)

    def test_pairwise_clear_winner_both_passes(self):
        """Verify superior candidate wins across both presentation orders."""
        def objective_model(prompt: str) -> str:
            # Model objectively evaluates based on content
            if "[Candidate 1]\nHigh quality" in prompt:
                # Pass 1: Cand 1 is A (high), Cand 2 is B (low)
                c1_score, c2_score = 4.8, 2.0
            else:
                # Pass 2: Cand 1 is B (low), Cand 2 is A (high)
                c1_score, c2_score = 2.0, 4.8

            data = {
                "rationale": "Evaluating based on factual depth.",
                "candidate_1": {
                    "accuracy": c1_score,
                    "faithfulness": c1_score,
                    "clarity": c1_score,
                    "conciseness": c1_score,
                },
                "candidate_2": {
                    "accuracy": c2_score,
                    "faithfulness": c2_score,
                    "clarity": c2_score,
                    "conciseness": c2_score,
                },
            }
            return json.dumps(data)

        judge = LLMJudge(model_fn=objective_model)
        result = judge.evaluate_pair(
            prompt="Explain flash attention.",
            response_a="High quality: FlashAttention tiles softmax computation in SRAM blocks.",
            response_b="Low quality: Flash attention is very fast attention.",
            golden_reference="FlashAttention uses IO-aware tiling in SRAM.",
        )

        self.assertEqual(result.winner, "A")
        self.assertEqual(result.pass1_winner, "A")
        self.assertEqual(result.pass2_winner, "A")
        self.assertGreater(result.score_delta, 2.0)
        self.assertAlmostEqual(result.position_inconsistency, 0.0, places=2)

    def test_mock_llm_json_fence_and_regex_fallback(self):
        """Verify robust parsing of markdown code fences and malformed outputs."""
        # 1. Output enclosed in markdown fence
        fenced_output = (
            "Thinking step by step...\n"
            "```json\n"
            "{\n"
            '  "rationale": "Strong response.",\n'
            '  "accuracy": 4.5,\n'
            '  "faithfulness": 4.2,\n'
            '  "clarity": 4.8,\n'
            '  "conciseness": 4.0\n'
            "}\n"
            "```\n"
        )
        judge = LLMJudge(model_fn=lambda p: fenced_output)
        score = judge.evaluate_single("Prompt", "Response")
        self.assertAlmostEqual(score.accuracy, 4.5)
        self.assertAlmostEqual(score.faithfulness, 4.2)
        self.assertAlmostEqual(score.clarity, 4.8)
        self.assertAlmostEqual(score.conciseness, 4.0)

        # 2. Output with broken JSON using regex fallback
        broken_output = (
            "Evaluation:\n"
            "accuracy: 4.2\n"
            "faithfulness: 3.8\n"
            "clarity: 4.0\n"
            "conciseness: 4.5\n"
            "The model answered clearly."
        )
        judge_fallback = LLMJudge(model_fn=lambda p: broken_output)
        score_fb = judge_fallback.evaluate_single("Prompt", "Response")
        self.assertAlmostEqual(score_fb.accuracy, 4.2)
        self.assertAlmostEqual(score_fb.faithfulness, 3.8)
        self.assertAlmostEqual(score_fb.clarity, 4.0)
        self.assertAlmostEqual(score_fb.conciseness, 4.5)

    def test_benchmark_case_pairwise_comparison(self):
        """Verify evaluate_benchmark_case() performs pairwise comparison when baseline is given."""
        case = self.case_map["core_peft_008"]
        candidate_resp = (
            "Prompt tokens must be masked to -100 to prevent computing gradient updates on user "
            "instruction tokens. In PyTorch CrossEntropyLoss, ignore_index=-100 ensures only assistant "
            "response tokens contribute to the loss, preventing prompt memorization and role bleed."
        )
        baseline_resp = "Prompt tokens are masked so we do not train on user questions."

        result = self.heuristic_judge.evaluate_benchmark_case(
            case=case,
            candidate_response=candidate_resp,
            baseline_response=baseline_resp,
        )
        self.assertIsInstance(result, PairwiseJudgeResult)
        self.assertEqual(result.winner, "A")
        self.assertGreater(result.score_a, result.score_b)
        self.assertGreater(result.score_delta, 0.4)

    def test_custom_rubric_weights_normalization(self):
        """Verify custom weights are correctly normalized and applied."""
        custom_weights = {
            "accuracy": 0.50,
            "faithfulness": 0.50,
            "clarity": 0.0,
            "conciseness": 0.0,
        }
        judge = LLMJudge(weights=custom_weights)
        raw, cal = judge.compute_composite_score(
            accuracy=4.0, faithfulness=5.0, clarity=1.0, conciseness=1.0, length_penalty=1.0
        )
        # Expected: 0.5 * 4.0 + 0.5 * 5.0 = 4.5
        self.assertAlmostEqual(raw, 4.5, places=3)
        self.assertAlmostEqual(cal, 4.5, places=3)

    def test_pairwise_tie_margin(self):
        """Verify nearly identical candidate responses result in a TIE."""
        resp_a = "LoRA decomposes updates into matrices B and A."
        resp_b = "LoRA decomposes updates into matrices A and B."
        result = self.heuristic_judge.evaluate_pair(
            prompt="Explain LoRA.",
            response_a=resp_a,
            response_b=resp_b,
            golden_reference="LoRA decomposes weight updates into matrices B and A.",
        )
        self.assertEqual(result.winner, "TIE")
        self.assertLessEqual(abs(result.score_delta), 0.15)

    def test_serialization_to_dict(self):
        """Verify JudgeScore and PairwiseJudgeResult serialize cleanly to dictionaries."""
        score = self.heuristic_judge.evaluate_single("Prompt", "Response")
        score_dict = score.to_dict()
        self.assertIn("accuracy", score_dict)
        self.assertIn("calibrated_score", score_dict)
        self.assertIn("length_penalty", score_dict)

        pair_result = self.heuristic_judge.evaluate_pair("Prompt", "A", "B")
        pair_dict = pair_result.to_dict()
        self.assertIn("winner", pair_dict)
        self.assertIn("score_delta", pair_dict)
        self.assertIn("position_inconsistency", pair_dict)

    def test_malformed_model_exception_triggers_heuristic_fallback(self):
        """Verify runtime exception in model_fn cleanly triggers heuristic fallback."""
        def crashing_model(prompt: str) -> str:
            raise RuntimeError("CUDA out of memory in evaluation sidecar!")

        resilient_judge = LLMJudge(model_fn=crashing_model)
        score = resilient_judge.evaluate_single(
            prompt="Explain LoRA.",
            response="LoRA adapts weights via low-rank decomposition.",
            golden_reference="LoRA adapts weights via low-rank decomposition.",
        )
        self.assertEqual(score.mode, "heuristic")
        self.assertGreaterEqual(score.accuracy, 4.0)


if __name__ == "__main__":
    unittest.main()


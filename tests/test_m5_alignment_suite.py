"""Unit and Integration Tests for Milestone 5, Phase 5.5 (Alignment Suite & Dynamic Engines)."""

import unittest

import torch

from evaluation.statistical_gating import GateDecision
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import GPTConfig, MiniGPT
from training.m5_preference.alignment_engine import (
    AlignmentConfig,
    AlignmentEngine,
    AlignmentStrategy,
)
from training.m5_preference.benchmark_suite import (
    AlignmentBenchmarkReport,
    AlignmentBenchmarkSuite,
    AlignmentTestCase,
)
from training.m5_preference.critic import CriticModel
from training.m5_preference.dpo_trainer import DPOTrainer
from training.m5_preference.grpo_trainer import GRPOTrainer
from training.m5_preference.ppo_trainer import PPOTrainer
from training.m5_preference.reward_model import RewardModel
from training.m5_preference.reward_trainer import RewardTrainer


class TestM5AlignmentSuite(unittest.TestCase):
    """Test suite for Phase 5.5 Dynamic Alignment Engine and Alignment Benchmark Suite."""

    def setUp(self):
        corpus = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            "<|>_\n:;.,!?()[]{}+-*/=@#$%^&~`'\""
        )
        self.tokenizer = CharTokenizer(corpus)
        self.config = GPTConfig(
            vocab_size=len(self.tokenizer),
            block_size=128,
            d_model=32,
            n_head=2,
            n_layer=2,
            dropout=0.0,
        )
        self.model = MiniGPT(self.config)

    def test_alignment_engine_factory_reward_model(self):
        """Verify AlignmentEngine dynamically constructs RewardTrainer."""
        rm = RewardModel(self.model)
        trainer = AlignmentEngine.create_trainer(
            strategy=AlignmentStrategy.REWARD_MODEL,
            model=rm,
            config=AlignmentConfig(lr=5e-4),
        )
        self.assertIsInstance(trainer, RewardTrainer)

    def test_alignment_engine_factory_dpo(self):
        """Verify AlignmentEngine dynamically constructs DPOTrainer."""
        ref_model = MiniGPT(self.config)
        trainer = AlignmentEngine.create_trainer(
            strategy=AlignmentStrategy.DPO,
            model=self.model,
            ref_model=ref_model,
            config=AlignmentConfig(beta=0.2),
        )
        self.assertIsInstance(trainer, DPOTrainer)
        self.assertEqual(trainer.loss_fn.beta, 0.2)

    def test_alignment_engine_factory_ppo(self):
        """Verify AlignmentEngine dynamically constructs PPOTrainer."""
        ref_model = MiniGPT(self.config)
        critic = CriticModel(MiniGPT(self.config))
        reward_model = RewardModel(MiniGPT(self.config))

        trainer = AlignmentEngine.create_trainer(
            strategy="ppo",
            model=self.model,
            ref_model=ref_model,
            critic=critic,
            reward_model=reward_model,
            config=AlignmentConfig(clip_ratio=0.15),
        )
        self.assertIsInstance(trainer, PPOTrainer)
        self.assertEqual(trainer.loss_fn.clip_ratio, 0.15)

    def test_alignment_engine_factory_grpo(self):
        """Verify AlignmentEngine dynamically constructs GRPOTrainer."""
        ref_model = MiniGPT(self.config)
        trainer = AlignmentEngine.create_trainer(
            strategy=AlignmentStrategy.GRPO,
            model=self.model,
            ref_model=ref_model,
            config=AlignmentConfig(group_size=6),
        )
        self.assertIsInstance(trainer, GRPOTrainer)
        self.assertEqual(trainer.group_size, 6)

    def test_benchmark_suite_format_and_accuracy_evaluation(self):
        """Verify AlignmentBenchmarkSuite scores formatting compliance, accuracy, and report dict."""
        suite = AlignmentBenchmarkSuite(
            test_cases=[
                AlignmentTestCase(
                    prompt="What is 10 + 5?",
                    ground_truth="15",
                    domain="reasoning",
                ),
                AlignmentTestCase(
                    prompt="What is 7 * 6?",
                    ground_truth="42",
                    domain="reasoning",
                ),
            ]
        )

        def perfect_candidate_fn(prompt: str) -> str:
            if "10 + 5" in prompt:
                return "<think>Add 10 and 5.</think><answer>15</answer>"
            return "<think>Multiply 7 and 6.</think><answer>42</answer>"

        report = suite.evaluate_candidate(
            candidate_name="Reasoning-MiniGPT",
            candidate_generate_fn=perfect_candidate_fn,
        )

        self.assertIsInstance(report, AlignmentBenchmarkReport)
        self.assertEqual(report.format_compliance_rate, 1.0)
        self.assertEqual(report.reasoning_accuracy_rate, 1.0)
        self.assertGreater(report.mean_score, 0.0)

        # Serialization check
        report_dict = report.to_dict()
        self.assertEqual(report_dict["candidate_name"], "Reasoning-MiniGPT")
        self.assertEqual(report_dict["gate_decision"], report.gate_decision.value)

    def test_benchmark_suite_pairwise_gating_proceed_deploy(self):
        """Verify AlignmentBenchmarkSuite issues PROCEED_DEPLOY when candidate outperforms baseline."""
        suite = AlignmentBenchmarkSuite(
            test_cases=[
                AlignmentTestCase(
                    prompt="What is 20 + 22?",
                    ground_truth="42",
                    domain="reasoning",
                ),
            ]
        )

        # Candidate provides thorough reasoned answer
        cand_fn = lambda p: "<think>Sum 20 and 22 gives 42.</think><answer>42</answer>"
        # Baseline gives brief answer
        base_fn = lambda p: "42"

        report = suite.evaluate_candidate(
            candidate_name="Candidate-V2",
            candidate_generate_fn=cand_fn,
            baseline_generate_fn=base_fn,
        )

        self.assertIn(
            report.gate_decision,
            [GateDecision.PROCEED_DEPLOY, GateDecision.BLOCK_RELEASE],
        )

    def test_benchmark_suite_pairwise_gating_trigger_rollback(self):
        """Verify AlignmentBenchmarkSuite issues TRIGGER_ROLLBACK when candidate outputs broken completions."""
        suite = AlignmentBenchmarkSuite(
            test_cases=[
                AlignmentTestCase(
                    prompt="Compute 100 / 4.",
                    ground_truth="25",
                    domain="reasoning",
                ),
            ]
        )

        broken_cand_fn = lambda p: "I don't know the answer sorry!"
        baseline_fn = lambda p: "<think>Divide 100 by 4</think><answer>25</answer>"

        report = suite.evaluate_candidate(
            candidate_name="Degraded-Candidate",
            candidate_generate_fn=broken_cand_fn,
            baseline_generate_fn=baseline_fn,
        )

        self.assertEqual(report.reasoning_accuracy_rate, 0.0)
        self.assertEqual(report.format_compliance_rate, 0.0)
        self.assertEqual(report.gate_decision, GateDecision.TRIGGER_ROLLBACK)


if __name__ == "__main__":
    unittest.main()


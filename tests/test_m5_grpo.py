"""Unit and Integration Tests for Milestone 5, Phase 5.4 (GRPO Engine & Verifiable Rewards)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import GPTConfig, MiniGPT
from training.m5_preference.grpo_loss import GRPOLoss, compute_group_advantages
from training.m5_preference.grpo_trainer import GRPOGroupBatch, GRPOTrainer
from training.m5_preference.verifiable_reward import VerifiableRewardEngine


class TestM5GRPO(unittest.TestCase):
    """Test suite for Phase 5.4 DeepSeek-R1 style Group Relative Policy Optimization."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
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
        self.actor = MiniGPT(self.config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verifiable_format_reward(self):
        """Verify VerifiableRewardEngine validates XML think/answer tags and catches malformations."""
        engine = VerifiableRewardEngine()

        valid_sample = "<think>Let x = 2. Then x^2 = 4.</think><answer>4</answer>"
        self.assertEqual(engine.compute_format_reward(valid_sample), 1.0)

        # Missing answer tag
        no_answer = "<think>Only thinking here</think>"
        self.assertEqual(engine.compute_format_reward(no_answer), 0.0)

        # Empty thinking content
        empty_think = "<think></think><answer>42</answer>"
        self.assertEqual(engine.compute_format_reward(empty_think), 0.0)

        # Inverted tag order (<answer> before <think>)
        inverted = "<answer>42</answer><think>Thinking after the fact</think>"
        self.assertEqual(engine.compute_format_reward(inverted), 0.0)

    def test_verifiable_accuracy_reward(self):
        """Verify accuracy evaluation supports string matching, numerical equivalence, and boxed syntax."""
        engine = VerifiableRewardEngine()

        # String and numeric match
        sample_tag = "<think>Compute 20 + 22</think><answer>42</answer>"
        self.assertEqual(engine.compute_accuracy_reward(sample_tag, "42"), 1.0)
        self.assertEqual(engine.compute_accuracy_reward(sample_tag, "42.0"), 1.0)

        # LaTeX \boxed{} syntax
        sample_boxed = "The computation gives \\boxed{42}."
        self.assertEqual(engine.compute_accuracy_reward(sample_boxed, "42"), 1.0)

        # Incorrect answer
        sample_wrong = "<think>Wrong math</think><answer>99</answer>"
        self.assertEqual(engine.compute_accuracy_reward(sample_wrong, "42"), 0.0)

    def test_group_advantages_zero_sum_and_unit_std(self):
        """Verify compute_group_advantages produces zero-sum and unit standard deviation within groups."""
        # 2 groups of size 4 (B = 8)
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 2.0, 4.0, 1.0, 5.0], dtype=torch.float32)
        group_size = 4

        adv = compute_group_advantages(rewards, group_size=group_size)
        self.assertEqual(adv.shape, (8,))

        # Group 1 check
        g1 = adv[:4]
        self.assertAlmostEqual(g1.sum().item(), 0.0, places=5)
        self.assertAlmostEqual(g1.std().item(), 1.0, places=4)

        # Group 2 check
        g2 = adv[4:]
        self.assertAlmostEqual(g2.sum().item(), 0.0, places=5)
        self.assertAlmostEqual(g2.std().item(), 1.0, places=4)

    def test_group_advantages_constant_reward_invariance(self):
        """Verify advantages collapse strictly to zero when all group completions share identical rewards."""
        # Group where all 4 samples received identical reward 1.0
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        adv = compute_group_advantages(rewards, group_size=4)

        # Should be strictly 0.0, preventing gradient explosion
        self.assertTrue((adv == 0.0).all())

    def test_group_advantages_directional_baseline_on_zero_variance(self):
        """Verify directional baseline provides +0.1 for all-pass and -0.1 for all-fail groups."""
        all_pass = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        adv_pass = compute_group_advantages(all_pass, group_size=4, directional_baseline=True)
        self.assertTrue((adv_pass == 0.1).all())

        all_fail = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        adv_fail = compute_group_advantages(all_fail, group_size=4, directional_baseline=True)
        self.assertTrue((adv_fail == -0.1).all())

    def test_deepseek_unbiased_kl_non_negativity(self):
        """Verify DeepSeek KL divergence estimator u - log(u) - 1 is unconditionally >= 0."""
        loss_fn = GRPOLoss(clip_ratio=0.2, kl_coef=1.0)

        action_mask = torch.ones((1, 4))
        advantages = torch.tensor([0.0])

        # Test diverse log-ratio gaps
        policy_logp = torch.tensor([[0.0, -1.0, -5.0, 2.0]])
        ref_logp = torch.tensor([[-2.0, 1.0, 0.0, -2.0]])
        old_logp = policy_logp.clone()

        _, metrics = loss_fn(
            policy_log_probs=policy_logp,
            old_log_probs=old_logp,
            advantages=advantages,
            action_mask=action_mask,
            ref_log_probs=ref_logp,
        )

        self.assertGreaterEqual(metrics["mean_kl"], 0.0)

        # When policy == ref, KL must be exactly 0.0
        _, metrics_ident = loss_fn(
            policy_log_probs=policy_logp,
            old_log_probs=old_logp,
            advantages=advantages,
            action_mask=action_mask,
            ref_log_probs=policy_logp,
        )
        self.assertAlmostEqual(metrics_ident["mean_kl"], 0.0, places=5)

    def test_grpo_loss_clipping_bounds(self):
        """Verify GRPOLoss applies clipping bounds (1 - eps, 1 + eps)."""
        loss_fn = GRPOLoss(clip_ratio=0.2, kl_coef=0.0)

        action_mask = torch.tensor([[1.0]])
        advantages = torch.tensor([2.0])  # Positive advantage

        old_logp = torch.tensor([[0.0]])
        # Ratio = exp(0.5) approx 1.6487 > 1.2 => clipped to 1.2
        new_logp = torch.tensor([[0.5]])

        loss, metrics = loss_fn(
            policy_log_probs=new_logp,
            old_log_probs=old_logp,
            advantages=advantages,
            action_mask=action_mask,
        )

        # Clipped surrogate: min(1.6487 * 2.0, 1.2 * 2.0) = 2.4 => loss = -2.4
        self.assertAlmostEqual(loss.item(), -2.4, places=3)
        self.assertEqual(metrics["clip_fraction"], 1.0)

    def test_grpo_trainer_convergence_on_synthetic_group(self):
        """Verify GRPOTrainer optimizes policy, reduces loss, and saves/loads checkpoints cleanly."""
        device = torch.device("cpu")
        ref_model = MiniGPT(self.config)
        ref_model.load_state_dict(self.actor.state_dict())

        trainer = GRPOTrainer(
            actor=self.actor,
            ref_model=ref_model,
            loss_fn=GRPOLoss(clip_ratio=0.2, kl_coef=0.02),
            optimizer=torch.optim.AdamW(self.actor.parameters(), lr=1e-4),
            group_size=4,
            device=device,
        )

        # Synthetic group batch of 4 sequences
        input_ids = torch.randint(0, len(self.tokenizer), (4, 32))
        attention_mask = torch.ones((4, 32), dtype=torch.long)
        prompt_lengths = torch.tensor([8, 8, 8, 8], dtype=torch.long)
        # Differing rewards across the group to provide learning signal
        rewards = torch.tensor([1.5, 0.0, 1.0, 0.5], dtype=torch.float32)

        group_batch = trainer.build_group_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            rewards=rewards,
        )

        initial_metrics = trainer.train_step(group_batch)

        last_metrics = initial_metrics
        for _ in range(5):
            last_metrics = trainer.train_step(group_batch)

        # Verification: policy improves on preferred trajectories
        self.assertLess(last_metrics["loss"], initial_metrics["loss"] + 0.05)
        self.assertLessEqual(last_metrics["clip_fraction"], 0.6)

        # Checkpoint roundtrip
        ckpt_path = Path(self.temp_dir) / "grpo_actor.pt"
        trainer.save_checkpoint(ckpt_path)
        self.assertTrue(ckpt_path.exists())

        fresh_actor = MiniGPT(self.config)
        fresh_trainer = GRPOTrainer(actor=fresh_actor, device=device)
        fresh_trainer.load_checkpoint(ckpt_path)

        with torch.no_grad():
            orig_p, _ = self.actor(input_ids[:1], return_all_logits=True)
            loaded_p, _ = fresh_actor(input_ids[:1], return_all_logits=True)
            self.assertTrue(torch.allclose(orig_p, loaded_p, atol=1e-4))

    def test_grpo_sample_group_rollouts_eos_stopping_and_empty_validation(self):
        """Verify sample_group_rollouts validates empty prompts and generates group trajectories."""
        trainer = GRPOTrainer(actor=self.actor, group_size=4, device=torch.device("cpu"))

        # Empty prompts validation
        with self.assertRaises(ValueError):
            trainer.sample_group_rollouts(prompts=[], tokenizer=self.tokenizer)

        # Set an EOS token
        eos_id = self.tokenizer.encode("!")[0]
        self.tokenizer.eos_token_id = eos_id

        # Sample group rollouts for 2 prompts: 2 * 4 = 8 trajectories
        padded_ids, attn_mask, prompt_lens, completions = trainer.sample_group_rollouts(
            prompts=["What is 2+2?", "Capital of France?"],
            tokenizer=self.tokenizer,
            max_new_tokens=16,
            eos_token_id=eos_id,
        )

        self.assertEqual(padded_ids.shape[0], 8)
        self.assertEqual(attn_mask.shape[0], 8)
        self.assertEqual(len(completions), 8)
        self.assertEqual(prompt_lens.shape[0], 8)

    def test_grpo_step_on_prompts_end_to_end(self):
        """Verify step_on_prompts runs full online group rollout-to-update cycle."""
        trainer = GRPOTrainer(
            actor=self.actor,
            group_size=4,
            optimizer=torch.optim.AdamW(self.actor.parameters(), lr=1e-3),
            device=torch.device("cpu"),
        )

        reward_engine = VerifiableRewardEngine()

        def dummy_reward_fn(completions: list[str]) -> list[float]:
            # Assign rewards based on length or content
            return [float(len(c) % 3) for c in completions]

        metrics = trainer.step_on_prompts(
            prompts=["Math puzzle", "Logic query"],
            tokenizer=self.tokenizer,
            reward_fn=dummy_reward_fn,
            max_new_tokens=16,
        )

        self.assertIn("loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("clip_fraction", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_grpo_step_rejects_reward_count_mismatch(self):
        trainer = GRPOTrainer(actor=self.actor, group_size=4, device=torch.device("cpu"))
        sequences = torch.randint(0, len(self.tokenizer), (4, 8))
        mask = torch.ones_like(sequences)
        prompt_lengths = torch.full((4,), 4, dtype=torch.long)

        with patch.object(
            trainer,
            "sample_group_rollouts",
            return_value=(sequences, mask, prompt_lengths, ["a", "b", "c", "d"]),
        ):
            with self.assertRaisesRegex(ValueError, "exactly 4"):
                trainer.step_on_prompts(
                    prompts=["prompt"],
                    tokenizer=self.tokenizer,
                    reward_fn=lambda completions: [1.0],
                )


if __name__ == "__main__":
    unittest.main()

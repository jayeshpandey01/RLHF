"""Unit and Integration Tests for Milestone 5, Phase 5.3 (PPO Actor-Critic Engine & GAE)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import GPTConfig, MiniGPT
from training.m5_preference.critic import CriticModel, ValueHead
from training.m5_preference.dataset import (
    PreferenceCollator,
    create_default_preference_pairs,
)
from training.m5_preference.gae import (
    PPORolloutBatch,
    PPORolloutBuffer,
    compute_gae_advantages,
)
from training.m5_preference.ppo_loss import PPOLoss
from training.m5_preference.ppo_trainer import (
    PPOTrainer,
    compute_token_log_probs_and_entropy,
)
from training.m5_preference.reward_model import RewardModel


class TestM5PPO(unittest.TestCase):
    """Test suite for Phase 5.3 PPO Actor-Critic, GAE, and Rollout Buffers."""

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
        self.critic = CriticModel(MiniGPT(self.config))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_critic_model_forward_shape_and_masking(self):
        """Verify CriticModel outputs per-token state values of shape (B, T) with mask zeroing."""
        input_ids = torch.randint(0, len(self.tokenizer), (2, 32))
        attention_mask = torch.ones((2, 32), dtype=torch.long)
        attention_mask[:, 24:] = 0  # Last 8 tokens are padding

        values = self.critic(input_ids, attention_mask)

        self.assertEqual(values.shape, (2, 32))
        # Active positions should have non-zero or finite values
        self.assertTrue(torch.isfinite(values[:, :24]).all())
        # Padding positions should be explicitly zeroed out
        self.assertTrue((values[:, 24:] == 0.0).all())

    def test_gae_advantages_analytical_exactness(self):
        """Verify compute_gae_advantages against manually calculated closed-form values."""
        # 1 sequence with T = 3 tokens
        rewards = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
        values = torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float32)
        masks = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)

        gamma = 1.0
        lam = 0.95

        # Manual math:
        # t = 2 (terminal): next_val = 0, delta_2 = 2.0 + 1.0 * 0 - 1.5 = 0.5. A_2 = 0.5
        # t = 1: next_val = 1.5, delta_1 = 0.0 + 1.0 * 1.5 - 1.0 = 0.5. A_1 = 0.5 + 1.0 * 0.95 * 0.5 = 0.975
        # t = 0: next_val = 1.0, delta_0 = 0.0 + 1.0 * 1.0 - 0.5 = 0.5. A_0 = 0.5 + 1.0 * 0.95 * 0.975 = 1.42625
        advantages, returns = compute_gae_advantages(
            rewards=rewards,
            values=values,
            masks=masks,
            gamma=gamma,
            lam=lam,
            normalize=False,
        )

        self.assertAlmostEqual(advantages[0, 2].item(), 0.5, places=4)
        self.assertAlmostEqual(advantages[0, 1].item(), 0.975, places=4)
        self.assertAlmostEqual(advantages[0, 0].item(), 1.42625, places=4)

        # Returns = Advantages + Values
        self.assertAlmostEqual(returns[0, 2].item(), 0.5 + 1.5, places=4)
        self.assertAlmostEqual(returns[0, 1].item(), 0.975 + 1.0, places=4)
        self.assertAlmostEqual(returns[0, 0].item(), 1.42625 + 0.5, places=4)

    def test_gae_boundary_terminal_condition(self):
        """Verify GAE terminal boundary condition: A_T = delta_T = r_T - V_T."""
        rewards = torch.tensor([[1.23]], dtype=torch.float32)
        values = torch.tensor([[0.45]], dtype=torch.float32)
        masks = torch.tensor([[1.0]], dtype=torch.float32)

        adv, ret = compute_gae_advantages(
            rewards, values, masks, gamma=1.0, lam=0.95, normalize=False
        )

        expected_delta = 1.23 - 0.45
        self.assertAlmostEqual(adv[0, 0].item(), expected_delta, places=5)
        self.assertAlmostEqual(ret[0, 0].item(), 1.23, places=5)

    def test_ppo_loss_clipping_bounds(self):
        """Verify PPOLoss activates ratio clipping when policy ratio exceeds 1 + epsilon."""
        loss_fn = PPOLoss(clip_ratio=0.2, clip_value=0.2, vf_coef=0.5, entropy_coef=0.0)

        # Action mask active for 1 token
        action_mask = torch.tensor([[1.0]])
        advantages = torch.tensor([[2.0]])  # Positive advantage
        returns = torch.tensor([[1.0]])
        values = torch.tensor([[1.0]])
        old_values = torch.tensor([[1.0]])

        # Case 1: unclipped ratio = 1.1 (< 1.2)
        # log_ratio = ln(1.1) approx 0.09531
        old_logp = torch.tensor([[0.0]])
        new_logp_unclipped = torch.tensor([[0.09531018]])

        loss_unclipped, metrics_unclipped = loss_fn(
            new_logp_unclipped, old_logp, values, old_values, advantages, returns, action_mask
        )
        # Loss should be -1.1 * 2.0 = -2.2
        self.assertAlmostEqual(loss_unclipped.item(), -2.2, places=3)
        self.assertEqual(metrics_unclipped["clip_fraction"], 0.0)

        # Case 2: heavily shifted ratio = 1.5 (> 1.2 => clipped to 1.2)
        new_logp_clipped = torch.tensor([[0.405465]])  # ln(1.5)
        loss_clipped, metrics_clipped = loss_fn(
            new_logp_clipped, old_logp, values, old_values, advantages, returns, action_mask
        )
        # Clipped surrogate: min(1.5 * 2.0, 1.2 * 2.0) = 2.4 => loss = -2.4
        self.assertAlmostEqual(loss_clipped.item(), -2.4, places=3)
        self.assertEqual(metrics_clipped["clip_fraction"], 1.0)

    def test_critic_value_clipping(self):
        """Verify value function loss clips updates exceeding clip_value threshold."""
        loss_fn = PPOLoss(clip_ratio=0.2, clip_value=0.2, vf_coef=1.0, entropy_coef=0.0)

        action_mask = torch.tensor([[1.0]])
        advantages = torch.tensor([[0.0]])
        returns = torch.tensor([[5.0]])
        old_values = torch.tensor([[1.0]])
        # Values moves up by 2.0 (from 1.0 to 3.0), exceeds clip threshold of 0.2
        new_values = torch.tensor([[3.0]])
        old_logp = torch.tensor([[0.0]])
        new_logp = torch.tensor([[0.0]])

        loss, metrics = loss_fn(
            new_logp, old_logp, new_values, old_values, advantages, returns, action_mask
        )

        # vf1 = (3.0 - 5.0)^2 = 4.0
        # val_clipped = 1.0 + 0.2 = 1.2. vf2 = (1.2 - 5.0)^2 = (-3.8)^2 = 14.44
        # max(4.0, 14.44) * 0.5 = 7.22
        self.assertAlmostEqual(metrics["vf_loss"], 7.22, places=3)

    def test_token_log_probs_and_entropy_exactness(self):
        """Verify compute_token_log_probs_and_entropy produces valid aligned tensors."""
        input_ids = torch.randint(0, len(self.tokenizer), (2, 16))
        log_probs, entropy = compute_token_log_probs_and_entropy(self.actor, input_ids)

        self.assertEqual(log_probs.shape, (2, 16))
        self.assertEqual(entropy.shape, (2, 16))
        # Entropy for categorical distribution must be >= 0
        self.assertTrue((entropy[:, 1:] >= 0.0).all())
        # Log-probs should be <= 0
        self.assertTrue((log_probs[:, 1:] <= 0.0).all())

    def test_rollout_buffer_mini_batching(self):
        """Verify PPORolloutBuffer slices and shuffles mini-batches correctly."""
        buffer = PPORolloutBuffer()

        for _ in range(3):
            batch = PPORolloutBatch(
                input_ids=torch.randint(0, len(self.tokenizer), (4, 16)),
                attention_mask=torch.ones((4, 16), dtype=torch.long),
                action_mask=torch.ones((4, 16), dtype=torch.float32),
                old_log_probs=torch.zeros((4, 16)),
                old_values=torch.zeros((4, 16)),
                advantages=torch.zeros((4, 16)),
                returns=torch.zeros((4, 16)),
            )
            buffer.add(batch)

        self.assertEqual(len(buffer), 12)

        mini_batches = list(buffer.get_mini_batches(mini_batch_size=4))
        self.assertEqual(len(mini_batches), 3)
        self.assertEqual(mini_batches[0].input_ids.shape, (4, 16))

    def test_ppo_trainer_convergence_on_synthetic_rollout(self):
        """Verify PPOTrainer optimizes Actor and Critic, reducing loss over multiple steps."""
        device = torch.device("cpu")
        ref_model = MiniGPT(self.config)
        ref_model.load_state_dict(self.actor.state_dict())
        reward_model = RewardModel(MiniGPT(self.config))

        trainer = PPOTrainer(
            actor=self.actor,
            critic=self.critic,
            ref_model=ref_model,
            reward_model=reward_model,
            loss_fn=PPOLoss(clip_ratio=0.2, clip_value=0.2, vf_coef=0.5, entropy_coef=0.01),
            actor_optimizer=torch.optim.AdamW(self.actor.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.AdamW(self.critic.parameters(), lr=2e-3),
            device=device,
        )

        collator = PreferenceCollator(self.tokenizer, max_seq_len=64)
        pairs = create_default_preference_pairs()[:2]
        pref_batch = collator(pairs)

        rollout_batch = trainer.build_rollout_batch(
            input_ids=pref_batch.chosen_input_ids,
            attention_mask=pref_batch.chosen_attention_mask,
            prompt_lengths=pref_batch.prompt_lengths,
        )

        initial_metrics = trainer.train_step(rollout_batch)

        last_metrics = initial_metrics
        for _ in range(15):
            last_metrics = trainer.train_step(rollout_batch)

        # Verifications
        self.assertLess(last_metrics["vf_loss"], initial_metrics["vf_loss"] + 0.1)
        self.assertLessEqual(last_metrics["clip_fraction"], 0.6)

        # Checkpoint roundtrip
        ckpt_path = Path(self.temp_dir) / "ppo_actor_critic.pt"
        trainer.save_checkpoint(ckpt_path)
        self.assertTrue(ckpt_path.exists())

        fresh_actor = MiniGPT(self.config)
        fresh_critic = CriticModel(MiniGPT(self.config))
        fresh_trainer = PPOTrainer(actor=fresh_actor, critic=fresh_critic, device=device)
        fresh_trainer.load_checkpoint(ckpt_path)

        with torch.no_grad():
            v1 = self.critic(pref_batch.chosen_input_ids[:1], pref_batch.chosen_attention_mask[:1])
            v2 = fresh_critic(pref_batch.chosen_input_ids[:1], pref_batch.chosen_attention_mask[:1])
            self.assertTrue(torch.allclose(v1, v2, atol=1e-4))

    def test_critic_temporal_alignment_shift_contract(self):
        """Verify CriticModel right-shift contract: V(s_t) = V(x, y_{<t}).

        Position 0 is identically 0.0 (unconditioned initial state).
        Position t >= 1 strictly reflects raw projection from position t-1.
        Padding tokens are zeroed out by attention mask.
        """
        input_ids = torch.randint(0, len(self.tokenizer), (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        attention_mask[:, 12:] = 0  # 4 padding tokens

        # Raw projection without right shift
        hidden = self.critic.extract_hidden_states(input_ids)
        raw_values = self.critic.value_head(hidden)

        shifted_values = self.critic(input_ids, attention_mask)

        # 1. Position 0 must be exactly 0.0
        self.assertTrue((shifted_values[:, 0] == 0.0).all())

        # 2. Position t in [1, 11] matches raw_values[:, t-1]
        self.assertTrue(torch.allclose(shifted_values[:, 1:12], raw_values[:, :11], atol=1e-5))

        # 3. Padding positions [12, 15] must be zero
        self.assertTrue((shifted_values[:, 12:] == 0.0).all())

    def test_ppo_mode_preservation_in_build_rollout_batch(self):
        """Verify build_rollout_batch preserves caller's train/eval mode for actor and critic."""
        trainer = PPOTrainer(actor=self.actor, critic=self.critic, device=torch.device("cpu"))
        input_ids = torch.randint(0, len(self.tokenizer), (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        prompt_lens = torch.tensor([8, 8], dtype=torch.long)

        # Case 1: Start in eval mode
        self.actor.eval()
        self.critic.eval()
        trainer.build_rollout_batch(input_ids, attention_mask, prompt_lens)
        self.assertFalse(self.actor.training)
        self.assertFalse(self.critic.training)

        # Case 2: Start in train mode
        self.actor.train()
        self.critic.train()
        trainer.build_rollout_batch(input_ids, attention_mask, prompt_lens)
        self.assertTrue(self.actor.training)
        self.assertTrue(self.critic.training)

    def test_ppo_sample_rollouts_eos_stopping_and_empty_validation(self):
        """Verify sample_rollouts validates empty prompts and terminates on EOS."""
        trainer = PPOTrainer(actor=self.actor, critic=self.critic, device=torch.device("cpu"))

        # Empty prompts validation
        with self.assertRaises(ValueError):
            trainer.sample_rollouts(prompts=[], tokenizer=self.tokenizer)

        # Set an EOS token
        eos_id = self.tokenizer.encode("!")[0]
        self.tokenizer.eos_token_id = eos_id

        # Generate rollout
        padded_ids, attn_mask, prompt_lens, completions = trainer.sample_rollouts(
            prompts=["Test prompt"],
            tokenizer=self.tokenizer,
            max_new_tokens=20,
            eos_token_id=eos_id,
        )

        self.assertEqual(padded_ids.shape[0], 1)
        self.assertEqual(attn_mask.shape[0], 1)
        self.assertEqual(len(completions), 1)

    def test_ppo_step_on_prompts_end_to_end(self):
        """Verify step_on_prompts runs full online rollout-to-update cycle."""
        trainer = PPOTrainer(
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=torch.optim.AdamW(self.actor.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.AdamW(self.critic.parameters(), lr=1e-3),
            device=torch.device("cpu"),
        )

        metrics = trainer.step_on_prompts(
            prompts=["Hello world", "What is RLHF?"],
            tokenizer=self.tokenizer,
            reward_fn=lambda completions: [1.0, -0.5],
            max_new_tokens=16,
        )

        self.assertIn("loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("vf_loss", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_ppo_external_reward_preserves_kl_shaping(self):
        """Verify custom terminal rewards are added without discarding token-level KL rewards."""
        ref_model = MiniGPT(self.config)
        ref_model.load_state_dict(self.actor.state_dict())
        with torch.no_grad():
            for parameter in ref_model.parameters():
                parameter.add_(0.01)

        trainer = PPOTrainer(
            actor=self.actor,
            critic=self.critic,
            ref_model=ref_model,
            kl_coef=0.2,
            device=torch.device("cpu"),
        )
        input_ids = torch.randint(0, len(self.tokenizer), (1, 5))
        attention_mask = torch.ones((1, 5), dtype=torch.long)
        prompt_lengths = torch.tensor([2], dtype=torch.long)
        action_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

        self.actor.eval()
        ref_model.eval()
        old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
        ref_log_probs, _ = compute_token_log_probs_and_entropy(ref_model, input_ids)
        expected_rewards = -trainer.kl_coef * (old_log_probs - ref_log_probs) * action_mask
        expected_rewards[0, 4] += 2.0
        captured = {}

        def capture_rewards(rewards, values, masks, gamma, lam, normalize):
            captured["rewards"] = rewards.clone()
            zeros = torch.zeros_like(rewards)
            return zeros, zeros

        with patch(
            "training.m5_preference.ppo_trainer.compute_gae_advantages",
            side_effect=capture_rewards,
        ):
            trainer.build_rollout_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
                terminal_rewards=torch.tensor([2.0]),
            )

        self.assertTrue(torch.allclose(captured["rewards"], expected_rewards, atol=1e-5))

    def test_ppo_step_rejects_reward_count_mismatch(self):
        trainer = PPOTrainer(actor=self.actor, critic=self.critic, device=torch.device("cpu"))
        sequences = torch.randint(0, len(self.tokenizer), (2, 8))
        mask = torch.ones_like(sequences)
        prompt_lengths = torch.tensor([4, 4])

        with patch.object(
            trainer,
            "sample_rollouts",
            return_value=(sequences, mask, prompt_lengths, ["first", "second"]),
        ):
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                trainer.step_on_prompts(
                    prompts=["a", "b"],
                    tokenizer=self.tokenizer,
                    reward_fn=lambda completions: [1.0],
                )


if __name__ == "__main__":
    unittest.main()

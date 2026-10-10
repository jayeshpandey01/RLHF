"""Reproduction: Active Dropout During Rollout Log-Probability Estimation Gap.

Paper reference:
- Schulman et al. (2017) Proximal Policy Optimization Algorithms (arXiv:1707.06347)
- Shao et al. (2024) DeepSeekMath (arXiv:2402.03300)

Theoretical Principle:
In PPO and GRPO, the policy optimization step uses importance sampling:
    rho_t(theta) = pi_theta(a_t | s_t) / pi_{theta_old}(a_t | s_t)
At the start of the update epoch (step 0), before any gradient step has occurred,
theta == theta_old. Therefore, by mathematical definition:
    rho_t = 1.0  and  log(rho_t) = 0.0 everywhere.

Code Discrepancy:
In training/m5_preference/ppo_trainer.py (PPOTrainer.build_rollout_batch) and
training/m5_preference/grpo_trainer.py (GRPOTrainer.build_group_batch):
Neither method sets `self.actor.eval()` before calling `compute_token_log_probs_and_entropy`.
MiniGPT defaults to active dropout (nn.Dropout in MLP, CausalMultiHeadAttention, and embeddings).
In PyTorch, `@torch.no_grad()` does NOT disable dropout; only `.eval()` disables dropout.
Consequently, old_log_probs are computed with one random dropout mask, and new_log_probs
are computed with a different dropout mask.
This causes rho_t != 1.0 at step 0, triggering PPO clipping prematurely and corrupting
the policy gradient with random dropout noise.
"""

import sys
import unittest
from pathlib import Path
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from training.m1_transformer.config import GPTConfig
from training.m1_transformer.model import MiniGPT
from training.m5_preference.ppo_trainer import compute_token_log_probs_and_entropy


class TestDropoutRolloutRatioGap(unittest.TestCase):
    """Verify and prove importance sampling ratio corruption due to active dropout."""

    def setUp(self):
        self.config = GPTConfig(
            vocab_size=100,
            block_size=32,
            n_layer=2,
            n_head=2,
            d_model=32,
            dropout=0.1,  # Active dropout as configured in MiniGPT
        )
        self.actor = MiniGPT(self.config)
        self.input_ids = torch.randint(0, 100, (4, 16))

    def test_train_mode_causes_non_unit_ratio_at_step_zero(self):
        """Show that without actor.eval(), ratio at step 0 deviates significantly from 1.0."""
        # Simulate build_rollout_batch execution in train mode
        self.actor.train()
        with torch.no_grad():
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, self.input_ids)

        # Simulate train_step execution in train mode (before any optimizer step)
        self.actor.train()
        new_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, self.input_ids)

        ratio = torch.exp(new_log_probs - old_log_probs)
        min_r = ratio.min().item()
        max_r = ratio.max().item()
        mean_r = ratio.mean().item()

        print(f"\n[Dropout Gap Verification] Step 0 ratio with active dropout (train mode):")
        print(f"  Min ratio: {min_r:.4f}")
        print(f"  Max ratio: {max_r:.4f}")
        print(f"  Mean ratio: {mean_r:.4f}")

        # In train mode, ratio deviates from 1.0 by up to 10-15%:
        max_deviation = max(abs(min_r - 1.0), abs(max_r - 1.0))
        self.assertGreater(
            max_deviation,
            0.02,
            f"Step 0 ratio should deviate due to dropout, got max deviation {max_deviation}",
        )

    def test_eval_mode_restores_exact_unit_ratio_at_step_zero(self):
        """Show that with actor.eval() during rollout, ratio is exactly 1.0 when deterministic."""
        self.actor.eval()
        with torch.no_grad():
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, self.input_ids)
            new_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, self.input_ids)

        ratio = torch.exp(new_log_probs - old_log_probs)
        min_r = ratio.min().item()
        max_r = ratio.max().item()

        print(f"[Dropout Gap Verification] Step 0 ratio in eval mode:")
        print(f"  Min ratio: {min_r:.4f}")
        print(f"  Max ratio: {max_r:.4f}")

        self.assertAlmostEqual(min_r, 1.0, places=5)
        self.assertAlmostEqual(max_r, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()


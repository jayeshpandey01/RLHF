"""Reproduction: PPO Critic Off-By-One & Action Information Leakage Gap.

Paper reference:
- Schulman et al. (2017) Proximal Policy Optimization Algorithms (arXiv:1707.06347)
- Schulman et al. (2015) High-Dimensional Continuous Control Using Generalized Advantage Estimation (arXiv:1506.02438)

Principle:
In autoregressive text generation, the state at step t is s_t = (x, y_{<t}).
The action taken at step t is a_t = y_t.
The state value V(s_t) = E[ sum gamma^k r_{t+k} | s_t ] must evaluate s_t BEFORE action a_t is taken.
In causal transformers, hidden_states[:, t] has already attended to input_ids[:, t] (the action).
Therefore, predicting values[:, t] directly from hidden_states[:, t] causes action information
to leak into the state value estimate, biasing advantages and breaking policy gradient baseline subtraction.
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
from training.m5_preference.critic import CriticModel


class TestCriticActionLeakageGap(unittest.TestCase):
    """Verify and prove action leakage in the current CriticModel."""

    def setUp(self):
        self.config = GPTConfig(
            vocab_size=100,
            block_size=32,
            n_layer=2,
            n_head=2,
            d_model=32,
            dropout=0.0,
        )
        self.backbone = MiniGPT(self.config)
        self.critic = CriticModel(self.backbone)
        self.critic.eval()

    def test_critic_value_leaks_current_action_token(self):
        """Demonstrate that changing the action token at position t changes values[:, t].

        If values[:, t] is truly V(s_t) = V(x, y_{<t}), it must be invariant to the
        token choice at position t.
        """
        # Prefix tokens [10, 20, 30] represent state s_3 = (x, y_{<3})
        # Sequence A takes action 40 at index 3
        seq_a = torch.tensor([[10, 20, 30, 40]])
        # Sequence B takes action 95 at index 3
        seq_b = torch.tensor([[10, 20, 30, 95]])

        with torch.no_grad():
            h_a = self.critic.extract_hidden_states(seq_a)
            raw_val_a = self.critic.value_head(h_a)
            h_b = self.critic.extract_hidden_states(seq_b)
            raw_val_b = self.critic.value_head(h_b)

        # 1. Past positions (t < 3) must be identical due to causal attention
        for t in range(3):
            self.assertTrue(
                torch.isclose(raw_val_a[0, t], raw_val_b[0, t], atol=1e-6),
                f"Position {t} should be identical in both sequences.",
            )

        # 2. Position 3 should be V(s_3), independent of action at position 3.
        # But in unshifted Critic, raw values[:, 3] differs because it embeds token 3!
        action_leakage_delta = torch.abs(raw_val_a[0, 3] - raw_val_b[0, 3]).item()
        print(f"\n[Critic Gap Verification] Action leakage delta at position 3 in unshifted hidden states: {action_leakage_delta:.6f}")
        print(f"  raw_val_a[0, 3] (action=40): {raw_val_a[0, 3].item():.6f}")
        print(f"  raw_val_b[0, 3] (action=95): {raw_val_b[0, 3].item():.6f}")

        # The unshifted implementation fails this invariance check:
        self.assertGreater(
            action_leakage_delta,
            1e-4,
            "Values at position 3 differ because hidden_states[:, 3] sees token 3 (action leakage).",
        )

    def test_shifted_critic_restores_action_invariance(self):
        """Demonstrate that shifting values by one position restores the required invariance.

        When V(s_t) is taken from hidden_states[:, t - 1], it only depends on (x, y_{<t}).
        """
        seq_a = torch.tensor([[10, 20, 30, 40]])
        seq_b = torch.tensor([[10, 20, 30, 95]])

        with torch.no_grad():
            val_a = self.critic(seq_a)
            val_b = self.critic(seq_b)

        # In our fixed CriticModel, values[:, 3] is V(s_3) = V(x, y_{<3}), independent of token 3
        self.assertTrue(
            torch.isclose(val_a[0, 3], val_b[0, 3], atol=1e-6),
            "Shifted CriticModel value at position 3 is strictly invariant to action token at position 3.",
        )


if __name__ == "__main__":
    unittest.main()

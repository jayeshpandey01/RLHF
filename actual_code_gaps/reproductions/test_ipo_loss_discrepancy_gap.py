"""Reproduction: IPO Loss Target Margin Discrepancy Gap.

Paper reference:
- Azar et al. (Google DeepMind, 2024) "A General Theoretical Paradigm to Understand Learning
  from Human Preferences" (AISTATS 2024 / arXiv:2310.12036).
- Hugging Face TRL DPOTrainer implementation of IPO loss.

Theoretical Formulation:
In Azar et al. (2024), the Identity Preference Optimization (IPO) objective is formulated as:
    L_IPO(theta) = E_{(x, y_w, y_l)} [ ( h_theta(x, y_w, y_l) - 1 / (2 * tau) )^2 ]
where h_theta = log(pi_theta(y_w|x) / pi_ref(y_w|x)) - log(pi_theta(y_l|x) / pi_ref(y_l|x))
is the unscaled log-ratio difference between preferred and rejected completions.
In TRL (with parameter beta):
    h = (pi_chosen_logps - ref_chosen_logps) - (pi_rejected_logps - ref_rejected_logps)
    loss = (h - 1 / (2 * beta)) ** 2
Target margin for h is 1 / (2 * beta).

Code Discrepancy:
In training/m5_preference/dpo_loss.py:
    chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen)
    rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected)
    delta = chosen_rewards - rejected_rewards   # delta = beta * h
    tau = 1.0 / (2.0 * self.beta)
    loss = torch.mean((delta - tau) ** 2)       # (beta * h - 1 / (2 * beta))^2
Expanding this expression:
    (beta * h - 1 / (2 * beta))^2 = beta^2 * (h - 1 / (2 * beta^2))^2
This inadvertently scales the target margin for h by an extra 1/beta factor, pushing the
target margin to 1 / (2 * beta^2) instead of 1 / (2 * beta), and scales the gradient by beta^2.
For beta=0.1, the intended target margin is 5.0, but dpo_loss.py enforces a target margin of 50.0!
"""

import sys
import unittest
from pathlib import Path
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from training.m5_preference.dpo_loss import DPOLoss


class TestIPOLossDiscrepancyGap(unittest.TestCase):
    """Verify and prove the mathematical discrepancy in dpo_loss.py IPO mode."""

    def test_ipo_loss_margin_and_scaling_discrepancy(self):
        beta = 0.1
        loss_fn = DPOLoss(beta=beta, loss_type="ipo")

        # Let unscaled log ratio difference h = 2.0
        # log pi(y_w) - log pi_ref(y_w) = 3.0
        # log pi(y_l) - log pi_ref(y_l) = 1.0
        pi_w = torch.tensor([3.0])
        pi_l = torch.tensor([1.0])
        ref_w = torch.tensor([0.0])
        ref_l = torch.tensor([0.0])

        h = (pi_w - ref_w) - (pi_l - ref_l)  # h = 2.0
        target_margin_paper = 1.0 / (2.0 * beta)  # 5.0

        # Azar et al. / TRL IPO loss
        expected_paper_loss = ((h - target_margin_paper) ** 2).item()  # (2.0 - 5.0)^2 = 9.0

        # Code implementation
        code_loss, metrics = loss_fn(pi_w, pi_l, ref_w, ref_l)
        actual_code_loss = code_loss.item()

        # Target margin implemented by code:
        # delta = tau => beta * h = 1 / (2 * beta) => h = 1 / (2 * beta^2) = 50.0
        target_margin_code = 1.0 / (2.0 * (beta ** 2))

        print(f"\n[IPO Gap Verification] beta={beta}")
        print(f"  Unscaled log-ratio difference h: {h.item():.2f}")
        print(f"  Target margin in Azar et al. / TRL: {target_margin_paper:.2f}")
        print(f"  Target margin in dpo_loss.py:       {target_margin_code:.2f} (10x distortion!)")
        print(f"  Expected paper loss:                {expected_paper_loss:.4f}")
        print(f"  Actual dpo_loss.py loss:            {actual_code_loss:.4f}")

        # The fixed DPOLoss matches the paper formula: (h - 1/(2*beta))^2 = (2.0 - 5.0)^2 = 9.0
        self.assertAlmostEqual(actual_code_loss, expected_paper_loss, places=4)
        print("  [Verified] DPOLoss now matches Azar et al. (2024) and TRL canonical formulation!")

    def test_ipo_loss_optimal_margin_point(self):
        """Demonstrate that code achieves minimum loss at h=5 (matching Azar et al.)."""
        beta = 0.1
        loss_fn = DPOLoss(beta=beta, loss_type="ipo")

        # 1. Test when h = 5.0 (the true optimal point in Azar et al.)
        pi_w_5 = torch.tensor([5.0])
        pi_l_5 = torch.tensor([0.0])
        ref_0 = torch.tensor([0.0])
        loss_at_5, _ = loss_fn(pi_w_5, pi_l_5, ref_0, ref_0)

        # 2. Test when h = 50.0
        pi_w_50 = torch.tensor([50.0])
        pi_l_50 = torch.tensor([0.0])
        loss_at_50, _ = loss_fn(pi_w_50, pi_l_50, ref_0, ref_0)

        print(f"  dpo_loss.py loss at h=5.0 (true target):   {loss_at_5.item():.4f}")
        print(f"  dpo_loss.py loss at h=50.0:                {loss_at_50.item():.4f}")

        # In fixed DPOLoss, loss at h=5 IS zero: (5.0 - 5.0)^2 = 0.0!
        self.assertAlmostEqual(loss_at_5.item(), 0.0, places=4)
        # Loss at h=50 is penalized: (50 - 5)^2 = 2025.0!
        self.assertAlmostEqual(loss_at_50.item(), 2025.0, places=2)


if __name__ == "__main__":
    unittest.main()


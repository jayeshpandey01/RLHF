"""Group Relative Policy Optimization (GRPO) loss criterion and advantage estimator."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
    directional_baseline: bool = False,
) -> torch.Tensor:
    """Compute group-relative standardized advantages across candidate completions.

    For each group of G outputs per query:
        A_i = (R_i - mean(R)) / (std(R) + eps)

    If all candidate completions in a group share identical reward (std < 1e-6),
    advantages for that group collapse to zero (or directional baseline when enabled).

    Args:
        rewards: Sequence-level reward tensor of shape (B,) where B is divisible by group_size.
        group_size: Number of completions per prompt group (G).
        eps: Small epsilon avoiding division by zero.
        directional_baseline: If True, assign directional non-zero baseline (+0.1 if mean>0 else -0.1)
                              to groups with zero variance (DeepSeek-R1 GRPO advantage stabilization).

    Returns:
        Standardized advantage tensor of shape (B,).
    """
    B = rewards.size(0)
    assert B % group_size == 0, (
        f"Batch size {B} must be an integer multiple of group_size {group_size}"
    )
    num_groups = B // group_size

    # Reshape into (num_groups, group_size)
    grouped_rewards = rewards.view(num_groups, group_size)
    mean_r = grouped_rewards.mean(dim=-1, keepdim=True)
    std_r = grouped_rewards.std(dim=-1, keepdim=True)

    # Standardize: handle zero variance groups
    valid_variance = std_r > 1e-6
    if directional_baseline:
        zero_var_advantages = torch.where(
            mean_r > 0.0,
            torch.full_like(grouped_rewards, 0.1),
            torch.full_like(grouped_rewards, -0.1),
        )
        advantages = torch.where(
            valid_variance,
            (grouped_rewards - mean_r) / (std_r + eps),
            zero_var_advantages,
        )
    else:
        advantages = torch.where(
            valid_variance,
            (grouped_rewards - mean_r) / (std_r + eps),
            torch.zeros_like(grouped_rewards),
        )

    return advantages.view(B)


class GRPOLoss(nn.Module):
    """DeepSeek-R1 style Group Relative Policy Optimization (GRPO) loss criterion.

    Formulation:
        L_GRPO = - 1/|o_i| * sum_t [ min(ρ_t * A_i, clip(ρ_t, 1 - ε, 1 + ε) * A_i) - β * D_KL ]
        D_KL(π || π_ref) = u - log(u) - 1,  where u = π_ref / π
    """

    def __init__(
        self,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.04,
    ):
        """Initialize the GRPOLoss criterion.

        Args:
            clip_ratio: Policy probability ratio clipping threshold ε (default 0.2).
            kl_coef: Unbiased KL penalty multiplier β (default 0.04).
        """
        super().__init__()
        assert clip_ratio > 0.0, f"clip_ratio must be positive, got {clip_ratio}"
        assert kl_coef >= 0.0, f"kl_coef must be non-negative, got {kl_coef}"
        self.clip_ratio = float(clip_ratio)
        self.kl_coef = float(kl_coef)

    def forward(
        self,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO loss with DeepSeek unbiased non-negative KL divergence.

        Args:
            policy_log_probs: Log-probabilities under active policy, shape (B, T).
            old_log_probs: Log-probabilities under rollout policy, shape (B, T).
            advantages: Group-relative advantages, shape (B,).
            action_mask: Binary mask for completion actions (1 for active, 0 for prompt/pad), shape (B, T).
            ref_log_probs: Optional log-probabilities under frozen reference policy, shape (B, T).

        Returns:
            (total_loss_tensor, metrics_dict)
        """
        device = policy_log_probs.device
        mask = action_mask.to(device).float()
        B, T = policy_log_probs.size()

        # Broadcast advantages of shape (B,) across sequence length (B, T)
        adv_expanded = advantages.unsqueeze(-1).expand(B, T)

        # Probability ratio ρ_{i,t}
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_expanded
        policy_surrogate = torch.min(surr1, surr2)  # Shape: (B, T)

        # DeepSeek unbiased non-negative KL divergence estimator: u - log(u) - 1
        if ref_log_probs is not None and self.kl_coef > 0.0:
            ref_logp = ref_log_probs.to(device)
            # u = exp(log π_ref - log π_policy)
            log_u = ref_logp - policy_log_probs
            u = torch.exp(log_u)
            kl_div = u - log_u - 1.0  # Unconditionally >= 0
        else:
            kl_div = torch.zeros_like(policy_log_probs)

        # Composite per-token objective: maximize (surrogate - β * KL) => minimize -surrogate + β * KL
        token_loss = -policy_surrogate + self.kl_coef * kl_div

        # Length normalization per sample (average over active completion tokens)
        completion_lengths = mask.sum(dim=-1).clamp(min=1.0)  # Shape: (B,)
        sample_losses = (token_loss * mask).sum(dim=-1) / completion_lengths
        total_loss = sample_losses.mean()

        # Monitoring diagnostics
        with torch.no_grad():
            active_bool = mask.bool()
            if active_bool.any():
                active_ratio = ratio[active_bool]
                clipped_mask = (active_ratio < 1.0 - self.clip_ratio) | (
                    active_ratio > 1.0 + self.clip_ratio
                )
                clip_fraction = clipped_mask.float().mean().item()
                mean_ratio = active_ratio.mean().item()
                mean_kl = (kl_div * mask).sum().item() / mask.sum().clamp(min=1.0).item()
            else:
                clip_fraction = 0.0
                mean_ratio = 1.0
                mean_kl = 0.0

            policy_loss_scalar = -(policy_surrogate * mask).sum().item() / mask.sum().clamp(min=1.0).item()

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss_scalar,
            "kl_loss": mean_kl * self.kl_coef,
            "mean_kl": mean_kl,
            "clip_fraction": clip_fraction,
            "mean_ratio": mean_ratio,
            "mean_advantage": advantages.mean().item(),
        }

        return total_loss, metrics


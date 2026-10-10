"""Proximal Policy Optimization (PPO) clipped surrogate loss criterion."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


class PPOLoss(nn.Module):
    """Numerically stable PPO Actor-Critic loss with clipped policy and value objectives.

    Formulation:
        L_CLIP(θ) = E_t [ min(ρ_t * A_t, clip(ρ_t, 1 - ε, 1 + ε) * A_t) ]
        L_VF(ϕ)   = 0.5 * E_t [ max((V_t - R_t)^2, (clip(V_t, V_old - ε_v, V_old + ε_v) - R_t)^2) ]
        L_PPO     = -L_CLIP(θ) + c_vf * L_VF(ϕ) - c_ent * S[π_θ]
    """

    def __init__(
        self,
        clip_ratio: float = 0.2,
        clip_value: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """Initialize the PPOLoss criterion.

        Args:
            clip_ratio: Policy probability ratio clipping threshold ε (default 0.2).
            clip_value: Value function clipping threshold ε_v (default 0.2).
            vf_coef: Value function loss weighting coefficient c_vf (default 0.5).
            entropy_coef: Policy entropy bonus coefficient c_ent (default 0.01).
        """
        super().__init__()
        assert clip_ratio > 0.0, f"clip_ratio must be positive, got {clip_ratio}"
        assert vf_coef >= 0.0, f"vf_coef must be non-negative, got {vf_coef}"
        self.clip_ratio = float(clip_ratio)
        self.clip_value = float(clip_value)
        self.vf_coef = float(vf_coef)
        self.entropy_coef = float(entropy_coef)

    def forward(
        self,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute composite PPO loss and monitoring metrics.

        Args:
            policy_log_probs: Log-probabilities under current policy, shape (B, T).
            old_log_probs: Log-probabilities under rollout behavior policy, shape (B, T).
            values: State-value predictions under current critic, shape (B, T).
            old_values: State-value predictions under rollout critic, shape (B, T).
            advantages: Normalized GAE-λ advantages, shape (B, T).
            returns: Target returns, shape (B, T).
            action_mask: Binary mask for completion actions (1 for active, 0 for prompt/pad).
            entropy: Optional per-token policy entropy distribution, shape (B, T).

        Returns:
            (total_loss_tensor, metrics_dict)
        """
        device = policy_log_probs.device
        mask = action_mask.to(device).float()
        num_active = mask.sum().clamp(min=1.0)

        # 1. Clipped Surrogate Policy Loss
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)  # ρ_t

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss_per_token = -torch.min(surr1, surr2)
        policy_loss = (policy_loss_per_token * mask).sum() / num_active

        # 2. Clipped Value Function Loss
        if self.clip_value > 0.0:
            val_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_value, self.clip_value
            )
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = (val_clipped - returns) ** 2
            vf_loss_per_token = 0.5 * torch.max(vf_loss1, vf_loss2)
        else:
            vf_loss_per_token = 0.5 * ((values - returns) ** 2)

        vf_loss = (vf_loss_per_token * mask).sum() / num_active

        # 3. Policy Entropy Regularization
        if entropy is not None:
            mean_entropy = (entropy * mask).sum() / num_active
        else:
            mean_entropy = torch.tensor(0.0, device=device)

        # 4. Total Composite PPO Loss
        total_loss = policy_loss + self.vf_coef * vf_loss - self.entropy_coef * mean_entropy

        # 5. Diagnostic Metrics
        with torch.no_grad():
            active_bool = mask.bool()
            if active_bool.any():
                active_ratio = ratio[active_bool]
                clipped_mask = (active_ratio < 1.0 - self.clip_ratio) | (
                    active_ratio > 1.0 + self.clip_ratio
                )
                clip_fraction = clipped_mask.float().mean().item()
                approx_kl = 0.5 * (log_ratio[active_bool] ** 2).mean().item()
                mean_ratio = active_ratio.mean().item()
            else:
                clip_fraction = 0.0
                approx_kl = 0.0
                mean_ratio = 1.0

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy": mean_entropy.item(),
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "mean_ratio": mean_ratio,
        }

        return total_loss, metrics


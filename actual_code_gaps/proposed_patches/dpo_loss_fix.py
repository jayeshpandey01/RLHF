"""Proposed fix for DPOLoss resolving IPO margin distortion and silent fallbacks (GAP-02, GAP-05)."""

from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedDPOLoss(nn.Module):
    """Numerically stable DPO, cDPO, and mathematically correct IPO loss criterion."""

    def __init__(
        self,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "cdpo", "ipo"] = "sigmoid",
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ):
        super().__init__()
        assert beta > 0.0, f"beta must be strictly positive, got {beta}"
        assert 0.0 <= label_smoothing < 0.5, (
            f"label_smoothing must be in [0.0, 0.5), got {label_smoothing}"
        )
        self.beta = float(beta)
        self.loss_type = loss_type
        self.label_smoothing = float(label_smoothing)
        self.reference_free = bool(reference_free)

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = policy_chosen_logps.device

        # Fix GAP-05: Reject silent fallback if reference_free is False but reference tensors are missing
        if not self.reference_free and (reference_chosen_logps is None or reference_rejected_logps is None):
            raise ValueError(
                "DPOLoss was initialized with reference_free=False, but reference log-probabilities "
                "were not provided in forward(). Pass reference log-probs or explicitly set reference_free=True."
            )

        # Compute unscaled log-ratio differences h
        if self.reference_free:
            log_ratios = policy_chosen_logps - policy_rejected_logps
            chosen_rewards = self.beta * policy_chosen_logps
            rejected_rewards = self.beta * policy_rejected_logps
        else:
            ref_chosen = reference_chosen_logps.to(device)
            ref_rejected = reference_rejected_logps.to(device)
            log_ratios = (policy_chosen_logps - ref_chosen) - (policy_rejected_logps - ref_rejected)
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected)

        delta = self.beta * log_ratios

        # Compute loss according to loss_type
        if self.loss_type == "ipo":
            # Fix GAP-02: IPO formulation (Azar et al., Google DeepMind 2024 / TRL)
            # Uses unscaled log-ratios h with target margin 1 / (2 * beta)
            tau = 1.0 / (2.0 * self.beta)
            loss = torch.mean((log_ratios - tau) ** 2)
        elif self.loss_type == "cdpo" or self.label_smoothing > 0.0:
            eps = self.label_smoothing
            loss = torch.mean(
                (1.0 - eps) * F.softplus(-delta) + eps * F.softplus(delta)
            )
        else:
            # Standard Sigmoid DPO: -log sigma(delta) = softplus(-delta)
            loss = torch.mean(F.softplus(-delta))

        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            mean_margin = delta.mean().item()
            mean_chosen_reward = chosen_rewards.mean().item()
            mean_rejected_reward = rejected_rewards.mean().item()

            if reference_chosen_logps is not None and not self.reference_free:
                ref_chosen = reference_chosen_logps.to(device)
                kl_div = (policy_chosen_logps - ref_chosen).mean().item()
            else:
                kl_div = 0.0

        metrics = {
            "loss": loss.item(),
            "chosen_rewards": mean_chosen_reward,
            "rejected_rewards": mean_rejected_reward,
            "mean_margin": mean_margin,
            "accuracy": accuracy,
            "kl_divergence": kl_div,
        }

        return loss, metrics


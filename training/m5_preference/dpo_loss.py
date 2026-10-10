"""Direct Preference Optimization (DPO), cDPO, and IPO loss criteria."""

from __future__ import annotations

from typing import Literal, Optional, Union

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """Numerically stable Direct Preference Optimization (DPO) criterion.

    Supports:
    1. Standard Sigmoid DPO (Rafailov et al., NeurIPS 2023)
    2. Conservative DPO with Label Smoothing (cDPO, Rafailov et al., 2023)
    3. Identity Preference Optimization (IPO, Azar et al., Google DeepMind 2024)
    4. Reference-free DPO mode (SimPO style, Meng et al., 2024)

    Formulation:
        r_w = beta * (log pi(y_w|x) - log pi_ref(y_w|x))
        r_l = beta * (log pi(y_l|x) - log pi_ref(y_l|x))
        delta = r_w - r_l
    """

    def __init__(
        self,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "cdpo", "ipo"] = "sigmoid",
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ):
        """Initialize the DPO loss criterion.

        Args:
            beta: Regularization temperature scaling the implicit reward (default 0.1).
            loss_type: Loss objective type ("sigmoid", "cdpo", or "ipo").
            label_smoothing: Noise tolerance parameter epsilon in [0, 0.5) for cDPO.
            reference_free: If True, ignores reference model log-probabilities.
        """
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
        """Compute DPO loss and comprehensive alignment diagnostics.

        Args:
            policy_chosen_logps: Log-probabilities of chosen completions under policy, shape (B,).
            policy_rejected_logps: Log-probabilities of rejected completions under policy, shape (B,).
            reference_chosen_logps: Log-probabilities of chosen under reference model, shape (B,).
            reference_rejected_logps: Log-probabilities of rejected under reference model, shape (B,).

        Returns:
            (loss_tensor, metrics_dict)
        """
        device = policy_chosen_logps.device

        # Compute implicit rewards and unscaled log-ratios h
        if self.reference_free or reference_chosen_logps is None or reference_rejected_logps is None:
            if not self.reference_free:
                logger.warning(
                    "DPOLoss: reference log-probabilities were not provided with reference_free=False. "
                    "Implicit rewards are running in reference-free mode."
                )
            log_ratios = policy_chosen_logps - policy_rejected_logps
            chosen_rewards = self.beta * policy_chosen_logps
            rejected_rewards = self.beta * policy_rejected_logps
        else:
            ref_chosen = reference_chosen_logps.to(device)
            ref_rejected = reference_rejected_logps.to(device)
            log_ratios = (policy_chosen_logps - ref_chosen) - (policy_rejected_logps - ref_rejected)
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected)

        delta = chosen_rewards - rejected_rewards  # Shape: (B,)

        # Compute loss according to loss_type
        if self.loss_type == "ipo":
            # IPO (Azar et al., Google DeepMind 2024 / TRL):
            # Target margin applies directly to unscaled log-ratio difference h:
            # L_IPO = (h - 1 / (2 * beta))^2
            tau = 1.0 / (2.0 * self.beta)
            loss = torch.mean((log_ratios - tau) ** 2)
        elif self.loss_type == "cdpo" or self.label_smoothing > 0.0:
            # cDPO: (1 - eps) * softplus(-delta) + eps * softplus(delta)
            eps = self.label_smoothing
            loss = torch.mean(
                (1.0 - eps) * F.softplus(-delta) + eps * F.softplus(delta)
            )
        else:
            # Standard Sigmoid DPO: -log sigma(delta) = softplus(-delta)
            loss = torch.mean(F.softplus(-delta))

        # Alignment diagnostics
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


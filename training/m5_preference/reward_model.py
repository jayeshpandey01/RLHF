"""Bradley-Terry Reward Model and Numerically Stable Loss Engine."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.m1_transformer.model import MiniGPT


class RewardModel(nn.Module):
    """Causal Transformer backbone augmented with a linear scalar reward head."""

    def __init__(self, backbone: MiniGPT):
        """Initialize the RewardModel.

        Args:
            backbone: Pre-LayerNorm MiniGPT transformer backbone.
        """
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.config.d_model, 1, bias=False)

        # Initialize reward head weights with small normal values for stable gradient start
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        """Infer active compute device from backbone parameters."""
        return self.backbone.device

    def extract_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone transformer layers to obtain final hidden states.

        Returns:
            Hidden states tensor of shape (B, T, d_model).
        """
        idx = idx.to(self.device)
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)

        tok_emb = self.backbone.transformer.wte(idx)
        pos_emb = self.backbone.transformer.wpe(pos)
        x = self.backbone.transformer.drop(tok_emb + pos_emb)

        for block in self.backbone.transformer.h:
            x, _ = block(x)

        x = self.backbone.transformer.ln_f(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar rewards for each sequence in the batch.

        Extracts hidden representation at the terminal non-padding token
        and projects to a scalar value via the reward head.

        Args:
            input_ids: Token ID tensor of shape (B, T).
            attention_mask: Binary mask of shape (B, T) where 1 is active, 0 is pad.

        Returns:
            Scalar rewards tensor of shape (B,).
        """
        input_ids = input_ids.to(self.device)
        B, T = input_ids.size()
        hidden = self.extract_hidden_states(input_ids)  # Shape: (B, T, d_model)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            # Find the last active non-padded token index for each sequence (supports both right and left padding)
            indices = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
            masked_indices = torch.where(attention_mask.bool(), indices, torch.full_like(indices, -1))
            last_token_idx = masked_indices.max(dim=-1).values.clamp(min=0)  # Shape: (B,)
            batch_idx = torch.arange(B, device=self.device)
            pooled_hidden = hidden[batch_idx, last_token_idx]  # Shape: (B, d_model)
        else:
            pooled_hidden = hidden[:, -1, :]  # Shape: (B, d_model)

        rewards = self.reward_head(pooled_hidden).squeeze(-1)  # Shape: (B,)
        return rewards


class BradleyTerryLoss(nn.Module):
    """Numerically stable margin-regularized Bradley-Terry negative log-likelihood loss.

    Formula:
        loss = softplus(-(r_w - r_l - gamma)) + (lambda_reg / 2) * (r_w^2 + r_l^2)
    """

    def __init__(self, margin: float = 0.0, l2_reg: float = 0.001):
        """Initialize the Bradley-Terry loss.

        Args:
            margin: Desired scalar preference separation threshold (default 0.0).
            l2_reg: L2 regularization coefficient to prevent reward scale drift (default 0.001).
        """
        super().__init__()
        self.margin = float(margin)
        self.l2_reg = float(l2_reg)

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        target_margins: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Bradley-Terry loss and calibration metrics.

        Args:
            chosen_rewards: Scalar rewards for chosen completions, shape (B,).
            rejected_rewards: Scalar rewards for rejected completions, shape (B,).
            target_margins: Optional per-sample margin offsets, shape (B,).

        Returns:
            (loss_tensor, metrics_dict)
        """
        delta_r = chosen_rewards - rejected_rewards

        if target_margins is not None:
            target_margins = target_margins.to(delta_r.device)
            loss_bt = F.softplus(-(delta_r - target_margins)).mean()
        elif self.margin > 0.0:
            loss_bt = F.softplus(-(delta_r - self.margin)).mean()
        else:
            loss_bt = F.softplus(-delta_r).mean()

        # L2 regularization on reward magnitudes
        loss_reg = 0.5 * self.l2_reg * (chosen_rewards.pow(2).mean() + rejected_rewards.pow(2).mean())
        total_loss = loss_bt + loss_reg

        # Compute empirical diagnostics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            mean_margin = delta_r.mean().item()
            r_w_mean = chosen_rewards.mean().item()
            r_l_mean = rejected_rewards.mean().item()

        metrics = {
            "loss": total_loss.item(),
            "loss_bt": loss_bt.item(),
            "loss_reg": loss_reg.item(),
            "accuracy": accuracy,
            "mean_margin": mean_margin,
            "chosen_reward": r_w_mean,
            "rejected_reward": r_l_mean,
        }

        return total_loss, metrics


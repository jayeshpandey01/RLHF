"""Generalized Advantage Estimation (GAE) and PPO Rollout Buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE-λ) and target returns.

    Formulation:
        δ_t = r_t + γ * V(s_{t+1}) * mask_{t+1} - V(s_t)
        A_t = δ_t + γ * λ * A_{t+1} * mask_{t+1}
        R_t = A_t + V(s_t)

    Args:
        rewards: Immediate rewards at each token step, shape (B, T).
        values: Critic state-value predictions, shape (B, T).
        masks: Binary mask (1 for active completion tokens, 0 for prompt or pad), shape (B, T).
        gamma: Discount factor γ (default 1.0 for episodic text generation).
        lam: GAE bias-variance trade-off parameter λ (default 0.95).
        normalize: Whether to normalize advantages to zero-mean and unit-variance across active tokens.

    Returns:
        (advantages, target_returns) each of shape (B, T).
    """
    B, T = rewards.size()
    device = rewards.device

    advantages = torch.zeros((B, T), dtype=torch.float32, device=device)
    last_gae = torch.zeros(B, dtype=torch.float32, device=device)

    # Backward temporal recursion from terminal step T - 1 down to 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_values = torch.zeros(B, dtype=torch.float32, device=device)
            next_masks = torch.zeros(B, dtype=torch.float32, device=device)
        else:
            next_values = values[:, t + 1]
            next_masks = masks[:, t + 1]

        # TD residual δ_t
        delta = rewards[:, t] + gamma * next_values * next_masks - values[:, t]

        # GAE accumulation A_t
        last_gae = delta + gamma * lam * next_masks * last_gae
        advantages[:, t] = last_gae

    # Target returns R_t = A_t + V_t
    returns = advantages + values

    # Zero out non-action positions (prompt and pad)
    advantages = advantages * masks
    returns = returns * masks

    if normalize:
        active_indices = masks.bool()
        if active_indices.any():
            adv_active = advantages[active_indices]
            mean_adv = adv_active.mean()
            std_adv = adv_active.std()
            norm_advantages = torch.zeros_like(advantages)
            if std_adv > 1e-6:
                norm_advantages[active_indices] = (adv_active - mean_adv) / (std_adv + 1e-8)
            else:
                norm_advantages[active_indices] = adv_active - mean_adv
            advantages = norm_advantages

    return advantages, returns


@dataclass
class PPORolloutBatch:
    """Collated rollout batch container for PPO training epochs."""

    input_ids: torch.Tensor        # Shape: (B, T)
    attention_mask: torch.Tensor   # Shape: (B, T)
    action_mask: torch.Tensor      # Shape: (B, T) (1 exclusively for active completion tokens)
    old_log_probs: torch.Tensor    # Shape: (B, T) (per-token log-probs under behavior policy)
    old_values: torch.Tensor       # Shape: (B, T) (per-token values under behavior critic)
    advantages: torch.Tensor       # Shape: (B, T) (normalized GAE-λ advantages)
    returns: torch.Tensor          # Shape: (B, T) (target returns for critic)

    def to(self, device: torch.device) -> PPORolloutBatch:
        """Transfer all batch tensors to target compute device."""
        return PPORolloutBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            old_values=self.old_values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )

    def __len__(self) -> int:
        return self.input_ids.size(0)


class PPORolloutBuffer:
    """In-memory collection buffer aggregating rollouts and generating mini-batches."""

    def __init__(self):
        self.batches: list[PPORolloutBatch] = []

    def add(self, batch: PPORolloutBatch) -> None:
        """Append a rollout batch to the buffer."""
        self.batches.append(batch)

    def clear(self) -> None:
        """Clear all stored rollout batches."""
        self.batches.clear()

    def __len__(self) -> int:
        return sum(len(b) for b in self.batches)

    def get_mini_batches(self, mini_batch_size: int) -> Iterator[PPORolloutBatch]:
        """Yield randomly sliced mini-batches across all accumulated rollouts.

        Args:
            mini_batch_size: Number of sequences per mini-batch.

        Yields:
            PPORolloutBatch mini-batches.
        """
        if not self.batches:
            return

        # Concatenate all batches across dimension 0
        all_input_ids = torch.cat([b.input_ids for b in self.batches], dim=0)
        all_attention_mask = torch.cat([b.attention_mask for b in self.batches], dim=0)
        all_action_mask = torch.cat([b.action_mask for b in self.batches], dim=0)
        all_old_log_probs = torch.cat([b.old_log_probs for b in self.batches], dim=0)
        all_old_values = torch.cat([b.old_values for b in self.batches], dim=0)
        all_advantages = torch.cat([b.advantages for b in self.batches], dim=0)
        all_returns = torch.cat([b.returns for b in self.batches], dim=0)

        total_samples = all_input_ids.size(0)
        indices = torch.randperm(total_samples)

        for start_idx in range(0, total_samples, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, total_samples)
            batch_idx = indices[start_idx:end_idx]

            yield PPORolloutBatch(
                input_ids=all_input_ids[batch_idx],
                attention_mask=all_attention_mask[batch_idx],
                action_mask=all_action_mask[batch_idx],
                old_log_probs=all_old_log_probs[batch_idx],
                old_values=all_old_values[batch_idx],
                advantages=all_advantages[batch_idx],
                returns=all_returns[batch_idx],
            )


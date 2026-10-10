"""Proposed fix for PPOTrainer resolving rollout dropout, critic shift, and silent fallbacks."""

from __future__ import annotations
from typing import Optional
import torch
from loguru import logger

from training.m1_transformer.model import MiniGPT
from training.m5_preference.critic import CriticModel
from training.m5_preference.gae import PPORolloutBatch, compute_gae_advantages
from training.m5_preference.ppo_loss import PPOLoss
from training.m5_preference.reward_model import RewardModel
from training.m5_preference.ppo_trainer import PPOTrainer, compute_token_log_probs_and_entropy


class FixedPPOTrainer(PPOTrainer):
    """Reinforcement learning PPOTrainer with eval-mode rollouts, shifted critic values, and warnings."""

    def __init__(
        self,
        actor: MiniGPT,
        critic: CriticModel,
        ref_model: Optional[MiniGPT] = None,
        reward_model: Optional[RewardModel] = None,
        loss_fn: Optional[PPOLoss] = None,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        kl_coef: float = 0.05,
        gamma: float = 1.0,
        lam: float = 0.95,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            ref_model=ref_model,
            reward_model=reward_model,
            loss_fn=loss_fn,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            kl_coef=kl_coef,
            gamma=gamma,
            lam=lam,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        # Fix GAP-05: Warn practitioner if KL regularization was requested but ref_model was omitted
        if self.ref_model is None and self.kl_coef > 0.0:
            logger.warning(
                f"FixedPPOTrainer: ref_model is None with kl_coef={self.kl_coef:.4f}. "
                "Token-level KL penalty will be zero! Provide a frozen reference policy to avoid reward hacking."
            )

    @torch.no_grad()
    def build_rollout_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> PPORolloutBatch:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)
        B, T = input_ids.size()

        action_mask = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        for i in range(B):
            p_len = int(prompt_lengths[i].item())
            action_mask[i, p_len:] = attention_mask[i, p_len:].float()

        # Fix GAP-03: Put actor and critic in eval mode during rollout extraction
        # to ensure deterministic log-probs and avoid dropout noise at step 0
        self.actor.eval()
        self.critic.eval()
        try:
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
            old_values = self.critic(input_ids, attention_mask)

            if self.ref_model is not None:
                ref_log_probs, _ = compute_token_log_probs_and_entropy(self.ref_model, input_ids)
            else:
                ref_log_probs = old_log_probs.detach().clone()
        finally:
            self.actor.train()
            self.critic.train()

        rewards = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        kl_penalty = -self.kl_coef * (old_log_probs - ref_log_probs)
        rewards = rewards + kl_penalty * action_mask

        if self.reward_model is not None:
            rm_scores = self.reward_model(input_ids, attention_mask)
            seq_lens = attention_mask.sum(dim=-1).clamp(min=1).long() - 1
            for i in range(B):
                active_indices = torch.nonzero(action_mask[i] > 0.0, as_tuple=False)
                last_idx = active_indices[-1].item() if len(active_indices) > 0 else seq_lens[i].item()
                rewards[i, last_idx] += rm_scores[i]

        advantages, returns = compute_gae_advantages(
            rewards=rewards,
            values=old_values,
            masks=action_mask,
            gamma=self.gamma,
            lam=self.lam,
            normalize=True,
        )

        return PPORolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
        )


"""Proximal Policy Optimization (PPO) Actor-Critic RL Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from training.m1_transformer.model import MiniGPT
from training.m5_preference.critic import CriticModel
from training.m5_preference.gae import (
    PPORolloutBatch,
    PPORolloutBuffer,
    compute_gae_advantages,
)
from training.m5_preference.ppo_loss import PPOLoss
from training.m5_preference.reward_model import RewardModel


def compute_token_log_probs_and_entropy(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute autoregressive per-token log-probabilities and entropy aligned to sequence shape.

    Args:
        model: Generative causal decoder (MiniGPT).
        input_ids: Token ID tensor of shape (B, T).

    Returns:
        (token_log_probs, token_entropy) each of shape (B, T).
    """
    device = input_ids.device
    B, T = input_ids.size()

    try:
        logits, _ = model(input_ids, return_all_logits=True)
    except TypeError:
        logits, _ = model(input_ids)

    # Shift: token at t predicts target at t + 1
    shift_logits = logits[:, :-1, :].contiguous()  # Shape: (B, T - 1, V)
    shift_tokens = input_ids[:, 1:].contiguous()    # Shape: (B, T - 1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # Shape: (B, T - 1, V)
    probs = log_probs.exp()

    per_token_logp = torch.gather(
        log_probs, dim=2, index=shift_tokens.unsqueeze(-1)
    ).squeeze(-1)  # Shape: (B, T - 1)

    per_token_entropy = -(probs * log_probs).sum(dim=-1)  # Shape: (B, T - 1)

    # Prepend zero at position 0 to restore original (B, T) shape
    aligned_logp = torch.cat(
        [torch.zeros((B, 1), device=device), per_token_logp], dim=1
    )
    aligned_entropy = torch.cat(
        [torch.zeros((B, 1), device=device), per_token_entropy], dim=1
    )

    return aligned_logp, aligned_entropy


class PPOTrainer:
    """Supervised orchestrator for PPO Actor-Critic Reinforcement Learning."""

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
        """Initialize the PPOTrainer.

        Args:
            actor: Active policy network MiniGPT to optimize.
            critic: State-value CriticModel network.
            ref_model: Optional frozen reference policy for token-level KL penalties.
            reward_model: Optional RewardModel scoring completed sequences.
            loss_fn: PPOLoss criterion (default clip_ratio=0.2, vf_coef=0.5).
            actor_optimizer: Optimizer for actor policy.
            critic_optimizer: Optimizer for critic value network.
            kl_coef: Multiplier β for token-level KL penalty.
            gamma: Discount factor γ for GAE.
            lam: Bias-variance parameter λ for GAE.
            max_grad_norm: Gradient clipping threshold (default 1.0).
            device: Compute device.
        """
        self.device = device or actor.device
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.ref_model = ref_model.to(self.device) if ref_model is not None else None
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.reward_model = reward_model.to(self.device) if reward_model is not None else None
        if self.reward_model is not None:
            self.reward_model.eval()
            for p in self.reward_model.parameters():
                p.requires_grad = False

        self.loss_fn = loss_fn or PPOLoss()
        self.actor_optimizer = actor_optimizer or torch.optim.AdamW(
            self.actor.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.critic_optimizer = critic_optimizer or torch.optim.AdamW(
            self.critic.parameters(), lr=5e-4, weight_decay=0.01
        )

        self.kl_coef = float(kl_coef)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.max_grad_norm = float(max_grad_norm)

        # Fix GAP-05: Warn practitioner if KL regularization was requested but ref_model is None
        if self.ref_model is None and self.kl_coef > 0.0:
            logger.warning(
                f"PPOTrainer: ref_model is None with kl_coef={self.kl_coef:.4f}. "
                "Token-level KL penalty will be zero! Provide a frozen reference policy to avoid reward hacking."
            )

    @torch.no_grad()
    def build_rollout_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        terminal_rewards: Optional[torch.Tensor] = None,
    ) -> PPORolloutBatch:
        """Construct a complete PPORolloutBatch with rewards and GAE advantages.

        Args:
            input_ids: Full sequence token IDs (prompt ∘ completion), shape (B, T).
            attention_mask: Binary attention mask (1 for active, 0 for pad), shape (B, T).
            prompt_lengths: Number of prompt tokens for each sequence, shape (B,).
            terminal_rewards: Optional external terminal rewards, shape (B,). If supplied,
                these replace the reward model's terminal score while retaining KL shaping.

        Returns:
            Populated PPORolloutBatch ready for PPO optimization epochs.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)
        B, T = input_ids.size()

        # 1. Compute action mask (1 exclusively for active completion tokens)
        action_mask = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        for i in range(B):
            p_len = int(prompt_lengths[i].item())
            action_mask[i, p_len:] = attention_mask[i, p_len:].float()

        # 2. Extract policy log-probs and critic state values in deterministic eval mode
        actor_was_training = self.actor.training
        critic_was_training = self.critic.training
        self.actor.eval()
        self.critic.eval()
        try:
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
            old_values = self.critic(input_ids, attention_mask)

            # 3. Extract reference log-probs
            if self.ref_model is not None:
                ref_log_probs, _ = compute_token_log_probs_and_entropy(self.ref_model, input_ids)
            else:
                ref_log_probs = old_log_probs.detach().clone()
        finally:
            self.actor.train(actor_was_training)
            self.critic.train(critic_was_training)

        # 4. Compute token-level rewards
        rewards = torch.zeros((B, T), dtype=torch.float32, device=self.device)

        # Per-token KL penalty: -β * (log π_actor - log π_ref)
        kl_penalty = -self.kl_coef * (old_log_probs - ref_log_probs)
        rewards = rewards + kl_penalty * action_mask

        # Terminal reward score from an external reward callback or the reward model.
        if terminal_rewards is not None:
            terminal_rewards = terminal_rewards.to(self.device, dtype=rewards.dtype)
            if terminal_rewards.shape != (B,):
                raise ValueError(f"terminal_rewards must have shape ({B},).")
            if not torch.isfinite(terminal_rewards).all():
                raise ValueError("terminal_rewards must contain only finite values.")
            rm_scores = terminal_rewards
        elif self.reward_model is not None:
            rm_scores = self.reward_model(input_ids, attention_mask)  # Shape: (B,)
        else:
            rm_scores = None

        if rm_scores is not None:
            seq_lens = attention_mask.sum(dim=-1).clamp(min=1).long() - 1
            for i in range(B):
                active_indices = torch.nonzero(action_mask[i] > 0.0, as_tuple=False)
                if len(active_indices) > 0:
                    last_idx = active_indices[-1].item()
                else:
                    last_idx = seq_lens[i].item()
                rewards[i, last_idx] += rm_scores[i]

        # 5. Compute GAE-λ advantages and target returns
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

    def train_step(self, rollout_batch: PPORolloutBatch) -> dict[str, float]:
        """Execute a single PPO Actor-Critic gradient update on a rollout batch.

        Args:
            rollout_batch: PPORolloutBatch containing trajectories and advantages.

        Returns:
            Dictionary of step metrics (loss, policy_loss, vf_loss, entropy, clip_fraction, approx_kl).
        """
        self.actor.train()
        self.critic.train()
        batch = rollout_batch.to(self.device)

        # 1. Forward current policy to get new log-probs and entropy
        new_log_probs, entropy = compute_token_log_probs_and_entropy(
            self.actor, batch.input_ids
        )

        # 2. Forward current critic to get new state-values
        new_values = self.critic(batch.input_ids, batch.attention_mask)

        # 3. Compute composite PPO loss
        total_loss, metrics = self.loss_fn(
            policy_log_probs=new_log_probs,
            old_log_probs=batch.old_log_probs,
            values=new_values,
            old_values=batch.old_values,
            advantages=batch.advantages,
            returns=batch.returns,
            action_mask=batch.action_mask,
            entropy=entropy,
        )

        # 4. Optimization step: Actor & Critic backpropagation
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return metrics

    def save_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Persist Actor and Critic checkpoints to disk."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "config": getattr(self.actor, "config", None),
            },
            path,
        )
        logger.info(f"Saved PPO Actor-Critic checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load Actor and Critic state dicts from checkpoint."""
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        logger.info(f"Loaded PPO Actor-Critic checkpoint from {path}")

    @torch.no_grad()
    def sample_rollouts(
        self,
        prompts: Sequence[str],
        tokenizer,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        eos_token_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """Autoregressive online rollout sampling for a batch of prompts.

        Samples 1 trajectory per prompt using current actor policy with EOS stopping.
        Returns padded input_ids, attention_mask, prompt_lengths, and completion texts.
        """
        if not prompts:
            raise ValueError("Prompts sequence cannot be empty.")

        was_training = self.actor.training
        self.actor.eval()
        all_sequences: list[torch.Tensor] = []
        all_prompt_lens: list[int] = []
        all_completion_texts: list[str] = []

        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)

        try:
            for prompt in prompts:
                prompt_tokens = tokenizer.encode(prompt)
                p_len = len(prompt_tokens)
                p_idx = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

                gen_seq = self.actor.generate(
                    p_idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_id=eos_token_id,
                )
                full_tokens = gen_seq[0].tolist()
                completion_tokens = full_tokens[p_len:]
                completion_str = tokenizer.decode(completion_tokens)

                all_sequences.append(gen_seq[0])
                all_prompt_lens.append(p_len)
                all_completion_texts.append(completion_str)

            max_len = max(seq.size(0) for seq in all_sequences)
            pad_id = getattr(tokenizer, "pad_token_id", 0)

            padded_ids = torch.full((len(all_sequences), max_len), pad_id, dtype=torch.long, device=self.device)
            attn_mask = torch.zeros((len(all_sequences), max_len), dtype=torch.long, device=self.device)

            for i, seq in enumerate(all_sequences):
                padded_ids[i, : seq.size(0)] = seq
                attn_mask[i, : seq.size(0)] = 1

            prompt_lens_tensor = torch.tensor(all_prompt_lens, dtype=torch.long, device=self.device)
            return padded_ids, attn_mask, prompt_lens_tensor, all_completion_texts
        finally:
            self.actor.train(was_training)

    def step_on_prompts(
        self,
        prompts: Sequence[str],
        tokenizer,
        reward_fn: Optional[Callable[[list[str]], list[float]]] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        eos_token_id: Optional[int] = None,
    ) -> dict[str, float]:
        """Execute a full online rollout-to-update PPO training step.

        1. Samples autoregressive rollouts from prompts with EOS stopping.
        2. Scores trajectories with reward_fn or internal RewardModel.
        3. Computes GAE advantages and target returns.
        4. Updates Actor-Critic networks and returns step metrics.
        """
        if reward_fn is None and self.reward_model is None:
            raise ValueError("PPO step_on_prompts requires reward_fn or a configured reward_model.")

        input_ids, attention_mask, prompt_lengths, completions = self.sample_rollouts(
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )

        terminal_rewards = None
        if reward_fn is not None:
            scores = torch.as_tensor(reward_fn(completions), dtype=torch.float32, device=self.device)
            if scores.shape != (len(completions),):
                raise ValueError(
                    f"reward_fn must return exactly {len(completions)} scalar rewards."
                )
            if not torch.isfinite(scores).all():
                raise ValueError("reward_fn must return only finite scores.")
            terminal_rewards = scores

        rollout_batch = self.build_rollout_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            terminal_rewards=terminal_rewards,
        )

        return self.train_step(rollout_batch)

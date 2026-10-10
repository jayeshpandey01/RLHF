"""Group Relative Policy Optimization (GRPO) Trainer with Verifiable Rewards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
from loguru import logger

from training.m1_transformer.model import MiniGPT
from training.m5_preference.grpo_loss import GRPOLoss, compute_group_advantages
from training.m5_preference.ppo_trainer import compute_token_log_probs_and_entropy
from training.m5_preference.verifiable_reward import VerifiableRewardEngine


@dataclass
class GRPOGroupBatch:
    """Collated group rollout trajectory container for GRPO optimization."""

    input_ids: torch.Tensor             # Shape: (B, T) where B = N * G
    attention_mask: torch.Tensor        # Shape: (B, T)
    action_mask: torch.Tensor           # Shape: (B, T) (1 exclusively for completion tokens)
    old_log_probs: torch.Tensor         # Shape: (B, T)
    advantages: torch.Tensor            # Shape: (B,)
    rewards: torch.Tensor               # Shape: (B,)
    ref_log_probs: Optional[torch.Tensor] = None # Shape: (B, T)

    def to(self, device: torch.device) -> GRPOGroupBatch:
        """Transfer batch tensors to target compute device."""
        return GRPOGroupBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            advantages=self.advantages.to(device),
            rewards=self.rewards.to(device),
            ref_log_probs=self.ref_log_probs.to(device) if self.ref_log_probs is not None else None,
        )

    def __len__(self) -> int:
        return self.input_ids.size(0)


class GRPOTrainer:
    """Supervised orchestrator for DeepSeek-R1 style Group Relative Policy Optimization."""

    def __init__(
        self,
        actor: MiniGPT,
        ref_model: Optional[MiniGPT] = None,
        reward_engine: Optional[VerifiableRewardEngine] = None,
        loss_fn: Optional[GRPOLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        group_size: int = 4,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize the GRPOTrainer.

        Args:
            actor: Active policy network MiniGPT to optimize.
            ref_model: Optional frozen reference policy for unbiased KL penalties.
            reward_engine: VerifiableRewardEngine scoring generated completions.
            loss_fn: GRPOLoss criterion.
            optimizer: Optimizer for actor policy (defaults to AdamW lr=1e-4).
            group_size: Number of completions per prompt group (G, default 4).
            max_grad_norm: Gradient norm clipping threshold (default 1.0).
            device: Compute device.
        """
        self.device = device or actor.device
        self.actor = actor.to(self.device)

        self.ref_model = ref_model.to(self.device) if ref_model is not None else None
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.reward_engine = reward_engine or VerifiableRewardEngine()
        self.loss_fn = loss_fn or GRPOLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.actor.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.group_size = int(group_size)
        self.max_grad_norm = float(max_grad_norm)

        # Fix GAP-05: Warn practitioner if KL regularization was requested but ref_model is None
        if self.ref_model is None and getattr(self.loss_fn, "kl_coef", 0.0) > 0.0:
            logger.warning(
                "GRPOTrainer initialized without ref_model, but loss_fn.kl_coef > 0. "
                "KL penalty will evaluate to zero (unpenalized policy drift)."
            )

    @torch.no_grad()
    def build_group_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        rewards: torch.Tensor,
    ) -> GRPOGroupBatch:
        """Construct a standardized GRPOGroupBatch with group advantages.

        Args:
            input_ids: Full sequence token IDs of shape (B, T) where B is divisible by group_size.
            attention_mask: Binary mask of shape (B, T).
            prompt_lengths: Number of prompt tokens per sequence, shape (B,).
            rewards: Scalar rewards evaluated for each sequence, shape (B,).

        Returns:
            Populated GRPOGroupBatch ready for gradient optimization.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)
        rewards = rewards.to(self.device)
        B, T = input_ids.size()

        # 1. Compute completion action mask
        action_mask = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        for i in range(B):
            p_len = int(prompt_lengths[i].item())
            action_mask[i, p_len:] = attention_mask[i, p_len:].float()

        # 2. Extract policy rollout log-probs in eval mode (Fix GAP-03)
        was_training = self.actor.training
        self.actor.eval()
        try:
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)

            # 3. Extract reference log-probs
            if self.ref_model is not None:
                ref_log_probs, _ = compute_token_log_probs_and_entropy(self.ref_model, input_ids)
            else:
                ref_log_probs = None
        finally:
            if was_training:
                self.actor.train()

        # 4. Compute group-relative standardized advantages
        advantages = compute_group_advantages(rewards=rewards, group_size=self.group_size)

        return GRPOGroupBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            old_log_probs=old_log_probs,
            advantages=advantages,
            rewards=rewards,
            ref_log_probs=ref_log_probs,
        )

    def train_step(self, group_batch: GRPOGroupBatch) -> dict[str, float]:
        """Execute a single GRPO gradient step on a group batch.

        Args:
            group_batch: GRPOGroupBatch containing trajectories and group advantages.

        Returns:
            Dictionary of metrics (loss, policy_loss, kl_loss, mean_ratio, clip_fraction).
        """
        self.actor.train()
        batch = group_batch.to(self.device)

        # 1. Forward current policy to get new log-probabilities
        new_log_probs, _ = compute_token_log_probs_and_entropy(
            self.actor, batch.input_ids
        )

        # 2. Compute GRPO loss with DeepSeek unbiased KL penalty
        loss, metrics = self.loss_fn(
            policy_log_probs=new_log_probs,
            old_log_probs=batch.old_log_probs,
            advantages=batch.advantages,
            action_mask=batch.action_mask,
            ref_log_probs=batch.ref_log_probs,
        )

        # 3. Optimization step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return metrics

    def save_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Persist Actor checkpoint to disk."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": getattr(self.actor, "config", None),
            },
            path,
        )
        logger.info(f"Saved GRPO Actor checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load Actor state dict from checkpoint."""
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded GRPO Actor checkpoint from {path}")

    @torch.no_grad()
    def sample_group_rollouts(
        self,
        prompts: Sequence[str],
        tokenizer,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        eos_token_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """Autoregressive online rollout sampling for a batch of prompts.

        For each prompt, generates group_size independent candidate trajectories with EOS stopping,
        returning padded token tensors, attention masks, prompt lengths, and completion strings.
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

                for _ in range(self.group_size):
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
        reward_fn: Callable[[list[str]], list[float]],
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        eos_token_id: Optional[int] = None,
    ) -> dict[str, float]:
        """Execute a full online group rollout-to-update GRPO training step.

        1. Samples group rollouts (group_size trajectories per prompt) with EOS stopping.
        2. Scores trajectories with reward_fn in exact group order.
        3. Computes normalized intra-group advantages via build_group_batch.
        4. Updates Actor policy with clipped surrogate objective and DeepSeek reverse KL.
        """
        input_ids, attention_mask, prompt_lengths, completions = self.sample_group_rollouts(
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )

        rewards_tensor = torch.as_tensor(reward_fn(completions), dtype=torch.float32, device=self.device)
        if rewards_tensor.shape != (len(completions),):
            raise ValueError(
                f"reward_fn must return exactly {len(completions)} scalar rewards."
            )
        if not torch.isfinite(rewards_tensor).all():
            raise ValueError("reward_fn must return only finite scores.")

        group_batch = self.build_group_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            rewards=rewards_tensor,
        )

        return self.train_step(group_batch)

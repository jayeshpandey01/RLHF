"""Proposed fix for GRPOTrainer resolving rollout dropout and providing online group generation."""

from __future__ import annotations
from typing import Optional, Sequence
import torch
from loguru import logger

from training.m1_transformer.model import MiniGPT
from training.m5_preference.grpo_loss import GRPOLoss, compute_group_advantages
from training.m5_preference.grpo_trainer import GRPOTrainer, GRPOGroupBatch
from training.m5_preference.ppo_trainer import compute_token_log_probs_and_entropy
from training.m5_preference.verifiable_reward import VerifiableRewardEngine


class FixedGRPOTrainer(GRPOTrainer):
    """DeepSeek-R1 style GRPOTrainer with eval-mode trajectory scoring and online group sampling."""

    @torch.no_grad()
    def build_group_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        rewards: torch.Tensor,
    ) -> GRPOGroupBatch:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)
        rewards = rewards.to(self.device)
        B, T = input_ids.size()

        action_mask = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        for i in range(B):
            p_len = int(prompt_lengths[i].item())
            action_mask[i, p_len:] = attention_mask[i, p_len:].float()

        # Fix GAP-03: Put actor in eval mode during rollout scoring
        self.actor.eval()
        try:
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
            if self.ref_model is not None:
                ref_log_probs, _ = compute_token_log_probs_and_entropy(self.ref_model, input_ids)
            else:
                ref_log_probs = None
        finally:
            self.actor.train()

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

    @torch.no_grad()
    def sample_group_rollouts(
        self,
        prompts: Sequence[str],
        tokenizer,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """Fix GAP-04: Online autoregressive rollout generation for a list of prompts.

        Samples G = group_size independent completions per prompt, returning collated
        input_ids, attention_mask, prompt_lengths, and generated completion strings.
        """
        self.actor.eval()
        all_sequences: list[torch.Tensor] = []
        all_prompt_lens: list[int] = []
        all_completion_texts: list[str] = []

        try:
            for prompt in prompts:
                prompt_tokens = tokenizer.encode(prompt)
                p_len = len(prompt_tokens)
                p_idx = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

                # Sample G completions
                for _ in range(self.group_size):
                    gen_seq = self.actor.generate(
                        p_idx, max_new_tokens=max_new_tokens, temperature=temperature
                    )
                    full_tokens = gen_seq[0].tolist()
                    completion_tokens = full_tokens[p_len:]
                    completion_str = tokenizer.decode(completion_tokens)

                    all_sequences.append(gen_seq[0])
                    all_prompt_lens.append(p_len)
                    all_completion_texts.append(completion_str)

            # Collate and pad sequences to uniform length
            max_len = max(seq.size(0) for seq in all_sequences)
            pad_id = getattr(tokenizer, "pad_token_id", 0)

            padded_ids = torch.full((len(all_sequences), max_len), pad_id, dtype=torch.long, device=self.device)
            attn_mask = torch.zeros((len(all_sequences), max_len), dtype=torch.long, device=self.device)

            for i, seq in enumerate(all_sequences):
                padded_ids[i, :seq.size(0)] = seq
                attn_mask[i, :seq.size(0)] = 1

            prompt_lens_tensor = torch.tensor(all_prompt_lens, dtype=torch.long, device=self.device)
            return padded_ids, attn_mask, prompt_lens_tensor, all_completion_texts
        finally:
            self.actor.train()


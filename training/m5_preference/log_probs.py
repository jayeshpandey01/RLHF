"""Autoregressive sequence log-probability extraction with completion masking."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.m1_transformer.model import MiniGPT


def extract_completion_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Extract causal autoregressive completion log-probabilities under a language model.

    For a sequence x ∘ y (prompt x followed by completion y), computes:
        log π(y | x) = sum_{t in completion} log P(y_t | x, y_<t)

    Prompt tokens and padding tokens are excluded from the summation via labels != -100
    and attention_mask == 1.

    Args:
        model: Causal decoder model (MiniGPT or compatible) returning logits (B, T, V).
        input_ids: Token ID tensor of shape (B, T).
        attention_mask: Optional binary attention mask of shape (B, T).
        labels: Optional label tensor of shape (B, T) where prompt and pad tokens are -100.
                If None, labels is derived from attention_mask (prompt tokens must be explicitly masked).
        length_normalize: If True, divides total completion log-prob by completion length |y|
                          (mitigates verbosity bias, SimPO style).

    Returns:
        Tensor of shape (B,) containing total (or length-normalized) completion log-probabilities.
    """
    device = input_ids.device
    B, T = input_ids.size()

    # Forward pass to obtain full sequence logits across all positions
    try:
        logits, _ = model(input_ids, return_all_logits=True)  # Shape: (B, T, V)
    except TypeError:
        logits, _ = model(input_ids)

    # Causal shift: token at t predicts target at t + 1
    shift_logits = logits[:, :-1, :].contiguous()  # Shape: (B, T - 1, V)
    shift_tokens = input_ids[:, 1:].contiguous()    # Shape: (B, T - 1)

    # Compute per-token log-probabilities across entire vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)  # Shape: (B, T - 1, V)

    # Gather log-probabilities at actual target token positions
    per_token_log_probs = torch.gather(
        log_probs, dim=2, index=shift_tokens.unsqueeze(-1)
    ).squeeze(-1)  # Shape: (B, T - 1)

    # Construct completion mask
    if labels is not None:
        shift_labels = labels[:, 1:].contiguous()  # Shape: (B, T - 1)
        loss_mask = shift_labels != -100
    else:
        loss_mask = torch.ones_like(shift_tokens, dtype=torch.bool)

    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous().bool()
        loss_mask = loss_mask & shift_mask

    # Mask out non-completion positions
    masked_log_probs = per_token_log_probs * loss_mask.float()
    seq_log_probs = masked_log_probs.sum(dim=-1)  # Shape: (B,)

    if length_normalize:
        seq_lengths = loss_mask.float().sum(dim=-1).clamp(min=1.0)
        seq_log_probs = seq_log_probs / seq_lengths

    return seq_log_probs


def extract_batch_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    prompt_lengths: Optional[torch.Tensor] = None,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Convenience helper extracting log-probabilities given explicit prompt lengths.

    Constructs labels with -100 masking for all prompt positions:
        labels[:, :prompt_len] = -100

    Args:
        model: Causal decoder model.
        input_ids: Token ID tensor of shape (B, T).
        attention_mask: Binary mask of shape (B, T).
        prompt_lengths: Tensor of shape (B,) containing prompt token count for each sequence.
        length_normalize: Whether to normalize by completion token count.

    Returns:
        Tensor of shape (B,) containing completion log-probabilities.
    """
    labels = input_ids.clone()
    if prompt_lengths is not None:
        for i, p_len in enumerate(prompt_lengths):
            labels[i, : int(p_len)] = -100

    return extract_completion_log_probs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        length_normalize=length_normalize,
    )

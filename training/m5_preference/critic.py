"""Value function Critic network for PPO reinforcement learning."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from training.m1_transformer.model import MiniGPT


class ValueHead(nn.Module):
    """Linear scalar projection head estimating state value from token hidden states."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states of shape (B, T, d_model) to values of shape (B, T)."""
        return self.proj(hidden_states).squeeze(-1)


class CriticModel(nn.Module):
    """Autoregressive Critic network predicting per-token state values V(s_t).

    Wraps a MiniGPT backbone architecture with a token-level ValueHead.
    For each position t in sequence x ∘ y, produces scalar value V(s_t) = V(x, y_{<t}).
    """

    def __init__(self, backbone: MiniGPT):
        """Initialize the CriticModel.

        Args:
            backbone: MiniGPT transformer decoder serving as feature extractor.
        """
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(backbone.config.d_model)

    @property
    def device(self) -> torch.device:
        """Infer active compute device from backbone parameters."""
        return self.backbone.device

    def extract_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward through backbone transformer layers to obtain hidden states.

        Args:
            idx: Token indices tensor of shape (B, T).

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
        """Forward pass computing per-token state values V(s_t).

        Args:
            input_ids: Token indices tensor of shape (B, T).
            attention_mask: Optional binary mask of shape (B, T).

        Returns:
            State values tensor of shape (B, T).
        """
        input_ids = input_ids.to(self.device)
        hidden = self.extract_hidden_states(input_ids)  # Shape: (B, T, d_model)
        raw_values = self.value_head(hidden)            # Shape: (B, T)

        # Causal temporal alignment: state s_t = (x, y_{<t}) precedes action token y_t.
        # In a causal transformer, hidden state at index t - 1 has observed tokens up to t - 1.
        # Shift values right by 1 so values[:, t] represents V(s_t) = V(x, y_{<t}).
        # Index 0 represents the unconditioned state before the first prompt token, initialized to 0.0.
        shifted_values = torch.zeros_like(raw_values)
        shifted_values[:, 1:] = raw_values[:, :-1]

        if attention_mask is not None:
            mask = attention_mask.to(self.device).float()
            shifted_values = shifted_values * mask

        return shifted_values


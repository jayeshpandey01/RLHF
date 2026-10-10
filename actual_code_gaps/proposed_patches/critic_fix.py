"""Proposed fix for CriticModel resolving action information leakage (GAP-01).

Ensures values[:, t] represents state value V(s_t) = V(x, y_{<t}) by taking
the value head projection from hidden state at t - 1.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from training.m1_transformer.model import MiniGPT
from training.m5_preference.critic import ValueHead


class FixedCriticModel(nn.Module):
    """Autoregressive Critic network predicting causal per-token state values V(s_t).

    Shifts value predictions by one position so hidden_states[:, t - 1] predicts
    the state value V(s_t) before action token y_t is observed.
    """

    def __init__(self, backbone: MiniGPT):
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(backbone.config.d_model)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    def extract_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
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
        """Forward pass computing causal state values V(s_t)."""
        input_ids = input_ids.to(self.device)
        hidden = self.extract_hidden_states(input_ids)  # Shape: (B, T, d_model)
        raw_values = self.value_head(hidden)            # Shape: (B, T)

        # Shift values right by 1 position:
        # State s_t = (x, y_{<t}) corresponds to prefix ending at index t - 1.
        # Its value is projected from hidden[:, t - 1].
        B, T = raw_values.size()
        shifted_values = torch.zeros_like(raw_values)
        shifted_values[:, 1:] = raw_values[:, :-1]

        if attention_mask is not None:
            mask = attention_mask.to(self.device).float()
            shifted_values = shifted_values * mask

        return shifted_values


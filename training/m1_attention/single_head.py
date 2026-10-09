"""Single-Head Causal Self-Attention Layer with Scaled Dot-Product & Lower-Triangular Masking."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadCausalAttention(nn.Module):
    """Single-Head Scaled Dot-Product Attention with causal masking."""

    def __init__(self, d_model: int, head_dim: int, block_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim**0.5)

        # Projections
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)

        # Causal mask registered as a persistent buffer (not a parameter)
        self.register_buffer(
            "tril_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model)

        Returns:
            out: Contextualized output representations (B, T, head_dim)
            weights: Attention probability matrix (B, 1, T, T)
        """
        B, T, C = x.shape

        # 1. Project inputs into Queries, Keys, Values
        q = self.q_proj(x)  # (B, T, head_dim)
        k = self.k_proj(x)  # (B, T, head_dim)
        v = self.v_proj(x)  # (B, T, head_dim)

        # Reshape to (B, 1, T, head_dim) for standardized matrix multiplication
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # 2. Scaled Dot-Product: (B, 1, T, head_dim) @ (B, 1, head_dim, T) -> (B, 1, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Apply Causal Masking: fill upper triangle (future tokens) with -inf
        mask = self.tril_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 4. Softmax over key dimension (-1)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # 5. Weighted aggregation of values: (B, 1, T, T) @ (B, 1, T, head_dim) -> (B, 1, T, head_dim)
        out = torch.matmul(weights, v).squeeze(1)  # (B, T, head_dim)

        return out, weights


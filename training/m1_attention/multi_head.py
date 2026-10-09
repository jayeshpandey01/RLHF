"""Multi-Head Causal Self-Attention Layer with Fused QKV Projections and Optional KV-Cache."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.m1_attention.kv_cache import KVCache


class CausalMultiHeadAttention(nn.Module):
    """Causal Multi-Head Attention (MHA) module matching modern GPT architectures."""

    def __init__(self, d_model: int, n_heads: int, block_size: int = 128, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / (self.head_dim**0.5)

        # Fused projection for all heads: Q, K, V computed in a single matrix multiplication
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask registered as a persistent buffer
        self.register_buffer(
            "tril_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model)
            return_attention_weights: If True, returns attention matrix
            kv_cache: Optional KVCache instance for accelerated incremental rollout

        Returns:
            (out, weights): Projected output tensor and optional attention weights
        """
        B, T, C = x.shape

        # 1. Fused Q, K, V projection: (B, T, d_model) -> (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape and transpose for multi-head parallel attention
        # (B, T, d_model) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. KV-Cache Handling or standard parallel attention
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            T_k = k.size(2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if T > 1:
                # Context prefill: tokens inside prompt cannot attend to future prompt tokens
                T_past = T_k - T
                mask = self.tril_mask[:, :, :T, :T]
                if T_past > 0:
                    past_ones = torch.ones(1, 1, T, T_past, device=x.device)
                    full_mask = torch.cat([past_ones, mask], dim=-1)
                else:
                    full_mask = mask
                scores = scores.masked_fill(full_mask == 0, float("-inf"))
        else:
            # 3. Scaled Dot-Product: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = self.tril_mask[:, :, :T, :T]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 4. Softmax attention probabilities
        weights = F.softmax(scores, dim=-1)
        weights_dropped = self.attn_dropout(weights)

        # 5. Values aggregation: (B, nh, T, T_k) @ (B, nh, T_k, hs) -> (B, nh, T, hs)
        head_out = torch.matmul(weights_dropped, v)

        # 6. Concatenate heads back into (B, T, d_model)
        out = head_out.transpose(1, 2).contiguous().view(B, T, C)

        # 7. Output projection and residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, (weights if return_attention_weights else None)

"""KV-Cache Architecture for Constant-Memory Incremental Autoregressive Decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Explicit key-value memory cache storing past token representations."""

    def __init__(self):
        self.k_cache: torch.Tensor | None = None  # (B, n_heads, T_past, head_dim)
        self.v_cache: torch.Tensor | None = None  # (B, n_heads, T_past, head_dim)

    def reset(self) -> None:
        """Clear the cache."""
        self.k_cache = None
        self.v_cache = None

    @property
    def seq_len(self) -> int:
        """Number of past tokens cached."""
        return self.k_cache.shape[2] if self.k_cache is not None else 0

    def update(
        self, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append newly computed keys and values to the historical cache.

        Args:
            new_k: (B, n_heads, T_new, head_dim)
            new_v: (B, n_heads, T_new, head_dim)

        Returns:
            (k_all, v_all): Full concatenated key and value tensors up to current position.
        """
        if self.k_cache is None or self.v_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)

        return self.k_cache, self.v_cache


class CausalAttentionWithKVCache(nn.Module):
    """Causal Multi-Head Attention supporting both full context and incremental KV-cache decoding."""

    def __init__(self, d_model: int, n_heads: int, block_size: int = 128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / (self.head_dim**0.5)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer(
            "tril_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(
        self, x: torch.Tensor, kv_cache: KVCache | None = None
    ) -> torch.Tensor:
        """Forward pass.

        If kv_cache is provided and seq_len > 0, performs incremental single-token decoding (O(T) FLOPs).
        Otherwise performs full context causal attention (O(T^2) FLOPs).
        """
        B, T, C = x.shape

        # 1. Project Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, d) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. KV-Cache handling
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            total_k_len = k.shape[2]

            # In incremental decoding (T=1), new query attends to all cached past tokens
            # No causal mask needed because past tokens are strictly in the past!
            if T == 1:
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nh, 1, total_k_len)
                weights = F.softmax(scores, dim=-1)
                head_out = torch.matmul(weights, v)  # (B, nh, 1, hs)
                out = head_out.transpose(1, 2).contiguous().view(B, T, C)
                return self.out_proj(out)

        # 3. Standard full context causal attention (Pre-fill / Training)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        mask = self.tril_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        head_out = torch.matmul(weights, v)
        out = head_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


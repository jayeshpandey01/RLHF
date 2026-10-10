"""Mini-GPT Decoder-Only Generative Transformer Architecture."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.m1_attention.kv_cache import KVCache
from training.m1_attention.multi_head import CausalMultiHeadAttention
from training.m1_transformer.config import GPTConfig


class MLP(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation and 4x expansion."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # Flag for GPT-2 residual projection variance scaling (1 / sqrt(2 * L))
        setattr(self.c_proj, "NANOGPT_SCALE_INIT", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with Causal Self-Attention and MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalMultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
        )
        # Flag MHA output projection for residual variance scaling
        setattr(self.attn.out_proj, "NANOGPT_SCALE_INIT", True)

        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Residual Stream 1: Pre-LN Attention
        attn_out, attn_weights = self.attn(
            self.ln_1(x),
            return_attention_weights=return_attention_weights,
            kv_cache=kv_cache,
        )
        x = x + attn_out

        # Residual Stream 2: Pre-LN MLP
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class MiniGPT(nn.Module):
    """Mini-GPT Decoder-Only Generative Transformer Model with Optional KV-Cache."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                wpe=nn.Embedding(config.block_size, config.d_model),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.d_model),
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (Press & Wolf, 2017): Tie input embeddings and output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights according to GPT-2 specification
        self.apply(self._init_weights)

    @property
    def device(self) -> torch.device:
        """Dynamically infer the active device of model parameters."""
        return next(self.parameters()).device

    def _init_weights(self, module: nn.Module) -> None:
        """Apply Gaussian initialization with residual projection scaling."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT") and getattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return parameter count (optionally excluding positional embeddings)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[KVCache]] = None,
        return_all_logits: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass supporting both full sequence training and cached incremental decoding.

        Args:
            idx: Token indices tensor of shape (B, T)
            targets: Target token indices tensor of shape (B, T)
            kv_caches: Optional list of KVCache objects (one per TransformerBlock)
            return_all_logits: If True and targets is None, returns full (B, T, vocab_size) logits
                               instead of standard last token (B, 1, vocab_size) logits.

        Returns:
            logits: Unnormalized logits of shape (B, T, vocab_size) or (B, 1, vocab_size)
            loss: Cross-entropy scalar loss (if targets is provided)
        """
        idx = idx.to(self.device)
        B, T = idx.size()

        # Handle position embedding offset during incremental KV-cache decoding
        if kv_caches is not None and kv_caches[0].seq_len > 0:
            past_len = kv_caches[0].seq_len
            pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=self.device)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=self.device)

        assert (
            pos[-1].item() < self.config.block_size
        ), f"Sequence position {pos[-1].item()} exceeds maximum context window {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through Transformer blocks (passing respective layer KV cache if enabled)
        for i, block in enumerate(self.transformer.h):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, _ = block(x, kv_cache=cache_i)

        x = self.transformer.ln_f(x)

        if targets is not None:
            targets = targets.to(self.device)
            logits = self.lm_head(x)
            ignore_idx = -100 if (targets == -100).any() else -1
            active_tokens = (targets != ignore_idx).sum()
            if active_tokens > 0:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_idx)
            else:
                loss = logits.sum() * 0.0
        else:
            if return_all_logits:
                logits = self.lm_head(x)
            else:
                logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


    def configure_optimizers(
        self, weight_decay: float = 0.01, learning_rate: float = 5e-4, betas: tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.Optimizer:
        """Configure AdamW with selective decoupled weight decay."""
        decay_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generate next tokens given conditioning prefix.

        Supports both ultra-fast cached rollout (O(T)) and standard context recomputation,
        with optional early stopping on eos_token_id.
        """
        idx = idx.to(self.device)
        self.eval()

        kv_caches = [KVCache() for _ in range(self.config.n_layer)] if use_cache else None

        # Prefill prompt tokens
        if use_cache:
            logits, _ = self(idx, kv_caches=kv_caches)
            curr_idx = idx
        else:
            curr_idx = idx

        for step in range(max_new_tokens):
            if use_cache:
                if step == 0:
                    logits_step = logits[:, -1, :]
                else:
                    logits_step, _ = self(idx_next, kv_caches=kv_caches)
                    logits_step = logits_step[:, -1, :]
            else:
                idx_cond = curr_idx if curr_idx.size(1) <= self.config.block_size else curr_idx[:, -self.config.block_size :]
                logits_step, _ = self(idx_cond)
                logits_step = logits_step[:, -1, :]

            if temperature <= 1e-5:
                idx_next = torch.argmax(logits_step, dim=-1, keepdim=True)
            else:
                logits_step = logits_step / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                    logits_step[logits_step < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits_step, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            curr_idx = torch.cat((curr_idx, idx_next), dim=1)

            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return curr_idx

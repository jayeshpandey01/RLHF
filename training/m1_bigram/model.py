"""Bigram Language Model in PyTorch with Cross-Entropy Loss & Sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pedagogical implementation of Cross-Entropy using the log-sum-exp trick for numerical stability."""
    m, _ = torch.max(logits, dim=-1, keepdim=True)
    shifted_logits = logits - m
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_probs = target_logits - (m.squeeze(-1) + log_sum_exp)

    return -torch.mean(log_probs)


class BigramLanguageModel(nn.Module):
    """Neural Bigram Language Model parameterizing token transitions."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    @property
    def device(self) -> torch.device:
        return self.token_embedding_table.weight.device

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with automatic device alignment."""
        device = self.device
        idx = idx.to(device)

        # (B, T) -> (B, T, C)
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            targets = targets.to(device)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate next tokens with temperature scaling."""
        device = self.device
        idx = idx.to(device)

        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])  # (B, 1, C)
            logits = logits[:, -1, :]  # (B, C)

            if temperature > 0:
                scaled_logits = logits / max(temperature, 1e-5)
                probs = F.softmax(scaled_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


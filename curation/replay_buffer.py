"""Convex combination experience replay buffer and zero-RAM binary export pipeline (Phase 2.4)."""

import random
from pathlib import Path
from typing import Optional, Union
import numpy as np
from loguru import logger

from core.schemas import CleanedDocumentSchema
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.dataset import export_tokens_to_binary


class ExperienceReplayBuffer:
    """Convex combination experience replay buffer preventing model collapse (70% anchor / 30% web)."""

    def __init__(
        self,
        anchor_ratio: float = 0.70,
        seed: int = 42,
    ):
        if not (0.0 < anchor_ratio < 1.0):
            raise ValueError(f"anchor_ratio must be between 0.0 and 1.0, got {anchor_ratio}")

        self.anchor_ratio = anchor_ratio
        self.web_ratio = 1.0 - anchor_ratio
        self.rng = random.Random(seed)

        self.anchor_pool: list[str] = []
        self.web_pool: list[str] = []

    def add_anchor_data(self, texts: list[str]) -> int:
        """Add golden baseline anchor documents to the buffer."""
        clean_texts = [t.strip() for t in texts if t and t.strip()]
        self.anchor_pool.extend(clean_texts)
        logger.info(f"ExperienceReplayBuffer: Added {len(clean_texts)} anchor documents (Total: {len(self.anchor_pool)})")
        return len(self.anchor_pool)

    def add_web_data(self, documents: Union[list[CleanedDocumentSchema], list[str]]) -> int:
        """Add cleaned and deduplicated domain web documents to the buffer."""
        for item in documents:
            if isinstance(item, CleanedDocumentSchema):
                text = item.clean_text.strip()
            else:
                text = item.strip()
            if text:
                self.web_pool.append(text)
        logger.info(f"ExperienceReplayBuffer: Added {len(documents)} web documents (Total: {len(self.web_pool)})")
        return len(self.web_pool)

    def sample_batch(self, batch_size: int) -> list[str]:
        """Sample a mini-batch with exact alpha=0.70 anchor ratio."""
        if not self.anchor_pool and not self.web_pool:
            raise ValueError("ExperienceReplayBuffer is completely empty.")

        n_anchor = int(round(batch_size * self.anchor_ratio))
        n_web = batch_size - n_anchor

        batch: list[str] = []

        # Sample anchor
        if self.anchor_pool:
            sampled_anchor = self.rng.choices(self.anchor_pool, k=n_anchor)
            batch.extend(sampled_anchor)
        else:
            n_web = batch_size

        # Sample web
        if self.web_pool:
            sampled_web = self.rng.choices(self.web_pool, k=n_web)
            batch.extend(sampled_web)
        elif not self.anchor_pool:
            raise ValueError("Both pools empty.")

        self.rng.shuffle(batch)
        return batch

    def get_stats(self) -> dict:
        """Return current buffer statistics and empirical mixture ratio."""
        total = len(self.anchor_pool) + len(self.web_pool)
        return {
            "total_documents": total,
            "anchor_count": len(self.anchor_pool),
            "web_count": len(self.web_pool),
            "configured_anchor_ratio": self.anchor_ratio,
            "empirical_anchor_ratio": len(self.anchor_pool) / max(1, total),
        }

    def export_sharded_binaries(
        self,
        tokenizer: Optional[CharTokenizer] = None,
        output_dir: Union[Path, str] = "data",
        target_tokens: int = 10_000,
        train_val_split: float = 0.90,
    ) -> tuple[Path, Path]:
        """Interleave the replay corpus, tokenize, and export memory-mapped binary files for training.
        
        Returns:
            (train_bin_path: Path, val_bin_path: Path)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Build combined corpus with exact 70/30 sampling
        combined_texts: list[str] = []
        tokens_collected = 0

        # Create tokenizer if not provided
        if tokenizer is None:
            # Build tokenizer from existing anchor + web texts
            all_text = " ".join(self.anchor_pool[:20] + self.web_pool[:20])
            tokenizer = CharTokenizer(all_text)

        while tokens_collected < target_tokens:
            sample_unit = self.sample_batch(batch_size=10)
            combined_texts.extend(sample_unit)
            tokens_collected = sum(len(tokenizer.encode(t)) for t in combined_texts)

        # Encode full interleaved corpus
        all_tokens: list[int] = []
        for text in combined_texts:
            all_tokens.extend(tokenizer.encode(text))
            all_tokens.append(tokenizer.encode(" ")[0])  # token delimiter

        # Train / Val Split
        split_idx = int(len(all_tokens) * train_val_split)
        train_tokens = all_tokens[:split_idx]
        val_tokens = all_tokens[split_idx:]

        train_bin = out_path / "replay_train.bin"
        val_bin = out_path / "replay_val.bin"

        export_tokens_to_binary(train_tokens, train_bin)
        export_tokens_to_binary(val_tokens, val_bin)

        logger.info(
            f"ExperienceReplayBuffer: Exported {len(train_tokens):,} train tokens to {train_bin} and {len(val_tokens):,} val tokens to {val_bin}"
        )
        return train_bin, val_bin

    def clear(self) -> None:
        """Clear all buffers."""
        self.anchor_pool.clear()
        self.web_pool.clear()


"""Frozen reference model engine and log-probability caching manager."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from training.m5_preference.dataset import PreferenceCollator, PreferenceDataset
from training.m5_preference.log_probs import extract_batch_log_probs
from training.m5_preference.schemas import PreferenceBatch


class ReferenceModelEngine:
    """Manages frozen reference policies and log-probability caching for DPO."""

    def __init__(
        self,
        reference_model: nn.Module,
        device: Optional[torch.device] = None,
        length_normalize: bool = False,
    ):
        """Initialize the ReferenceModelEngine.

        Args:
            reference_model: Causal decoder model (MiniGPT) to serve as frozen anchor.
            device: Compute device for inference (defaults to model device).
            length_normalize: Whether to length-normalize completion log-probs.
        """
        self.device = device or next(reference_model.parameters()).device
        self.model = reference_model.to(self.device)
        self.length_normalize = length_normalize

        # Ensure reference model is strictly frozen in evaluation mode
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def compute_batch_log_probs(
        self, batch: PreferenceBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reference log-probabilities for chosen and rejected completions.

        Args:
            batch: PreferenceBatch containing chosen and rejected sequences.

        Returns:
            (ref_chosen_logps, ref_rejected_logps) as shape (B,) tensors.
        """
        self.model.eval()
        batch = batch.to(self.device)

        ref_chosen_logps = extract_batch_log_probs(
            model=self.model,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
            prompt_lengths=batch.prompt_lengths,
            length_normalize=self.length_normalize,
        )

        ref_rejected_logps = extract_batch_log_probs(
            model=self.model,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
            prompt_lengths=batch.prompt_lengths,
            length_normalize=self.length_normalize,
        )

        return ref_chosen_logps, ref_rejected_logps

    @torch.no_grad()
    def populate_batch_cache(self, batch: PreferenceBatch) -> PreferenceBatch:
        """Enrich a PreferenceBatch with precomputed reference log-probabilities.

        Args:
            batch: PreferenceBatch to augment.

        Returns:
            New PreferenceBatch instance with populated ref_chosen_logps and ref_rejected_logps.
        """
        ref_chosen, ref_rejected = self.compute_batch_log_probs(batch)
        batch.ref_chosen_logps = ref_chosen
        batch.ref_rejected_logps = ref_rejected
        return batch

    @torch.no_grad()
    def precompute_dataset_cache(
        self,
        dataset: PreferenceDataset,
        collator: PreferenceCollator,
        batch_size: int = 16,
    ) -> list[PreferenceBatch]:
        """Precompute reference log-probabilities for an entire dataset into cached batches.

        This reduces VRAM/CPU memory consumption during policy training by ~50%
        because the reference model does not need to remain active in memory.

        Args:
            dataset: PreferenceDataset to evaluate.
            collator: PreferenceCollator for batching.
            batch_size: Batch size for inference passes.

        Returns:
            List of cached PreferenceBatch instances.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        cached_batches: list[PreferenceBatch] = []
        logger.info(
            f"Precomputing reference log-probabilities across {len(dataset)} preference pairs..."
        )

        for batch in dataloader:
            cached_batch = self.populate_batch_cache(batch)
            cached_batches.append(cached_batch)

        logger.info(
            f"Successfully cached reference log-probabilities for {len(cached_batches)} batches."
        )
        return cached_batches


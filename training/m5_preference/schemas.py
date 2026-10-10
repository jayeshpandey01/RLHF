"""Preference data schemas and batch tensor structures for Milestone 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

from core.schemas import PreferencePairSchema


@dataclass
class PreferenceSample:
    """Internal dataclass representation of a preference sample."""

    prompt: str
    chosen: str
    rejected: str
    margin: float = 0.0
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_schema(self) -> PreferencePairSchema:
        """Convert to Pydantic PreferencePairSchema."""
        return PreferencePairSchema(
            prompt=self.prompt,
            chosen=self.chosen,
            rejected=self.rejected,
            margin=self.margin,
            category=self.category,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_schema(cls, schema: PreferencePairSchema) -> PreferenceSample:
        """Construct PreferenceSample from Pydantic schema."""
        return cls(
            prompt=schema.prompt,
            chosen=schema.chosen,
            rejected=schema.rejected,
            margin=schema.margin,
            category=schema.category,
            metadata=dict(schema.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize sample to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferenceSample:
        """Construct sample from dictionary."""
        return cls(
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            margin=float(data.get("margin", 0.0)),
            category=data.get("category", "general"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PreferenceBatch:
    """Collated PyTorch batch tensors for pairwise preference modeling."""

    chosen_input_ids: torch.Tensor        # Shape: (B, T) - torch.long
    chosen_attention_mask: torch.Tensor   # Shape: (B, T) - torch.long (1 for active, 0 for pad)
    rejected_input_ids: torch.Tensor      # Shape: (B, T) - torch.long
    rejected_attention_mask: torch.Tensor # Shape: (B, T) - torch.long
    margins: Optional[torch.Tensor] = None # Shape: (B,) - torch.float32
    prompt_lengths: Optional[torch.Tensor] = None # Shape: (B,) - torch.long
    ref_chosen_logps: Optional[torch.Tensor] = None # Shape: (B,) - torch.float32 (cached)
    ref_rejected_logps: Optional[torch.Tensor] = None # Shape: (B,) - torch.float32 (cached)

    def to(self, device: torch.device) -> PreferenceBatch:
        """Transfer all batch tensors to target compute device."""
        return PreferenceBatch(
            chosen_input_ids=self.chosen_input_ids.to(device),
            chosen_attention_mask=self.chosen_attention_mask.to(device),
            rejected_input_ids=self.rejected_input_ids.to(device),
            rejected_attention_mask=self.rejected_attention_mask.to(device),
            margins=self.margins.to(device) if self.margins is not None else None,
            prompt_lengths=self.prompt_lengths.to(device) if self.prompt_lengths is not None else None,
            ref_chosen_logps=self.ref_chosen_logps.to(device) if self.ref_chosen_logps is not None else None,
            ref_rejected_logps=self.ref_rejected_logps.to(device) if self.ref_rejected_logps is not None else None,
        )


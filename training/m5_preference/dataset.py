"""Preference dataset, formatting, and dual-sequence batch collator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from core.schemas import PreferencePairSchema
from training.m1_bigram.tokenizer import CharTokenizer
from training.m3_sft.prompt import ChatMLFormatter
from training.m5_preference.schemas import PreferenceBatch, PreferenceSample


class PreferenceCollator:
    """Vectorized collator that formats, tokenizes, and pads chosen and rejected pairs."""

    def __init__(
        self,
        tokenizer: CharTokenizer,
        max_seq_len: int = 128,
        pad_token_id: Optional[int] = None,
    ):
        """Initialize the PreferenceCollator.

        Args:
            tokenizer: CharTokenizer instance with stoi mapping.
            max_seq_len: Maximum context window length (default 128).
            pad_token_id: Token ID used for sequence padding.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        elif " " in self.tokenizer.stoi:
            self.pad_token_id = self.tokenizer.stoi[" "]
        else:
            self.pad_token_id = 0

    def format_pair(self, prompt: str, completion: str) -> str:
        """Format a prompt and completion into canonical ChatML string."""
        prompt_clean = prompt.strip()
        comp_clean = completion.strip()
        return (
            f"{ChatMLFormatter.START_TAG}user\n{prompt_clean}{ChatMLFormatter.END_TAG}\n"
            f"{ChatMLFormatter.START_TAG}assistant\n{comp_clean}{ChatMLFormatter.END_TAG}\n"
        )

    def collate_single(
        self, sample: PreferenceSample
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Tokenize and pad a single preference sample.

        Returns:
            (chosen_ids, chosen_mask, rejected_ids, rejected_mask)
        """
        chosen_text = self.format_pair(sample.prompt, sample.chosen)
        rejected_text = self.format_pair(sample.prompt, sample.rejected)

        toks_w = self.tokenizer.encode(chosen_text)
        toks_l = self.tokenizer.encode(rejected_text)

        # Truncate if sequence exceeds max_seq_len
        if len(toks_w) > self.max_seq_len:
            toks_w = toks_w[: self.max_seq_len]
        if len(toks_l) > self.max_seq_len:
            toks_l = toks_l[: self.max_seq_len]

        pad_w = self.max_seq_len - len(toks_w)
        pad_l = self.max_seq_len - len(toks_l)

        chosen_ids = toks_w + [self.pad_token_id] * pad_w
        chosen_mask = [1] * len(toks_w) + [0] * pad_w

        rejected_ids = toks_l + [self.pad_token_id] * pad_l
        rejected_mask = [1] * len(toks_l) + [0] * pad_l

        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

    def prompt_prefix_length(self, prompt: str) -> int:
        """Calculate token length of prompt prefix in ChatML format."""
        prompt_prefix = (
            f"{ChatMLFormatter.START_TAG}user\n{prompt.strip()}{ChatMLFormatter.END_TAG}\n"
            f"{ChatMLFormatter.START_TAG}assistant\n"
        )
        return min(len(self.tokenizer.encode(prompt_prefix)), self.max_seq_len)

    def __call__(self, batch: list[PreferenceSample]) -> PreferenceBatch:
        """Collate a list of PreferenceSample instances into a PreferenceBatch."""
        chosen_ids_list = []
        chosen_mask_list = []
        rejected_ids_list = []
        rejected_mask_list = []
        margins_list = []
        prompt_lengths_list = []

        for sample in batch:
            c_ids, c_mask, r_ids, r_mask = self.collate_single(sample)
            chosen_ids_list.append(c_ids)
            chosen_mask_list.append(c_mask)
            rejected_ids_list.append(r_ids)
            rejected_mask_list.append(r_mask)
            margins_list.append(sample.margin)
            prompt_lengths_list.append(self.prompt_prefix_length(sample.prompt))

        return PreferenceBatch(
            chosen_input_ids=torch.tensor(chosen_ids_list, dtype=torch.long),
            chosen_attention_mask=torch.tensor(chosen_mask_list, dtype=torch.long),
            rejected_input_ids=torch.tensor(rejected_ids_list, dtype=torch.long),
            rejected_attention_mask=torch.tensor(rejected_mask_list, dtype=torch.long),
            margins=torch.tensor(margins_list, dtype=torch.float32),
            prompt_lengths=torch.tensor(prompt_lengths_list, dtype=torch.long),
        )


class PreferenceDataset(Dataset):
    """PyTorch Dataset container for pairwise preference samples."""

    def __init__(self, samples: list[PreferenceSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PreferenceSample:
        return self.samples[idx]

    @classmethod
    def from_schemas(cls, schemas: list[PreferencePairSchema]) -> PreferenceDataset:
        """Construct dataset from core Pydantic schemas."""
        samples = [PreferenceSample.from_schema(s) for s in schemas]
        return cls(samples)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> PreferenceDataset:
        """Load preference dataset from a JSON list file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [PreferenceSample.from_dict(d) for d in data]
        return cls(samples)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """Export dataset to JSON file."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.samples], f, indent=2)


def create_default_preference_pairs() -> list[PreferenceSample]:
    """Curate canonical preference pairs across ML theory, transformers, and alignment."""
    pairs = [
        PreferenceSample(
            prompt="Explain why AdamW decouples weight decay from gradient updates.",
            chosen=(
                "AdamW applies weight decay directly to model parameters W = W - lr * lambda * W, "
                "decoupling it from adaptive moment estimation. In standard Adam with L2 regularization, "
                "decay gradients are normalized by the second moment, penalizing frequent parameters less."
            ),
            rejected=(
                "AdamW is just normal Adam with a different learning rate decay schedule to make it faster."
            ),
            margin=1.0,
            category="ml_theory",
        ),
        PreferenceSample(
            prompt="How does FlashAttention avoid high-bandwidth memory (HBM) bottlenecks?",
            chosen=(
                "FlashAttention tiles query, key, and value matrices into blocks that fit within on-chip SRAM. "
                "It computes softmax incrementally using the online scaling technique, fusing attention "
                "computation and avoiding writing intermediate N x N attention matrices back to HBM [Dao et al., 2022]."
            ),
            rejected=(
                "FlashAttention uses a faster GPU architecture and bigger memory bandwidth to speed up training."
            ),
            margin=1.0,
            category="transformer_internals",
        ),
        PreferenceSample(
            prompt="Why is low-rank adapter matrix B initialized to zero in LoRA?",
            chosen=(
                "In LoRA weight update Delta W = (alpha / r) * B @ A, matrix A is initialized from a Gaussian "
                "distribution and matrix B is initialized to strictly zero. This ensures Delta W = 0 at step zero, "
                "preserving exact pretrained model outputs before optimization begins [Hu et al., 2021]."
            ),
            rejected=(
                "Both matrices A and B in LoRA are randomly initialized with Gaussian weights so the model learns fast."
            ),
            margin=1.0,
            category="peft_sft",
        ),
        PreferenceSample(
            prompt="What is the key benefit of Group Relative Policy Optimization (GRPO) over PPO?",
            chosen=(
                "GRPO samples a group of completions for each prompt and normalizes advantages intra-group. "
                "This completely eliminates the value network (critic), saving approximately 50% GPU memory "
                "and removing critic approximation instability [Shao et al., 2024]."
            ),
            rejected=(
                "GRPO trains a bigger critic network to evaluate multi-step reasoning."
            ),
            margin=1.0,
            category="rlhf_grpo",
        ),
        PreferenceSample(
            prompt="Why must prompt tokens be masked with -100 during Supervised Fine-Tuning?",
            chosen=(
                "Masking prompt tokens to ignore_index=-100 ensures cross-entropy loss is computed strictly "
                "on assistant completion tokens. This prevents the model from wasting capacity memorizing prompt "
                "syntax and prevents role bleed during multi-turn generation."
            ),
            rejected=(
                "Prompt tokens are masked to save disk space and reduce batch tensor size."
            ),
            margin=1.0,
            category="peft_sft",
        ),
    ]
    return pairs


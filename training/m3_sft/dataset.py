"""Dataset and batch collation for Supervised Fine-Tuning with completion-only loss masking."""

from dataclasses import dataclass
from typing import Optional, Union
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

from training.m1_bigram.tokenizer import CharTokenizer
from training.m3_sft.prompt import ChatMessage, InstructionExample, ChatMLFormatter


@dataclass
class BatchCollatorOutput:
    """Collated PyTorch batch tensors ready for transformer training."""

    input_ids: torch.Tensor       # Shape: (B, T) - torch.long
    targets: torch.Tensor         # Shape: (B, T) - torch.long (prompt tokens = -100)
    attention_mask: torch.Tensor  # Shape: (B, T) - torch.long (1 for token, 0 for pad)


class CompletionMaskCollator:
    """Collator that identifies assistant response spans and masks prompts to -100.

    Ensures that loss and gradients are computed strictly over assistant tokens
    and turn termination tokens, preventing prompt memorization and role bleed.
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        max_seq_len: int = 256,
        ignore_index: int = -100,
        pad_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        # Default pad token id to 0 or space if available
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif " " in self.tokenizer.stoi:
            self.pad_token_id = self.tokenizer.stoi[" "]
        else:
            self.pad_token_id = 0

    def collate_single(self, text: str) -> tuple[list[int], list[int], list[int]]:
        """Tokenize a single ChatML dialogue and construct completion-masked input/target pair.

        Returns:
            input_ids: list of input token IDs [t_0, ..., t_{N-2}]
            targets: list of target token IDs [y_0, ..., y_{N-2}], where y_i = t_{i+1} if
                     t_{i+1} is in an assistant span, else ignore_index (-100).
            mask: list of binary flags [1, ..., 1] for active sequence length.
        """
        # Determine character-level active assistant spans
        char_is_assistant = [False] * len(text)
        spans = ChatMLFormatter.extract_assistant_spans(text)
        for start_char, end_char in spans:
            # Mark assistant content as active
            for c_idx in range(start_char, end_char):
                char_is_assistant[c_idx] = True
            # Also mark the trailing <|im_end|> token as active so model learns when to stop
            end_tag_len = len(ChatMLFormatter.END_TAG)
            if end_char + end_tag_len <= len(text):
                for c_idx in range(end_char, end_char + end_tag_len):
                    char_is_assistant[c_idx] = True

        # Encode characters to tokens while preserving character-to-token active flag
        tokens: list[int] = []
        is_active_token: list[bool] = []
        for c_idx, ch in enumerate(text):
            if ch in self.tokenizer.stoi:
                tokens.append(self.tokenizer.stoi[ch])
                is_active_token.append(char_is_assistant[c_idx])

        if len(tokens) < 2:
            # Fallback for degenerate empty text
            return [0], [self.ignore_index], [1]

        # Truncate sequence to max_seq_len + 1 while guaranteeing assistant tokens are preserved
        if len(tokens) > self.max_seq_len + 1:
            if True in is_active_token:
                first_active = is_active_token.index(True)
                # Keep prompt context preceding assistant start
                prompt_budget = max(4, (self.max_seq_len + 1) // 3)
                start_idx = max(0, first_active - prompt_budget)
                end_idx = min(len(tokens), start_idx + self.max_seq_len + 1)
                tokens = tokens[start_idx:end_idx]
                is_active_token = is_active_token[start_idx:end_idx]
            else:
                tokens = tokens[: self.max_seq_len + 1]
                is_active_token = is_active_token[: self.max_seq_len + 1]

        # Causal shift: input is tokens[:-1], target predicts tokens[1:]
        input_ids = tokens[:-1]
        targets = []
        for i in range(len(input_ids)):
            target_token = tokens[i + 1]
            # If the next token is an assistant token, compute loss on it
            if is_active_token[i + 1]:
                targets.append(target_token)
            else:
                targets.append(self.ignore_index)

        mask = [1] * len(input_ids)
        return input_ids, targets, mask

    def __call__(self, texts: list[str]) -> BatchCollatorOutput:
        """Batch-level collation with right-padding and completion masking."""
        batch_inputs = []
        batch_targets = []
        batch_masks = []

        for text in texts:
            inp, tgt, msk = self.collate_single(text)
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            batch_masks.append(msk)

        # Determine batch maximum sequence length
        max_len = max(len(inp) for inp in batch_inputs)
        max_len = min(max_len, self.max_seq_len)

        padded_inputs = []
        padded_targets = []
        padded_masks = []

        for inp, tgt, msk in zip(batch_inputs, batch_targets, batch_masks):
            cur_len = len(inp)
            pad_len = max_len - cur_len
            if pad_len > 0:
                p_inp = inp + [self.pad_token_id] * pad_len
                p_tgt = tgt + [self.ignore_index] * pad_len
                p_msk = msk + [0] * pad_len
            else:
                p_inp = inp[:max_len]
                p_tgt = tgt[:max_len]
                p_msk = msk[:max_len]

            padded_inputs.append(p_inp)
            padded_targets.append(p_tgt)
            padded_masks.append(p_msk)

        return BatchCollatorOutput(
            input_ids=torch.tensor(padded_inputs, dtype=torch.long),
            targets=torch.tensor(padded_targets, dtype=torch.long),
            attention_mask=torch.tensor(padded_masks, dtype=torch.long),
        )


class HarmonizedSFTDataset(Dataset):
    """PyTorch Dataset implementing experience replay mixture sampling.

    Draws samples with convex combination:
        P_replay = anchor_ratio * P_anchor + (1 - anchor_ratio) * P_web
    """

    def __init__(
        self,
        anchor_examples: list[InstructionExample],
        web_examples: Optional[list[InstructionExample]] = None,
        anchor_ratio: float = 0.70,
        epoch_length: int = 1000,
        seed: int = 42,
    ):
        if not anchor_examples:
            raise ValueError("HarmonizedSFTDataset requires at least one anchor instruction example.")

        self.anchor_examples = anchor_examples
        self.web_examples = web_examples or []
        self.anchor_ratio = anchor_ratio
        self.epoch_length = epoch_length
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> str:
        """Sample an instruction dialogue and return its serialized ChatML string."""
        # Decide whether to draw from anchor or web
        draw_anchor = True
        if self.web_examples and self.rng.random() > self.anchor_ratio:
            draw_anchor = False

        if draw_anchor or not self.web_examples:
            example = self.rng.choice(self.anchor_examples)
        else:
            example = self.rng.choice(self.web_examples)

        return ChatMLFormatter.format_dialogue(example.messages)


def create_default_anchor_examples() -> list[InstructionExample]:
    """Factory creating diverse baseline anchor instruction examples."""
    examples = [
        InstructionExample(
            messages=[
                ChatMessage(role="system", content="You are a helpful and concise AI assistant."),
                ChatMessage(role="user", content="What is the purpose of attention in Transformers?"),
                ChatMessage(
                    role="assistant",
                    content="Attention allows models to dynamically weight and contextualize all tokens across a sequence.",
                ),
            ],
            provenance_source="golden_anchor",
            category="reasoning",
        ),
        InstructionExample(
            messages=[
                ChatMessage(role="system", content="You are an expert Shakespearean scholar."),
                ChatMessage(role="user", content="Speak of honour and virtue in poetic verse."),
                ChatMessage(
                    role="assistant",
                    content="True honour lives not in the praise of men, but in the steadfast virtue of the soul.",
                ),
            ],
            provenance_source="golden_anchor",
            category="literature",
        ),
        InstructionExample(
            messages=[
                ChatMessage(role="system", content="You are a helpful and concise AI assistant."),
                ChatMessage(role="user", content="Calculate the sum of twelve and twenty-eight."),
                ChatMessage(
                    role="assistant",
                    content="The sum of 12 and 28 is 40.",
                ),
            ],
            provenance_source="golden_anchor",
            category="arithmetic",
        ),
        InstructionExample(
            messages=[
                ChatMessage(role="system", content="You are a helpful and concise AI assistant."),
                ChatMessage(role="user", content="What is machine learning?"),
                ChatMessage(
                    role="assistant",
                    content="Machine learning is a field of artificial intelligence focused on training algorithms to learn patterns from data.",
                ),
            ],
            provenance_source="golden_anchor",
            category="factual",
        ),
    ]
    return examples


def create_web_curation_examples(
    documents: Union[list[str], list["CleanedDocumentSchema"]],
) -> list[InstructionExample]:
    """Convert curated web articles into structured instruction-response examples."""
    from core.schemas import CleanedDocumentSchema

    web_examples = []
    for doc in documents:
        if isinstance(doc, CleanedDocumentSchema):
            clean_text = doc.clean_text.strip()
        else:
            clean_text = str(doc).strip()

        if not clean_text or len(clean_text) < 40:
            continue
        # Construct summary instruction
        summary = clean_text[:200].rsplit(".", 1)[0] + "."
        example = InstructionExample(
            messages=[
                ChatMessage(role="system", content="You are a knowledge assistant synthesizing facts."),
                ChatMessage(role="user", content="Summarize the core knowledge from this domain document."),
                ChatMessage(role="assistant", content=summary),
            ],
            provenance_source="web_curation",
            category="domain_knowledge",
        )
        web_examples.append(example)
    return web_examples



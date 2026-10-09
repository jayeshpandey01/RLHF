"""ChatML conversation formatting, instruction schemas, and delimiter mechanics."""

from dataclasses import dataclass, field
from typing import Literal
import re


@dataclass
class ChatMessage:
    """Represents a single utterance in a conversational dialogue."""

    role: Literal["system", "user", "assistant"]
    content: str

    @classmethod
    def from_schema(cls, schema: "ChatMessageSchema") -> "ChatMessage":
        """Construct ChatMessage from Pydantic ChatMessageSchema."""
        return cls(role=schema.role, content=schema.content)

    def to_schema(self) -> "ChatMessageSchema":
        """Convert to Pydantic ChatMessageSchema."""
        from core.schemas import ChatMessageSchema
        return ChatMessageSchema(role=self.role, content=self.content)


@dataclass
class InstructionExample:
    """Encapsulates a complete multi-turn dialogue with provenance tracking."""

    messages: list[ChatMessage] = field(default_factory=list)
    provenance_source: str = "golden_anchor"  # e.g., "golden_anchor" or "web_curation"
    category: str = "general"

    def add_turn(self, role: Literal["system", "user", "assistant"], content: str) -> None:
        """Append a conversational turn."""
        self.messages.append(ChatMessage(role=role, content=content))

    @classmethod
    def from_schema(cls, schema: "TrainingSampleSchema") -> "InstructionExample":
        """Construct InstructionExample from Pydantic TrainingSampleSchema."""
        messages = [ChatMessage.from_schema(m) for m in schema.messages]
        return cls(
            messages=messages,
            provenance_source=schema.origin,
            category="general",
        )

    def to_schema(self, source_url: str = "", source_hash: str = "") -> "TrainingSampleSchema":
        """Convert to Pydantic TrainingSampleSchema."""
        from core.schemas import TrainingSampleSchema
        origin = "anchor_golden" if "anchor" in self.provenance_source else "web_scraped"
        return TrainingSampleSchema(
            messages=[m.to_schema() for m in self.messages],
            source_url=source_url,
            source_hash=source_hash,
            origin=origin,
        )


class ChatMLFormatter:
    """Canonical ChatML serialization and assistant span detection.

    Schema:
    <|im_start|>system
    {system_content}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    {assistant_content}<|im_end|>
    """

    START_TAG = "<|im_start|>"
    END_TAG = "<|im_end|>"
    ASSISTANT_HEADER = "<|im_start|>assistant\n"

    @classmethod
    def format_dialogue(cls, messages: list[ChatMessage]) -> str:
        """Serialize a full conversational trajectory into a canonical ChatML string."""
        formatted_turns = []
        for msg in messages:
            clean_content = msg.content.strip()
            turn_str = f"{cls.START_TAG}{msg.role}\n{clean_content}{cls.END_TAG}\n"
            formatted_turns.append(turn_str)
        return "".join(formatted_turns)

    @classmethod
    def format_prompt_only(cls, messages: list[ChatMessage]) -> str:
        """Serialize system and user turns, appending '<|im_start|>assistant\\n' for inference generation."""
        # Find all turns up to before the final assistant response
        prompt_messages = []
        for msg in messages:
            if msg.role != "assistant":
                prompt_messages.append(msg)
            else:
                break
        formatted_prompt = cls.format_dialogue(prompt_messages)
        return f"{formatted_prompt}{cls.ASSISTANT_HEADER}"

    @classmethod
    def extract_assistant_spans(cls, formatted_text: str) -> list[tuple[int, int]]:
        """Identify (start_char_idx, end_char_idx) character spans of assistant completions.

        Start index is right after `<|im_start|>assistant\\n`.
        End index is right before `<|im_end|>`.
        """
        spans: list[tuple[int, int]] = []
        pattern = re.compile(
            re.escape(cls.ASSISTANT_HEADER) + r"(.*?)" + re.escape(cls.END_TAG),
            re.DOTALL,
        )
        for match in pattern.finditer(formatted_text):
            # match.start(1) is the first character of assistant content
            # match.end(1) is right after the last character of assistant content
            spans.append((match.start(1), match.end(1)))
        return spans


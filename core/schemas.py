"""Pydantic V2 data contracts and validation schemas."""

import hashlib
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator


class UserQuerySchema(BaseModel):
    """Input query received from user chat interface."""

    query: str = Field(..., min_length=1, description="Raw query from user")
    session_id: str = Field(default="default-session", description="Unique session tracking ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp"
    )


class ScrapedDocumentSchema(BaseModel):
    """Raw document scraped from the web before filtering."""

    url: str = Field(..., description="Source URL of scraped document")
    raw_html: str = Field(..., description="Raw HTML or unprocessed body text")
    clean_markdown: str = Field(default="", description="Extracted main content markdown")
    sha256_hash: str = Field(default="", description="Cryptographic SHA-256 hash of content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Ingestion timestamp"
    )

    def compute_hash(self) -> str:
        """Compute and set SHA-256 hash of clean_markdown."""
        content = self.clean_markdown or self.raw_html
        self.sha256_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.sha256_hash


class CleanedDocumentSchema(BaseModel):
    """Validated, deduplicated document ready for knowledge indexing or dataset generation."""

    url: str
    clean_text: str
    word_count: int
    line_repetition_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of duplicate lines (must be < 0.20)"
    )
    sha256_hash: str
    minhash_signature: list[int] = Field(default_factory=list, description="MinHash signature vector")
    is_valid: bool = True


class ChatMessageSchema(BaseModel):
    """Individual message in ChatML / Llama-3 format."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_chat_message(self) -> "ChatMessage":
        """Convert Pydantic schema to training ChatMessage."""
        from training.m3_sft.prompt import ChatMessage
        return ChatMessage(role=self.role, content=self.content)


class TrainingSampleSchema(BaseModel):
    """Validated sample formatted for SFT or preference training."""

    messages: list[ChatMessageSchema] = Field(..., min_length=2)
    source_url: str = ""
    source_hash: str = ""
    origin: Literal["anchor_golden", "web_scraped"] = "web_scraped"

    @field_validator("messages")
    @classmethod
    def validate_conversation_flow(cls, v: list[ChatMessageSchema]) -> list[ChatMessageSchema]:
        """Ensure last message is from assistant for completion loss masking."""
        if v[-1].role != "assistant":
            raise ValueError("Training sample must end with an assistant completion.")
        return v

    def to_instruction_example(self) -> "InstructionExample":
        """Convert Pydantic schema to training InstructionExample."""
        from training.m3_sft.prompt import InstructionExample
        return InstructionExample.from_schema(self)



class BenchmarkCaseSchema(BaseModel):
    """Golden evaluation question for regression testing."""

    case_id: str
    question: str
    ground_truth_context: str
    required_citations: list[str] = Field(default_factory=list)
    golden_keywords: list[str] = Field(default_factory=list)


class EvaluationReportSchema(BaseModel):
    """Automated benchmark result and release gate verdict."""

    candidate_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    perplexity: float
    citation_faithfulness: float = Field(..., ge=0.0, le=1.0)
    base_accuracy: float = Field(..., ge=0.0, le=1.0)
    candidate_accuracy: float = Field(..., ge=0.0, le=1.0)
    regression_delta: float
    decision: Literal["APPROVED", "REJECTED"]
    failure_reason: str = ""


class ProvenanceRecordSchema(BaseModel):
    """Immutable audit trail entry for ingested documents."""

    url: str
    sha256_hash: str
    source_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status_code: int = 200
    latency_ms: float = 0.0
    content_length: int = 0

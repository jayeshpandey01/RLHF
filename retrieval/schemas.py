from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RetrievalDocument:
    """Represents a document stored in the retrieval index."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    url: Optional[str] = None


@dataclass
class SearchResult:
    """Represents a ranked document result returned by retrieval search."""
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    url: Optional[str] = None
    rank: int = 1
    sub_scores: Dict[str, float] = field(default_factory=dict)


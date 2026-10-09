"""Train-test benchmark contamination detector (Dodge et al., 2021 13-gram decontamination)."""

import re
from typing import Optional
from core.schemas import BenchmarkCaseSchema


class BenchmarkContaminationGuard:
    """Detects and quarantines web documents that leak held-out test evaluation queries or answers."""

    def __init__(self, ngram_size: int = 13):
        self.ngram_size = ngram_size
        self._forbidden_ngrams: set[str] = set()
        self._registered_benchmarks: int = 0

    @staticmethod
    def normalize_tokens(text: str) -> list[str]:
        """Normalize text into lowercase alphanumeric tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def _extract_ngrams(self, tokens: list[str]) -> list[str]:
        """Extract contiguous N-grams from token sequence."""
        if len(tokens) < self.ngram_size:
            return []
        return [
            " ".join(tokens[i : i + self.ngram_size])
            for i in range(len(tokens) - self.ngram_size + 1)
        ]

    def add_benchmark_text(self, text: str) -> int:
        """Extract and index forbidden 13-grams from benchmark text."""
        tokens = self.normalize_tokens(text)
        ngrams = self._extract_ngrams(tokens)
        self._forbidden_ngrams.update(ngrams)
        self._registered_benchmarks += 1
        return len(ngrams)

    def add_benchmark_case(self, case: BenchmarkCaseSchema) -> int:
        """Index question, ground-truth context, and golden keywords from benchmark case."""
        combined_text = f"{case.question} {case.ground_truth_context} {' '.join(case.golden_keywords)}"
        return self.add_benchmark_text(combined_text)

    def is_contaminated(self, document_text: str) -> tuple[bool, Optional[str]]:
        """Check if document contains any forbidden 13-grams from the evaluation suite.
        
        Returns:
            (is_contaminated: bool, matched_ngram: Optional[str])
        """
        if not self._forbidden_ngrams:
            return False, None

        tokens = self.normalize_tokens(document_text)
        if len(tokens) < self.ngram_size:
            return False, None

        for i in range(len(tokens) - self.ngram_size + 1):
            cand_ngram = " ".join(tokens[i : i + self.ngram_size])
            if cand_ngram in self._forbidden_ngrams:
                return True, cand_ngram

        return False, None

    def size(self) -> int:
        """Return number of forbidden N-grams tracked."""
        return len(self._forbidden_ngrams)

    def clear(self) -> None:
        """Reset the contamination index."""
        self._forbidden_ngrams.clear()
        self._registered_benchmarks = 0


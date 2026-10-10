"""Industrial data cleaning, quality filtering, and heuristic validation (C4 / FineWeb standards)."""

import re

# Standard English high-frequency stopwords for density filtering
COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

VALID_TERMINAL_PUNCTUATION = {".", "!", "?", "\"", "'", "”", "’"}


class C4QualityFilter:
    """Rigorous heuristic quality filter enforcing C4, RefinedWeb, and FineWeb bounds."""

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100_000,
        max_line_repetition_rate: float = 0.20,
        max_char_repetition_rate: float = 0.15,
        min_stopword_density: float = 0.10,
        min_terminal_punct_ratio: float = 0.70,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.max_line_repetition_rate = max_line_repetition_rate
        self.max_char_repetition_rate = max_char_repetition_rate
        self.min_stopword_density = min_stopword_density
        self.min_terminal_punct_ratio = min_terminal_punct_ratio

    def evaluate(self, text: str) -> tuple[bool, str, dict]:
        """Evaluate text against C4 quality heuristics.
        
        Returns:
            (is_valid: bool, failure_reason: str, metrics: dict)
        """
        if not text or not text.strip():
            return False, "EMPTY_TEXT", {"word_count": 0}

        cleaned = text.strip()
        words = re.findall(r"\b\w+\b", cleaned.lower())
        word_count = len(words)

        # 1. Word Count Bound [min_words, max_words]
        if word_count < self.min_words:
            return False, f"WORD_COUNT_TOO_LOW ({word_count} < {self.min_words})", {"word_count": word_count}
        if word_count > self.max_words:
            return False, f"WORD_COUNT_TOO_HIGH ({word_count} > {self.max_words})", {"word_count": word_count}

        # 2. Line Repetition Rate R_line = 1 - (|unique lines| / |total lines|)
        lines = [line.strip().lower() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return False, "NO_VALID_LINES", {"word_count": word_count}

        unique_lines = set(lines)
        line_rep_rate = 1.0 - (len(unique_lines) / float(len(lines)))
        if line_rep_rate > self.max_line_repetition_rate:
            return False, f"EXCESSIVE_LINE_REPETITION ({line_rep_rate:.3f} > {self.max_line_repetition_rate})", {
                "word_count": word_count,
                "line_repetition_rate": line_rep_rate,
            }

        # 3. Stopword Density (filters SEO spam and keyword stuffing)
        stopword_count = sum(1 for w in words if w in COMMON_STOPWORDS)
        stopword_density = stopword_count / float(word_count)
        if stopword_density < self.min_stopword_density:
            return False, f"LOW_STOPWORD_DENSITY ({stopword_density:.3f} < {self.min_stopword_density})", {
                "word_count": word_count,
                "stopword_density": stopword_density,
            }

        # 4. Character / N-gram Repetition Rate (detects generator loops / corrupted dumps)
        char_rep_rate = self._compute_char_ngram_repetition(cleaned)
        if char_rep_rate > self.max_char_repetition_rate:
            return False, f"EXCESSIVE_CHAR_REPETITION ({char_rep_rate:.3f} > {self.max_char_repetition_rate})", {
                "word_count": word_count,
                "char_repetition_rate": char_rep_rate,
            }

        # 5. Terminal Punctuation Ratio (detects broken sentences / truncated scraping)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        sentences = [
            line
            for line in lines
            if not line.startswith("#")
            and not re.match(r"[-*]\s+\*\*[^*]+\*\*:", line)
        ]
        if sentences:
            punct_count = sum(1 for s in sentences if s[-1] in VALID_TERMINAL_PUNCTUATION)
            punct_ratio = punct_count / float(len(sentences))
            if punct_ratio < self.min_terminal_punct_ratio:
                return False, f"LOW_TERMINAL_PUNCTUATION ({punct_ratio:.3f} < {self.min_terminal_punct_ratio})", {
                    "word_count": word_count,
                    "terminal_punct_ratio": punct_ratio,
                }
        else:
            punct_ratio = 1.0

        metrics = {
            "word_count": word_count,
            "line_repetition_rate": line_rep_rate,
            "stopword_density": stopword_density,
            "char_repetition_rate": char_rep_rate,
            "terminal_punct_ratio": punct_ratio,
        }
        return True, "PASSED", metrics

    def _compute_char_ngram_repetition(self, text: str, n: int = 4) -> float:
        """Measure whether one character n-gram dominates the normalized text."""
        normalized = re.sub(r"\s+", "", text.lower())
        total_ngrams = len(normalized) - n + 1
        if total_ngrams < n:
            return 0.0

        counts: dict[str, int] = {}
        for i in range(total_ngrams):
            gram = normalized[i : i + n]
            counts[gram] = counts.get(gram, 0) + 1

        return max(counts.values()) / float(total_ngrams)

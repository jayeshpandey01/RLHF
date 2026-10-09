"""Citation Extraction, Token Faithfulness, and RAG Triad Attribution Verifier (Phase 4.2).

Implements ALCE Citation Precision and Recall (Gao et al., EMNLP 2023) and
RAG Triad Faithfulness (Es et al., EACL 2024) to programmatically audit model
responses for factual grounding and citation fidelity.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union
import re
from loguru import logger

from evaluation.benchmark_suite import GoldenTestCase

# Standard English stopwords to isolate content keywords during lexical entailment
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "that",
    "this", "these", "those", "it", "its", "as", "of",
}

# Conversational boilerplate patterns that carry zero factual payload
BOILERPLATE_PATTERNS = [
    r"^(?:here is|below is|sure|certainly|in conclusion|to summarize|summary|note that)\b",
    r"^(?:i hope this helps|feel free to ask|let me know|thanks for asking)\b",
    r"^(?:as mentioned|overall|in summary|firstly|secondly|finally)\b",
]


@dataclass
class CitationTag:
    """Parsed citation marker with normalized reference indices."""

    raw_tag: str
    tag_type: Literal["numeric", "numeric_range", "author_year", "markdown_url", "named_source"]
    normalized_indices: list[str]
    start_char: int
    end_char: int


@dataclass
class SentenceClaim:
    """Individual segmented sentence with attached citations and entailment audit."""

    sentence_idx: int
    raw_text: str
    clean_claim: str
    citations: list[CitationTag] = field(default_factory=list)
    is_factual: bool = True
    supported_by_citations: bool = False
    supporting_citations: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    context_entailed: bool = False
    entailment_score: float = 0.0


@dataclass
class CitationMetrics:
    """Quantitative citation precision, recall, and factual faithfulness metrics."""

    citation_precision: float
    citation_recall: float
    citation_f1: float
    faithfulness_score: float
    total_sentences: int
    factual_sentences: int
    total_citations: int
    valid_citations: int
    unsupported_claims: list[str] = field(default_factory=list)
    hallucinated_citations: list[str] = field(default_factory=list)
    summary: str = ""


class CitationVerifier:
    """Extracts citation markers, segments atomic propositions, and verifies entailment."""

    # Regex patterns for citation markers
    NUMERIC_LIST_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
    NUMERIC_RANGE_PATTERN = re.compile(r"\[(\d+)\s*[-–—]\s*(\d+)\]")
    AUTHOR_YEAR_BRACKET = re.compile(
        r"\[([A-Z][a-zA-Z0-9\s&.,-]+(?:,\s*\d{4})?)\]"
    )
    AUTHOR_YEAR_PAREN = re.compile(
        r"\(([A-Z][a-zA-Z0-9\s&.,-]+,\s*\d{4})\)"
    )

    MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    NAMED_SOURCE_PATTERN = re.compile(
        r"\[(?:Source|Doc|Ref|Document)\s*([0-9a-zA-Z_-]+)\]", re.IGNORECASE
    )

    # Common abbreviations to prevent premature sentence boundary splitting
    ABBREVIATIONS = [
        "et al.", "i.e.", "e.g.", "fig.", "vs.", "dr.", "mr.", "mrs.", "ms.",
        "no.", "p.", "pp.", "vol.", "al.", "approx.", "etc.", "cf.",
    ]

    def __init__(
        self,
        entailment_threshold: float = 0.50,
        strict_mode: bool = True,
        stopwords: Optional[set[str]] = None,
    ):
        self.entailment_threshold = entailment_threshold
        self.strict_mode = strict_mode
        self.stopwords = stopwords or DEFAULT_STOPWORDS

    @staticmethod
    def tokenize_words(text: str) -> list[str]:
        """Normalize text into lowercase alphanumeric word tokens."""
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    @staticmethod
    def _tokens_match(t1: str, t2: str) -> bool:
        """Morphological prefix/stem matching for English terms."""
        if t1 == t2:
            return True
        if len(t1) >= 4 and len(t2) >= 4:
            min_p = min(4, len(t1), len(t2))
            if t1[:min_p] == t2[:min_p] and abs(len(t1) - len(t2)) <= 6:
                return True
        return False

    def extract_citations(self, text: str) -> list[CitationTag]:
        """Parse all inline citation markers from text into structured CitationTag objects."""
        tags: list[CitationTag] = []

        # 1. Numeric ranges: [1-3], [2-4]
        for match in self.NUMERIC_RANGE_PATTERN.finditer(text):
            start_num = int(match.group(1))
            end_num = int(match.group(2))
            if start_num <= end_num and end_num - start_num < 20:
                indices = [str(n) for n in range(start_num, end_num + 1)]
                tags.append(
                    CitationTag(
                        raw_tag=match.group(0),
                        tag_type="numeric_range",
                        normalized_indices=indices,
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )

        # 2. Numeric single or comma lists: [1], [1, 2]
        for match in self.NUMERIC_LIST_PATTERN.finditer(text):
            raw = match.group(0)
            if any(t.start_char == match.start() and t.end_char == match.end() for t in tags):
                continue
            indices = [x.strip() for x in match.group(1).split(",")]
            tags.append(
                CitationTag(
                    raw_tag=raw,
                    tag_type="numeric",
                    normalized_indices=indices,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        # 3. Named sources: [Source 1], [Doc 2]
        for match in self.NAMED_SOURCE_PATTERN.finditer(text):
            tags.append(
                CitationTag(
                    raw_tag=match.group(0),
                    tag_type="named_source",
                    normalized_indices=[match.group(1).strip()],
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        # 4. Author-Year in brackets: [Hu et al., 2021]
        for match in self.AUTHOR_YEAR_BRACKET.finditer(text):
            raw = match.group(0)
            if any(t.start_char == match.start() for t in tags):
                continue
            author_tag = match.group(1).strip()
            tags.append(
                CitationTag(
                    raw_tag=raw,
                    tag_type="author_year",
                    normalized_indices=[author_tag],
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        # 5. Author-Year in parentheses: (Vaswani et al., 2017)
        for match in self.AUTHOR_YEAR_PAREN.finditer(text):
            raw = match.group(0)
            author_tag = match.group(1).strip()
            tags.append(
                CitationTag(
                    raw_tag=raw,
                    tag_type="author_year",
                    normalized_indices=[author_tag],
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        # 6. Markdown links: [Source](https://...)
        for match in self.MARKDOWN_LINK_PATTERN.finditer(text):
            title = match.group(1).strip()
            url = match.group(2).strip()
            tags.append(
                CitationTag(
                    raw_tag=match.group(0),
                    tag_type="markdown_url",
                    normalized_indices=[url, title],
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        tags.sort(key=lambda t: t.start_char)
        return tags

    def split_sentences_safely(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences while protecting abbreviations and citations."""
        masked = text
        placeholders = {}
        for idx, abbr in enumerate(self.ABBREVIATIONS):
            token = f"__ABBR_{idx}__"
            placeholders[token] = abbr
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            masked = pattern.sub(token, masked)

        sentence_spans = []
        pattern = re.compile(r"([^.!?\n]+[.!?\n]*)", re.DOTALL)
        for match in pattern.finditer(masked):
            s_text = match.group(1).strip()
            if not s_text:
                continue
            for token, abbr in placeholders.items():
                s_text = s_text.replace(token, abbr)
            sentence_spans.append((s_text, match.start(), match.end()))

        if not sentence_spans and text.strip():
            sentence_spans = [(text.strip(), 0, len(text.strip()))]

        return sentence_spans

    def is_factual_claim(self, clean_text: str) -> bool:
        """Classify if a sentence contains factual claims vs conversational pleasantries."""
        norm = clean_text.strip().lower()
        if len(norm) < 12:
            return False
        for pat in BOILERPLATE_PATTERNS:
            if re.search(pat, norm):
                return False
        return True

    def segment_claims(self, text: str) -> list[SentenceClaim]:
        """Segment response into claims, associate citations, and strip bracketed markers."""
        raw_citations = self.extract_citations(text)
        spans = self.split_sentences_safely(text)
        claims: list[SentenceClaim] = []

        for s_idx, (s_text, start_pos, end_pos) in enumerate(spans):
            # Find all citations belonging to this sentence
            s_citations = [
                c for c in raw_citations
                if c.start_char >= start_pos and c.end_char <= end_pos + 10
            ]

            # Strip bracketed numeric and link citations from the claim
            clean_claim = s_text
            for c in s_citations:
                if c.tag_type in ["numeric", "numeric_range", "named_source", "markdown_url"]:
                    clean_claim = clean_claim.replace(c.raw_tag, "")
            clean_claim = re.sub(r"\s+", " ", clean_claim).strip()

            factual = self.is_factual_claim(clean_claim)
            claims.append(
                SentenceClaim(
                    sentence_idx=s_idx,
                    raw_text=s_text,
                    clean_claim=clean_claim,
                    citations=s_citations,
                    is_factual=factual,
                )
            )

        return claims

    def check_entailment(self, claim: str, context: str) -> tuple[bool, float]:
        """Compute hybrid grounding entailment score S_ground between claim and context.

        Formula:
            C_lex = |ContentTokens(claim) intersect Tokens(context)| / |ContentTokens(claim)|
            J_token = |Tokens(claim) intersect Tokens(context)| / |Tokens(claim) union Tokens(context)|
            S_ground = 0.70 * C_lex + 0.30 * J_token
            Entailed iff S_ground >= entailment_threshold
        """
        claim_tokens = self.tokenize_words(claim)
        context_tokens = set(self.tokenize_words(context))

        if not claim_tokens or not context_tokens:
            return False, 0.0

        content_claim_tokens = [t for t in claim_tokens if t not in self.stopwords]
        if not content_claim_tokens:
            content_claim_tokens = claim_tokens

        # Lexical containment with morphological prefix matching
        contained_hits = sum(
            1 for t in content_claim_tokens
            if any(self._tokens_match(t, c) for c in context_tokens)
        )
        c_lex = contained_hits / len(content_claim_tokens)

        # Jaccard token overlap
        claim_set = set(claim_tokens)
        intersection_count = sum(
            1 for t in claim_set if any(self._tokens_match(t, c) for c in context_tokens)
        )
        union_size = len(claim_set.union(context_tokens))
        j_token = intersection_count / union_size if union_size else 0.0

        s_ground = 0.70 * c_lex + 0.30 * j_token
        is_entailed = s_ground >= self.entailment_threshold

        return is_entailed, s_ground

    def verify_response(
        self,
        response: str,
        ground_truth_context: str,
        reference_documents: Optional[dict[str, str]] = None,
    ) -> CitationMetrics:
        """Execute full ALCE citation evaluation and RAGAS faithfulness verification."""
        claims = self.segment_claims(response)
        doc_map = reference_documents or {}

        def get_context_for_citation(tag: CitationTag) -> str:
            for idx in tag.normalized_indices:
                if idx in doc_map:
                    return doc_map[idx]
                clean_idx = re.sub(r"[\[\(\]\)]", "", idx).strip().lower()
                for k, v in doc_map.items():
                    if clean_idx in k.lower() or k.lower() in clean_idx:
                        return v
            return ground_truth_context

        factual_claims = [c for c in claims if c.is_factual]
        total_citations_count = 0
        valid_citations_count = 0
        supported_sentences_count = 0
        context_entailed_claims_count = 0
        unsupported_claims: list[str] = []
        hallucinated_citations: list[str] = []

        for claim in factual_claims:
            # 1. Context Faithfulness (RAGAS)
            is_context_entailed, entail_score = self.check_entailment(
                claim.clean_claim, ground_truth_context
            )
            claim.context_entailed = is_context_entailed
            claim.entailment_score = entail_score
            if is_context_entailed:
                context_entailed_claims_count += 1
            else:
                unsupported_claims.append(claim.clean_claim)

            # 2. ALCE Citation Precision & Recall
            if not claim.citations:
                claim.supported_by_citations = False
            else:
                has_supporting_citation = False
                for cite in claim.citations:
                    num_indices = len(cite.normalized_indices)
                    total_citations_count += num_indices
                    cite_context = get_context_for_citation(cite)
                    is_cite_valid, _ = self.check_entailment(claim.clean_claim, cite_context)

                    if is_cite_valid:
                        valid_citations_count += num_indices
                        has_supporting_citation = True
                        claim.supporting_citations.extend(cite.normalized_indices)
                    else:
                        hallucinated_citations.extend(cite.normalized_indices)
                        claim.unsupported_citations.extend(cite.normalized_indices)

                claim.supported_by_citations = has_supporting_citation
                if has_supporting_citation:
                    supported_sentences_count += 1

        num_fact = max(1, len(factual_claims))
        citation_recall = (
            (supported_sentences_count / num_fact) if factual_claims else 1.0
        )
        citation_precision = (
            (valid_citations_count / max(1, total_citations_count))
            if total_citations_count > 0
            else 0.0
        )
        denom = citation_precision + citation_recall
        citation_f1 = (
            (2.0 * citation_precision * citation_recall / denom) if denom > 0.0 else 0.0
        )
        faithfulness = (
            (context_entailed_claims_count / num_fact) if factual_claims else 1.0
        )

        summary = (
            f"Citation Evaluation: Precision={citation_precision:.2%}, "
            f"Recall={citation_recall:.2%}, F1={citation_f1:.2%}, "
            f"Faithfulness={faithfulness:.2%} | Factual Claims: {len(factual_claims)}, "
            f"Citations: {valid_citations_count}/{total_citations_count} valid."
        )

        logger.info(summary)

        return CitationMetrics(
            citation_precision=round(citation_precision, 4),
            citation_recall=round(citation_recall, 4),
            citation_f1=round(citation_f1, 4),
            faithfulness_score=round(faithfulness, 4),
            total_sentences=len(claims),
            factual_sentences=len(factual_claims),
            total_citations=total_citations_count,
            valid_citations=valid_citations_count,
            unsupported_claims=unsupported_claims,
            hallucinated_citations=hallucinated_citations,
            summary=summary,
        )

    def verify_benchmark_case(
        self, case: GoldenTestCase, response: str
    ) -> CitationMetrics:
        """Verify response against a GoldenTestCase ground-truth context and required citations."""
        full_context = f"{case.ground_truth_context} {' '.join(case.golden_keywords)}"
        ref_docs = {
            "1": full_context,
            "default": full_context,
        }
        for cite in case.required_citations:
            ref_docs[cite] = full_context
            clean_name = re.sub(r"[\[\(\]\)]", "", cite).strip()
            ref_docs[clean_name] = full_context

        metrics = self.verify_response(
            response=response,
            ground_truth_context=full_context,
            reference_documents=ref_docs,
        )
        return metrics

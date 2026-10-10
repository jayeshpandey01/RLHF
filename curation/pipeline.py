"""End-to-end data curation pipeline connecting quality filtering, contamination checks, and MinHash LSH."""

from typing import Any, Optional
from loguru import logger

from core.schemas import CleanedDocumentSchema, ScrapedDocumentSchema
from curation.contamination import BenchmarkContaminationGuard
from curation.filters import C4QualityFilter
from curation.minhash_lsh import MinHashLSH


class CurationPipeline:
    """Unified curation pipeline executing deterministic filters and fuzzy deduplication."""

    def __init__(
        self,
        quality_filter: Optional[C4QualityFilter] = None,
        minhash_lsh: Optional[MinHashLSH] = None,
        contamination_guard: Optional[BenchmarkContaminationGuard] = None,
    ):
        self.quality_filter = quality_filter or C4QualityFilter()
        self.minhash_lsh = minhash_lsh or MinHashLSH()
        self.contamination_guard = contamination_guard or BenchmarkContaminationGuard()
        self._seen_exact_hashes: set[str] = set()

    def process_document(
        self, doc: ScrapedDocumentSchema
    ) -> tuple[bool, Optional[CleanedDocumentSchema], str]:
        """Process a single scraped document through the complete quality and deduplication DAG.
        
        Returns:
            (is_accepted: bool, cleaned_doc: Optional[CleanedDocumentSchema], reason: str)
        """
        # Ensure SHA-256 hash
        if not doc.sha256_hash:
            doc.compute_hash()

        # 1. Exact SHA-256 Hash Filter O(1)
        if doc.sha256_hash in self._seen_exact_hashes:
            return False, None, f"EXACT_DUPLICATE_HASH ({doc.sha256_hash[:8]})"

        content = (doc.clean_markdown or doc.raw_html).strip()
        if not content:
            return False, None, "EMPTY_CONTENT"

        # 2. C4 / FineWeb Quality Heuristics
        is_quality_valid, reason, metrics = self.quality_filter.evaluate(content)
        if not is_quality_valid:
            return False, None, f"QUALITY_REJECT: {reason}"

        # 3. Benchmark Contamination Guard (Dodge et al., 2021)
        is_contam, leaked_ngram = self.contamination_guard.is_contaminated(content)
        if is_contam:
            return False, None, f"BENCHMARK_CONTAMINATION: '{leaked_ngram}'"

        # 4. MinHash LSH Fuzzy Deduplication
        signature = self.minhash_lsh.compute_signature(content)
        is_dup, sim, match_id = self.minhash_lsh.is_duplicate(signature)
        if is_dup:
            return False, None, f"NEAR_DUPLICATE_LSH (sim={sim:.3f} >= {self.minhash_lsh.threshold}, match='{match_id}')"

        # Document is pristine: Index and return CleanedDocumentSchema
        self._seen_exact_hashes.add(doc.sha256_hash)
        self.minhash_lsh.index(doc.url, signature)

        cleaned_doc = CleanedDocumentSchema(
            url=doc.url,
            clean_text=content,
            word_count=metrics["word_count"],
            line_repetition_rate=metrics["line_repetition_rate"],
            sha256_hash=doc.sha256_hash,
            minhash_signature=signature,
            is_valid=True,
        )
        return True, cleaned_doc, "ACCEPTED"

    def process_batch(
        self, docs: list[ScrapedDocumentSchema]
    ) -> tuple[list[CleanedDocumentSchema], dict]:
        """Process a batch of scraped documents, returning accepted documents and audit stats."""
        accepted: list[CleanedDocumentSchema] = []
        stats: dict[str, int] = {
            "total_input": len(docs),
            "accepted": 0,
            "rejected_exact_duplicate": 0,
            "rejected_quality": 0,
            "rejected_contamination": 0,
            "rejected_near_duplicate": 0,
        }

        for doc in docs:
            ok, cleaned_doc, reason = self.process_document(doc)
            if ok and cleaned_doc:
                accepted.append(cleaned_doc)
                stats["accepted"] += 1
            else:
                if "EXACT_DUPLICATE" in reason:
                    stats["rejected_exact_duplicate"] += 1
                elif "QUALITY_REJECT" in reason or "EMPTY" in reason:
                    stats["rejected_quality"] += 1
                elif "BENCHMARK_CONTAMINATION" in reason:
                    stats["rejected_contamination"] += 1
                elif "NEAR_DUPLICATE" in reason:
                    stats["rejected_near_duplicate"] += 1

        logger.info(
            f"CurationPipeline: Ingested {len(docs)} -> Accepted {len(accepted)} ({stats['accepted']/max(1, len(docs))*100:.1f}%)"
        )
        return accepted, stats

    def register_benchmark_suite(self, suite: Any) -> int:
        """Register a GoldenBenchmarkSuite or benchmark cases with the contamination guard.

        Returns:
            Total number of forbidden 13-grams registered.
        """
        if hasattr(suite, "register_with_contamination_guard"):
            return suite.register_with_contamination_guard(self.contamination_guard)
        elif hasattr(suite, "cases"):
            added = 0
            for case in suite.cases:
                if hasattr(case, "to_schema"):
                    added += self.contamination_guard.add_benchmark_case(case.to_schema())
                elif hasattr(case, "question"):
                    added += self.contamination_guard.add_benchmark_case(case)
            return added
        elif hasattr(suite, "test_cases"):
            added = 0
            for case in suite.test_cases:
                text = f"{getattr(case, 'prompt', '')} {getattr(case, 'ground_truth', '')}"
                if text.strip():
                    added += self.contamination_guard.add_benchmark_text(text)
            return added
        elif isinstance(suite, (list, tuple)):
            added = 0
            for case in suite:
                if hasattr(case, "to_schema"):
                    added += self.contamination_guard.add_benchmark_case(case.to_schema())
                elif hasattr(case, "question"):
                    added += self.contamination_guard.add_benchmark_case(case)
                elif isinstance(case, str):
                    added += self.contamination_guard.add_benchmark_text(case)
            return added
        return 0

    def clear(self) -> None:
        """Reset internal deduplication and contamination indices."""
        self._seen_exact_hashes.clear()
        self.minhash_lsh.clear()
        self.contamination_guard.clear()


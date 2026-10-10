"""Live web collection and persistent curated knowledge for the serving gateway."""

import asyncio
import json
from pathlib import Path

from loguru import logger

from core.config import settings
from core.schemas import ScrapedDocumentSchema
from curation.cluster import ClusterDeduplicator
from curation.orchestrator import DEFAULT_ANCHOR_BASELINE
from curation.pipeline import CurationPipeline
from curation.replay_buffer import ExperienceReplayBuffer
from evaluation.benchmark_suite import GoldenBenchmarkSuite
from ingestion.provenance import ProvenanceLogger
from ingestion.router import IngestionRouter
from retrieval.hybrid import HybridRetriever
from retrieval.schemas import RetrievalDocument


class LiveKnowledgeService:
    """Collect, curate, persist, and index source-backed documents for queries."""

    def __init__(
        self,
        retriever: HybridRetriever,
        corpus_path: Path | None = None,
        provenance_path: Path | None = None,
    ):
        self.retriever = retriever
        self.corpus_path = corpus_path or settings.DATA_DIR / "knowledge_corpus.jsonl"
        provenance_path = provenance_path or settings.DATA_DIR / "provenance_manifest.jsonl"
        self.provenance = ProvenanceLogger(manifest_path=str(provenance_path))
        self.router = IngestionRouter(provenance_logger=self.provenance)
        self.curation = CurationPipeline()
        self.curation.register_benchmark_suite(GoldenBenchmarkSuite.load_default_suite())
        self.cluster_deduplicator = ClusterDeduplicator()
        self.replay_buffer = ExperienceReplayBuffer(anchor_ratio=settings.ANCHOR_DATA_RATIO)
        self.replay_buffer.add_anchor_data(DEFAULT_ANCHOR_BASELINE)
        self._lock = asyncio.Lock()
        self._load_corpus()

    def _load_corpus(self) -> None:
        if not self.corpus_path.exists():
            return
        loaded = 0
        with self.corpus_path.open(encoding="utf-8") as corpus_file:
            for line in corpus_file:
                if not line.strip():
                    continue
                item = json.loads(line)
                document = RetrievalDocument(
                    doc_id=item["sha256_hash"],
                    text=item["text"],
                    url=item["url"],
                    metadata={"sha256_hash": item["sha256_hash"]},
                )
                self.curation.process_document(
                    ScrapedDocumentSchema(
                        url=item["url"],
                        raw_html=item["text"],
                        clean_markdown=item["text"],
                        sha256_hash=item["sha256_hash"],
                    )
                )
                self.retriever.add_document(document)
                self.replay_buffer.add_web_data([item["text"]])
                loaded += 1
        logger.info(f"Loaded {loaded} persistent knowledge documents from {self.corpus_path}")

    async def collect_and_index(self, query: str, max_results: int = 5) -> dict:
        async with self._lock:
            intent, scraped_docs = await self.router.route_and_fetch(
                query,
                max_results=max_results,
            )
            if not scraped_docs:
                return {
                    "intent": intent.value,
                    "fetched": 0,
                    "accepted": 0,
                    "indexed": 0,
                    "filter_stats": {},
                }

            cleaned_docs, filter_stats = self.curation.process_batch(scraped_docs)
            candidate_pairs = []
            for index, left in enumerate(cleaned_docs):
                for right in cleaned_docs[index + 1 :]:
                    similarity = self.curation.minhash_lsh.jaccard_similarity(
                        left.minhash_signature,
                        right.minhash_signature,
                    )
                    if similarity >= self.curation.minhash_lsh.threshold:
                        candidate_pairs.append((left.url, right.url))

            canonical_docs, cluster_stats = self.cluster_deduplicator.deduplicate(
                cleaned_docs,
                candidate_pairs,
            )
            indexed = 0
            for document in canonical_docs:
                accepted_count = self.replay_buffer.add_web_data([document])
                if not accepted_count:
                    continue
                record = {
                    "url": document.url,
                    "text": document.clean_text,
                    "sha256_hash": document.sha256_hash,
                }
                self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
                with self.corpus_path.open("a", encoding="utf-8") as corpus_file:
                    corpus_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.retriever.add_document(
                    RetrievalDocument(
                        doc_id=document.sha256_hash,
                        text=document.clean_text,
                        url=document.url,
                        metadata={"sha256_hash": document.sha256_hash},
                    )
                )
                indexed += 1

            return {
                "intent": intent.value,
                "fetched": len(scraped_docs),
                "accepted": len(cleaned_docs),
                "indexed": indexed,
                "filter_stats": filter_stats,
                "cluster_stats": cluster_stats,
                "replay_stats": self.replay_buffer.get_stats(),
            }

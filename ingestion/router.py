"""Multi-source ingestion router with intent classification and fallback cascades.

Routes queries according to semantic intent (Geospatial/Map, Encyclopedic, Academic Research,
Technical Documentation, General Web) to specialized authoritative data sources.
"""

import asyncio
import re
import time
from enum import Enum
from typing import Optional
from loguru import logger

from core.schemas import ScrapedDocumentSchema
from ingestion.provenance import ProvenanceLogger
from ingestion.sources.base import canonicalize_url
from ingestion.sources.arxiv import ArXivSource
from ingestion.sources.crawler import SitemapCrawlerSource
from ingestion.sources.duckduckgo import DuckDuckGoSource
from ingestion.sources.openstreetmap import OpenStreetMapSource
from ingestion.sources.wikipedia import WikipediaSource


class QueryIntent(str, Enum):
    """Semantic intent classification of ingestion queries."""

    MAP_GEOSPATIAL = "map_geospatial"
    ENCYCLOPEDIC = "encyclopedic"
    ACADEMIC_RESEARCH = "academic_research"
    TECHNICAL_DOCS = "technical_docs"
    GENERAL_WEB = "general_web"


# Intent heuristic keyword and regex patterns
INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.MAP_GEOSPATIAL: [
        r"\b(coordinates|latitude|longitude|lat\s*lon|map\s+of|osm|overpass|nominatim)\b",
        r"\b(location\s+of|where\s+is|address\s+of|near|border\s+between|distance\s+between)\b",
        r"\b(boundaries|poi|points\s+of\s+interest|topological|subway\s+route|highway)\b",
        r"\b(capital\s+of|city\s+of|monument\s+in|mount\s+|river\s+)\b",
    ],
    QueryIntent.ACADEMIC_RESEARCH: [
        r"\b(arxiv|paper|theorem|proof|lemma|corollary|mathematical\s+derivation)\b",
        r"\b(transformer\s+architecture|attention\s+mechanism|backpropagation|gradient\s+descent)\b",
        r"\b(loss\s+function|convergence\s+rate|empirical\s+evaluation|ablation\s+study)\b",
        r"\b(pre-training|post-training|dpo|grpo|rlhf|kv-cache|flashattention)\b",
    ],
    QueryIntent.TECHNICAL_DOCS: [
        r"\b(pytorch|python|cuda|torchrun|distributed|huggingface|transformers\s+lib)\b",
        r"\b(api\s+docs|documentation|class\s+reference|function\s+signature|sdk|pip\s+install)\b",
        r"\b(github\.com|docs\.|developer\.|spec\b|standard\s+library)\b",
    ],
    QueryIntent.ENCYCLOPEDIC: [
        r"\b(who\s+is|who\s+was|what\s+is|what\s+was|definition\s+of|biography\s+of)\b",
        r"\b(history\s+of|invented\s+by|discovered\s+by|founded\s+in|nobel\s+prize)\b",
        r"\b(wikipedia|encyclopedia|overview\s+of|summary\s+of|origin\s+of)\b",
    ],
}


def classify_intent(query: str) -> QueryIntent:
    """Classify the semantic intent of a query using compiled regular expressions."""
    cleaned = query.strip().lower()

    # Match in priority order: Academic -> Geospatial -> Technical -> Encyclopedic -> General
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                return intent

    return QueryIntent.GENERAL_WEB


class IngestionRouter:
    """Unified ingestion router dispatching queries to specialized engines."""

    def __init__(
        self,
        provenance_logger: Optional[ProvenanceLogger] = None,
        timeout_seconds: float = 15.0,
    ):
        self.provenance = provenance_logger or ProvenanceLogger()
        self.timeout = timeout_seconds

        # Instantiate specialized engines
        self.osm_source = OpenStreetMapSource()
        self.wiki_source = WikipediaSource()
        self.arxiv_source = ArXivSource()
        self.crawler_source = SitemapCrawlerSource()
        self.ddg_source = DuckDuckGoSource()

    def classify(self, query: str) -> QueryIntent:
        """Classify query intent."""
        return classify_intent(query)

    async def route_and_fetch(
        self,
        query: str,
        max_results: int = 5,
        forced_intent: Optional[QueryIntent] = None,
    ) -> tuple[QueryIntent, list[ScrapedDocumentSchema]]:
        """Route a query to the primary source with fallback cascades and provenance logging."""
        intent = forced_intent or self.classify(query)
        logger.info(f"IngestionRouter: query='{query}' classified as intent={intent.value}")

        primary_source, fallback_source = self._resolve_sources(intent)

        # 1. Primary source execution
        start_time = time.monotonic()
        results: list[ScrapedDocumentSchema] = []
        source_name = primary_source.name

        try:
            results = await asyncio.wait_for(
                primary_source.fetch(query, max_results=max_results),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Primary source {primary_source.name} timed out for query '{query}'")
        except Exception as e:
            logger.warning(f"Primary source {primary_source.name} failed for query '{query}': {e}")

        # 2. Fallback execution if primary returned empty results
        if not results and fallback_source:
            logger.info(f"Falling back to {fallback_source.name} for query '{query}'")
            source_name = fallback_source.name
            try:
                results = await asyncio.wait_for(
                    fallback_source.fetch(query, max_results=max_results),
                    timeout=self.timeout,
                )
            except Exception as e:
                logger.warning(f"Fallback source {fallback_source.name} failed for query '{query}': {e}")

        latency_ms = (time.monotonic() - start_time) * 1000.0

        # 3. Provenance recording and deduplication filtering
        accepted_docs: list[ScrapedDocumentSchema] = []
        for doc in results:
            if not doc.sha256_hash:
                doc.compute_hash()

            clean_url = canonicalize_url(doc.url) or doc.url

            # Filter out already seen hashes or URLs to save training compute
            if self.provenance.is_seen_hash(doc.sha256_hash) or (clean_url and self.provenance.is_seen_url(clean_url)):
                logger.debug(f"Skipping duplicate document {doc.sha256_hash[:8]} from {doc.url}")
                continue

            doc.url = clean_url
            self.provenance.record_document(
                doc=doc,
                source_name=source_name,
                latency_ms=latency_ms,
                status_code=200,
            )
            accepted_docs.append(doc)

        logger.info(
            f"IngestionRouter: successfully ingested {len(accepted_docs)} documents from {source_name} ({latency_ms:.1f}ms)"
        )
        return intent, accepted_docs

    async def fetch_multi(
        self,
        query: str,
        intents: list[QueryIntent],
        max_per_source: int = 3,
    ) -> dict[QueryIntent, list[ScrapedDocumentSchema]]:
        """Fetch simultaneously across multiple intents for rich multi-modal grounding."""
        tasks = [
            self.route_and_fetch(query, max_results=max_per_source, forced_intent=intent)
            for intent in intents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: dict[QueryIntent, list[ScrapedDocumentSchema]] = {}
        for intent, res in zip(intents, results):
            if isinstance(res, Exception):
                logger.error(f"Multi-fetch error for intent {intent}: {res}")
                merged[intent] = []
            else:
                _, docs = res
                merged[intent] = docs
        return merged

    def _resolve_sources(self, intent: QueryIntent):
        """Map intent to primary and fallback source engines."""
        if intent == QueryIntent.MAP_GEOSPATIAL:
            return self.osm_source, self.wiki_source
        elif intent == QueryIntent.ACADEMIC_RESEARCH:
            return self.arxiv_source, self.ddg_source
        elif intent == QueryIntent.TECHNICAL_DOCS:
            return self.crawler_source, self.ddg_source
        elif intent == QueryIntent.ENCYCLOPEDIC:
            return self.wiki_source, self.ddg_source
        else:  # GENERAL_WEB
            return self.ddg_source, self.wiki_source


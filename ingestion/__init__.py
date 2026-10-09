"""Enterprise Multi-Source Ingestion Engine for Continual LLM Post-Training."""

from ingestion.provenance import ProvenanceLogger
from ingestion.router import IngestionRouter, QueryIntent, classify_intent
from ingestion.sources import (
    ArXivSource,
    BaseIngestionSource,
    DuckDuckGoSource,
    OpenStreetMapSource,
    SitemapCrawlerSource,
    TrafilaturaExtractor,
    WikipediaSource,
    canonicalize_url,
)

__all__ = [
    "BaseIngestionSource",
    "canonicalize_url",
    "OpenStreetMapSource",
    "WikipediaSource",
    "ArXivSource",
    "SitemapCrawlerSource",
    "TrafilaturaExtractor",
    "DuckDuckGoSource",
    "ProvenanceLogger",
    "IngestionRouter",
    "QueryIntent",
    "classify_intent",
]


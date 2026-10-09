"""Specialized ingestion sources for multi-modal and textual data acquisition."""

from ingestion.sources.arxiv import ArXivSource
from ingestion.sources.base import BaseIngestionSource, canonicalize_url
from ingestion.sources.crawler import SitemapCrawlerSource, TrafilaturaExtractor
from ingestion.sources.duckduckgo import DuckDuckGoSource
from ingestion.sources.openstreetmap import OpenStreetMapSource
from ingestion.sources.wikipedia import WikipediaSource

__all__ = [
    "BaseIngestionSource",
    "canonicalize_url",
    "OpenStreetMapSource",
    "WikipediaSource",
    "ArXivSource",
    "SitemapCrawlerSource",
    "TrafilaturaExtractor",
    "DuckDuckGoSource",
]


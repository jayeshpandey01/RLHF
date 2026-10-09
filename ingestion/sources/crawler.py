"""Trafilatura HTML DOM extractor and sitemap crawler for technical documentation."""

import re
from typing import Optional
import httpx
from loguru import logger
import trafilatura

from core.schemas import ScrapedDocumentSchema
from ingestion.sources.base import BaseIngestionSource, canonicalize_url

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


class TrafilaturaExtractor:
    """High-accuracy HTML DOM extractor using Text-Density Ratio pruning."""

    @staticmethod
    def extract_text(html: str, url: Optional[str] = None) -> Optional[str]:
        """Extract main body text, tables, and code blocks, stripping nav and ads."""
        if not html or not html.strip():
            return None

        # 1. Primary extraction via Trafilatura (F1 > 0.90)
        try:
            extracted = trafilatura.extract(
                html,
                url=url,
                include_tables=True,
                include_comments=False,
                include_links=False,
                favor_precision=True,
                output_format="markdown",
            )
            if extracted and len(extracted.strip()) > 30:
                return extracted.strip()
        except Exception as e:
            logger.debug(f"Trafilatura primary extraction failed: {e}")

        # 2. Fallback lightweight DOM parser
        return TrafilaturaExtractor._fallback_clean(html)

    @classmethod
    def extract_markdown(cls, html: str, url: Optional[str] = None) -> str:
        """Alias for extract_text returning clean Markdown string."""
        return cls.extract_text(html, url=url) or ""

    @staticmethod
    def _fallback_clean(html: str) -> Optional[str]:
        """Fallback cleaner stripping script, style, navigation, and tags."""
        try:
            text = re.sub(r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text if len(text) > 30 else None
        except Exception:
            return None


class SitemapCrawlerSource(BaseIngestionSource):
    """Deep crawler extracting documentation pages and sitemaps via Trafilatura."""

    def __init__(self, timeout: float = 10.0):
        super().__init__(name="sitemap_crawler")
        self.timeout = timeout
        self.extractor = TrafilaturaExtractor()

    async def fetch(self, query: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Fetch content from direct URL or list of URLs."""
        results: list[ScrapedDocumentSchema] = []
        urls = [query] if isinstance(query, str) else list(query)

        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
            for raw_url in urls[:max_results]:
                canon_url = canonicalize_url(raw_url)
                if not canon_url:
                    continue
                doc = await self.fetch_single_url(client, canon_url)
                if doc:
                    results.append(doc)

        return results

    async def fetch_single_url(self, client: httpx.AsyncClient, url: str) -> Optional[ScrapedDocumentSchema]:
        """Fetch and extract clean markdown from a single URL."""
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.debug(f"HTTP {resp.status_code} fetching '{url}'")
                return None

            clean_markdown = self.extractor.extract_text(resp.text, url=url)
            if not clean_markdown:
                return None

            doc = ScrapedDocumentSchema(
                url=url,
                raw_html=resp.text[:50000],  # Truncate raw storage to 50KB to preserve disk
                clean_markdown=clean_markdown,
            )
            doc.compute_hash()
            return doc
        except Exception as e:
            logger.debug(f"Failed to crawl URL '{url}': {e}")
            return None

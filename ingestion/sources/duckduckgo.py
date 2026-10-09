"""Auxiliary DuckDuckGo search client with randomized jitter exponential backoff."""

import asyncio
import random
import re
from typing import Optional
from urllib.parse import unquote
import httpx
from loguru import logger

from core.schemas import ScrapedDocumentSchema
from ingestion.sources.base import BaseIngestionSource, canonicalize_url

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]


class DuckDuckGoSource(BaseIngestionSource):
    """Auxiliary long-tail search engine with exponential backoff and jitter."""

    def __init__(self, timeout: float = 8.0, max_retries: int = 2):
        super().__init__(name="duckduckgo")
        self.timeout = timeout
        self.max_retries = max_retries

    async def fetch(self, query: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Execute DuckDuckGo HTML search and extract ranked candidate links and snippets."""
        results: list[ScrapedDocumentSchema] = []

        for attempt in range(self.max_retries + 1):
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
                    data = {"q": query, "b": ""}
                    resp = await client.post(DDG_HTML_URL, data=data)

                    if resp.status_code == 200:
                        parsed_items = self._parse_results(resp.text)
                        for item in parsed_items[:max_results]:
                            doc = ScrapedDocumentSchema(
                                url=item["url"],
                                raw_html=item["snippet"],
                                clean_markdown=f"# {item['title']}\n\n{item['snippet']}",
                            )
                            doc.compute_hash()
                            results.append(doc)
                        return results
                    elif resp.status_code in (429, 202):
                        # Exponential backoff with jitter: 1.5 * 2^k + Uniform(0, 1)
                        wait_sec = min(10.0, 1.5 * (2**attempt) + random.uniform(0.1, 1.0))
                        logger.debug(f"DuckDuckGo rate-limited (HTTP {resp.status_code}). Backing off {wait_sec:.2f}s...")
                        await asyncio.sleep(wait_sec)
                    else:
                        logger.debug(f"DuckDuckGo returned HTTP {resp.status_code}")
                        break
            except Exception as e:
                logger.debug(f"DuckDuckGo attempt {attempt} failed: {e}")
                await asyncio.sleep(1.0)

        return results

    def parse_html(self, html: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Parse raw HTML and return list of ScrapedDocumentSchema documents."""
        parsed_items = self._parse_results(html)
        results = []
        for item in parsed_items[:max_results]:
            doc = ScrapedDocumentSchema(
                url=item["url"],
                raw_html=item["snippet"],
                clean_markdown=f"# {item['title']}\n\n{item['snippet']}",
            )
            doc.compute_hash()
            results.append(doc)
        return results

    def _parse_results(self, html: str) -> list[dict[str, str]]:
        """Extract links, titles, and snippets from DuckDuckGo HTML response."""
        items = []
        # Pattern 1: DDG standard template with h2 title and snippet
        blocks = re.findall(
            r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>.*?<h2[^>]*>.*?<a[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>',
            html,
            flags=re.DOTALL,
        )

        if blocks:
            for raw_link, raw_title, raw_snippet in blocks:
                real_url = self._extract_real_url(raw_link)
                canon = canonicalize_url(real_url) if real_url else None
                if not canon:
                    continue
                clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
                items.append({"url": canon, "title": clean_title, "snippet": clean_snippet})
            return items

        # Pattern 2: Result items with url and snippet pairs
        pairs = re.findall(
            r'<a\s+class="result__url"\s+href="([^"]+)"[^>]*>.*?</a>.*?<a\s+class="result__snippet"[^>]*>(.*?)</a>',
            html,
            flags=re.DOTALL,
        )
        if pairs:
            for raw_link, raw_snippet in pairs:
                real_url = self._extract_real_url(raw_link)
                canon = canonicalize_url(real_url) if real_url else None
                if not canon:
                    continue
                clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
                items.append({"url": canon, "title": "Web Search Result", "snippet": clean_snippet})
            return items

        # Fallback regex for pure link anchors
        links = re.findall(r'<a\s+class="result__url"\s+href="([^"]+)"', html)
        for link in links:
            real_url = self._extract_real_url(link)
            canon = canonicalize_url(real_url) if real_url else None
            if canon:
                items.append({"url": canon, "title": "Web Search Result", "snippet": canon})
        return items

    def _extract_real_url(self, ddg_link: str) -> Optional[str]:
        """Decode DuckDuckGo redirection link (/l/?uddg=...)."""
        if "uddg=" in ddg_link:
            match = re.search(r"uddg=([^&]+)", ddg_link)
            if match:
                return unquote(match.group(1))
        if ddg_link.startswith("http"):
            return ddg_link
        return None

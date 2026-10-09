"""Wikipedia REST API and Geosearch factual knowledge ingestion engine."""

from typing import Optional
from urllib.parse import quote
import httpx
from loguru import logger

from core.schemas import ScrapedDocumentSchema
from ingestion.sources.base import BaseIngestionSource

WIKI_REST_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "RLHF-Autonomous-Continual-Learning/1.0 (contact: info@local-rlhf-pipeline.org)"


class WikipediaSource(BaseIngestionSource):
    """Encyclopedic engine accessing authoritative, neutral Wikipedia summaries and geosearch."""

    def __init__(self, timeout: float = 10.0):
        super().__init__(name="wikipedia")
        self.timeout = timeout

    async def fetch(self, query: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Fetch Wikipedia articles matching title query."""
        results: list[ScrapedDocumentSchema] = []
        headers = {"User-Agent": USER_AGENT}

        # 1. Search for matching page titles via OpenSearch action API
        titles = await self._search_titles(query, limit=max_results)
        if not titles:
            titles = [query]

        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            for title in titles[:max_results]:
                doc = await self._fetch_summary(client, title)
                if doc:
                    results.append(doc)

        return results

    async def geosearch(
        self,
        lat: float,
        lon: float,
        radius: int = 10000,
        limit: int = 5,
        radius_meters: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> list[ScrapedDocumentSchema]:
        """Fetch Wikipedia articles located within radius (meters) of coordinates."""
        radius = radius_meters if radius_meters is not None else radius
        limit = max_results if max_results is not None else limit
        results: list[ScrapedDocumentSchema] = []
        headers = {"User-Agent": USER_AGENT}

        params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": str(radius),
            "gslimit": str(limit),
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                resp = await client.get(WIKI_ACTION_API, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    geosearch_results = data.get("query", {}).get("geosearch", [])
                    for item in geosearch_results:
                        title = item.get("title")
                        if title:
                            doc = await self._fetch_summary(client, title)
                            if doc:
                                results.append(doc)
        except Exception as e:
            logger.debug(f"Wikipedia Geosearch failed for ({lat}, {lon}): {e}")

        return results

    async def _search_titles(self, query: str, limit: int = 5) -> list[str]:
        """Find Wikipedia article titles via Opensearch prefix API."""
        params = {
            "action": "opensearch",
            "search": query,
            "limit": str(limit),
            "namespace": "0",
            "format": "json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": USER_AGENT}) as client:
                resp = await client.get(WIKI_ACTION_API, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if len(data) >= 2 and isinstance(data[1], list):
                        return data[1]
        except Exception as e:
            logger.debug(f"Opensearch failed for '{query}': {e}")
        return []

    async def _fetch_summary(self, client: httpx.AsyncClient, title: str) -> Optional[ScrapedDocumentSchema]:
        """Fetch clean JSON summary from Wikipedia REST API."""
        encoded_title = quote(title.replace(" ", "_"))
        url = f"{WIKI_REST_BASE}/{encoded_title}"

        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None

            data = resp.json()
            display_title = data.get("title", title)
            description = data.get("description", "")
            extract = data.get("extract", "")
            canonical_url = data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded_title}")

            if not extract:
                return None

            # Format clean Markdown
            lines = [
                f"# {display_title}",
                "",
                f"*{description}*" if description else "",
                "",
                "## Summary",
                extract,
            ]
            content = "\n".join(line for line in lines if line is not None)

            doc = ScrapedDocumentSchema(
                url=canonical_url,
                raw_html=str(resp.text) if resp.text else "{}",
                clean_markdown=content,
            )
            doc.compute_hash()
            return doc
        except Exception as e:
            logger.debug(f"Failed to fetch summary for '{title}': {e}")
            return None

    # Alias for API consistency
    fetch_geosearch = geosearch

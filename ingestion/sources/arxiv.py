"""ArXiv Open API scientific and mathematical paper ingestion engine."""

import re
import xml.etree.ElementTree as ET

import httpx
from loguru import logger

from core.schemas import ScrapedDocumentSchema
from ingestion.sources.base import BaseIngestionSource

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArXivSource(BaseIngestionSource):
    """Scientific literature engine extracting peer-reviewed abstracts and LaTeX formulations."""

    def __init__(self, timeout: float = 12.0):
        super().__init__(name="arxiv")
        self.timeout = timeout

    @staticmethod
    def build_search_query(query: str) -> str:
        """Convert common RLHF acronym queries into focused ArXiv phrase searches."""
        cleaned = " ".join(query.split())
        lower_query = cleaned.lower()
        if re.search(r"\b(?:dpo|direct preference optimization)\b", lower_query):
            search_query = 'all:"Direct Preference Optimization"'
        elif re.search(r"\b(?:grpo|group relative policy optimization)\b", lower_query):
            search_query = 'all:"Group Relative Policy Optimization"'
        else:
            return f"all:{cleaned}"

        if "reward model" in lower_query:
            search_query += ' AND all:"reward model"'
        return search_query

    async def fetch(self, query: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Query ArXiv API and parse paper abstracts into structured Markdown documents."""
        results: list[ScrapedDocumentSchema] = []
        params = {
            "search_query": self.build_search_query(query),
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(ARXIV_API_BASE, params=params)
                if resp.status_code != 200:
                    logger.warning(f"ArXiv query failed with HTTP {resp.status_code}")
                    return results

                ET.fromstring(resp.text)
                results = self.parse_atom_xml(resp.text)
        except Exception as e:
            logger.error(f"Error querying ArXiv for '{query}': {e}")

        return results

    def parse_atom_xml(self, raw_xml: str) -> list[ScrapedDocumentSchema]:
        """Parse raw Atom XML feed and extract valid ScrapedDocumentSchema papers."""
        results: list[ScrapedDocumentSchema] = []
        try:
            root = ET.fromstring(raw_xml)
            for entry in root.findall("atom:entry", ATOM_NS):
                doc = self._parse_entry(entry, raw_xml)
                if doc:
                    results.append(doc)
        except Exception as e:
            logger.debug(f"Failed to parse ArXiv XML feed: {e}")
        return results

    def _parse_entry(self, entry: ET.Element, raw_xml: str) -> ScrapedDocumentSchema | None:
        """Parse individual Atom entry into a clean Markdown paper document."""
        try:
            title_elem = entry.find("atom:title", ATOM_NS)
            summary_elem = entry.find("atom:summary", ATOM_NS)
            id_elem = entry.find("atom:id", ATOM_NS)
            published_elem = entry.find("atom:published", ATOM_NS)

            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Untitled"
            summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
            url = id_elem.text.strip() if id_elem is not None and id_elem.text else "https://arxiv.org"
            if url.startswith("http://"):
                url = "https://" + url[len("http://") :]
            published = published_elem.text.strip() if published_elem is not None and published_elem.text else "Unknown"

            # Extract authors
            authors = []
            for author_elem in entry.findall("atom:author", ATOM_NS):
                name_elem = author_elem.find("atom:name", ATOM_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Extract primary category
            category_elem = entry.find("arxiv:primary_category", ATOM_NS)
            category = category_elem.attrib.get("term", "cs.AI") if category_elem is not None else "General"

            # Format structured Markdown
            content_lines = [
                f"# Research Paper: {title}",
                "",
                "## Metadata",
                f"- **ArXiv Identifier**: {url}",
                f"- **Primary Category**: {category}",
                f"- **Published Date**: {published}",
                f"- **Authors**: {', '.join(authors) if authors else 'Unknown'}",
                "",
                "## Abstract & Mathematical Formulations",
                summary,
            ]
            content = "\n".join(content_lines)

            doc = ScrapedDocumentSchema(
                url=url,
                raw_html=raw_xml,
                clean_markdown=content,
            )
            doc.compute_hash()
            return doc
        except Exception as e:
            logger.debug(f"Failed to parse ArXiv entry: {e}")
            return None

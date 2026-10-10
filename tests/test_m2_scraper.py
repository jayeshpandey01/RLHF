"""Unit and integration test suite for Milestone 2 / Phase 2.1: Multi-Source Ingestion Engine."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import ScrapedDocumentSchema
from ingestion.provenance import ProvenanceLogger
from ingestion.router import IngestionRouter, QueryIntent, classify_intent
from ingestion.sources.arxiv import ArXivSource
from ingestion.sources.base import canonicalize_url
from ingestion.sources.crawler import TrafilaturaExtractor
from ingestion.sources.duckduckgo import DuckDuckGoSource
from ingestion.sources.openstreetmap import OpenStreetMapSource
from ingestion.sources.wikipedia import WikipediaSource


class TestM2Scraper(unittest.TestCase):
    """Thorough verification of URL canonicalization, sources, provenance, and router."""

    def test_canonicalize_url(self):
        """Verify URL cleaning, parameter stripping, and binary rejection."""
        # Tracking params stripping
        url1 = "https://example.com/search?utm_source=twitter&utm_medium=cpc&id=42&fbclid=xyz"
        clean1 = canonicalize_url(url1)
        self.assertEqual(clean1, "https://example.com/search?id=42")

        # Fragment stripping
        url2 = "https://docs.python.org/3/library/unittest.html#basic-example"
        clean2 = canonicalize_url(url2)
        self.assertEqual(clean2, "https://docs.python.org/3/library/unittest.html")

        # Port normalization and lowercase domain
        url3 = "HTTPS://WWW.EXAMPLE.ORG:443/api/v1"
        clean3 = canonicalize_url(url3)
        self.assertEqual(clean3, "https://www.example.org/api/v1")

        # Disallowed binary rejection
        self.assertIsNone(canonicalize_url("https://example.com/paper.pdf"))
        self.assertIsNone(canonicalize_url("https://example.com/archive.zip"))
        self.assertIsNone(canonicalize_url("https://example.com/photo.jpeg"))
        self.assertIsNone(canonicalize_url("ftp://ftp.example.com/data.txt"))

    def test_trafilatura_extractor(self):
        """Verify extraction and DOM boilerplate stripping."""
        messy_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Article</title></head>
        <body>
            <header><nav><a href="/home">Home</a><a href="/login">Login</a></nav></header>
            <main>
                <h1>Deep Reinforcement Learning</h1>
                <p>Reinforcement learning from human feedback (RLHF) aligns LLMs with user intent.</p>
                <p>Proximal Policy Optimization and Direct Preference Optimization are standard.</p>
            </main>
            <footer><p>&copy; 2026 AI Lab. All rights reserved. Follow our Twitter.</p></footer>
        </body>
        </html>
        """
        extracted = TrafilaturaExtractor.extract_markdown(messy_html)
        self.assertIn("Reinforcement learning", extracted)
        self.assertIn("RLHF", extracted)
        # Navbar should be pruned
        self.assertNotIn("Login", extracted)

    def test_openstreetmap_parsing(self):
        """Verify OpenStreetMap Nominatim and Overpass parsing."""
        osm = OpenStreetMapSource()

        # Mock Nominatim
        mock_nominatim = [
            {
                "display_name": "Eiffel Tower, 5, Avenue Anatole France, 7th Arrondissement, Paris, France",
                "lat": "48.8583701",
                "lon": "2.2944813",
                "osm_type": "way",
                "osm_id": 5013364,
                "boundingbox": ["48.8574753", "48.8592043", "2.2933084", "2.2956897"],
            }
        ]

        async def run_nominatim():
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_nominatim
            mock_client.get = AsyncMock(return_value=mock_resp)

            docs = await osm._fetch_nominatim(mock_client, "Eiffel Tower", max_results=1)
            return docs

        docs = asyncio.run(run_nominatim())
        self.assertEqual(len(docs), 1)
        self.assertIn("Eiffel Tower", docs[0].clean_markdown)
        self.assertIn("48.858370", docs[0].clean_markdown)
        self.assertIn("2.294481", docs[0].clean_markdown)
        self.assertTrue(docs[0].url.startswith("https://www.openstreetmap.org/way/5013364"))

        # Mock Overpass
        mock_overpass = {
            "elements": [
                {
                    "type": "node",
                    "id": 123456,
                    "lat": 48.8584,
                    "lon": 2.2945,
                    "tags": {
                        "name": "Jardins du Trocadéro",
                        "leisure": "park",
                        "tourism": "attraction",
                    },
                }
            ]
        }

        async def run_overpass():
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_overpass
            mock_client.post = AsyncMock(return_value=mock_resp)

            docs = await osm._fetch_overpass(mock_client, "park", max_results=1)
            return docs

        docs_op = asyncio.run(run_overpass())
        self.assertEqual(len(docs_op), 1)
        self.assertIn("Jardins du Trocadéro", docs_op[0].clean_markdown)
        self.assertIn("tourism=attraction", docs_op[0].clean_markdown)

    def test_wikipedia_summary_and_geosearch(self):
        """Verify Wikipedia REST summary and geosearch endpoint parsing."""
        wiki = WikipediaSource()

        mock_summary = {
            "title": "Group Relative Policy Optimization",
            "extract": "GRPO is a reinforcement learning algorithm that evaluates groups of completions.",
            "content_urls": {
                "desktop": {
                    "page": "https://en.wikipedia.org/wiki/Group_Relative_Policy_Optimization"
                }
            },
        }

        async def run_wiki():
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_summary
            mock_resp.text = json.dumps(mock_summary)
            mock_client.get = AsyncMock(return_value=mock_resp)

            doc = await wiki._fetch_summary(mock_client, "GRPO")
            return doc

        doc = asyncio.run(run_wiki())
        self.assertIsNotNone(doc)
        self.assertEqual(doc.url, "https://en.wikipedia.org/wiki/Group_Relative_Policy_Optimization")
        self.assertIn("GRPO is a reinforcement learning algorithm", doc.clean_markdown)

        # Mock Geosearch
        mock_geosearch = {
            "query": {
                "geosearch": [
                    {
                        "pageid": 999,
                        "title": "Musée d'Orsay",
                        "lat": 48.8599,
                        "lon": 2.3266,
                        "dist": 150.0,
                    }
                ]
            }
        }

        async def run_geo():
            mock_client = AsyncMock()
            mock_resp_geo = MagicMock()
            mock_resp_geo.status_code = 200
            mock_resp_geo.json.return_value = mock_geosearch

            mock_resp_sum = MagicMock()
            mock_resp_sum.status_code = 200
            mock_resp_sum.text = '{"title": "Musée d\'Orsay"}'
            mock_resp_sum.json.return_value = {
                "title": "Musée d'Orsay",
                "extract": "A museum on the Left Bank of the Seine.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Musée_d'Orsay"}},
            }

            mock_client.get = AsyncMock(side_effect=[mock_resp_geo, mock_resp_sum])
            docs = await wiki.fetch_geosearch(48.8599, 2.3266, radius_meters=500, max_results=1)
            return docs

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_resp_geo = MagicMock()
            mock_resp_geo.status_code = 200
            mock_resp_geo.json.return_value = mock_geosearch

            mock_resp_sum = MagicMock()
            mock_resp_sum.status_code = 200
            mock_resp_sum.text = '{"title": "Musée d\'Orsay"}'
            mock_resp_sum.json.return_value = {
                "title": "Musée d'Orsay",
                "extract": "A museum on the Left Bank of the Seine.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Mus%C3%A9e_d%27Orsay"}},
            }
            mock_inst.get = AsyncMock(side_effect=[mock_resp_geo, mock_resp_sum])
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            geo_docs = asyncio.run(wiki.fetch_geosearch(48.8599, 2.3266, radius_meters=500, max_results=1))
            self.assertEqual(len(geo_docs), 1)
            self.assertIn("Musée d'Orsay", geo_docs[0].clean_markdown)

    def test_arxiv_atom_xml_parsing(self):
        """Verify ArXiv Atom XML parsing with titles, abstracts, and authors."""
        arxiv = ArXivSource()

        xml_data = """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2402.03300v1</id>
            <title>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</title>
            <summary>We introduce DeepSeekMath 7B. We also introduce Group Relative Policy Optimization (GRPO).</summary>
            <author><name>Zhihong Shao</name></author>
            <author><name>Peiyi Wang</name></author>
            <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL"/>
            <category term="cs.AI"/>
            <link title="pdf" href="http://arxiv.org/pdf/2402.03300v1" rel="related" type="application/pdf"/>
          </entry>
        </feed>
        """

        docs = arxiv.parse_atom_xml(xml_data)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].url, "https://arxiv.org/abs/2402.03300v1")
        self.assertIn("DeepSeekMath", docs[0].clean_markdown)
        self.assertIn("Group Relative Policy Optimization", docs[0].clean_markdown)
        self.assertIn("Zhihong Shao", docs[0].clean_markdown)
        self.assertIn("cs.CL", docs[0].clean_markdown)

    def test_arxiv_search_query_targets_canonical_dpo_paper(self):
        self.assertEqual(
            ArXivSource.build_search_query(
                "How does Direct Preference Optimization eliminate the reward model?"
            ),
            'all:"Direct Preference Optimization" AND all:"reward model"',
        )
        self.assertEqual(
            ArXivSource.build_search_query("latest advances in GRPO"),
            'all:"Group Relative Policy Optimization"',
        )

    def test_duckduckgo_parsing(self):
        """Verify DuckDuckGo HTML parser extracts links and snippets."""
        ddg = DuckDuckGoSource()

        html_data = """
        <html>
        <body>
            <div class="result results_links">
                <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Findex.html">pytorch.org</a>
                <a class="result__snippet" href="#">PyTorch documentation and official library reference.</a>
            </div>
            <div class="result results_links">
                <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fhuggingface.co%2Fdocs">huggingface.co</a>
                <a class="result__snippet" href="#">Hugging Face ecosystem models and datasets.</a>
            </div>
        </body>
        </html>
        """

        docs = ddg.parse_html(html_data)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].url, "https://pytorch.org/docs/stable/index.html")
        self.assertIn("PyTorch documentation", docs[0].clean_markdown)
        self.assertEqual(docs[1].url, "https://huggingface.co/docs")

    def test_provenance_logger(self):
        """Verify provenance logging, deduplication checks, and persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "test_manifest.jsonl"
            logger = ProvenanceLogger(manifest_path=str(manifest_file), auto_load=False)

            doc1 = ScrapedDocumentSchema(
                url="https://example.com/page1",
                raw_html="<p>Doc 1 Content</p>",
                clean_markdown="Doc 1 Content",
            )
            doc1.compute_hash()

            self.assertFalse(logger.is_seen_url(doc1.url))
            self.assertFalse(logger.is_seen_hash(doc1.sha256_hash))

            # Record
            record = logger.record_document(doc1, source_name="wikipedia", latency_ms=12.5)
            self.assertEqual(record.url, doc1.url)
            self.assertEqual(record.sha256_hash, doc1.sha256_hash)
            self.assertTrue(logger.is_seen_url(doc1.url))
            self.assertTrue(logger.is_seen_hash(doc1.sha256_hash))

            stats = logger.get_stats()
            self.assertEqual(stats["total_records"], 1)
            self.assertEqual(stats["unique_urls"], 1)

            # Re-instantiate logger to verify disk rehydration
            logger2 = ProvenanceLogger(manifest_path=str(manifest_file), auto_load=True)
            self.assertTrue(logger2.is_seen_url(doc1.url))
            self.assertTrue(logger2.is_seen_hash(doc1.sha256_hash))
            self.assertEqual(logger2.get_stats()["total_records"], 1)

    def test_intent_classification(self):
        """Verify query intent classification heuristic mapping."""
        # Map/Geospatial
        self.assertEqual(
            classify_intent("What are the coordinates of Mount Everest?"),
            QueryIntent.MAP_GEOSPATIAL,
        )
        self.assertEqual(
            classify_intent("latitude and longitude of Tokyo Tower"),
            QueryIntent.MAP_GEOSPATIAL,
        )

        # Academic / Research
        self.assertEqual(
            classify_intent("arxiv paper on transformer architecture attention mechanism"),
            QueryIntent.ACADEMIC_RESEARCH,
        )
        self.assertEqual(
            classify_intent("loss function convergence theorem derivation"),
            QueryIntent.ACADEMIC_RESEARCH,
        )
        self.assertEqual(
            classify_intent(
                "How does Direct Preference Optimization eliminate the reward model?"
            ),
            QueryIntent.ACADEMIC_RESEARCH,
        )

        # Technical Documentation
        self.assertEqual(
            classify_intent("pytorch torchrun distributed training api docs"),
            QueryIntent.TECHNICAL_DOCS,
        )
        self.assertEqual(
            classify_intent("python standard library module reference"),
            QueryIntent.TECHNICAL_DOCS,
        )

        # Encyclopedic
        self.assertEqual(
            classify_intent("Who was Marie Curie biography and nobel prize"),
            QueryIntent.ENCYCLOPEDIC,
        )
        self.assertEqual(
            classify_intent("history of the Roman Empire wikipedia"),
            QueryIntent.ENCYCLOPEDIC,
        )

        # General
        self.assertEqual(
            classify_intent("best coffee shops nearby with wifi"),
            QueryIntent.GENERAL_WEB,
        )

    def test_ingestion_router_dispatch_and_deduplication(self):
        """Verify IngestionRouter dispatches and rejects duplicate hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.jsonl"
            prov_logger = ProvenanceLogger(manifest_path=str(manifest_file), auto_load=False)
            router = IngestionRouter(provenance_logger=prov_logger)

            # Mock OSM Source
            doc_sample = ScrapedDocumentSchema(
                url="https://www.openstreetmap.org/node/100",
                raw_html="<osm>Colosseum</osm>",
                clean_markdown="Colosseum in Rome, coordinates 41.8902, 12.4922",
            )
            doc_sample.compute_hash()

            router.osm_source.fetch = AsyncMock(return_value=[doc_sample])

            async def run_dispatch():
                intent, docs = await router.route_and_fetch("coordinates of Colosseum Rome")
                return intent, docs

            intent, docs = asyncio.run(run_dispatch())
            self.assertEqual(intent, QueryIntent.MAP_GEOSPATIAL)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].url, doc_sample.url)

            # Ingesting the same query again must filter out duplicate hash
            _, docs2 = asyncio.run(run_dispatch())
            self.assertEqual(len(docs2), 0)  # Filtered out duplicate!

    def test_ingestion_router_multi_fetch(self):
        """Verify multi-intent parallel dispatching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "multi_manifest.jsonl"
            prov_logger = ProvenanceLogger(manifest_path=str(manifest_file), auto_load=False)
            router = IngestionRouter(provenance_logger=prov_logger)

            doc_wiki = ScrapedDocumentSchema(
                url="https://en.wikipedia.org/wiki/Transformer",
                raw_html="<wiki>Transformer</wiki>",
                clean_markdown="Transformer architecture overview.",
            )
            doc_arxiv = ScrapedDocumentSchema(
                url="https://arxiv.org/abs/1706.03762",
                raw_html="<arxiv>Attention is all you need</arxiv>",
                clean_markdown="Attention Is All You Need paper abstract.",
            )

            router.wiki_source.fetch = AsyncMock(return_value=[doc_wiki])
            router.arxiv_source.fetch = AsyncMock(return_value=[doc_arxiv])

            async def run_multi():
                return await router.fetch_multi(
                    query="Transformer architecture",
                    intents=[QueryIntent.ENCYCLOPEDIC, QueryIntent.ACADEMIC_RESEARCH],
                    max_per_source=1,
                )

            multi_results = asyncio.run(run_multi())
            self.assertIn(QueryIntent.ENCYCLOPEDIC, multi_results)
            self.assertIn(QueryIntent.ACADEMIC_RESEARCH, multi_results)
            self.assertEqual(len(multi_results[QueryIntent.ENCYCLOPEDIC]), 1)
            self.assertEqual(len(multi_results[QueryIntent.ACADEMIC_RESEARCH]), 1)


if __name__ == "__main__":
    unittest.main()

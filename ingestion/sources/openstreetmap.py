"""OpenStreetMap geospatial and map knowledge ingestion engine."""

import json
from typing import Optional
import httpx
from loguru import logger

from core.schemas import ScrapedDocumentSchema
from ingestion.sources.base import BaseIngestionSource

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "RLHF-Autonomous-Continual-Learning/1.0 (contact: info@local-rlhf-pipeline.org)"


class OpenStreetMapSource(BaseIngestionSource):
    """Geospatial engine ingesting structured coordinates, bounding boxes, and POI topology."""

    def __init__(self, timeout: float = 12.0):
        super().__init__(name="openstreetmap")
        self.timeout = timeout

    async def fetch(self, query: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Geocode query and extract spatial features and surrounding POIs."""
        results: list[ScrapedDocumentSchema] = []
        headers = {"User-Agent": USER_AGENT}

        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                results = await self._fetch_nominatim(client, query, max_results=max_results)
        except Exception as e:
            logger.error(f"Error executing OpenStreetMap ingestion for '{query}': {e}")

        return results

    async def _fetch_nominatim(
        self, client: httpx.AsyncClient, query: str, max_results: int = 5
    ) -> list[ScrapedDocumentSchema]:
        """Query Nominatim endpoint and process matching places."""
        results: list[ScrapedDocumentSchema] = []
        params = {
            "q": query,
            "format": "json",
            "addressdetails": "1",
            "limit": str(max_results),
        }
        resp = await client.get(NOMINATIM_URL, params=params)
        if resp.status_code != 200:
            logger.warning(f"Nominatim query failed with status {resp.status_code}")
            return results

        places = resp.json()
        for place in places:
            doc = await self._process_place(client, place)
            if doc:
                results.append(doc)
        return results

    async def _process_place(self, client: httpx.AsyncClient, place: dict) -> Optional[ScrapedDocumentSchema]:
        """Process a Nominatim geocoded place and fetch nearby POIs via Overpass API."""
        try:
            display_name = place.get("display_name", "Unknown Location")
            lat = float(place.get("lat", 0.0))
            lon = float(place.get("lon", 0.0))
            bbox = place.get("boundingbox", [])
            place_type = place.get("type", "feature")
            category = place.get("class", "general")
            osm_type = place.get("osm_type", "node")
            osm_id = place.get("osm_id", "")
            address = place.get("address", {})

            # Build structured Markdown spatial document
            lines = [
                f"# Geospatial Entity: {display_name}",
                "",
                "## Geographic Coordinates & Boundary",
                f"- **Latitude / Longitude**: ({lat:.6f}, {lon:.6f})",
                f"- **Place Type**: {place_type} (Category: {category})",
                f"- **OSM Identifier**: {osm_type}/{osm_id}",
            ]

            if len(bbox) == 4:
                lines.append(f"- **Bounding Box [South, North, West, East]**: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

            if address:
                lines.extend([
                    "",
                    "## Administrative Hierarchy",
                    f"- Country: {address.get('country', 'N/A')}",
                    f"- State / Region: {address.get('state', address.get('region', 'N/A'))}",
                    f"- County / District: {address.get('county', 'N/A')}",
                    f"- City / Town: {address.get('city', address.get('town', 'N/A'))}",
                    f"- Postcode: {address.get('postcode', 'N/A')}",
                ])

            # Query Overpass for nearby POIs within small delta box (0.01 deg ≈ 1.1 km)
            pois = await self._fetch_overpass_pois(client, lat, lon)
            if pois:
                lines.extend([
                    "",
                    "## Surrounding Points of Interest & Topology",
                    "\n".join(f"- {p}" for p in pois[:15]),
                ])

            content = "\n".join(lines)
            url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_id else f"https://nominatim.openstreetmap.org/place/{lat},{lon}"

            doc = ScrapedDocumentSchema(
                url=url,
                raw_html=json.dumps(place),
                clean_markdown=content,
            )
            doc.compute_hash()
            return doc
        except Exception as e:
            logger.debug(f"Failed to process place entry: {e}")
            return None

    async def _fetch_overpass_pois(self, client: httpx.AsyncClient, lat: float, lon: float, delta: float = 0.008) -> list[str]:
        """Fetch surrounding POIs (amenities, tourism, transport) via Overpass QL."""
        s, n = lat - delta, lat + delta
        w, e = lon - delta, lon + delta
        ql = f"""[out:json][timeout:8];
(
  node["amenity"]({s},{w},{n},{e});
  node["tourism"]({s},{w},{n},{e});
  node["historic"]({s},{w},{n},{e});
);
out body 20;"""
        try:
            resp = await client.post(OVERPASS_URL, data={"data": ql})
            if resp.status_code == 200:
                data = resp.json()
                pois = []
                for element in data.get("elements", []):
                    tags = element.get("tags", {})
                    name = tags.get("name")
                    amenity = tags.get("amenity") or tags.get("tourism") or tags.get("historic")
                    if name and amenity:
                        pois.append(f"**{name}** ({amenity})")
                return pois
        except Exception as e:
            logger.debug(f"Overpass POI query failed (graceful degradation): {e}")
        return []

    async def _fetch_overpass(
        self, client: httpx.AsyncClient, query_tag: str, max_results: int = 5
    ) -> list[ScrapedDocumentSchema]:
        """Directly query Overpass API by tag or amenity and return document schemas."""
        ql = f"""[out:json][timeout:10];
node[{query_tag}];
out body {max_results};"""
        docs = []
        try:
            resp = await client.post(OVERPASS_URL, data={"data": ql})
            if resp.status_code == 200:
                data = resp.json()
                for el in data.get("elements", []):
                    tags = el.get("tags", {})
                    name = tags.get("name", "OSM Feature")
                    lat, lon = el.get("lat", 0.0), el.get("lon", 0.0)
                    tag_str = ", ".join(f"{k}={v}" for k, v in tags.items())
                    md = f"# {name}\n- **Coordinates**: ({lat}, {lon})\n- **Tags**: {tag_str}"
                    doc = ScrapedDocumentSchema(
                        url=f"https://www.openstreetmap.org/node/{el.get('id', 0)}",
                        raw_html=json.dumps(el),
                        clean_markdown=md,
                    )
                    doc.compute_hash()
                    docs.append(doc)
        except Exception as e:
            logger.debug(f"Direct Overpass query failed: {e}")
        return docs

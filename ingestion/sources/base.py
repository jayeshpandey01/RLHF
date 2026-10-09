"""Base abstraction and URL normalization utilities for ingestion sources."""

from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from loguru import logger

from core.schemas import ScrapedDocumentSchema

# Binary and non-text extensions to automatically reject
DISALLOWED_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".exe",
    ".dmg",
    ".iso",
    ".bin",
    ".apk",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
}

# Tracking query parameters to strip
TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "source",
    "session_id",
}


def canonicalize_url(raw_url: str) -> Optional[str]:
    """Normalize URL, remove tracking query parameters, strip fragments, and discard binaries."""
    try:
        parsed = urlparse(raw_url.strip())
        if parsed.scheme not in ("http", "https"):
            return None

        # Check for disallowed binary extensions in path
        path_lower = parsed.path.lower()
        for ext in DISALLOWED_EXTENSIONS:
            if path_lower.endswith(ext):
                return None

        # Filter out tracking parameters
        clean_query = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in TRACKING_QUERY_PARAMS and not k.lower().startswith("utm_")
        ]

        # Normalize hostname to lowercase and strip default ports
        netloc = parsed.netloc.lower()
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif netloc.endswith(":443"):
            netloc = netloc[:-4]

        # Strip fragment (#) and trailing slashes from path
        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

        canonical = urlunparse(
            (
                parsed.scheme.lower(),
                netloc,
                path,
                "",  # params
                urlencode(clean_query),
                "",  # fragment
            )
        )
        return canonical
    except Exception as e:
        logger.debug(f"Failed to canonicalize URL '{raw_url}': {e}")
        return None


class BaseIngestionSource(ABC):
    """Abstract base class for all specialized ingestion engines."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def fetch(self, query: str, max_results: int = 5) -> list[ScrapedDocumentSchema]:
        """Fetch and return a list of standardized scraped document schemas."""
        pass


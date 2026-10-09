"""Cryptographic provenance logger for multi-source ingestion.

Maintains an immutable append-only JSONL audit manifest tracking every scraped document
with its source URL, SHA-256 cryptographic hash, ingestion timestamp, and performance metrics.
"""

import json
import threading
from pathlib import Path
from typing import Optional
from loguru import logger

from core.schemas import ProvenanceRecordSchema, ScrapedDocumentSchema


class ProvenanceLogger:
    """Thread-safe append-only provenance manifest recorder and deduplication index."""

    def __init__(self, manifest_path: str = "data/provenance_manifest.jsonl", auto_load: bool = True):
        self.manifest_path = Path(manifest_path)
        self._lock = threading.Lock()
        self._seen_urls: set[str] = set()
        self._seen_hashes: set[str] = set()
        self._total_records: int = 0

        # Ensure parent directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if auto_load and self.manifest_path.exists():
            self.load_manifest()

    def load_manifest(self) -> int:
        """Load pre-existing records from manifest file into in-memory bloom/set index."""
        with self._lock:
            loaded = 0
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            url = data.get("url")
                            h = data.get("sha256_hash")
                            if url:
                                self._seen_urls.add(url)
                            if h:
                                self._seen_hashes.add(h)
                            loaded += 1
                        except json.JSONDecodeError:
                            continue
                self._total_records = loaded
                logger.info(
                    f"ProvenanceLogger loaded {loaded} records ({len(self._seen_urls)} URLs, {len(self._seen_hashes)} hashes) from {self.manifest_path}"
                )
                return loaded
            except Exception as e:
                logger.warning(f"Failed to load provenance manifest at {self.manifest_path}: {e}")
                return 0

    def is_seen_url(self, url: str) -> bool:
        """Check if URL has already been recorded in the provenance manifest."""
        with self._lock:
            return url in self._seen_urls

    def is_seen_hash(self, sha256_hash: str) -> bool:
        """Check if content hash has already been recorded in the provenance manifest."""
        with self._lock:
            return sha256_hash in self._seen_hashes

    def record_document(
        self,
        doc: ScrapedDocumentSchema,
        source_name: str,
        latency_ms: float = 0.0,
        status_code: int = 200,
    ) -> ProvenanceRecordSchema:
        """Record scraped document to the manifest and update index."""
        if not doc.sha256_hash:
            doc.compute_hash()

        record = ProvenanceRecordSchema(
            url=doc.url,
            sha256_hash=doc.sha256_hash,
            source_name=source_name,
            status_code=status_code,
            latency_ms=latency_ms,
            content_length=len(doc.clean_markdown or doc.raw_html),
        )

        with self._lock:
            self._seen_urls.add(record.url)
            self._seen_hashes.add(record.sha256_hash)
            self._total_records += 1

            # Append to JSONL file
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")

        return record

    def get_stats(self) -> dict:
        """Return provenance tracking statistics."""
        with self._lock:
            return {
                "total_records": self._total_records,
                "unique_urls": len(self._seen_urls),
                "unique_hashes": len(self._seen_hashes),
                "manifest_path": str(self.manifest_path),
            }

    def clear(self) -> None:
        """Reset in-memory sets and truncate manifest file (for testing purposes)."""
        with self._lock:
            self._seen_urls.clear()
            self._seen_hashes.clear()
            self._total_records = 0
            if self.manifest_path.exists():
                self.manifest_path.write_text("", encoding="utf-8")


"""Graph-based connected components cluster deduplication and canonical document election."""

import math
from typing import Optional
from loguru import logger

from core.schemas import CleanedDocumentSchema
from curation.filters import COMMON_STOPWORDS


class DisjointSet:
    """Disjoint-Set Union (DSU / Union-Find) data structure with path compression and rank union."""

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        """Find the representative root of the set containing item with path compression."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item

        # Path compression: point node directly to root
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> bool:
        """Merge sets containing elements a and b using union by rank."""
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False  # Already in same set

        # Attach smaller rank tree under root of larger rank tree
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True

    def get_components(self) -> dict[str, list[str]]:
        """Return all connected component clusters grouped by root representative."""
        components: dict[str, list[str]] = {}
        for item in list(self.parent.keys()):
            root = self.find(item)
            if root not in components:
                components[root] = []
            components[root].append(item)
        return components


class ClusterDeduplicator:
    """Partitions near-duplicate candidate pairs into connected components and elects canonicals."""

    @staticmethod
    def compute_quality_score(doc: CleanedDocumentSchema) -> float:
        """Compute heuristic information density score for canonical election."""
        text = doc.clean_text.lower()
        words = text.split()
        word_count = max(1, len(words))

        # Stopword ratio
        stopword_count = sum(1 for w in words if w in COMMON_STOPWORDS)
        stopword_density = max(0.01, stopword_count / float(word_count))

        # Length bonus (logarithmic scaling)
        length_factor = math.log(max(10, len(doc.clean_text)))

        # Repetition penalty
        clean_rep_factor = max(0.01, 1.0 - doc.line_repetition_rate)

        # Quality = ln(len) * (1 - R_line) * stopword_density
        score = length_factor * clean_rep_factor * stopword_density
        return score

    def deduplicate(
        self,
        documents: list[CleanedDocumentSchema],
        candidate_pairs: list[tuple[str, str]],
    ) -> tuple[list[CleanedDocumentSchema], dict]:
        """Group documents into graph components and elect one canonical document per cluster.
        
        Args:
            documents: List of cleaned documents passing quality checks.
            candidate_pairs: List of (url_a, url_b) pairs with Jaccard similarity >= threshold.
            
        Returns:
            (canonical_documents: list[CleanedDocumentSchema], audit_metrics: dict)
        """
        doc_map = {doc.url: doc for doc in documents}
        dsu = DisjointSet()

        # Initialize all documents as singletons
        for doc in documents:
            dsu.find(doc.url)

        # Union near-duplicate pairs
        merged_edges = 0
        for url_a, url_b in candidate_pairs:
            if url_a in doc_map and url_b in doc_map:
                if dsu.union(url_a, url_b):
                    merged_edges += 1

        components = dsu.get_components()
        canonical_docs: list[CleanedDocumentSchema] = []
        multi_doc_clusters = 0
        pruned_count = 0

        for root, cluster_urls in components.items():
            if len(cluster_urls) == 1:
                # Singleton document: retain directly
                canonical_docs.append(doc_map[cluster_urls[0]])
            else:
                multi_doc_clusters += 1
                # Elect canonical document with highest quality score
                cluster_docs = [doc_map[u] for u in cluster_urls if u in doc_map]
                if not cluster_docs:
                    continue

                best_doc = max(cluster_docs, key=self.compute_quality_score)
                canonical_docs.append(best_doc)
                pruned_count += len(cluster_docs) - 1
                logger.debug(
                    f"Cluster of size {len(cluster_docs)} resolved: Elected '{best_doc.url}' (score={self.compute_quality_score(best_doc):.2f}), pruned {len(cluster_docs) - 1} duplicates"
                )

        audit_metrics = {
            "total_input_docs": len(documents),
            "candidate_edges": len(candidate_pairs),
            "merged_edges": merged_edges,
            "total_clusters": len(components),
            "multi_doc_clusters": multi_doc_clusters,
            "pruned_duplicates": pruned_count,
            "canonical_retained": len(canonical_docs),
        }

        logger.info(
            f"ClusterDeduplicator: {len(documents)} input docs -> {len(canonical_docs)} canonicals retained ({pruned_count} pruned across {multi_doc_clusters} clusters)"
        )
        return canonical_docs, audit_metrics


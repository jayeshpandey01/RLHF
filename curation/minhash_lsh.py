"""MinHash Locality-Sensitive Hashing (LSH) for fuzzy near-duplicate document deduplication."""

import random
import re
from typing import Optional
import mmh3

MERSENNE_PRIME = (1 << 61) - 1  # 2^61 - 1


class MinHashLSH:
    """MinHash and LSH index implementing Broder's theorem with S-curve banding."""

    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 32,
        shingle_size: int = 5,
        threshold: float = 0.80,
        seed: int = 42,
    ):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands  # e.g., 128 // 32 = 4
        self.shingle_size = shingle_size
        self.threshold = threshold

        # Generate deterministic universal hash coefficients a_i, b_i
        rng = random.Random(seed)
        self.a_coeffs = [rng.randint(1, MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self.b_coeffs = [rng.randint(0, MERSENNE_PRIME - 1) for _ in range(num_perm)]

        # LSH Hash Tables: (band_idx, band_hash) -> list[doc_id]
        self.buckets: dict[tuple[int, int], list[str]] = {}
        # Storage of registered signatures: doc_id -> list[int]
        self.signatures: dict[str, list[int]] = {}

    def tokenize_and_shingle(self, text: str) -> list[str]:
        """Convert text into k-shingles of normalized words."""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < self.shingle_size:
            # Fallback to single words or characters if text is very short
            return words if words else [text.lower()]

        shingles = [
            " ".join(words[i : i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        ]
        return shingles

    def compute_signature(self, text: str) -> list[int]:
        """Compute the m-dimensional MinHash signature vector for given text."""
        shingles = self.tokenize_and_shingle(text)
        if not shingles:
            return [0] * self.num_perm

        # 64-bit integer hashes for each shingle
        shingle_hashes = [
            mmh3.hash64(s)[0] & 0x7FFFFFFFFFFFFFFF  # non-negative 63-bit int
            for s in shingles
        ]

        # MinHash vector: min_s ((a_i * s + b_i) mod p)
        signature = []
        for a, b in zip(self.a_coeffs, self.b_coeffs):
            min_val = min(((a * h + b) % MERSENNE_PRIME) for h in shingle_hashes)
            signature.append(min_val)

        return signature

    @staticmethod
    def jaccard_similarity(sig_a: list[int], sig_b: list[int]) -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if len(sig_a) != len(sig_b) or not sig_a:
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / float(len(sig_a))

    def _hash_band(self, band_slice: list[int]) -> int:
        """Hash a band chunk into a single 64-bit bucket identifier."""
        raw_bytes = b"".join(val.to_bytes(8, byteorder="little", signed=False) for val in band_slice)
        return mmh3.hash64(raw_bytes)[0]

    def query_candidates(self, signature: list[int]) -> set[str]:
        """Retrieve all candidate doc IDs sharing at least one band."""
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = signature[start:end]
            band_hash = self._hash_band(band_slice)

            key = (band_idx, band_hash)
            if key in self.buckets:
                candidates.update(self.buckets[key])
        return candidates

    def is_duplicate(
        self, signature: list[int], threshold: Optional[float] = None
    ) -> tuple[bool, float, Optional[str]]:
        """Check if signature matches any indexed document with Jaccard >= threshold.
        
        Returns:
            (is_duplicate: bool, max_similarity: float, matching_doc_id: Optional[str])
        """
        thresh = threshold if threshold is not None else self.threshold
        candidates = self.query_candidates(signature)
        if not candidates:
            return False, 0.0, None

        max_sim = 0.0
        best_match = None

        for doc_id in candidates:
            cand_sig = self.signatures.get(doc_id)
            if cand_sig is None:
                continue
            sim = self.jaccard_similarity(signature, cand_sig)
            if sim > max_sim:
                max_sim = sim
                best_match = doc_id

            if max_sim >= thresh:
                return True, max_sim, best_match

        return (max_sim >= thresh), max_sim, best_match

    def index(self, doc_id: str, signature: list[int]) -> None:
        """Register document signature into LSH hash tables."""
        self.signatures[doc_id] = signature

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = signature[start:end]
            band_hash = self._hash_band(band_slice)

            key = (band_idx, band_hash)
            if key not in self.buckets:
                self.buckets[key] = []
            self.buckets[key].append(doc_id)

    def size(self) -> int:
        """Return total number of indexed documents."""
        return len(self.signatures)

    def clear(self) -> None:
        """Reset the LSH index."""
        self.buckets.clear()
        self.signatures.clear()


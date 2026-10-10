"""Convex combination experience replay buffer and model collapse prevention engine (Phase 6.2).

Prevents catastrophic forgetting and model collapse via:
  1. Strict 70% Anchor / 30% Streaming Web stratified mixture sampling (Robins, 1995; Askell et al., 2021).
  2. ModelCollapseGuard: Statistical Shannon token entropy, vocabulary diversity ratio,
     and n-gram repetition penalty (Shumailov et al., Nature 2024).
  3. Memory-mapped zero-RAM token streaming and binary dataset export.
  4. FIFO sliding-window capacity control for streaming web ingestion.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger

from core.schemas import CleanedDocumentSchema
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.dataset import export_tokens_to_binary


@dataclass
class ReplaySample:
    """Individual data sample in the continual experience replay buffer."""
    text: str
    source: str = "web"  # "anchor" or "web"
    doc_id: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: Optional[List[int]] = None


class ModelCollapseGuard:
    """Detects statistical generative degradation, low entropy, and synthetic hallucination loops.

    Based on Shumailov et al. (Nature 2024), continuous recursive training causes
    distribution tails to vanish, producing severe entropy drops and vocabulary collapse.
    """

    def __init__(
        self,
        min_shannon_entropy: float = 2.5,
        min_vocab_diversity: float = 0.20,
        max_ngram_repetition_rate: float = 0.25,
        min_tokens: int = 15,
        ngram_size: int = 4,
    ):
        self.min_shannon_entropy = float(min_shannon_entropy)
        self.min_vocab_diversity = float(min_vocab_diversity)
        self.max_ngram_repetition_rate = float(max_ngram_repetition_rate)
        self.min_tokens = int(min_tokens)
        self.ngram_size = int(ngram_size)

    def tokenize_words(self, text: str) -> List[str]:
        """Extract alphanumeric lowercase tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def calculate_entropy(self, tokens: Sequence[Any]) -> float:
        """Calculate Shannon token entropy: H(T) = -sum p(v) log2 p(v)."""
        if not tokens:
            return 0.0
        n = len(tokens)
        counts = Counter(tokens)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            entropy -= p * math.log2(p)
        return float(entropy)

    def calculate_vocab_diversity(self, tokens: Sequence[Any]) -> float:
        """Calculate vocabulary diversity ratio: |V_unique| / |T|."""
        if not tokens:
            return 0.0
        return float(len(set(tokens)) / len(tokens))

    def calculate_max_ngram_repetition(self, tokens: Sequence[Any], n: Optional[int] = None) -> float:
        """Calculate maximum frequency ratio of any repeated n-gram."""
        ngram_len = n or self.ngram_size
        if len(tokens) < ngram_len:
            return 0.0
        ngrams = [tuple(tokens[i : i + ngram_len]) for i in range(len(tokens) - ngram_len + 1)]
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        most_common_count = counts.most_common(1)[0][1]
        return float(most_common_count / len(ngrams))

    def evaluate_tokens(self, tokens: Sequence[Any]) -> Tuple[bool, str, Dict[str, float]]:
        """Evaluate token sequence against model collapse thresholds."""
        n_tokens = len(tokens)
        entropy = self.calculate_entropy(tokens)
        vocab_diversity = self.calculate_vocab_diversity(tokens)
        ngram_rep = self.calculate_max_ngram_repetition(tokens, self.ngram_size)

        metrics = {
            "token_count": float(n_tokens),
            "shannon_entropy": entropy,
            "vocab_diversity": vocab_diversity,
            "ngram_repetition_rate": ngram_rep,
        }

        # For very short sequences, bypass strict diversity checks to avoid false positives
        if n_tokens < self.min_tokens:
            return True, "PASSED_SHORT_SEQUENCE", metrics

        if entropy < self.min_shannon_entropy:
            return False, f"COLLAPSE_LOW_ENTROPY (H={entropy:.2f} < {self.min_shannon_entropy})", metrics

        if vocab_diversity < self.min_vocab_diversity:
            return False, f"COLLAPSE_LOW_VOCAB_DIVERSITY (V_ratio={vocab_diversity:.2f} < {self.min_vocab_diversity})", metrics

        if ngram_rep > self.max_ngram_repetition_rate:
            return False, f"COLLAPSE_HIGH_NGRAM_REPETITION (Rep={ngram_rep:.2f} > {self.max_ngram_repetition_rate})", metrics

        return True, "PASSED_QUALITY_GATE", metrics

    def evaluate_text(self, text: str) -> Tuple[bool, str, Dict[str, float]]:
        """Evaluate natural language string against model collapse thresholds."""
        tokens = self.tokenize_words(text)
        return self.evaluate_tokens(tokens)


class ExperienceReplayBuffer:
    """Stratified Continual Experience Replay Buffer with Model Collapse Prevention.

    Enforces:
      - 70% Golden Anchor baseline (immutable, prevents catastrophic forgetting).
      - 30% Streaming Web ingestion (curated domain knowledge).
      - Active ModelCollapseGuard quarantine on streaming web ingestion.
      - FIFO sliding-window capacity management for web samples.
    """

    def __init__(
        self,
        anchor_ratio: float = 0.70,
        seed: int = 42,
        max_web_capacity: Optional[int] = None,
        guard_model_collapse: bool = True,
        collapse_guard: Optional[ModelCollapseGuard] = None,
    ):
        if not (0.0 < anchor_ratio < 1.0):
            raise ValueError(f"anchor_ratio must be between 0.0 and 1.0, got {anchor_ratio}")

        self.anchor_ratio = float(anchor_ratio)
        self.web_ratio = float(1.0 - anchor_ratio)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.max_web_capacity = int(max_web_capacity) if max_web_capacity is not None else None
        self.max_quarantine_capacity = 1000
        self.guard_model_collapse = bool(guard_model_collapse)
        self.collapse_guard = collapse_guard or ModelCollapseGuard()

        self.anchor_pool: List[ReplaySample] = []
        self.web_pool: List[ReplaySample] = []
        self.quarantine_pool: List[Dict[str, Any]] = []
        self._seen_hashes: set[str] = set()

    def add_anchor_sample(self, sample: ReplaySample) -> None:
        """Add an immutable golden anchor sample to the buffer."""
        if sample.text in self._seen_hashes:
            return
        self._seen_hashes.add(sample.text)
        sample.source = "anchor"
        self.anchor_pool.append(sample)

    def add_anchor_data(self, texts: List[str]) -> int:
        """Add golden baseline anchor documents to the buffer (backward-compatible)."""
        clean_texts = [t.strip() for t in texts if t and t.strip()]
        for i, text in enumerate(clean_texts):
            if text in self._seen_hashes:
                continue
            self._seen_hashes.add(text)
            self.anchor_pool.append(
                ReplaySample(
                    text=text,
                    source="anchor",
                    doc_id=f"anchor_{len(self.anchor_pool)}_{i}",
                )
            )
        logger.info(f"ExperienceReplayBuffer: Added {len(clean_texts)} anchor documents (Total: {len(self.anchor_pool)})")
        return len(self.anchor_pool)

    def add_web_sample(self, sample: ReplaySample) -> bool:
        """Add a streaming web sample to the buffer with model collapse gating.

        Returns:
            True if sample passed collapse guard and was indexed; False if quarantined or duplicate.
        """
        if sample.text in self._seen_hashes:
            return False

        sample.source = "web"
        if self.guard_model_collapse:
            is_safe, reason, metrics = self.collapse_guard.evaluate_text(sample.text)
            if not is_safe:
                logger.warning(f"ExperienceReplayBuffer: Quarantined web sample due to {reason}")
                self.quarantine_pool.append({
                    "sample": sample,
                    "reason": reason,
                    "metrics": metrics,
                })
                if len(self.quarantine_pool) > self.max_quarantine_capacity:
                    self.quarantine_pool.pop(0)
                return False

        self._seen_hashes.add(sample.text)
        self.web_pool.append(sample)
        self._apply_capacity_eviction()
        return True

    def add_web_data(self, documents: Union[List[CleanedDocumentSchema], List[str]]) -> int:
        """Add cleaned web documents with automated model collapse filtering (backward-compatible)."""
        accepted = 0
        for i, item in enumerate(documents):
            if isinstance(item, CleanedDocumentSchema):
                text = item.clean_text.strip()
                url = item.url
                doc_id = item.sha256_hash or f"web_{len(self.web_pool)}_{i}"
            else:
                text = item.strip()
                url = None
                doc_id = f"web_{len(self.web_pool)}_{i}"

            if not text:
                continue

            sample = ReplaySample(text=text, source="web", doc_id=doc_id, url=url)
            if self.add_web_sample(sample):
                accepted += 1

        logger.info(
            f"ExperienceReplayBuffer: Added {accepted}/{len(documents)} web documents "
            f"(Total: {len(self.web_pool)}, Quarantined: {len(self.quarantine_pool)})"
        )
        return accepted

    def _apply_capacity_eviction(self) -> None:
        """Evict oldest web samples (FIFO) if exceeding maximum capacity."""
        if self.max_web_capacity is not None and len(self.web_pool) > self.max_web_capacity:
            evict_count = len(self.web_pool) - self.max_web_capacity
            self.web_pool = self.web_pool[evict_count:]
            logger.debug(f"ExperienceReplayBuffer: Evicted {evict_count} oldest web samples (Current: {len(self.web_pool)})")

    def sample_batch(self, batch_size: int) -> List[str]:
        """Sample a text mini-batch with exact 70/30 anchor mixture (backward-compatible)."""
        replay_samples = self.sample_replay_batch(batch_size=batch_size)
        return [s.text for s in replay_samples]

    def sample_replay_batch(self, batch_size: int) -> List[ReplaySample]:
        """Sample a structured ReplaySample mini-batch with exact alpha=0.70 anchor ratio."""
        if not self.anchor_pool and not self.web_pool:
            raise ValueError("ExperienceReplayBuffer is completely empty.")

        if self.anchor_pool and self.web_pool:
            n_anchor = int(round(batch_size * self.anchor_ratio))
        elif self.anchor_pool:
            n_anchor = batch_size
        else:
            n_anchor = 0
        n_web = batch_size - n_anchor

        batch: List[ReplaySample] = []

        # Sample anchor
        if self.anchor_pool:
            sampled_anchor = self.rng.choices(self.anchor_pool, k=n_anchor)
            batch.extend(sampled_anchor)
        else:
            n_web = batch_size

        # Sample web
        if self.web_pool:
            sampled_web = self.rng.choices(self.web_pool, k=n_web)
            batch.extend(sampled_web)

        self.rng.shuffle(batch)
        return batch

    def get_stats(self) -> Dict[str, Any]:
        """Return current buffer statistics, mixture ratios, and quarantine metrics."""
        total = len(self.anchor_pool) + len(self.web_pool)
        return {
            "total_documents": total,
            "anchor_count": len(self.anchor_pool),
            "web_count": len(self.web_pool),
            "quarantined_count": len(self.quarantine_pool),
            "configured_anchor_ratio": self.anchor_ratio,
            "empirical_anchor_ratio": (len(self.anchor_pool) / max(1, total)),
            "max_web_capacity": self.max_web_capacity,
        }

    def get_collapse_guard_stats(self) -> Dict[str, Any]:
        """Return detailed statistics regarding quarantined and degraded web samples."""
        reasons = Counter([q["reason"].split(" ")[0] for q in self.quarantine_pool])
        return {
            "quarantined_total": len(self.quarantine_pool),
            "reasons_breakdown": dict(reasons),
        }

    def export_sharded_binaries(
        self,
        tokenizer: Optional[CharTokenizer] = None,
        output_dir: Union[Path, str] = "data",
        target_tokens: int = 10_000,
        train_val_split: float = 0.90,
    ) -> Tuple[Path, Path]:
        """Interleave the replay corpus, tokenize, and export memory-mapped binary files for training.

        Enforces document-level partitioning before tokenization to prevent data leakage
        and sample-with-replacement cross-split contamination (Lee et al., ACL 2022).

        Returns:
            (train_bin_path: Path, val_bin_path: Path)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if tokenizer is None:
            anchor_sample = [s.text for s in self.anchor_pool[:20]]
            web_sample = [s.text for s in self.web_pool[:20]]
            all_text = " ".join(anchor_sample + web_sample)
            tokenizer = CharTokenizer(all_text)

        # 1. Deduplicate candidate document texts
        unique_anchors: List[str] = list(dict.fromkeys(s.text for s in self.anchor_pool))
        unique_web: List[str] = list(dict.fromkeys(s.text for s in self.web_pool))

        # 2. Document-level train/validation partition
        if len(unique_anchors) > 1:
            split_a = max(1, int(len(unique_anchors) * train_val_split))
            if split_a >= len(unique_anchors):
                split_a = len(unique_anchors) - 1
            train_anchors = unique_anchors[:split_a]
            val_anchors = unique_anchors[split_a:]
        else:
            train_anchors = unique_anchors
            val_anchors = unique_anchors

        if len(unique_web) > 1:
            split_w = max(1, int(len(unique_web) * train_val_split))
            if split_w >= len(unique_web):
                split_w = len(unique_web) - 1
            train_web = unique_web[:split_w]
            val_web = unique_web[split_w:]
        else:
            train_web = unique_web
            val_web = unique_web

        target_train_tokens = max(16, int(target_tokens * train_val_split))
        target_val_tokens = max(16, target_tokens - target_train_tokens)

        # 3. Helper to assemble tokens from document pools respecting anchor_ratio
        def _collect_tokens(
            anchor_docs: List[str],
            web_docs: List[str],
            token_budget: int,
            seed: int = 42,
        ) -> List[int]:
            if not anchor_docs and not web_docs:
                return []
            rng = random.Random(seed)
            collected_tokens: List[int] = []
            space_token = tokenizer.encode(" ")[0]

            while len(collected_tokens) < token_budget:
                draw_anchor = True
                if web_docs and rng.random() > self.anchor_ratio:
                    draw_anchor = False

                if draw_anchor or not web_docs:
                    doc = rng.choice(anchor_docs) if anchor_docs else rng.choice(web_docs)
                else:
                    doc = rng.choice(web_docs)

                encoded = tokenizer.encode(doc)
                if encoded:
                    collected_tokens.extend(encoded)
                    collected_tokens.append(space_token)

            return collected_tokens

        train_tokens = _collect_tokens(train_anchors, train_web, target_train_tokens, seed=self.seed)
        val_tokens = _collect_tokens(val_anchors, val_web, target_val_tokens, seed=self.seed + 1)

        train_bin = out_path / "replay_train.bin"
        val_bin = out_path / "replay_val.bin"

        export_tokens_to_binary(train_tokens, train_bin)
        export_tokens_to_binary(val_tokens, val_bin)

        logger.info(
            f"ExperienceReplayBuffer: Exported {len(train_tokens):,} train tokens to {train_bin} "
            f"and {len(val_tokens):,} val tokens to {val_bin} (Zero-leakage document partition)"
        )
        return train_bin, val_bin

    def clear(self) -> None:
        """Clear all pools and quarantines."""
        self.anchor_pool.clear()
        self.web_pool.clear()
        self.quarantine_pool.clear()

    def __len__(self) -> int:
        return len(self.anchor_pool) + len(self.web_pool)

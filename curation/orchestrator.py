"""Master Continual Learning Data Orchestrator.

Ties together Multi-Source Ingestion (Phase 2.1), C4 Quality & Contamination Filtering (Phase 2.2),
Graph-Based Cluster Deduplication (Phase 2.3), and Experience Replay Binary Serialization (Phase 2.4)
into a single turnkey, production-grade post-training data pipeline.
"""

from pathlib import Path
from typing import Optional, Union
from loguru import logger

from core.schemas import CleanedDocumentSchema, ScrapedDocumentSchema
from curation.cluster import ClusterDeduplicator
from curation.pipeline import CurationPipeline
from curation.replay_buffer import ExperienceReplayBuffer
from ingestion.provenance import ProvenanceLogger
from ingestion.router import IngestionRouter, QueryIntent
from training.m1_bigram.tokenizer import CharTokenizer

# Default golden anchor conversational demonstrations for baseline distribution stability
DEFAULT_ANCHOR_BASELINE = [
    "User: Explain the difference between supervised learning and reinforcement learning.\n"
    "Assistant: In supervised learning, a model learns to map inputs to outputs using labeled training pairs with direct supervision. "
    "In reinforcement learning, an autonomous agent interacts with an environment, learning an optimal policy through trial, error, "
    "and delayed scalar reward signals without explicit input-output pairs.",
    
    "User: What is gradient descent in deep learning?\n"
    "Assistant: Gradient descent is a first-order iterative optimization algorithm that minimizes a differentiable scalar objective function. "
    "It computes the negative gradient of the loss surface with respect to parameters and updates weights along the steepest descent direction.",
    
    "User: How does the attention mechanism solve the bottleneck in Recurrent Neural Networks?\n"
    "Assistant: Standard RNNs compress sequential inputs into a single fixed-length hidden state vector, creating an information bottleneck. "
    "The attention mechanism allows each token to directly attend to all other token representations with dynamic weightings, "
    "enabling constant path length O(1) information routing across arbitrary sequence lengths.",
]


class ContinualDataOrchestrator:
    """Master orchestrator connecting Ingestion (2.1), Curation (2.2), Clustering (2.3), and Replay (2.4)."""

    def __init__(
        self,
        router: Optional[IngestionRouter] = None,
        curation_pipeline: Optional[CurationPipeline] = None,
        cluster_deduplicator: Optional[ClusterDeduplicator] = None,
        replay_buffer: Optional[ExperienceReplayBuffer] = None,
    ):
        self.router = router or IngestionRouter()
        self.curation = curation_pipeline or CurationPipeline()
        self.cluster_dedup = cluster_deduplicator or ClusterDeduplicator()
        self.replay = replay_buffer or ExperienceReplayBuffer(anchor_ratio=0.70)

    async def ingest_curate_and_export(
        self,
        query: str,
        max_scraped: int = 5,
        anchor_samples: Optional[list[str]] = None,
        output_dir: Union[Path, str] = "data",
        target_tokens: int = 2_000,
        tokenizer: Optional[CharTokenizer] = None,
    ) -> tuple[Path, Path, dict]:
        """Execute the complete end-to-end continual learning dataflow.
        
        Steps:
            1. Multi-source scraping with cryptographic SHA-256 provenance logging (Phase 2.1).
            2. C4 quality filtering and 13-gram benchmark contamination guard (Phase 2.2).
            3. MinHash candidate pairing and Disjoint-Set cluster deduplication (Phase 2.3).
            4. Experience replay mixing (70% anchor / 30% web) and binary token export (Phase 2.4).
            
        Returns:
            (train_bin_path: Path, val_bin_path: Path, metrics: dict)
        """
        logger.info(f"ContinualDataOrchestrator: Starting pipeline for query='{query}'...")

        # 1. Multi-Source Ingestion (Phase 2.1)
        intent, scraped_docs = await self.router.route_and_fetch(query, max_results=max_scraped)
        logger.info(f"Phase 2.1: Ingested {len(scraped_docs)} documents with intent '{intent.value}'")

        # 2. C4 Quality & Contamination Filtering (Phase 2.2)
        cleaned_docs, filter_stats = self.curation.process_batch(scraped_docs)
        logger.info(f"Phase 2.2: {len(cleaned_docs)}/{len(scraped_docs)} passed quality & contamination gates")

        # 3. MinHash LSH Candidate Discovery & Graph Cluster Resolution (Phase 2.3)
        candidate_pairs = []
        for i in range(len(cleaned_docs)):
            for j in range(i + 1, len(cleaned_docs)):
                sim = self.curation.minhash_lsh.jaccard_similarity(
                    cleaned_docs[i].minhash_signature, cleaned_docs[j].minhash_signature
                )
                if sim >= self.curation.minhash_lsh.threshold:
                    candidate_pairs.append((cleaned_docs[i].url, cleaned_docs[j].url))

        canonical_docs, cluster_metrics = self.cluster_dedup.deduplicate(cleaned_docs, candidate_pairs)
        logger.info(f"Phase 2.3: {len(canonical_docs)} canonical documents retained after cluster deduplication")

        # 4. Experience Replay Buffer & Binary Serialization (Phase 2.4)
        anchors = anchor_samples or DEFAULT_ANCHOR_BASELINE
        self.replay.add_anchor_data(anchors)
        self.replay.add_web_data(canonical_docs)

        # Build combined tokenizer if not provided
        if tokenizer is None:
            all_corpus = " ".join(anchors + [d.clean_text for d in canonical_docs])
            tokenizer = CharTokenizer(all_corpus)

        train_bin, val_bin = self.replay.export_sharded_binaries(
            tokenizer=tokenizer,
            output_dir=output_dir,
            target_tokens=target_tokens,
        )

        metrics = {
            "query": query,
            "intent": intent.value,
            "scraped_count": len(scraped_docs),
            "cleaned_count": len(cleaned_docs),
            "canonical_count": len(canonical_docs),
            "canonical_docs": canonical_docs,
            "cluster_metrics": cluster_metrics,
            "filter_stats": filter_stats,
            "replay_stats": self.replay.get_stats(),
            "train_bin_path": str(train_bin),
            "val_bin_path": str(val_bin),
            "tokenizer": tokenizer,
        }
        logger.info(f"ContinualDataOrchestrator: E2E pipeline complete. Binaries exported to {output_dir}")
        return train_bin, val_bin, metrics

    async def ingest_curate_and_sft_align(
        self,
        query: str,
        base_model: "MiniGPT",
        tokenizer: Optional[CharTokenizer] = None,
        max_scraped: int = 5,
        anchor_ratio: float = 0.70,
        sft_config: Optional[object] = None,
        lora_config: Optional[object] = None,
    ) -> tuple[object, dict]:
        """Turnkey autonomous bridge connecting Ingestion (M2) -> SFT Continual Alignment (M3).
        
        Fetches web knowledge, filters and cluster-deduplicates it, and runs LoRA SFT with
        the anti-catastrophic forgetting regression gate.
        """
        from training.m3_sft.orchestrator import ContinualSFTOrchestrator

        # 1. Ingest, curate, and deduplicate
        intent, scraped_docs = await self.router.route_and_fetch(query, max_results=max_scraped)
        cleaned_docs, filter_stats = self.curation.process_batch(scraped_docs)

        candidate_pairs = []
        for i in range(len(cleaned_docs)):
            for j in range(i + 1, len(cleaned_docs)):
                sim = self.curation.minhash_lsh.jaccard_similarity(
                    cleaned_docs[i].minhash_signature, cleaned_docs[j].minhash_signature
                )
                if sim >= self.curation.minhash_lsh.threshold:
                    candidate_pairs.append((cleaned_docs[i].url, cleaned_docs[j].url))

        canonical_docs, cluster_metrics = self.cluster_dedup.deduplicate(cleaned_docs, candidate_pairs)

        # 2. Build or reuse tokenizer
        if tokenizer is None:
            all_text = " ".join([d.clean_text for d in canonical_docs] + DEFAULT_ANCHOR_BASELINE)
            tokenizer = CharTokenizer(all_text)

        # 3. Continual SFT Orchestrator
        sft_orchestrator = ContinualSFTOrchestrator(
            base_model=base_model,
            tokenizer=tokenizer,
            lora_config=lora_config,  # type: ignore
            sft_config=sft_config,    # type: ignore
        )
        pipeline_result = sft_orchestrator.run_continual_alignment(
            web_documents=canonical_docs,
            anchor_ratio=anchor_ratio,
        )

        audit_metrics = {
            "query": query,
            "intent": intent.value,
            "scraped_count": len(scraped_docs),
            "cleaned_count": len(cleaned_docs),
            "canonical_count": len(canonical_docs),
            "cluster_metrics": cluster_metrics,
            "filter_stats": filter_stats,
            "pipeline_success": pipeline_result.success,
            "deployment_status": pipeline_result.deployment_status,
        }
        return pipeline_result, audit_metrics


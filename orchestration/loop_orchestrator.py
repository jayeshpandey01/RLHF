"""Autonomous Continual Learning Loop Orchestrator (Milestone 6 - Phase 6.3).

Coordinates the decoupled high-throughput serving and out-of-band training lifecycle:
  1. L1 Semantic Cache (< 5ms hit)
  2. L2 Hybrid Retrieval (BM25 + Dense RRF)
  3. Non-blocking event bus & request coalescing (debounce window)
  4. Experience replay buffer with ModelCollapseGuard (70% anchor / 30% web)
  5. Decoupled training triggers (volume, time, drift)
  6. Milestone 4/5 statistical gating & zero-downtime atomic model hot-swapping
"""

import copy
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from curation.replay_buffer import ExperienceReplayBuffer, ReplaySample
from evaluation.citation_verifier import CitationVerifier
from evaluation.statistical_gating import GateDecision, StatisticalReleaseGate
from orchestration.async_queue import AsyncEventQueue, TelemetryEvent
from orchestration.coalescer import RequestCoalescer
from orchestration.semantic_cache import SemanticCache
from retrieval.hybrid import HybridRetriever
from retrieval.schemas import RetrievalDocument, SearchResult
from training.m5_preference.alignment_engine import AlignmentEngine, AlignmentStrategy
from training.m5_preference.benchmark_suite import AlignmentBenchmarkSuite, AlignmentTestCase

_USE_ORCHESTRATOR_MODEL = object()


class ContinualLearningOrchestrator:
    """Master orchestrator for decoupled high-concurrency continual learning and serving."""

    def __init__(
        self,
        active_model: Optional[Any] = None,
        anchor_model: Optional[Any] = None,
        retriever: Optional[HybridRetriever] = None,
        semantic_cache: Optional[SemanticCache] = None,
        coalescer: Optional[RequestCoalescer] = None,
        event_queue: Optional[AsyncEventQueue] = None,
        replay_buffer: Optional[ExperienceReplayBuffer] = None,
        benchmark_suite: Optional[AlignmentBenchmarkSuite] = None,
        statistical_gate: Optional[StatisticalReleaseGate] = None,
        training_trigger_threshold: int = 5,
        citation_verifier: Optional[CitationVerifier] = None,
    ):
        self.active_model = active_model
        self.anchor_model = anchor_model or active_model
        self.candidate_model: Optional[Any] = None

        self.retriever = retriever or HybridRetriever()
        self.semantic_cache = semantic_cache or SemanticCache()
        self.coalescer = coalescer or RequestCoalescer(debounce_window_seconds=1.0)
        self.event_queue = event_queue or AsyncEventQueue()
        self.replay_buffer = replay_buffer or ExperienceReplayBuffer(anchor_ratio=0.70)
        self.benchmark_suite = benchmark_suite or AlignmentBenchmarkSuite()
        self.statistical_gate = statistical_gate or StatisticalReleaseGate()
        self.citation_verifier = citation_verifier or CitationVerifier()

        self.training_trigger_threshold = int(training_trigger_threshold)
        self.new_web_samples_since_train: int = 0
        self.model_version: int = 1
        self.total_queries_served: int = 0
        self.training_cycles_completed: int = 0
        self.latest_gating_decision: Optional[str] = None

        self._model_lock = threading.Lock()

    @staticmethod
    def _extractive_fallback(query: str, evidence: list[SearchResult]) -> str:
        """Return the most query-relevant sentence across cited retrieved sources."""
        if not evidence:
            return "I couldn't find source evidence to answer this query."

        query_terms = {
            token
            for token in re.findall(r"\b[a-z0-9]+\b", query.lower())
            if len(token) > 2 and token not in {"how", "does", "what", "the", "from", "that", "this"}
        }
        best_sentence = ""
        best_source_index = 1
        best_score = -1
        for source_index, result in enumerate(evidence, start=1):
            abstract_lines: list[str] = []
            in_abstract = False
            for line in result.text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    if "abstract" in stripped.lower():
                        in_abstract = True
                        continue
                    if in_abstract:
                        break
                    continue
                if not stripped or stripped.startswith("- **"):
                    continue
                if in_abstract:
                    abstract_lines.append(stripped)
                elif not abstract_lines:
                    abstract_lines.append(stripped)

            content = " ".join(abstract_lines) or result.text
            content = re.sub(r"\s+", " ", content).strip()
            content = re.sub(r"\s*\[\d+\]", "", content)
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", content)
                if sentence.strip()
            ]
            for sentence in sentences:
                sentence_terms = re.findall(r"\b[a-z0-9]+\b", sentence.lower())
                matched_terms = {
                    term
                    for term in query_terms
                    if any(token.startswith(term[:4]) or term.startswith(token[:4]) for token in sentence_terms)
                }
                if len(matched_terms) > best_score:
                    best_sentence = sentence
                    best_source_index = source_index
                    best_score = len(matched_terms)

        if not best_sentence:
            return "I couldn't extract a readable answer from the retrieved sources."
        sentence = best_sentence
        if len(sentence) > 600:
            sentence = sentence[:597].rsplit(" ", maxsplit=1)[0] + "..."
        return f"{sentence} [{best_source_index}]"

    def process_query(
        self,
        query: str,
        top_k: int = 3,
        generator_fn: Optional[Callable[[str, List[SearchResult], Any], str]] = None,
        active_model: Any = _USE_ORCHESTRATOR_MODEL,
    ) -> Dict[str, Any]:
        """High-concurrency query path: Semantic Cache -> Hybrid Retrieval -> Active Model.

        Bypasses retrieval and inference if query hits L1 semantic cache (< 5ms).
        Emits non-blocking telemetry events for out-of-band continuous ingestion.
        """
        self.total_queries_served += 1
        clean_query = query.strip()

        # Step 1: L1 Semantic Cache lookup
        cache_hit = self.semantic_cache.lookup(clean_query)
        if cache_hit is not None:
            cached_resp, sim, meta = cache_hit
            # Emit telemetry
            self.event_queue.enqueue(
                TelemetryEvent(
                    event_type="query_cache_hit",
                    payload={"query": clean_query, "similarity": sim},
                )
            )
            return {
                "query": clean_query,
                "response": cached_resp,
                "cached": True,
                "similarity": sim,
                "evidence": meta.get("evidence", []),
                "model_version": self.model_version,
            }

        # Step 2: L2 Hybrid Retrieval lookup
        evidence = self.retriever.retrieve(clean_query, top_k=top_k)

        # Step 3: Inference generation
        if active_model is _USE_ORCHESTRATOR_MODEL:
            with self._model_lock:
                current_model = self.active_model
        else:
            current_model = active_model

        if generator_fn is not None and not evidence:
            response = "I couldn't find source evidence to answer this query."
        elif generator_fn is not None:
            response = generator_fn(clean_query, evidence, current_model)
        elif hasattr(current_model, "transformer") or hasattr(current_model, "config"):
            from training.m1_bigram.tokenizer import CharTokenizer
            tokenizer = getattr(current_model, "tokenizer", None) or CharTokenizer()
            prompt_context = f"Context: {' '.join([e.text for e in evidence])}\nQuestion: {clean_query}\nAnswer:"
            tokens = tokenizer.encode(prompt_context)
            device = getattr(current_model, "device", torch.device("cpu"))
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            max_block = getattr(current_model.config, "block_size", 256)
            if idx.size(1) > max_block:
                idx = idx[:, -max_block:]
            gen_tokens = current_model.generate(idx, max_new_tokens=40)
            response = tokenizer.decode(gen_tokens[0].tolist()[idx.size(1):])
        elif hasattr(current_model, "generate") and callable(current_model.generate):
            prompt_context = f"Context: {' '.join([e.text for e in evidence])}\nQuestion: {clean_query}\nAnswer:"
            try:
                response = str(current_model.generate(prompt_context))
            except Exception:
                evidence_summary = " ".join([e.text for e in evidence]) if evidence else "No evidence found."
                response = f"Grounded response for '{clean_query}': {evidence_summary}"
        else:
            # Fallback formatted response with evidence grounding
            evidence_summary = " ".join([e.text for e in evidence]) if evidence else "No evidence found."
            response = f"Grounded response for '{clean_query}': {evidence_summary}"

        # In-flight factual grounding verification
        if evidence and response:
            full_context = " ".join([e.text for e in evidence])
            ref_docs = {
                str(index): result.text
                for index, result in enumerate(evidence, start=1)
            }
            ref_docs.update(
                {result.doc_id: result.text for result in evidence if result.doc_id}
            )
            ver_metrics = self.citation_verifier.verify_response(
                response=response,
                ground_truth_context=full_context,
                reference_documents=ref_docs,
            )
            if ver_metrics.factual_sentences > 0 and (
                ver_metrics.citation_recall < 1.0
                or ver_metrics.citation_precision < 1.0
                or ver_metrics.faithfulness_score < 0.5
                or ver_metrics.hallucinated_citations
            ):
                logger.warning(
                    "Generated answer failed citation/faithfulness checks; returning a cited "
                    "extract from the most query-relevant retrieved source."
                )
                response = self._extractive_fallback(clean_query, evidence)

        # Step 4: Populate L1 Semantic Cache
        evidence_dicts = [
            {"doc_id": e.doc_id, "text": e.text, "url": e.url, "score": e.score}
            for e in evidence
        ]
        self.semantic_cache.put(
            query=clean_query,
            response=response,
            metadata={"evidence": evidence_dicts},
        )

        # Step 5: Non-blocking asynchronous event emission for request coalescing
        self.coalescer.add_query(clean_query)
        self.event_queue.enqueue(
            TelemetryEvent(
                event_type="query_served",
                payload={"query": clean_query, "evidence_count": len(evidence)},
            )
        )

        return {
            "query": clean_query,
            "response": response,
            "cached": False,
            "similarity": 0.0,
            "evidence": evidence_dicts,
            "model_version": self.model_version,
        }

    def ingest_coalesced_knowledge(
        self,
        scraper_fn: Optional[Callable[[List[str]], List[str]]] = None,
        force_drain: bool = False,
    ) -> int:
        """Drain coalesced queries and ingest fresh web knowledge into replay buffer."""
        canonical_queries = self.coalescer.drain_coalesced_queries(force=force_drain)
        if not canonical_queries:
            return 0

        # Fetch external documents via scraper or mock synthesis
        if scraper_fn is not None:
            documents = scraper_fn(canonical_queries)
        else:
            documents = [
                f"Curated fresh web knowledge regarding: {q}. Validated factual evidence."
                for q in canonical_queries
            ]

        accepted_samples: list[ReplaySample] = []
        ingestion_id = time.time_ns()
        for index, text in enumerate(documents):
            if not text.strip():
                continue
            sample = ReplaySample(
                text=text,
                doc_id=f"continual_doc_{ingestion_id}_{index}",
            )
            if self.replay_buffer.add_web_sample(sample):
                accepted_samples.append(sample)

        accepted = len(accepted_samples)
        self.new_web_samples_since_train += accepted

        # Keep retrieval aligned with the samples accepted into the replay corpus.
        retrieval_docs = [
            RetrievalDocument(doc_id=sample.doc_id, text=sample.text, url=sample.url or "")
            for sample in accepted_samples
            if sample.doc_id is not None
        ]
        self.retriever.add_documents(retrieval_docs)

        logger.info(
            f"ContinualLearningOrchestrator: Ingested {accepted} web documents from "
            f"{len(canonical_queries)} coalesced queries (Pending for train: {self.new_web_samples_since_train})"
        )
        return accepted

    def should_trigger_continual_training(self, force: bool = False) -> Tuple[bool, str]:
        """Check whether out-of-band continual post-training should be triggered."""
        if force:
            return True, "TRIGGERED_MANUALLY"

        if self.new_web_samples_since_train >= self.training_trigger_threshold:
            return True, f"TRIGGERED_BUFFER_VOLUME ({self.new_web_samples_since_train} >= {self.training_trigger_threshold})"

        return False, "THRESHOLD_NOT_MET"

    def execute_continual_learning_cycle(
        self,
        force: bool = False,
        candidate_generate_fn: Optional[Callable[[str], str]] = None,
        baseline_generate_fn: Optional[Callable[[str], str]] = None,
        train_fn: Optional[Callable[[Any, List[str]], Any]] = None,
    ) -> Dict[str, Any]:
        """Execute complete out-of-band Continual Training -> Statistical Gating -> Atomic Hot-Swap."""
        self.ingest_coalesced_knowledge(force_drain=True)

        should_train, reason = self.should_trigger_continual_training(force=force)
        if not should_train:
            return {
                "status": "SKIPPED",
                "reason": reason,
                "model_version": self.model_version,
            }

        if train_fn is None:
            return {
                "status": "SKIPPED",
                "reason": "TRAIN_FN_NOT_CONFIGURED",
                "model_version": self.model_version,
            }
        if len(self.replay_buffer) == 0:
            return {
                "status": "SKIPPED",
                "reason": "REPLAY_BUFFER_EMPTY",
                "model_version": self.model_version,
            }

        logger.info(f"ContinualLearningOrchestrator: Initiating continual training cycle ({reason})...")

        # Step 3: Clone candidate model in secondary memory buffer
        with self._model_lock:
            if hasattr(self.active_model, "state_dict"):
                self.candidate_model = copy.deepcopy(self.active_model)
            else:
                self.candidate_model = self.active_model

        # Step 4: Sample stratified 70/30 replay training batch
        replay_batch = self.replay_buffer.sample_batch(batch_size=min(len(self.replay_buffer), 10))
        self.candidate_model = train_fn(self.candidate_model, replay_batch)
        if self.candidate_model is None:
            raise RuntimeError("Training callback did not return a candidate model.")

        # Step 5: Benchmark Evaluation & M4 Statistical Gating
        cand_fn = candidate_generate_fn
        if cand_fn is None:
            if hasattr(self.candidate_model, "transformer") or hasattr(self.candidate_model, "config"):
                def _cand_gen(p: str) -> str:
                    from training.m1_bigram.tokenizer import CharTokenizer
                    tokenizer = getattr(self.candidate_model, "tokenizer", None) or CharTokenizer()
                    tokens = tokenizer.encode(p)
                    device = getattr(self.candidate_model, "device", torch.device("cpu"))
                    idx = torch.tensor([tokens], dtype=torch.long, device=device)
                    max_block = getattr(self.candidate_model.config, "block_size", 256)
                    if idx.size(1) > max_block:
                        idx = idx[:, -max_block:]
                    gen = self.candidate_model.generate(idx, max_new_tokens=40)
                    return tokenizer.decode(gen[0].tolist()[idx.size(1):])
                cand_fn = _cand_gen
            elif hasattr(self.candidate_model, "generate") and callable(self.candidate_model.generate):
                cand_fn = lambda p: str(self.candidate_model.generate(p))
            else:
                logger.warning("ContinualLearningOrchestrator: Candidate model cannot generate text. Failing closed.")
                return {
                    "status": "REJECTED",
                    "reason": "CANDIDATE_MODEL_CANNOT_GENERATE",
                    "decision": GateDecision.BLOCK_RELEASE.value,
                    "model_version": self.model_version,
                }

        base_fn = baseline_generate_fn
        if base_fn is None:
            if hasattr(self.active_model, "transformer") or hasattr(self.active_model, "config"):
                def _base_gen(p: str) -> str:
                    from training.m1_bigram.tokenizer import CharTokenizer
                    tokenizer = getattr(self.active_model, "tokenizer", None) or CharTokenizer()
                    tokens = tokenizer.encode(p)
                    device = getattr(self.active_model, "device", torch.device("cpu"))
                    idx = torch.tensor([tokens], dtype=torch.long, device=device)
                    max_block = getattr(self.active_model.config, "block_size", 256)
                    if idx.size(1) > max_block:
                        idx = idx[:, -max_block:]
                    gen = self.active_model.generate(idx, max_new_tokens=40)
                    return tokenizer.decode(gen[0].tolist()[idx.size(1):])
                base_fn = _base_gen
            elif hasattr(self.active_model, "generate") and callable(self.active_model.generate):
                base_fn = lambda p: str(self.active_model.generate(p))
            else:
                base_fn = lambda p: "Baseline response"

        try:
            report_obj = self.benchmark_suite.evaluate_candidate(
                candidate_name=f"Candidate-V{self.model_version + 1}",
                candidate_generate_fn=cand_fn,
                baseline_generate_fn=base_fn,
            )
            eval_report = report_obj.to_dict()
            decision = eval_report.get("gate_decision", GateDecision.BLOCK_RELEASE.value)
        except Exception as e:
            logger.error(f"Candidate evaluation crashed (fail-closed): {e}")
            decision = GateDecision.BLOCK_RELEASE.value
            eval_report = {"gate_decision": decision, "error": str(e)}

        self.latest_gating_decision = decision

        # Step 6: Atomic Hot-Swap or Rollback
        if decision == GateDecision.PROCEED_DEPLOY.value:
            with self._model_lock:
                self.active_model = self.candidate_model
                self.model_version += 1
            self.new_web_samples_since_train = 0
            self.training_cycles_completed += 1
            # Evict semantic cache on model upgrade to avoid stale completions
            self.semantic_cache.clear()
            logger.info(f"ContinualLearningOrchestrator: PROCEED_DEPLOY -> Model atomically hot-swapped to V{self.model_version}")
            status = "PROMOTED"

        elif decision in (GateDecision.TRIGGER_ROLLBACK.value, GateDecision.BLOCK_RELEASE.value):
            # Candidate failed pre-deployment gating; reject candidate and retain active model untouched
            logger.info(
                f"ContinualLearningOrchestrator: {decision} -> Candidate rejected; retaining current active model V{self.model_version}."
            )
            status = "REJECTED"
        else:
            # ALERT_DRIFT or unknown: retain current active model
            logger.info(f"ContinualLearningOrchestrator: {decision} -> Holding candidate; retaining current active model.")
            status = "REJECTED"

        return {
            "status": status,
            "reason": reason,
            "decision": decision,
            "model_version": self.model_version,
            "win_rate": eval_report.get("win_rate", 0.0),
            "format_compliance_rate": eval_report.get("format_compliance_rate", 0.0),
            "verifiable_accuracy_rate": eval_report.get("verifiable_accuracy_rate", 0.0),
            "cycles_completed": self.training_cycles_completed,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive orchestrator state and telemetry diagnostics."""
        return {
            "model_version": self.model_version,
            "total_queries_served": self.total_queries_served,
            "training_cycles_completed": self.training_cycles_completed,
            "latest_gating_decision": self.latest_gating_decision,
            "pending_samples_for_train": self.new_web_samples_since_train,
            "trigger_threshold": self.training_trigger_threshold,
            "semantic_cache": self.semantic_cache.get_stats(),
            "coalescer": self.coalescer.get_stats(),
            "event_queue": self.event_queue.get_stats(),
            "replay_buffer": self.replay_buffer.get_stats(),
            "retriever_documents": len(self.retriever),
        }

    def rollback_to_anchor(self, reason: str = "Active model rollback triggered") -> Dict[str, Any]:
        """Atomically restore active production model to golden anchor baseline."""
        with self._model_lock:
            self.active_model = self.anchor_model
            self.model_version = 0
        self.semantic_cache.clear()
        logger.warning(f"ContinualLearningOrchestrator: Active model rolled back to anchor. Reason: {reason}")
        return {
            "status": "ROLLED_BACK",
            "model_version": 0,
            "reason": reason,
        }

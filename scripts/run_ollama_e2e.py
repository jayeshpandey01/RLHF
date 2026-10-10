#!/usr/bin/env python3
"""Run live ingestion, curation, retrieval, Ollama inference, and answer evaluation."""

import argparse
import asyncio
import tempfile
from contextlib import nullcontext
from pathlib import Path

from core.config import settings
from core.ollama_client import OllamaClient
from curation.cluster import ClusterDeduplicator
from curation.filters import C4QualityFilter
from curation.minhash_lsh import MinHashLSH
from curation.pipeline import CurationPipeline
from curation.replay_buffer import ExperienceReplayBuffer
from evaluation.citation_verifier import CitationVerifier
from ingestion.provenance import ProvenanceLogger
from ingestion.router import IngestionRouter
from retrieval.hybrid import HybridRetriever
from retrieval.schemas import RetrievalDocument
from training.m1_bigram.tokenizer import CharTokenizer

DEFAULT_QUERY = "How does Direct Preference Optimization eliminate the reward model?"
ANCHOR_SAMPLES = [
    "User: Explain supervised learning. Assistant: A model learns from labeled input-output examples.",
    "User: Explain gradient descent. Assistant: It updates model parameters to reduce a loss function.",
    "User: Explain attention. Assistant: Attention weights information from different sequence positions.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise live web ingestion through evaluation with an actual local Ollama model."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Topic to collect and ask about.")
    parser.add_argument("--model", default=settings.DEFAULT_MODEL, help="Installed Ollama model name.")
    parser.add_argument("--ollama-url", default=settings.OLLAMA_BASE_URL, help="Ollama API base URL.")
    parser.add_argument("--max-docs", type=int, default=5, help="Maximum documents requested from sources.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved documents passed to Ollama.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Ollama inference timeout in seconds.")
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.5,
        help="Minimum citation-verifier faithfulness score required for a passing evaluation.",
    )
    parser.add_argument("--pull-model", action="store_true", help="Pull the requested model if it is not installed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep provenance and replay binaries in this directory (temporary directory by default).",
    )
    args = parser.parse_args()
    if args.max_docs < 1 or args.top_k < 1 or args.timeout <= 0:
        parser.error("--max-docs and --top-k must be positive; --timeout must be positive.")
    if not 0.0 <= args.min_faithfulness <= 1.0:
        parser.error("--min-faithfulness must be between 0 and 1.")
    return args


def report(stage: str, detail: str) -> None:
    print(f"\n[{stage}] {detail}", flush=True)


async def run(args: argparse.Namespace) -> bool:
    client = OllamaClient(base_url=args.ollama_url, timeout=args.timeout)
    if not await client.check_health():
        report("FAIL: Ollama", f"Cannot reach {args.ollama_url}. Start `ollama serve` and retry.")
        return False

    model_names = await client.list_models()
    model_installed = args.model in model_names or (
        ":" not in args.model and f"{args.model}:latest" in model_names
    )
    if not model_installed and args.pull_model:
        report("Ollama", f"Pulling {args.model}; this may take a while.")
        if not await client.pull_model(args.model):
            report("FAIL: Ollama", f"Could not pull {args.model}.")
            return False
        model_names = await client.list_models()
        model_installed = args.model in model_names or (
            ":" not in args.model and f"{args.model}:latest" in model_names
        )
    if not model_installed:
        report(
            "FAIL: Ollama",
            f"Model {args.model!r} is not installed. Available models: {model_names or 'none'}. "
            f"Run `ollama pull {args.model}` or pass --pull-model.",
        )
        return False
    report("PASS: Ollama", f"Connected to {args.ollama_url}; using installed model {args.model}.")

    output_context = (
        nullcontext(args.output_dir)
        if args.output_dir is not None
        else tempfile.TemporaryDirectory(prefix="rlhf-ollama-e2e-")
    )
    with output_context as output_dir:
        artifacts = Path(output_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        provenance = ProvenanceLogger(
            manifest_path=str(artifacts / "provenance_manifest.jsonl"),
            auto_load=False,
        )
        router = IngestionRouter(provenance_logger=provenance)

        report("Ingestion", f"Collecting up to {args.max_docs} live documents for: {args.query}")
        intent, scraped = await router.route_and_fetch(args.query, max_results=args.max_docs)
        report(
            "Ingestion",
            f"intent={intent.value}; fetched={len(scraped)}; "
            f"provenance_records={provenance.get_stats()['total_records']}",
        )
        if not scraped:
            report(
                "FAIL: Ingestion",
                "No documents were returned. Check network/source availability or try another query.",
            )
            return False

        curation = CurationPipeline(
            quality_filter=C4QualityFilter(),
            minhash_lsh=MinHashLSH(),
        )
        cleaned, filter_stats = curation.process_batch(scraped)
        report(
            "Curation",
            f"accepted={len(cleaned)}/{len(scraped)}; filter_stats={filter_stats}",
        )
        if not cleaned:
            diagnostics = CurationPipeline()
            rejection_reasons: dict[str, int] = {}
            for doc in scraped:
                _, _, reason = diagnostics.process_document(doc)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            report("Curation diagnostics", str(rejection_reasons))
            report("Page extraction", "Search snippets were inadequate; crawling their result pages.")
            crawled = await router.crawler_source.fetch(
                [doc.url for doc in scraped],
                max_results=args.max_docs,
            )
            for doc in crawled:
                if not doc.sha256_hash:
                    doc.compute_hash()
                provenance.record_document(doc, source_name="page_crawler")
            if crawled:
                scraped = crawled
                curation = CurationPipeline(
                    quality_filter=C4QualityFilter(),
                    minhash_lsh=MinHashLSH(),
                )
                cleaned, filter_stats = curation.process_batch(scraped)
                report(
                    "Page extraction",
                    f"crawled={len(crawled)}; curated={len(cleaned)}; filter_stats={filter_stats}",
                )
        if not cleaned:
            report(
                "FAIL: Curation",
                "No documents passed quality and contamination checks, including page extraction.",
            )
            return False

        candidate_pairs: list[tuple[str, str]] = []
        for index, left in enumerate(cleaned):
            for right in cleaned[index + 1 :]:
                similarity = curation.minhash_lsh.jaccard_similarity(
                    left.minhash_signature,
                    right.minhash_signature,
                )
                if similarity >= curation.minhash_lsh.threshold:
                    candidate_pairs.append((left.url, right.url))

        canonical, cluster_stats = ClusterDeduplicator().deduplicate(cleaned, candidate_pairs)
        report(
            "Deduplication",
            f"canonical={len(canonical)}/{len(cleaned)}; near_duplicate_pairs={len(candidate_pairs)}; "
            f"cluster_stats={cluster_stats}",
        )
        if not canonical:
            report("FAIL: Deduplication", "No canonical documents remain.")
            return False

        replay = ExperienceReplayBuffer(anchor_ratio=0.70)
        replay.add_anchor_data(ANCHOR_SAMPLES)
        accepted_web = replay.add_web_data(canonical)
        replay_stats = replay.get_stats()
        if accepted_web == 0:
            report(
                "FAIL: Replay",
                f"No curated web documents passed the replay safety guard; stats={replay_stats}",
            )
            return False

        corpus_text = " ".join(ANCHOR_SAMPLES + [doc.clean_text for doc in canonical])
        tokenizer = CharTokenizer(corpus_text)
        train_path, val_path = replay.export_sharded_binaries(
            tokenizer=tokenizer,
            output_dir=artifacts,
            target_tokens=2_000,
        )
        report(
            "Replay/export",
            f"buffer={replay_stats}; web_accepted={accepted_web}; "
            f"train_tokens_file={train_path.stat().st_size} bytes; "
            f"validation_tokens_file={val_path.stat().st_size} bytes; directory={artifacts}",
        )

        retriever = HybridRetriever()
        retriever.add_documents(
            [
                RetrievalDocument(
                    doc_id=str(index + 1),
                    text=doc.clean_text,
                    url=doc.url,
                    metadata={"sha256": doc.sha256_hash},
                )
                for index, doc in enumerate(canonical)
            ]
        )
        evidence = retriever.retrieve(args.query, top_k=args.top_k)
        if not evidence:
            report("FAIL: Retrieval", "No evidence was returned for the query.")
            return False
        references = {
            str(index): result.text
            for index, result in enumerate(evidence, start=1)
        }
        context = "\n\n".join(
            f"[{index}] {result.text}\nSource: {result.url or 'unknown'}"
            for index, result in enumerate(evidence, start=1)
        )
        report(
            "Retrieval",
            f"returned={len(evidence)}; sources={[result.url for result in evidence]}",
        )

        report("Inference", f"Sending retrieved context to Ollama model {args.model}.")
        answer = await client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied evidence. Do not add facts from memory. "
                        "Cite every factual sentence using the evidence number, for example [1]. "
                        "Do not repeat source URLs in the answer. "
                        "If the evidence is insufficient, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{context}\n\nQuestion: {args.query}",
                },
            ],
            model=args.model,
            temperature=0.0,
        )
        if not answer.strip():
            report("FAIL: Inference", "Ollama returned an empty answer; inspect the Ollama service logs.")
            return False
        print("\n--- Ollama answer ---")
        print(answer)
        print("--- end answer ---")
        report("PASS: Inference", f"Received {len(answer)} characters from Ollama.")

        metrics = CitationVerifier().verify_response(
            response=answer,
            ground_truth_context=" ".join(references.values()),
            reference_documents=references,
        )
        print(
            "\nEvaluation: "
            f"citation_precision={metrics.citation_precision:.3f}, "
            f"citation_recall={metrics.citation_recall:.3f}, "
            f"citation_f1={metrics.citation_f1:.3f}, "
            f"faithfulness={metrics.faithfulness_score:.3f}, "
            f"factual_claims={metrics.factual_sentences}"
        )
        print(f"Evaluation details: {metrics.summary}")
        if metrics.unsupported_claims:
            print(f"Unsupported claims: {metrics.unsupported_claims}")
        if metrics.hallucinated_citations:
            print(f"Invalid citations: {metrics.hallucinated_citations}")
        evaluation_passed = (
            metrics.factual_sentences > 0
            and metrics.citation_precision >= args.min_faithfulness
            and metrics.citation_recall >= args.min_faithfulness
            and metrics.faithfulness_score >= args.min_faithfulness
            and not metrics.hallucinated_citations
        )
        report(
            "PASS: Evaluation" if evaluation_passed else "FAIL: Evaluation",
            f"threshold={args.min_faithfulness:.2f}",
        )

        report(
            "Training/deployment",
            "NOT RUN by this smoke-test script. To train the persistent gateway corpus, "
            "start the API and call POST /trigger_learning; poll GET /learning/jobs/{job_id}.",
        )
        if args.output_dir is None:
            report("Artifacts", "Temporary provenance and token files will be removed when the script exits.")
        else:
            report("Artifacts", f"Saved provenance and token files under {artifacts}.")
        return evaluation_passed


def main() -> int:
    args = parse_args()
    try:
        return 0 if asyncio.run(run(args)) else 1
    except (OSError, RuntimeError, ValueError) as exc:
        report("FAIL", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

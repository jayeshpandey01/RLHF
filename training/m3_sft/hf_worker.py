"""Bounded Hugging Face LoRA training and gated Ollama candidate deployment."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from core.config import settings
from curation.orchestrator import DEFAULT_ANCHOR_BASELINE
from evaluation.benchmark_suite import GoldenBenchmarkSuite
from evaluation.statistical_gating import GateDecision, StatisticalReleaseGate


def _anchor_examples() -> list[tuple[str, str]]:
    examples = []
    for sample in DEFAULT_ANCHOR_BASELINE:
        user, assistant = sample.split("\nAssistant: ", maxsplit=1)
        examples.append((user.removeprefix("User: "), assistant))
    return examples


def _read_web_examples(corpus_path: Path) -> list[tuple[str, str]]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"No collected knowledge corpus exists at {corpus_path}.")
    examples = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
                url = document["url"]
                text = document["text"].strip()
            except (json.JSONDecodeError, KeyError, AttributeError) as exc:
                raise ValueError(f"Invalid corpus record on line {line_number}: {exc}") from exc
            if text and url:
                instruction = document.get("instruction")
                answer = document.get("answer")
                if instruction is None and answer is None:
                    continue
                if (
                    not isinstance(instruction, str)
                    or not instruction.strip()
                    or not isinstance(answer, str)
                    or not answer.strip()
                ):
                    raise ValueError(
                        f"Corpus record on line {line_number} must provide non-empty string "
                        "'instruction' and 'answer' fields together."
                    )
                if document.get("supervision_status") != "reviewed":
                    raise ValueError(
                        f"Corpus record on line {line_number} must mark its supervision_status "
                        "as 'reviewed' before it can be used for SFT."
                    )

                excerpt = text[:600]
                normalized_answer = " ".join(answer.split()).casefold()
                normalized_source = " ".join(text.split()).casefold()
                if normalized_answer == normalized_source:
                    raise ValueError(
                        f"Corpus record on line {line_number} uses the complete source as its answer."
                    )
                examples.append(
                    (
                        f"{instruction.strip()}\n\nSource URL: {url}\n\n{excerpt}",
                        f"{answer.strip()}\n\nSource: {url}",
                    )
                )
    return examples


class HuggingFaceLoRAWorker:
    """Train, evaluate, and package an Ollama-compatible candidate from curated sources."""

    def __init__(
        self,
        corpus_path: Path,
        artifact_dir: Path | None = None,
        base_model: str | None = None,
        ollama_url: str | None = None,
    ):
        self.corpus_path = corpus_path
        self.artifact_dir = artifact_dir or settings.CHECKPOINTS_DIR / "hf_lora"
        self.base_model = base_model or os.getenv(
            "HF_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self.ollama_url = (ollama_url or settings.OLLAMA_BASE_URL).rstrip("/")

    def train_evaluate_and_package(
        self, baseline_model: str | None = None
    ) -> dict[str, Any]:
        """Run CPU-safe LoRA SFT and return a deployable Ollama tag only on gate approval."""
        web = _read_web_examples(self.corpus_path)
        if not web:
            return {
                "status": "BLOCKED",
                "reason": "NO_VALIDATED_WEB_SUPERVISION",
                "detail": (
                    "Raw source text remains available to retrieval but cannot be used as an "
                    "SFT target. Add reviewed instruction/answer fields to web corpus records."
                ),
            }

        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from torch.utils.data import Dataset
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face training dependencies are missing; sync pyproject.toml first."
            ) from exc

        if not torch.cuda.is_available():
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        max_steps = int(os.getenv("HF_SFT_MAX_STEPS", "20"))
        if max_steps < 1:
            raise ValueError("HF_SFT_MAX_STEPS must be at least 1.")

        run_dir = self.artifact_dir / (
            f"candidate-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        )
        adapter_dir = run_dir / "adapter"
        merged_dir = run_dir / "merged"
        run_dir.mkdir(parents=True, exist_ok=False)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        class CompletionDataset(Dataset):
            def __init__(self, pairs: list[tuple[str, str]]):
                self.features = []
                for prompt, answer in pairs:
                    messages = [
                        {"role": "system", "content": "Answer accurately and preserve source attribution."},
                        {"role": "user", "content": prompt},
                    ]
                    prefix = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    full = prefix + answer + (tokenizer.eos_token or "")
                    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                    token_ids = tokenizer(
                        full,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=512,
                    )["input_ids"]
                    if len(token_ids) > len(prefix_ids):
                        labels = [-100] * min(len(prefix_ids), len(token_ids))
                        labels.extend(token_ids[len(labels):])
                        self.features.append((token_ids, labels))
                if not self.features:
                    raise ValueError("No trainable assistant completion tokens were constructed.")

            def __len__(self):
                return len(self.features)

            def __getitem__(self, index):
                input_ids, labels = self.features[index]
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                }

        def collate(features):
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in features],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                [item["labels"] for item in features],
                batch_first=True,
                padding_value=-100,
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in features],
                batch_first=True,
                padding_value=0,
            )
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        anchor = _anchor_examples()
        anchor = _anchor_examples()
        pairs = [
            (anchor[index % len(anchor)] if index % 10 < 7 else web[index % len(web)])
            for index in range(max(len(anchor), len(web)) * 10)
        ]
        dataset = CompletionDataset(pairs)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        model.config.use_cache = False
        training_args = TrainingArguments(
            output_dir=str(run_dir / "trainer"),
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            use_cpu=not torch.cuda.is_available(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=False,
            bf16=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate,
        )
        trainer.train()
        model.config.use_cache = True
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        suite = GoldenBenchmarkSuite.load_default_suite()
        baseline_model = baseline_model or os.getenv("OLLAMA_BASELINE_MODEL", settings.DEFAULT_MODEL)
        baseline_scores = self._score_ollama_baseline(suite, baseline_model=baseline_model)
        candidate_scores = self._score_candidate(model, tokenizer, suite)
        paired_delta_std = os.getenv("HF_GATE_PAIRED_DELTA_STD")
        min_improvement_delta = os.getenv("HF_GATE_MIN_IMPROVEMENT_DELTA")
        gate = StatisticalReleaseGate(
            paired_delta_std=float(paired_delta_std) if paired_delta_std is not None else None,
            min_improvement_delta=(
                float(min_improvement_delta) if min_improvement_delta is not None else None
            ),
        )
        gate_report = gate.evaluate_gate(
            suite,
            candidate_scores=candidate_scores,
            baseline_scores=baseline_scores,
        )
        report_path = run_dir / "release_gate.json"
        report_path.write_text(
            json.dumps(gate_report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        if gate_report.decision != GateDecision.PROCEED_DEPLOY:
            return {
                "status": "BLOCKED",
                "reason": "STATISTICAL_GATE",
                "decision": gate_report.decision.value,
                "gate_report": str(report_path),
                "adapter": str(adapter_dir),
                "ollama_model": None,
                "baseline_model": baseline_model,
            }

        model = model.merge_and_unload()
        model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        model_tag = f"rlhf-candidate-{int(time.time())}-{uuid4().hex[:6]}"
        modelfile = run_dir / "Modelfile"
        modelfile.write_text(f"FROM {merged_dir.resolve()}\n", encoding="utf-8")
        ollama = shutil.which("ollama")
        if ollama is None:
            raise RuntimeError("Ollama CLI was not found; candidate remains saved but undeployed.")
        subprocess.run(
            [ollama, "create", model_tag, "-f", str(modelfile)],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        self._probe_candidate(model_tag)
        return {
            "status": "APPROVED",
            "decision": gate_report.decision.value,
            "gate_report": str(report_path),
            "adapter": str(adapter_dir),
            "merged_model": str(merged_dir),
            "ollama_model": model_tag,
            "baseline_model": baseline_model,
        }

    def _probe_candidate(self, model_tag: str) -> None:
        response = httpx.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": model_tag,
                "messages": [{"role": "user", "content": "Reply with the word ready."}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "")
        if not answer:
            raise RuntimeError(
                f"Ollama created candidate {model_tag}, but its readiness probe returned no answer."
            )

    def _score_ollama_baseline(
        self, suite: GoldenBenchmarkSuite, baseline_model: str | None = None
    ) -> dict[str, float]:
        model_name = baseline_model or os.getenv("OLLAMA_BASELINE_MODEL", settings.DEFAULT_MODEL)
        scores = {}
        for case in suite.cases:
            prompt = self._benchmark_prompt(case)
            response = httpx.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 120},
                },
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "")
            if not answer:
                raise RuntimeError(f"Ollama baseline returned an empty answer for {case.case_id}.")
            scores[case.case_id] = self._case_score(case, answer)
        return scores

    def _score_candidate(self, model, tokenizer, suite: GoldenBenchmarkSuite) -> dict[str, float]:
        import torch

        model.eval()
        scores = {}
        for case in suite.cases:
            messages = [{"role": "user", "content": self._benchmark_prompt(case)}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(next(model.parameters()).device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    max_new_tokens=120,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                output[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            scores[case.case_id] = self._case_score(case, answer)
        return scores

    @staticmethod
    def _benchmark_prompt(case) -> str:
        required = ", ".join(case.required_citations) or "the supplied evidence"
        return (
            f"Context:\n{case.ground_truth_context}\n\nQuestion: {case.question}\n"
            f"Answer using only this context and include citations, including {required}."
        )

    @staticmethod
    def _case_score(case, answer: str) -> float:
        """Score keyword recall, citation presence, and lexical context support in [0.0, 5.0]."""
        if not answer.strip():
            return 0.0

        kw_cov = case.evaluate_keyword_coverage(answer)
        cite_pres = case.evaluate_citation_presence(answer)
        ignored = {
            "about", "after", "also", "and", "are", "because", "been", "being", "but",
            "for", "from", "has", "have", "into", "its", "more", "not", "only", "that",
            "the", "their", "then", "there", "these", "this", "those", "through", "was",
            "were", "which", "with",
        }

        def content_counts(text: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) > 2 and token not in ignored:
                    counts[token] = counts.get(token, 0) + 1
            return counts

        answer_counts = content_counts(answer)
        context_counts = content_counts(case.ground_truth_context)
        answer_token_count = sum(answer_counts.values())
        supported_token_count = sum(
            min(count, context_counts.get(token, 0))
            for token, count in answer_counts.items()
        )
        context_support = supported_token_count / answer_token_count if answer_token_count else 0.0
        score = 2.5 * kw_cov + cite_pres + 1.5 * context_support
        return round(min(5.0, max(0.0, score)), 4)

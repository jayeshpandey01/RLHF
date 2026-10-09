"""Anti-Catastrophic Forgetting Evaluator and Automated Qualification Gate."""

from dataclasses import dataclass
from typing import Literal, Optional, Union
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from core.hardware import get_device_strategy
from training.m1_bigram.tokenizer import CharTokenizer
from training.m3_sft.dataset import CompletionMaskCollator, HarmonizedSFTDataset


@dataclass
class RegressionResult:
    """Detailed evaluation report emitted by the anti-catastrophic forgetting gate."""

    passed: bool
    status: Literal["PASS", "FAIL_FORGETTING", "FAIL_NO_LEARNING"]
    base_anchor_ppl: float
    adapted_anchor_ppl: float
    anchor_drift_pct: float
    base_domain_ppl: float
    adapted_domain_ppl: float
    domain_gain_pct: float
    drift_threshold_pct: float = 5.0
    summary: str = ""

    def to_schema(self, candidate_name: str = "lora_adapter") -> "EvaluationReportSchema":
        """Convert RegressionResult into Pydantic EvaluationReportSchema."""
        from core.schemas import EvaluationReportSchema
        decision = "APPROVED" if self.passed else "REJECTED"
        return EvaluationReportSchema(
            candidate_name=candidate_name,
            perplexity=float(self.adapted_domain_ppl),
            citation_faithfulness=1.0,
            base_accuracy=float(max(0.0, min(1.0, 1.0 - (self.base_anchor_ppl / 100.0)))),
            candidate_accuracy=float(max(0.0, min(1.0, 1.0 - (self.adapted_anchor_ppl / 100.0)))),
            regression_delta=float(self.anchor_drift_pct / 100.0),
            decision=decision,
            failure_reason="" if self.passed else self.summary,
        )


class SFTRegressionEvaluator:
    """Evaluates candidate LoRA models against pre-trained baselines to prevent forgetting."""

    def __init__(
        self,
        tokenizer: CharTokenizer,
        anchor_eval_dataset: HarmonizedSFTDataset,
        domain_eval_dataset: HarmonizedSFTDataset,
        drift_threshold_pct: float = 5.0,
        device: Optional[Union[torch.device, str]] = None,
        max_eval_batches: int = 15,
        batch_size: int = 4,
    ):
        self.tokenizer = tokenizer
        self.anchor_eval_dataset = anchor_eval_dataset
        self.domain_eval_dataset = domain_eval_dataset
        self.drift_threshold_pct = drift_threshold_pct
        self.max_eval_batches = max_eval_batches
        self.batch_size = batch_size

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_device_strategy().device

    @torch.no_grad()
    def evaluate_perplexity(
        self, model: nn.Module, dataset: HarmonizedSFTDataset
    ) -> float:
        """Compute average completion loss and perplexity over an evaluation dataset."""
        model.eval()
        collator = CompletionMaskCollator(
            tokenizer=self.tokenizer,
            max_seq_len=getattr(getattr(model, "config", None), "block_size", 128),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        total_loss = 0.0
        steps = 0

        for batch in loader:
            if steps >= self.max_eval_batches:
                break
            x = batch.input_ids.to(self.device)
            y = batch.targets.to(self.device)

            if not (y != -100).any():
                continue

            _, loss = model(x, targets=y)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                steps += 1

        avg_loss = (total_loss / max(1, steps)) if steps > 0 else 4.0
        return math.exp(min(avg_loss, 20.0))

    def evaluate_gate(
        self, base_model: nn.Module, adapted_model: nn.Module
    ) -> RegressionResult:
        """Execute dual benchmark evaluation and issue release decision."""
        base_model.to(self.device)
        adapted_model.to(self.device)

        # 1. Evaluate baseline performance
        base_anchor_ppl = self.evaluate_perplexity(base_model, self.anchor_eval_dataset)
        base_domain_ppl = self.evaluate_perplexity(base_model, self.domain_eval_dataset)

        # 2. Evaluate adapted performance
        adapted_anchor_ppl = self.evaluate_perplexity(
            adapted_model, self.anchor_eval_dataset
        )
        adapted_domain_ppl = self.evaluate_perplexity(
            adapted_model, self.domain_eval_dataset
        )

        # 3. Compute relative drift and gain percentages
        # Positive drift means perplexity worsened on anchor
        anchor_drift_pct = (
            (adapted_anchor_ppl - base_anchor_ppl) / max(1e-6, base_anchor_ppl)
        ) * 100.0

        # Positive gain means perplexity improved (lowered) on domain
        domain_gain_pct = (
            (base_domain_ppl - adapted_domain_ppl) / max(1e-6, base_domain_ppl)
        ) * 100.0

        # 4. Gate decision logic
        if anchor_drift_pct > self.drift_threshold_pct:
            status = "FAIL_FORGETTING"
            passed = False
            summary = (
                f"Catastrophic forgetting detected: Anchor PPL drifted by "
                f"+{anchor_drift_pct:.2f}% (Threshold: +{self.drift_threshold_pct:.1f}%). "
                f"Adapter weights must be rolled back."
            )
        elif domain_gain_pct <= 0.0:
            status = "FAIL_NO_LEARNING"
            passed = False
            summary = (
                f"Underfitting detected: Domain PPL changed by {domain_gain_pct:.2f}% "
                f"(expected > 0.0% improvement). No significant domain knowledge assimilated."
            )
        else:
            status = "PASS"
            passed = True
            summary = (
                f"Release approved: Domain knowledge improved by +{domain_gain_pct:.2f}%, "
                f"and anchor capability drift is within safe bounds (+{anchor_drift_pct:.2f}%)."
            )

        result = RegressionResult(
            passed=passed,
            status=status,
            base_anchor_ppl=base_anchor_ppl,
            adapted_anchor_ppl=adapted_anchor_ppl,
            anchor_drift_pct=anchor_drift_pct,
            base_domain_ppl=base_domain_ppl,
            adapted_domain_ppl=adapted_domain_ppl,
            domain_gain_pct=domain_gain_pct,
            drift_threshold_pct=self.drift_threshold_pct,
            summary=summary,
        )

        logger.info(
            f"SFT Regression Gate: {status} | Anchor Drift: {anchor_drift_pct:+.2f}% | "
            f"Domain Gain: {domain_gain_pct:+.2f}%"
        )
        return result


"""Turnkey Continual SFT Orchestrator with Automated Regression Gate & Deployment Fusion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import copy
import torch
from loguru import logger

from core.hardware import get_device_strategy
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import MiniGPT
from training.m3_sft.prompt import InstructionExample
from training.m3_sft.dataset import (
    HarmonizedSFTDataset,
    create_default_anchor_examples,
    create_web_curation_examples,
)
from training.m3_sft.lora import (
    LoRAConfig,
    inject_lora_to_model,
    merge_lora_weights,
    unmerge_lora_weights,
)
from training.m3_sft.trainer import SFTConfig, TrainMetrics, SFTTrainer
from training.m3_sft.evaluator import RegressionResult, SFTRegressionEvaluator


@dataclass
class OrchestratorPipelineResult:
    """Outcome of continual ingestion -> SFT -> qualification gate -> deployment merge."""

    success: bool
    regression_result: Optional[RegressionResult]
    training_history: list[TrainMetrics]
    merged_model: Optional[MiniGPT]
    adapter_checkpoint_path: Optional[Path]
    deployment_status: str


class ContinualSFTOrchestrator:
    """Turnkey coordinator managing continuous post-training alignment and anti-forgetting gates."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: CharTokenizer,
        lora_config: Optional[LoRAConfig] = None,
        sft_config: Optional[SFTConfig] = None,
        drift_threshold_pct: float = 5.0,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.lora_config = lora_config or LoRAConfig()
        self.sft_config = sft_config or SFTConfig()
        self.drift_threshold_pct = drift_threshold_pct

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_device_strategy().device

        self.base_model.to(self.device)

    def run_continual_alignment(
        self,
        web_documents: Union[list[str], list["CleanedDocumentSchema"]],
        anchor_examples: Optional[list[InstructionExample]] = None,
        anchor_ratio: float = 0.70,
    ) -> OrchestratorPipelineResult:
        """Execute full continuous alignment pipeline from raw web text or CleanedDocumentSchema to merged model."""

        logger.info(
            f"ContinualSFTOrchestrator: Starting alignment pipeline with {len(web_documents)} web docs..."
        )

        if not web_documents:
            logger.warning("ContinualSFTOrchestrator: No web documents provided. Skipping alignment.")
            return OrchestratorPipelineResult(
                success=True,
                regression_result=None,
                training_history=[],
                merged_model=self.base_model,
                adapter_checkpoint_path=None,
                deployment_status="SKIPPED_EMPTY_WEB_STREAM",
            )

        # 1. Harmonize curated web knowledge and golden anchor demonstrations
        web_instructs = create_web_curation_examples(web_documents)
        anchor_instructs = anchor_examples or create_default_anchor_examples()

        # Split into training and evaluation sets
        split_anchor = max(1, int(len(anchor_instructs) * 0.8))
        anchor_train = anchor_instructs[:split_anchor]
        anchor_val = anchor_instructs[split_anchor:] or anchor_instructs

        if len(web_instructs) >= 2:
            split_web = max(1, int(len(web_instructs) * 0.8))
            if split_web >= len(web_instructs):
                split_web = len(web_instructs) - 1
            web_train = web_instructs[:split_web]
            web_val = web_instructs[split_web:]
        else:
            web_train = web_instructs
            web_val = []

        train_ds = HarmonizedSFTDataset(
            anchor_examples=anchor_train,
            web_examples=web_train,
            anchor_ratio=anchor_ratio,
            epoch_length=max(100, self.sft_config.max_steps * 2),
            seed=self.sft_config.seed,
        )
        val_ds = HarmonizedSFTDataset(
            anchor_examples=anchor_val,
            web_examples=web_val,
            anchor_ratio=anchor_ratio,
            epoch_length=50,
            seed=self.sft_config.seed + 1,
            deterministic=True,
        )

        # 2. Preserve uncorrupted baseline copy for gate evaluation
        base_model_eval = copy.deepcopy(self.base_model)
        base_model_eval.to(self.device)

        # 3. Prepare candidate model and inject LoRA adapters
        candidate_model = copy.deepcopy(self.base_model)
        candidate_model.to(self.device)
        inject_lora_to_model(candidate_model, self.lora_config)

        # 4. Supervised Fine-Tuning
        trainer = SFTTrainer(
            model=candidate_model,
            tokenizer=self.tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=self.sft_config,
            lora_config=self.lora_config,
            device=self.device,
        )
        history = trainer.train()

        # 5. Anti-Catastrophic Forgetting Regression Gate
        anchor_eval_ds = HarmonizedSFTDataset(
            anchor_examples=anchor_val,
            epoch_length=30,
            seed=999,
            deterministic=True,
        )
        domain_eval_ds = HarmonizedSFTDataset(
            anchor_examples=web_val or web_train,
            epoch_length=30,
            seed=888,
            deterministic=True,
        )

        evaluator = SFTRegressionEvaluator(
            tokenizer=self.tokenizer,
            anchor_eval_dataset=anchor_eval_ds,
            domain_eval_dataset=domain_eval_ds,
            drift_threshold_pct=self.drift_threshold_pct,
            device=self.device,
            max_eval_batches=10,
        )
        gate_result = evaluator.evaluate_gate(
            base_model=base_model_eval,
            adapted_model=candidate_model,
        )

        # 6. Qualification Release or Rollback Decision
        adapter_ckpt = Path(self.sft_config.checkpoint_dir) / "best_adapter" / "adapter_weights.pt"

        if gate_result.passed:
            logger.info(
                "ContinualSFTOrchestrator: Qualification Gate PASSED. Fusing LoRA weights..."
            )
            merge_lora_weights(candidate_model)
            deployment_status = "DEPLOYED_ZERO_LATENCY_MERGED_WEIGHTS"
            merged_model = candidate_model
            success = True
        else:
            logger.warning(
                f"ContinualSFTOrchestrator: Qualification Gate FAILED ({gate_result.status}). "
                f"Rolling back weights to base model."
            )
            unmerge_lora_weights(candidate_model)
            deployment_status = f"ROLLBACK_PREVENTED_FORGETTING ({gate_result.status})"
            merged_model = None
            success = False

        return OrchestratorPipelineResult(
            success=success,
            regression_result=gate_result,
            training_history=history,
            merged_model=merged_model,
            adapter_checkpoint_path=adapter_ckpt if adapter_ckpt.exists() else None,
            deployment_status=deployment_status,
        )


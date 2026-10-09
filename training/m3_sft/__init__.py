"""Milestone 3 (M3): Supervised Fine-Tuning (SFT) & LoRA Adaptation Subsystem."""

from training.m3_sft.prompt import (
    ChatMessage,
    InstructionExample,
    ChatMLFormatter,
)
from training.m3_sft.dataset import (
    BatchCollatorOutput,
    CompletionMaskCollator,
    HarmonizedSFTDataset,
    create_default_anchor_examples,
    create_web_curation_examples,
)
from training.m3_sft.lora import (
    LoRAConfig,
    LoRALinear,
    inject_lora_to_model,
    mark_only_lora_as_trainable,
    merge_lora_weights,
    unmerge_lora_weights,
    count_trainable_parameters,
    save_adapter_weights,
    load_adapter_weights,
)
from training.m3_sft.trainer import (
    SFTConfig,
    TrainMetrics,
    SFTTrainer,
)
from training.m3_sft.evaluator import (
    RegressionResult,
    SFTRegressionEvaluator,
)
from training.m3_sft.orchestrator import (
    OrchestratorPipelineResult,
    ContinualSFTOrchestrator,
)

__all__ = [
    # Phase 3.1: Prompts & Collation
    "ChatMessage",
    "InstructionExample",
    "ChatMLFormatter",
    "BatchCollatorOutput",
    "CompletionMaskCollator",
    "HarmonizedSFTDataset",
    "create_default_anchor_examples",
    "create_web_curation_examples",
    # Phase 3.2: LoRA Architecture
    "LoRAConfig",
    "LoRALinear",
    "inject_lora_to_model",
    "mark_only_lora_as_trainable",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "count_trainable_parameters",
    "save_adapter_weights",
    "load_adapter_weights",
    # Phase 3.3: SFT Training Engine
    "SFTConfig",
    "TrainMetrics",
    "SFTTrainer",
    # Phase 3.4: Anti-Forgetting Gate & Orchestrator
    "RegressionResult",
    "SFTRegressionEvaluator",
    "OrchestratorPipelineResult",
    "ContinualSFTOrchestrator",
]

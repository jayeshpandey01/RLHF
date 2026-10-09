"""Native PyTorch LoRA (Low-Rank Adaptation) module and model injection engine."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union
import json
import math
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class LoRAConfig:
    """Hyperparameters configuring low-rank adapter injection."""

    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["qkv_proj", "out_proj", "c_fc", "c_proj"]
    )
    bias: str = "none"  # "none", "all", or "lora_only"


class LoRALinear(nn.Module):
    """Native PyTorch LoRA linear layer wrapping a frozen base nn.Linear.

    Forward computes:
        h = base_linear(x) + (alpha / r) * (dropout(x) @ A.T @ B.T)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.is_merged = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        dtype = base_linear.weight.dtype
        device = base_linear.weight.device

        # Trainable low-rank adapter parameters
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype, device=device))

        # Gaussian initialization for A: N(0, 1/r)
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(r))
        # Strictly zero initialization for B: guarantees identity preservation at step 0
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Freeze base linear parameters
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. In merged state, executes standard dense GEMM with zero overhead."""
        if self.is_merged:
            return self.base_linear(x)

        base_out = self.base_linear(x)
        # Compute low-rank adaptation branch
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """Permanently fold (alpha / r) * B * A into base_linear.weight for deployment."""
        if self.is_merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_linear.weight.data.add_(delta_w)
        self.is_merged = True

    def unmerge(self) -> None:
        """Subtract (alpha / r) * B * A to restore original baseline weights."""
        if not self.is_merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_linear.weight.data.sub_(delta_w)
        self.is_merged = False


def inject_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Recursively traverse model and replace target nn.Linear layers with LoRALinear modules."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and name in config.target_modules:
            lora_layer = LoRALinear(
                base_linear=module,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
            setattr(model, name, lora_layer)
        else:
            inject_lora_to_model(module, config)

    mark_only_lora_as_trainable(model)
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Enforces requires_grad = True strictly on LoRA parameters and False everywhere else."""
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def merge_lora_weights(model: nn.Module) -> None:
    """Merges all LoRALinear layers across the model for zero-overhead inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    """Unmerges all LoRALinear layers across the model, restoring baseline weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def count_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
    """Calculate (trainable_params, total_params, trainable_percentage)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    return trainable_params, total_params, trainable_pct


def save_adapter_weights(
    model: nn.Module, config: LoRAConfig, save_dir: Union[str, Path]
) -> Path:
    """Serializes adapter parameters and config to compact disk storage (< 1 MB)."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_state = {
        name: param.data.cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    ckpt_path = out_dir / "adapter_weights.pt"
    torch.save(adapter_state, ckpt_path)

    config_dict = asdict(config)
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(
        f"Saved LoRA adapter checkpoint ({len(adapter_state)} tensors, "
        f"{ckpt_path.stat().st_size / 1024:.1f} KB) to {out_dir}"
    )
    return ckpt_path


def load_adapter_weights(model: nn.Module, adapter_path: Union[str, Path]) -> None:
    """Loads adapter parameters from disk into injected LoRA layers."""
    adapter_file = Path(adapter_path)
    if adapter_file.is_dir():
        adapter_file = adapter_file / "adapter_weights.pt"

    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter file not found at {adapter_file}")

    state_dict = torch.load(adapter_file, map_location="cpu", weights_only=True)
    model_params = dict(model.named_parameters())

    loaded_count = 0
    for name, tensor in state_dict.items():
        if name in model_params:
            model_params[name].data.copy_(tensor.to(model_params[name].device))
            loaded_count += 1

    logger.info(f"Loaded {loaded_count} LoRA adapter tensors from {adapter_file}")


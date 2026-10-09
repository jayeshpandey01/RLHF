"""Hardware detection and execution strategy abstraction layer."""

import os
from dataclasses import dataclass
import torch
from loguru import logger


@dataclass
class DeviceStrategy:
    """Strategy parameters customized for host hardware."""

    device_name: str
    torch_device: torch.device
    dtype: torch.dtype
    micro_batch_size: int
    gradient_accumulation_steps: int
    use_gradient_checkpointing: bool
    description: str

    @property
    def device(self) -> torch.device:
        """Alias for torch_device."""
        return self.torch_device

    @property
    def precision(self) -> str:
        """Return precision string representation (e.g. bfloat16, float32)."""
        return str(self.dtype).replace("torch.", "")



def get_device_strategy() -> DeviceStrategy:
    """Auto-detect host hardware and return optimal training/inference strategy.

    Supports:
    - Apple Silicon (Mac M-series Metal Performance Shaders with unified memory & bfloat16)
    - NVIDIA CUDA (Cloud GPU / RunPod / Colab)
    - Standard CPU (GitHub Codespaces, Google Cloud Shell)
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected NVIDIA CUDA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        is_bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if is_bf16_supported else torch.float16

        return DeviceStrategy(
            device_name="cuda",
            torch_device=torch.device("cuda"),
            dtype=dtype,
            micro_batch_size=4 if vram_gb >= 16 else 2,
            gradient_accumulation_steps=4,
            use_gradient_checkpointing=True,
            description=f"CUDA ({gpu_name}, {vram_gb:.1f}GB, dtype={dtype})",
        )

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon Mac (M-series unified memory)
        logger.info("Detected Apple Silicon (MPS). Enabling Metal Performance Shaders with bfloat16.")
        return DeviceStrategy(
            device_name="mps",
            torch_device=torch.device("mps"),
            dtype=torch.bfloat16,  # Modern PyTorch 2.x supports native MPS bfloat16
            micro_batch_size=2,
            gradient_accumulation_steps=4,
            use_gradient_checkpointing=True,
            description="Apple Silicon MPS (Metal Performance Shaders + Unified Memory + bfloat16)",
        )

    else:
        # Constrained CPU environment (Codespaces / Cloud Shell)
        num_cores = os.cpu_count() or 2
        active_threads = max(1, num_cores - 1)
        torch.set_num_threads(active_threads)
        logger.info(f"Detected CPU environment. Configured {active_threads} PyTorch threads.")

        return DeviceStrategy(
            device_name="cpu",
            torch_device=torch.device("cpu"),
            dtype=torch.float32,
            micro_batch_size=1,
            gradient_accumulation_steps=8,  # Simulate batch size 8 on CPU
            use_gradient_checkpointing=False,
            description=f"CPU-only ({active_threads} active threads, Codespaces/Cloud Shell optimized)",
        )

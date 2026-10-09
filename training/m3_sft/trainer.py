"""Supervised Fine-Tuning (SFT) Training Loop and Optimizer Engine for MiniGPT + LoRA."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from core.hardware import get_device_strategy
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import MiniGPT
from training.m3_sft.prompt import ChatMessage, ChatMLFormatter
from training.m3_sft.dataset import (
    CompletionMaskCollator,
    HarmonizedSFTDataset,
    BatchCollatorOutput,
)
from training.m3_sft.lora import (
    LoRAConfig,
    save_adapter_weights,
    count_trainable_parameters,
)


@dataclass
class SFTConfig:
    """Hyperparameters configuring SFT training loop and optimization."""

    max_steps: int = 200
    batch_size: int = 4
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    warmup_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_steps: int = 10
    checkpoint_dir: str = "models/sft_adapters"
    seed: int = 42
    ignore_index: int = -100


@dataclass
class TrainMetrics:
    """Metrics recorded during SFT training and evaluation intervals."""

    step: int
    train_loss: float
    val_loss: float
    val_perplexity: float
    learning_rate: float
    elapsed_seconds: float
    best_val_loss: float


class SFTTrainer:
    """Industrial-strength Supervised Fine-Tuning (SFT) trainer with completion-only masking."""

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: CharTokenizer,
        train_dataset: HarmonizedSFTDataset,
        val_dataset: HarmonizedSFTDataset,
        config: SFTConfig,
        lora_config: Optional[LoRAConfig] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.lora_config = lora_config or LoRAConfig()

        # Hardware and execution strategy detection
        if device is not None:
            self.device = torch.device(device)
            self.dtype = torch.bfloat16 if self.device.type in ["mps", "cuda"] else torch.float32
        else:
            strategy = get_device_strategy()
            self.device = strategy.device
            self.dtype = strategy.dtype

        self.model.to(self.device)

        # Build collator aligned with model block size
        self.collator = CompletionMaskCollator(
            tokenizer=self.tokenizer,
            max_seq_len=self.model.config.block_size,
            ignore_index=self.config.ignore_index,
        )

        # Filter optimizer parameters: STRICTLY trainable LoRA weights
        self.lora_params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.lora_params:
            raise ValueError(
                "No trainable parameters found in model. Ensure inject_lora_to_model() was called."
            )

        self.optimizer = torch.optim.AdamW(
            self.lora_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        trainable_p, total_p, pct = count_trainable_parameters(self.model)
        logger.info(
            f"SFTTrainer initialized on {self.device} ({self.dtype}) | "
            f"Trainable params: {trainable_p:,} / {total_p:,} ({pct:.2f}%)"
        )

    def get_lr(self, step: int) -> float:
        """Calculate learning rate with linear warmup and cosine annealing decay."""
        # 1. Linear warmup phase
        if step < self.config.warmup_steps:
            return self.config.learning_rate * float(step) / max(1, self.config.warmup_steps)

        # 2. Post-training floor
        if step >= self.config.max_steps:
            return self.config.min_learning_rate

        # 3. Cosine decay phase
        decay_ratio = float(step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_learning_rate + coeff * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluate completion loss and perplexity on the validation dataset."""
        self.model.eval()
        total_loss = 0.0
        steps = 0

        for batch in self.val_loader:
            if steps >= self.config.eval_steps:
                break
            x = batch.input_ids.to(self.device)
            y = batch.targets.to(self.device)

            # Avoid evaluate crash if batch has zero active completion tokens
            if not (y != self.config.ignore_index).any():
                continue

            _, loss = self.model(x, targets=y)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                steps += 1

        avg_loss = (total_loss / max(1, steps)) if steps > 0 else 4.0
        perplexity = math.exp(min(avg_loss, 20.0))
        return avg_loss, perplexity

    def generate_sample(
        self, prompt_text: str, max_new_tokens: int = 60, temperature: float = 0.8
    ) -> str:
        """Autoregressively sample tokens from prompt using model generator."""
        self.model.eval()
        # Ensure prompt has assistant delimiter if raw user text
        if "<|im_start|>" not in prompt_text:
            formatted_prompt = ChatMLFormatter.format_prompt_only([
                ChatMessage(role="user", content=prompt_text)
            ])
        else:
            formatted_prompt = prompt_text

        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        max_prompt = max(1, self.model.config.block_size - max_new_tokens - 1)
        prompt_tokens = prompt_tokens[-max_prompt:]
        if not prompt_tokens:
            prompt_tokens = [0]

        context = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=10,
        )
        return self.tokenizer.decode(generated[0].tolist())

    def train(self) -> list[TrainMetrics]:
        """Execute full SFT training loop with evaluation intervals and adapter checkpointing."""
        logger.info(f"Starting SFT optimization for {self.config.max_steps} steps...")
        self.model.train()
        start_time = time.time()
        best_val_loss = float("inf")
        history: list[TrainMetrics] = []

        train_iter = iter(self.train_loader)
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for step in range(self.config.max_steps):
            # Dynamic learning rate update
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Next mini-batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            x = batch.input_ids.to(self.device)
            y = batch.targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision execution on supported hardware
            if self.device.type in ["mps", "cuda"] and self.dtype == torch.bfloat16:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    _, loss = self.model(x, targets=y)
            else:
                _, loss = self.model(x, targets=y)

            # Backward pass strictly on LoRA parameters
            loss.backward()

            # Pascanu gradient clipping
            torch.nn.utils.clip_grad_norm_(self.lora_params, self.config.grad_clip)

            self.optimizer.step()

            # Periodic evaluation and adapter checkpointing
            if (
                step % self.config.eval_interval == 0
                or step == self.config.max_steps - 1
                or step == 0
            ):
                val_loss, val_ppl = self.evaluate()
                elapsed = time.time() - start_time

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = save_adapter_weights(
                        self.model, self.lora_config, ckpt_dir / "best_adapter"
                    )
                    logger.info(
                        f"Step {step:03d} | New best val_loss={best_val_loss:.4f} -> "
                        f"Saved adapter checkpoint to {best_ckpt_path}"
                    )

                metrics = TrainMetrics(
                    step=step,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                    learning_rate=lr,
                    elapsed_seconds=elapsed,
                    best_val_loss=best_val_loss,
                )
                history.append(metrics)

                logger.info(
                    f"SFT Step {step:03d}/{self.config.max_steps} | "
                    f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val PPL: {val_ppl:.2f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s"
                )
                self.model.train()

        # Final adapter checkpoint save
        save_adapter_weights(self.model, self.lora_config, ckpt_dir / "final_adapter")
        logger.info(f"SFT training complete in {time.time() - start_time:.2f} seconds.")
        return history


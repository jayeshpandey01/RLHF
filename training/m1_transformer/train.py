"""Training pipeline for Mini-GPT Transformer on TinyShakespeare."""

import math
from pathlib import Path
import sys
import time
from typing import Optional, Union
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hardware import get_device_strategy
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.config import GPTConfig
from training.m1_transformer.dataset import (
    MemoryMappedTokenDataset,
    get_batch,
    get_shakespeare_data,
)
from training.m1_transformer.model import MiniGPT


def get_lr(
    step: int,
    warmup_steps: int = 50,
    max_steps: int = 500,
    max_lr: float = 5e-4,
    min_lr: float = 5e-5,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: Union[torch.Tensor, MemoryMappedTokenDataset],
    val_data: Union[torch.Tensor, MemoryMappedTokenDataset],
    eval_iters: int = 25,
    batch_size: int = 32,
    block_size: int = 128,
) -> dict[str, float]:
    """Estimate cross-entropy loss over multiple train and val batches."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if isinstance(data, MemoryMappedTokenDataset):
                x, y = data.get_batch(batch_size, block_size, device=model.device)
            else:
                x, y = get_batch(data, batch_size, block_size, device=model.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    max_steps: int = 500,
    batch_size: int = 32,
    block_size: int = 128,
    eval_interval: int = 50,
    learning_rate: float = 5e-4,
    save_checkpoint: bool = True,
    custom_data: Optional[
        tuple[
            Union[torch.Tensor, MemoryMappedTokenDataset],
            Union[torch.Tensor, MemoryMappedTokenDataset],
            CharTokenizer,
        ]
    ] = None,
) -> MiniGPT:
    """Train MiniGPT on Shakespeare text or custom/memory-mapped replay corpora."""
    strategy = get_device_strategy()
    device = strategy.device

    if custom_data is not None:
        train_data, val_data, tokenizer = custom_data
    else:
        train_data, val_data, tokenizer = get_shakespeare_data()

    config = GPTConfig(
        block_size=block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        d_model=128,
        dropout=0.1,
    )
    model = MiniGPT(config).to(device)

    # Cast model to bfloat16 if hardware strategy specifies it
    if strategy.precision == "bfloat16" and device.type in ("mps", "cuda"):
        model = model.to(dtype=torch.bfloat16)

    num_params = model.get_num_params()
    logger.info(f"Model initialized: {num_params:,} parameters | Device: {device} | Precision: {strategy.precision}")

    theoretical_init_loss = math.log(tokenizer.vocab_size)
    logger.info(f"Theoretical Uniform Entropy: {theoretical_init_loss:.4f} nats (vocab_size={tokenizer.vocab_size})")

    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=learning_rate)

    t0 = time.time()
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(max_steps):
        # Update learning rate according to cosine schedule
        lr = get_lr(step, warmup_steps=50, max_steps=max_steps, max_lr=learning_rate, min_lr=learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters=20, batch_size=batch_size, block_size=block_size)
            elapsed = time.time() - t0
            logger.info(
                f"Step {step:03d}/{max_steps} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Sample batch & forward pass
        if isinstance(train_data, MemoryMappedTokenDataset):
            x, y = train_data.get_batch(batch_size, block_size, device=device)
        else:
            x, y = get_batch(train_data, batch_size, block_size, device=device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)

        # Backward pass & gradient norm clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total_time = time.time() - t0
    logger.info(f"Training completed in {total_time:.2f} seconds ({max_steps / total_time:.1f} steps/s)")

    if save_checkpoint:
        ckpt_path = checkpoint_dir / "minigpt_shakespeare.pt"
        torch.save(
            {
                "config": config,
                "model_state_dict": model.state_dict(),
                "vocab": tokenizer.vocab,
                "final_val_loss": losses["val"],
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved to {ckpt_path}")

    # Generate sample
    model.eval()
    prompt = "ROMEO:\n" if all(c in tokenizer.vocab for c in "ROMEO:\n") else tokenizer.vocab[0]
    prompt_tokens = tokenizer.encode(prompt)[: max(1, config.block_size // 2)]
    if not prompt_tokens:
        prompt_tokens = [0]
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    max_gen = max(1, min(150, config.block_size - context.size(1)))
    generated = model.generate(context, max_new_tokens=max_gen, temperature=0.8, top_k=10)
    sample_text = tokenizer.decode(generated[0].tolist())

    print("\n" + "=" * 60)
    print("🎭 GENERATED AUTOREGRESSIVE SHAKESPEARE SAMPLE:")
    print("=" * 60)
    print(sample_text)
    print("=" * 60 + "\n")

    return model


if __name__ == "__main__":
    train(max_steps=500, batch_size=32, block_size=128, eval_interval=50)


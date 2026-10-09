"""Training demonstration for Bigram Language Model: Counting Baseline vs. Neural Model."""

from pathlib import Path
import sys
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hardware import get_device_strategy
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_bigram.model import BigramLanguageModel

# Compact Shakespearean excerpt corpus for training
CORPUS = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!
"""


def compute_statistical_bigram_entropy(data_indices: list[int], vocab_size: int) -> float:
    """Compute empirical cross-entropy of statistical counting-based bigram MLE."""
    counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    for ch1, ch2 in zip(data_indices[:-1], data_indices[1:]):
        counts[ch1, ch2] += 1

    # Add-1 Laplace smoothing to prevent log(0)
    probs = (counts + 1e-8) / (counts + 1e-8).sum(dim=1, keepdim=True)

    nll = 0.0
    for ch1, ch2 in zip(data_indices[:-1], data_indices[1:]):
        nll += -torch.log(probs[ch1, ch2]).item()

    return nll / (len(data_indices) - 1)


def train_bigram_model(steps: int = 500, lr: float = 0.1) -> tuple[BigramLanguageModel, CharTokenizer]:
    """Train neural bigram model on text corpus and compare against statistical MLE."""
    strategy = get_device_strategy()
    device = strategy.torch_device

    # 1. Tokenize corpus
    tokenizer = CharTokenizer(CORPUS)
    data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
    vocab_size = tokenizer.vocab_size

    # 2. Compute theoretical statistical limit
    stat_entropy = compute_statistical_bigram_entropy(data.tolist(), vocab_size)
    print(f"\n[Theoretical Limit] Statistical Bigram Entropy: {stat_entropy:.4f} nats (PPL: {torch.exp(torch.tensor(stat_entropy)):.2f})")

    # 3. Prepare inputs and targets (B=1, T=N-1)
    x = data[:-1].unsqueeze(0).to(device)
    y = data[1:].unsqueeze(0).to(device)

    # 4. Neural Bigram model
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initial loss before training (should be ~ -ln(1/vocab_size))
    initial_logits, initial_loss = model(x, y)
    print(f"[Untrained Model] Initial Loss: {initial_loss.item():.4f} (Uniform expectation: {-torch.log(torch.tensor(1.0/vocab_size)):.4f})")

    # 5. Training loop
    for step in range(steps):
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step:03d} | Neural Loss: {loss.item():.4f} | Delta to Theory: {abs(loss.item() - stat_entropy):.4f}")

    # 6. Autoregressive text generation
    print("\n--- Autoregressive Samples (Temperature = 0.8) ---")
    context = torch.tensor([[tokenizer.encode("F")[0]]], dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=100, temperature=0.8)[0].tolist()
    print(tokenizer.decode(generated_indices))
    print("---------------------------------------------------\n")

    return model, tokenizer


if __name__ == "__main__":
    train_bigram_model()


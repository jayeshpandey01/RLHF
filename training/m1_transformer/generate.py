"""Autoregressive text generation utility using trained MiniGPT checkpoint."""

import argparse
from pathlib import Path
import sys
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hardware import get_device_strategy
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import MiniGPT


def generate_text(
    checkpoint_path: Path | str = "checkpoints/minigpt_shakespeare.pt",
    prompt: str = "JULIET:\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 10,
) -> str:
    """Load model checkpoint and generate text given prompt."""
    strategy = get_device_strategy()
    device = strategy.device

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}. Run train.py first.")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    tokenizer = CharTokenizer("".join(vocab))
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    context_tokens = tokenizer.encode(prompt)
    x = torch.tensor([context_tokens], dtype=torch.long, device=device)

    generated_tokens = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(generated_tokens[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using trained MiniGPT model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/minigpt_shakespeare.pt")
    parser.add_argument("--prompt", type=str, default="HAMLET:\nTo be, or not to be, ")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    output = generate_text(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n" + "=" * 50)
    print(output)
    print("=" * 50 + "\n")

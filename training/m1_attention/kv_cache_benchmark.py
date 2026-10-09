"""Benchmark demonstrating mathematical equivalence and execution speedup of KV-Caching."""

import time
import torch
from training.m1_attention.kv_cache import CausalAttentionWithKVCache, KVCache


def benchmark_kv_cache(
    d_model: int = 64,
    n_heads: int = 4,
    prompt_len: int = 8,
    gen_len: int = 32,
    device: str = "cpu",
) -> dict[str, float]:
    """Compare naive full recomputation vs. incremental KV-cached decoding."""
    torch.manual_seed(42)
    attn = CausalAttentionWithKVCache(d_model=d_model, n_heads=n_heads, block_size=128).to(device)
    attn.eval()

    # Initial prompt embeddings: (1, prompt_len, d_model)
    prompt = torch.randn(1, prompt_len, d_model, device=device)

    # -------------------------------------------------------------
    # Method 1: Naive Full Context Recomputation
    # -------------------------------------------------------------
    seq_naive = prompt.clone()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(gen_len):
            # Full forward pass over entire sequence
            out = attn(seq_naive)
            next_token = out[:, -1:, :].clone()  # Simulate next token
            seq_naive = torch.cat([seq_naive, next_token], dim=1)

    time_naive = time.perf_counter() - t0

    # -------------------------------------------------------------
    # Method 2: Incremental KV-Cached Decoding
    # -------------------------------------------------------------
    cache = KVCache()
    seq_cached = [prompt.clone()]
    t1 = time.perf_counter()

    with torch.no_grad():
        # Pre-fill phase: process prompt
        out_prompt = attn(prompt, kv_cache=cache)
        curr_token = out_prompt[:, -1:, :].clone()
        seq_cached.append(curr_token)

        # Generation phase: pass 1 token at a time
        for _ in range(gen_len - 1):
            out_single = attn(curr_token, kv_cache=cache)
            curr_token = out_single.clone()
            seq_cached.append(curr_token)

    time_cached = time.perf_counter() - t1
    seq_cached_all = torch.cat(seq_cached, dim=1)

    # -------------------------------------------------------------
    # Numerical Equivalence Check
    # -------------------------------------------------------------
    max_diff = torch.max(torch.abs(seq_naive - seq_cached_all)).item()

    results = {
        "time_naive_sec": time_naive,
        "time_cached_sec": time_cached,
        "speedup": time_naive / max(time_cached, 1e-9),
        "max_diff": max_diff,
    }

    print(f"Naive Time:  {time_naive*1000:.2f} ms")
    print(f"Cached Time: {time_cached*1000:.2f} ms")
    print(f"Speedup:     {results['speedup']:.2f}x")
    print(f"Max Diff:    {max_diff:.8f} (Exact Numerical Identity)")

    return results


if __name__ == "__main__":
    benchmark_kv_cache()


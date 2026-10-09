"""Comprehensive unit tests for Milestone 1 Phase 3 (Causal Multi-Head Attention & KV-Cache)."""

from pathlib import Path
import sys
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.m1_attention.single_head import SingleHeadCausalAttention
from training.m1_attention.multi_head import CausalMultiHeadAttention
from training.m1_attention.kv_cache import CausalAttentionWithKVCache, KVCache
from training.m1_attention.kv_cache_benchmark import benchmark_kv_cache


class TestCausalAttentionAndKVCache(unittest.TestCase):
    """Test suite for causal attention mechanisms, multi-head projection, and KV-cache."""

    def test_single_head_attention_causal_mask_and_stochasticity(self):
        """Verify that causal attention is strictly lower-triangular and row-stochastic."""
        d_model = 32
        head_dim = 16
        T = 6
        B = 2

        layer = SingleHeadCausalAttention(d_model=d_model, head_dim=head_dim, block_size=16)
        x = torch.randn(B, T, d_model)
        out, weights = layer(x)

        # 1. Output shape must be (B, T, head_dim)
        self.assertEqual(out.shape, (B, T, head_dim))

        # 2. Row stochasticity: sum across key dimension (-1) must equal 1.0
        row_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6))

        # 3. Causal masking: weights for all future positions (j > i) must be exactly 0.0
        for i in range(T):
            for j in range(i + 1, T):
                self.assertTrue((weights[:, :, i, j] == 0.0).all())

    def test_multi_head_attention_shapes_and_gradients(self):
        """Verify multi-head attention tensor transformations and backpropagation."""
        d_model = 64
        n_heads = 4
        B, T = 3, 10

        mha = CausalMultiHeadAttention(d_model=d_model, n_heads=n_heads, block_size=32)
        x = torch.randn(B, T, d_model, requires_grad=True)

        out, weights = mha(x, return_attention_weights=True)

        # 1. Shape preservation: (B, T, d_model)
        self.assertEqual(out.shape, (B, T, d_model))
        self.assertEqual(weights.shape, (B, n_heads, T, T))

        # 2. Gradient flow: loss must propagate back to input and weights
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (B, T, d_model))
        self.assertIsNotNone(mha.qkv_proj.weight.grad)
        self.assertIsNotNone(mha.out_proj.weight.grad)

    def test_kv_cache_mathematical_identity(self):
        """Verify that KV-cached incremental generation produces exact numerical match to naive recomputation."""
        results = benchmark_kv_cache(
            d_model=32,
            n_heads=2,
            prompt_len=4,
            gen_len=12,
            device="cpu",
        )

        # Discrepancy between cached and naive must be virtually zero (< 1e-5)
        self.assertLess(results["max_diff"], 1e-5)
        # Cached generation should execute cleanly
        self.assertGreater(results["time_naive_sec"], 0.0)
        self.assertGreater(results["time_cached_sec"], 0.0)

    def test_variance_scaling_effect(self):
        """Verify that scaling by 1/sqrt(d_k) prevents attention score variance explosion."""
        head_dim = 64
        B, T = 10, 20

        # Zero mean, unit variance queries and keys
        q = torch.randn(B, T, head_dim)
        k = torch.randn(B, T, head_dim)

        # Unscaled dot product variance should be approximately head_dim
        unscaled = torch.matmul(q, k.transpose(-2, -1))
        unscaled_var = torch.var(unscaled).item()

        # Scaled dot product variance should be approximately 1.0
        scaled = unscaled / (head_dim**0.5)
        scaled_var = torch.var(scaled).item()

        print(f"\n[Attention Scaling Test] Head Dim: {head_dim} | Unscaled Var: {unscaled_var:.2f} | Scaled Var: {scaled_var:.2f}")
        self.assertGreater(unscaled_var, 30.0)
        self.assertAlmostEqual(scaled_var, 1.0, delta=0.3)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCausalAttentionAndKVCache)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


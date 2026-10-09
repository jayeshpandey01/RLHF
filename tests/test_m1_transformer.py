"""Unit and integration test suite for Milestone 1 Phase 4 (Mini-GPT Transformer)."""

import math
from pathlib import Path
import sys
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hardware import get_device_strategy
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.config import GPTConfig
from training.m1_transformer.model import MiniGPT
from training.m1_transformer.dataset import (
    MemoryMappedTokenDataset,
    export_tokens_to_binary,
    get_batch,
)


class TestMiniGPTTransformer(unittest.TestCase):
    """Test suite for MiniGPT architecture, initialization, gradients, and convergence."""

    def setUp(self):
        self.device = get_device_strategy().device
        self.vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,:;!?\n-'")
        self.tokenizer = CharTokenizer("".join(self.vocab))
        self.config = GPTConfig(
            block_size=32,
            vocab_size=self.tokenizer.vocab_size,
            n_layer=2,
            n_head=2,
            d_model=32,
            dropout=0.0,
        )
        self.model = MiniGPT(self.config).to(self.device)

    def test_model_forward_shapes_and_loss(self):
        """Verify logits shape and cross-entropy loss computation."""
        B, T = 4, 16
        idx = torch.randint(0, self.config.vocab_size, (B, T), device=self.device)
        targets = torch.randint(0, self.config.vocab_size, (B, T), device=self.device)

        # Forward with targets (training mode)
        logits, loss = self.model(idx, targets)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)

        # Forward without targets (inference mode)
        logits_eval, loss_eval = self.model(idx)
        self.assertEqual(logits_eval.shape, (B, 1, self.config.vocab_size))
        self.assertIsNone(loss_eval)

    def test_initial_loss_near_uniform_entropy(self):
        """Verify that initialization yields loss close to theoretical ln(V)."""
        V = self.config.vocab_size
        expected_entropy = math.log(V)

        B, T = 16, 32
        idx = torch.randint(0, V, (B, T), device=self.device)
        targets = torch.randint(0, V, (B, T), device=self.device)

        with torch.no_grad():
            _, loss = self.model(idx, targets)

        loss_val = loss.item()
        delta = abs(loss_val - expected_entropy)
        print(f"\n[Uniform Entropy Test] Vocab: {V} | Expected: {expected_entropy:.4f} | Actual: {loss_val:.4f} | Delta: {delta:.4f}")
        # Allow small deviation from random sampling fluctuation
        self.assertLess(delta, 0.45, "Initial loss deviates too far from theoretical ln(V)")

    def test_weight_tying_and_parameters(self):
        """Verify that lm_head and wte are strictly weight-tied."""
        self.assertTrue(
            self.model.transformer.wte.weight is self.model.lm_head.weight,
            "lm_head.weight must point to the exact same tensor as wte.weight (Weight Tying)",
        )

        n_params = self.model.get_num_params()
        self.assertGreater(n_params, 10_000)

    def test_gradient_flow_and_no_nan(self):
        """Verify backward pass populates non-zero finite gradients across all parameters."""
        B, T = 2, 8
        idx = torch.randint(0, self.config.vocab_size, (B, T), device=self.device)
        targets = torch.randint(0, self.config.vocab_size, (B, T), device=self.device)

        self.model.zero_grad()
        _, loss = self.model(idx, targets)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} was not populated")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf")

    def test_fast_overfit_convergence(self):
        """Verify that MiniGPT can rapidly overfit a small batch (loss 4.17 -> <0.5)."""
        B, T = 2, 8
        idx = torch.randint(0, self.config.vocab_size, (B, T), device=self.device)
        targets = torch.randint(0, self.config.vocab_size, (B, T), device=self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3, weight_decay=0.0)

        initial_loss = None
        for step in range(100):
            optimizer.zero_grad()
            _, loss = self.model(idx, targets)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        print(f"\n[Overfit Test] Step 0: {initial_loss:.4f} -> Step 100: {final_loss:.4f}")
        self.assertLess(final_loss, 0.50, "Model failed to overfit small batch within 100 steps")

    def test_autoregressive_generation_modes(self):
        """Verify greedy and temperature autoregressive generation."""
        prompt = "test"
        tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

        # Deterministic greedy generation (T=0)
        out1 = self.model.generate(tokens, max_new_tokens=10, temperature=0.0)
        out2 = self.model.generate(tokens, max_new_tokens=10, temperature=0.0)
        self.assertTrue(torch.equal(out1, out2), "Greedy generation (T=0) must be deterministic")

        # Temperature generation (T=0.8)
        out_temp = self.model.generate(tokens, max_new_tokens=15, temperature=0.8, top_k=5)
        text = self.tokenizer.decode(out_temp[0].tolist())
        self.assertTrue(text.startswith(prompt))
        self.assertEqual(len(text), len(prompt) + 15)

    def test_kv_cache_generation_parity(self):
        """Verify that KV-cached MiniGPT generation yields identical tokens to naive recomputation."""
        prompt = "ROMEO:\n"
        tokens = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

        out_naive = self.model.generate(tokens, max_new_tokens=12, temperature=0.0, use_cache=False)
        out_cached = self.model.generate(tokens, max_new_tokens=12, temperature=0.0, use_cache=True)

        self.assertTrue(
            torch.equal(out_naive, out_cached),
            "KV-cached generation tokens must match naive recomputation tokens identically",
        )

    def test_memory_mapped_dataset_large_data_readiness(self):
        """Verify binary serialization and MemoryMappedTokenDataset O(1) RAM streaming for M2."""
        sample_tokens = list(range(1000))
        temp_bin = Path("data/test_scratch.bin")

        try:
            export_tokens_to_binary(sample_tokens, temp_bin)
            dataset = MemoryMappedTokenDataset(temp_bin)
            self.assertEqual(len(dataset), 1000)

            x, y = get_batch(dataset, batch_size=4, block_size=16, device=self.device)
            self.assertEqual(x.shape, (4, 16))
            self.assertEqual(y.shape, (4, 16))

            # Verify shifted alignment: y_t must equal x_{t+1}
            for b in range(4):
                self.assertTrue(torch.equal(x[b, 1:], y[b, :-1]))
        finally:
            if temp_bin.exists():
                temp_bin.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)

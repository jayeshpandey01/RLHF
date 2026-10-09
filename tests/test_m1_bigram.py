"""Comprehensive unit tests for Milestone 1 Phase 2 (Bigram Language Model & Cross-Entropy)."""

from pathlib import Path
import sys
import unittest
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_bigram.model import BigramLanguageModel, manual_cross_entropy
from training.m1_bigram.train_bigram import CORPUS, train_bigram_model


class TestBigramLanguageModel(unittest.TestCase):
    """Test suite for character tokenization, cross-entropy, and bigram modeling."""

    def test_tokenizer_roundtrip(self):
        """Verify that encoding and decoding preserves exact string content."""
        sample_text = "The quick brown fox jumps over the lazy dog! 12345"
        tokenizer = CharTokenizer(sample_text)
        self.assertGreater(tokenizer.vocab_size, 20)

        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, sample_text)

    def test_cross_entropy_logsumexp_parity(self):
        """Verify that manual log-sum-exp cross-entropy matches PyTorch F.cross_entropy to 6 decimal places."""
        torch.manual_seed(42)
        batch_size = 4
        vocab_size = 32

        logits = torch.randn(batch_size, vocab_size, dtype=torch.float64)
        targets = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long)

        pytorch_loss = F.cross_entropy(logits, targets)
        manual_loss = manual_cross_entropy(logits, targets)

        self.assertAlmostEqual(manual_loss.item(), pytorch_loss.item(), places=6)

    def test_model_forward_shapes(self):
        """Verify tensor output shapes from BigramLanguageModel."""
        vocab_size = 25
        model = BigramLanguageModel(vocab_size)

        B, T = 2, 8
        idx = torch.randint(0, vocab_size, (B, T))
        targets = torch.randint(0, vocab_size, (B, T))

        logits, loss = model(idx, targets)
        self.assertEqual(logits.shape, (B, T, vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)  # Scalar

    def test_temperature_sampling_modes(self):
        """Verify temperature effects: greedy (T=0) yields deterministic argmax."""
        vocab_size = 10
        model = BigramLanguageModel(vocab_size)
        start_token = torch.tensor([[0]])

        # 1. Greedy decoding (T=0) must be deterministic
        out1 = model.generate(start_token, max_new_tokens=10, temperature=0.0)
        out2 = model.generate(start_token, max_new_tokens=10, temperature=0.0)
        self.assertTrue(torch.equal(out1, out2))

        # 2. Stochastic sampling (T=1.0) produces valid tokens
        out_sampled = model.generate(start_token, max_new_tokens=10, temperature=1.0)
        self.assertEqual(out_sampled.shape, (1, 11))
        self.assertTrue((out_sampled >= 0).all() and (out_sampled < vocab_size).all())

    def test_training_convergence_towards_statistical_entropy(self):
        """Verify that training the neural bigram model reduces cross-entropy loss towards statistical limit."""
        model, tokenizer = train_bigram_model(steps=300, lr=0.15)
        device = model.device
        data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long, device=device).unsqueeze(0)
        x = data[:, :-1]
        y = data[:, 1:]

        logits, final_loss = model(x, y)
        initial_expected_loss = -torch.log(torch.tensor(1.0 / tokenizer.vocab_size)).item()

        # Final loss must have decreased significantly from uniform expectation
        self.assertLess(final_loss.item(), initial_expected_loss - 0.7)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBigramLanguageModel)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


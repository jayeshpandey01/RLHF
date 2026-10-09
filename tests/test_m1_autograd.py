"""Comprehensive unit tests for Milestone 1 Phase 1 (Autograd Engine & Numerical Verification)."""

import math
from pathlib import Path
import sys
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.m1_autograd.engine import Value
from training.m1_autograd.grad_check import verify_gradients
from training.m1_autograd.nn import MLP
from training.m1_autograd.train_mlp import train_mlp


class TestAutogradEngine(unittest.TestCase):
    """Test suite for the scalar automatic differentiation engine."""

    def test_primitive_ops(self):
        """Verify forward and analytical backward passes of primitive operations."""
        # 1. Addition
        a = Value(3.0)
        b = Value(4.0)
        c = a + b
        c.backward()
        self.assertEqual(c.data, 7.0)
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

        # 2. Multiplication & Power
        x = Value(3.0)
        y = x**2
        y.backward()
        self.assertEqual(y.data, 9.0)
        self.assertEqual(x.grad, 6.0)  # d(x^2)/dx = 2x = 6

        # 3. Division
        p = Value(6.0)
        q = Value(2.0)
        r = p / q
        r.backward()
        self.assertEqual(r.data, 3.0)
        self.assertAlmostEqual(p.grad, 0.5)  # 1/2
        self.assertAlmostEqual(q.grad, -1.5)  # -6 / 2^2 = -1.5

        # 4. Exponential and Log
        m = Value(2.0)
        n = m.exp()
        n.backward()
        self.assertAlmostEqual(n.data, math.exp(2.0))
        self.assertAlmostEqual(m.grad, math.exp(2.0))

        u = Value(5.0)
        v = u.log()
        v.backward()
        self.assertAlmostEqual(v.data, math.log(5.0))
        self.assertAlmostEqual(u.grad, 0.2)

    def test_gradient_accumulation(self):
        """Verify multivariable chain rule when variables are reused."""
        # Case 1: x + x -> grad must be 2.0
        x = Value(3.0)
        y = x + x
        y.backward()
        self.assertEqual(y.data, 6.0)
        self.assertEqual(x.grad, 2.0)

        # Case 2: x * x -> grad must be 2x = 6.0
        a = Value(3.0)
        b = a * a
        b.backward()
        self.assertEqual(b.data, 9.0)
        self.assertEqual(a.grad, 6.0)

        # Case 3: x * x + x -> grad must be 2x + 1 = 7.0
        m = Value(3.0)
        n = m * m + m
        n.backward()
        self.assertEqual(n.data, 12.0)
        self.assertEqual(m.grad, 7.0)

    def test_pytorch_parity(self):
        """Verify mathematical parity against PyTorch C++ autograd engine."""
        # PyTorch forward & backward
        torch_a = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
        torch_b = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
        torch_c = torch.tensor(4.0, requires_grad=True, dtype=torch.float64)
        torch_d = torch.tensor(1.5, requires_grad=True, dtype=torch.float64)

        torch_e = (torch_a * torch_b + torch_c) / torch_d
        torch_f = torch.tanh(torch_e) + torch.relu(torch_a * torch_c)
        torch_f.backward()

        # Custom Value forward & backward
        val_a = Value(2.0)
        val_b = Value(-3.0)
        val_c = Value(4.0)
        val_d = Value(1.5)

        val_e = (val_a * val_b + val_c) / val_d
        val_f = val_e.tanh() + (val_a * val_c).relu()
        val_f.backward()

        # Check forward output
        self.assertAlmostEqual(val_f.data, torch_f.item(), places=7)

        # Check all gradients against PyTorch
        self.assertAlmostEqual(val_a.grad, torch_a.grad.item(), places=7)
        self.assertAlmostEqual(val_b.grad, torch_b.grad.item(), places=7)
        self.assertAlmostEqual(val_c.grad, torch_c.grad.item(), places=7)
        self.assertAlmostEqual(val_d.grad, torch_d.grad.item(), places=7)

    def test_finite_difference_numerical_gradient(self):
        """Verify analytical gradients against central finite difference estimates."""
        def composite_function(vals: list[Value]) -> Value:
            x1, x2, x3 = vals
            return (x1 * x2).tanh() + (x2 / x3).exp() - (x1**2)

        inputs = [1.2, -0.8, 2.5]
        all_passed, details = verify_gradients(composite_function, inputs, h=1e-5, tolerance=1e-5)

        for d in details:
            self.assertTrue(
                d["passed"],
                f"Gradient mismatch at index {d['index']}: ana={d['analytical']}, num={d['numerical']}, rel_error={d['rel_error']}",
            )
        self.assertTrue(all_passed)

    def test_mlp_training_convergence(self):
        """Verify that micro-MLP trains with vanilla SGD and reduces loss."""
        model, loss_history = train_mlp(epochs=40, learning_rate=0.08)
        self.assertGreater(loss_history[0], 0.3)
        self.assertLess(loss_history[-1], 0.15)
        # Loss must show overall monotonic downward trend
        self.assertLess(loss_history[-1], loss_history[0])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAutogradEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


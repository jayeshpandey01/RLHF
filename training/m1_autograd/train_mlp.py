"""Training demonstration: Non-linear binary classification using pure scalar autograd."""

import random
from training.m1_autograd.engine import Value
from training.m1_autograd.nn import MLP


def generate_synthetic_dataset(n_samples: int = 40) -> tuple[list[list[float]], list[float]]:
    """Generate 2D non-linear classification dataset (circles/rings)."""
    random.seed(42)
    xs: list[list[float]] = []
    ys: list[float] = []

    for _ in range(n_samples // 2):
        # Class -1: Inner circle (radius < 0.5)
        r = random.uniform(0.05, 0.45)
        theta = random.uniform(0, 2 * 3.14159)
        xs.append([r * random.uniform(0.8, 1.2), r * random.uniform(0.8, 1.2)])
        ys.append(-1.0)

    for _ in range(n_samples // 2):
        # Class +1: Outer ring (radius > 0.8)
        r = random.uniform(0.75, 1.1)
        theta = random.uniform(0, 2 * 3.14159)
        xs.append([r * random.uniform(0.8, 1.2), r * random.uniform(0.8, 1.2)])
        ys.append(1.0)

    return xs, ys


def train_mlp(epochs: int = 50, learning_rate: float = 0.05) -> tuple[MLP, list[float]]:
    """Train MLP on non-linear data using scalar autograd engine."""
    xs, ys = generate_synthetic_dataset()

    # Network: 2 inputs -> 16 hidden (tanh) -> 16 hidden (tanh) -> 1 output
    model = MLP(nin=2, nouts=[16, 16, 1], activation="tanh")
    loss_history: list[float] = []

    print(f"Total trainable parameters: {len(model.parameters())}")

    for epoch in range(epochs):
        # 1. Forward pass
        predictions = [model(x) for x in xs]

        # 2. Compute Mean Squared Error (MSE) loss
        losses = [(p - y) ** 2 for p, y in zip(predictions, ys)]
        total_loss = sum(losses) * (1.0 / len(xs))

        # 3. Zero gradients
        model.zero_grad()

        # 4. Backward pass (reverse-mode automatic differentiation)
        total_loss.backward()

        # 5. Stochastic Gradient Descent update
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        loss_val = total_loss.data
        loss_history.append(loss_val)

        if epoch % 10 == 0 or epoch == epochs - 1:
            # Calculate classification accuracy
            correct = sum((1 if (p.data > 0) == (y > 0) else 0) for p, y in zip(predictions, ys))
            acc = (correct / len(ys)) * 100.0
            print(f"Epoch {epoch:02d} | Loss: {loss_val:.6f} | Accuracy: {acc:.1f}%")

    return model, loss_history


if __name__ == "__main__":
    train_mlp()


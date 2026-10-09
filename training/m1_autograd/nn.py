"""Neural Network modules constructed purely with the scalar Value autograd engine."""

import random
from typing import Literal
from training.m1_autograd.engine import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        """Return list of trainable parameters."""
        return []


class Neuron(Module):
    """A single artificial neuron with learned weights and bias."""

    def __init__(
        self,
        nin: int,
        activation: Literal["linear", "relu", "tanh"] = "relu",
    ):
        # Xavier / Glorot uniform initialization: [-1/sqrt(nin), 1/sqrt(nin)]
        bound = 1.0 / (nin**0.5) if nin > 0 else 1.0
        self.w = [Value(random.uniform(-bound, bound), label=f"w{i}") for i in range(nin)]
        self.b = Value(0.0, label="b")
        self.activation = activation

    def __call__(self, x: list[Value | float]) -> Value:
        x_vals = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        if len(x_vals) != len(self.w):
            raise ValueError(f"Expected {len(self.w)} inputs, got {len(x_vals)}")

        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x_vals)), self.b)

        if self.activation == "relu":
            return act.relu()
        elif self.activation == "tanh":
            return act.tanh()
        return act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    """A dense layer of parallel neurons."""

    def __init__(
        self,
        nin: int,
        nout: int,
        activation: Literal["linear", "relu", "tanh"] = "relu",
    ):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x: list[Value | float]) -> list[Value] | Value:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Multi-Layer Perceptron (feed-forward neural network)."""

    def __init__(
        self,
        nin: int,
        nouts: list[int],
        activation: Literal["linear", "relu", "tanh"] = "relu",
    ):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            # Final layer is linear by default for regression/logits
            act = "linear" if i == len(nouts) - 1 else activation
            self.layers.append(Layer(sz[i], sz[i + 1], activation=act))

    def __call__(self, x: list[Value | float]) -> list[Value] | Value:
        for layer in self.layers:
            x = layer(x) if isinstance(x, list) else layer([x])
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


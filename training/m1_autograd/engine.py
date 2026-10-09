"""Scalar Automatic Differentiation Engine (micrograd-style).

Implements reverse-mode automatic differentiation over a dynamically constructed
Directed Acyclic Graph (DAG) of scalar operations.
"""

from __future__ import annotations
import math
from typing import Callable, Union

Numeric = Union[int, float, "Value"]


class Value:
    """A scalar value that tracks its computation history for reverse-mode automatic differentiation."""

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: set[Value] = set(_children)
        self._op: str = _op
        self.label: str = label
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f}, op='{self._op}')"

    def __add__(self, other: Numeric) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Numeric) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Numeric) -> Value:
        return self + (-other)

    def __rsub__(self, other: Numeric) -> Value:
        return Value(other) + (-self)

    def __mul__(self, other: Numeric) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Numeric) -> Value:
        return self * other

    def __pow__(self, exponent: Union[int, float]) -> Value:
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Exponent must be numeric, got {type(exponent)}")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: Numeric) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1.0)

    def __rtruediv__(self, other: Numeric) -> Value:
        return Value(other) * (self**-1.0)

    def relu(self) -> Value:
        """Rectified Linear Unit activation function."""
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Hyperbolic tangent activation function."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Natural exponential function."""
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Natural logarithm function."""
        if self.data <= 0:
            raise ValueError(f"Natural log undefined for non-positive values: {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Execute reverse-mode automatic differentiation via topological sort."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Base case seed: dL/dL = 1.0
        self.grad = 1.0

        # Traverse in reverse topological order
        for node in reversed(topo):
            node._backward()


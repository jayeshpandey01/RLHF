"""Numerical finite-difference gradient verification protocol."""

from typing import Callable
from training.m1_autograd.engine import Value


def compute_numerical_gradient(
    forward_fn: Callable[[list[Value]], Value],
    inputs: list[Value],
    target_idx: int,
    h: float = 1e-5,
) -> float:
    """Compute numerical derivative using central finite differences:

    f'(x) = [f(x + h) - f(x - h)] / (2h)
    """
    orig_val = inputs[target_idx].data

    # f(x + h)
    inputs[target_idx].data = orig_val + h
    out_plus = forward_fn(inputs)
    val_plus = out_plus.data

    # f(x - h)
    inputs[target_idx].data = orig_val - h
    out_minus = forward_fn(inputs)
    val_minus = out_minus.data

    # Restore original value
    inputs[target_idx].data = orig_val

    # Central difference estimate
    return (val_plus - val_minus) / (2.0 * h)


def verify_gradients(
    forward_fn: Callable[[list[Value]], Value],
    input_values: list[float],
    h: float = 1e-5,
    tolerance: float = 1e-5,
) -> tuple[bool, list[dict[str, float]]]:
    """Verify analytical backpropagation gradients against numerical gradients.

    Returns:
        (passed: bool, details: list of dicts with numerical/analytical/relative errors)
    """
    # 1. Analytical backward pass
    inputs = [Value(x) for x in input_values]
    out = forward_fn(inputs)
    out.backward()

    results = []
    all_passed = True

    # 2. Check each input parameter
    for i, inp in enumerate(inputs):
        ana_grad = inp.grad
        num_grad = compute_numerical_gradient(forward_fn, inputs, i, h=h)

        denom = max(abs(ana_grad), abs(num_grad), 1e-8)
        rel_error = abs(ana_grad - num_grad) / denom

        passed = rel_error < tolerance
        if not passed:
            all_passed = False

        results.append(
            {
                "index": i,
                "analytical": ana_grad,
                "numerical": num_grad,
                "rel_error": rel_error,
                "passed": passed,
            }
        )

    return all_passed, results


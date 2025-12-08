from __future__ import annotations

from typing import Protocol
import numpy as np


class ManifoldMap(Protocol):
    """Protocol for a manifold mapping Ψ(u, Θ) -> z_hat."""

    def __call__(self, u: np.ndarray, theta: np.ndarray) -> np.ndarray: ...


def linear_manifold(u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Linear manifold z = Θ u.

    Θ is represented as a flattened (n_states * n_inputs) vector.
    """

    if u.ndim != 1:
        raise ValueError("u must be a 1D vector")

    m = u.shape[0]
    if theta.size % m != 0:
        raise ValueError(f"theta size {theta.size} not compatible with input {m}")

    n_states = theta.size // m
    theta_mat = theta.reshape(n_states, m)
    return theta_mat @ u
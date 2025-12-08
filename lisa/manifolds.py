from __future__ import annotations

import numpy as np
from typing import Callable

Array = np.ndarray


def linear_manifold(dim_state: int, dim_input: int) -> Callable[[Array, Array], Array]:
    """Construct Ψ(u, Θ) = W u with Θ = vec(W)."""

    n = dim_state
    m = dim_input
    expected_params = n * m

    def manifold(u: Array, theta: Array) -> Array:
        if theta.shape[0] != expected_params:
            raise ValueError(
                f"Expected θ of length {expected_params}, got {theta.shape[0]}"
            )

        w = theta.reshape(n, m)
        return w @ u

    return manifold
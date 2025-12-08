from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


Array = np.ndarray


class ManifoldMap(Protocol):
    """
    Mapping Ψ(u, Θ): input u and structural parameters Θ ↦ manifold state in ℝ^n.

    In the linear case, Ψ(u, Θ) = W u with Θ = vec(W).
    """

    def __call__(self, u: Array, theta: Array) -> Array:
        ...


class Regressor(Protocol):
    """
    Regressor ϕ(z, u, Θ) appearing in the slow structural update law.
    """

    def __call__(self, z: Array, u: Array, theta: Array) -> Array:
        ...


@dataclass
class LISAConfig:
    """
    Dual-timescale LISA configuration.

    Attributes:
        n_states: dim(z) (fast state) as in Eq. (6).
        n_params: dim(Θ) (structural parameters) as in Eq. (13).
        epsilon: ε perturbation (0 < ε << 1) that separates timescales.
        gamma: scalar scale of the positive-definite gain Γ.
        k_fast: contraction gain for the fast manifold dynamics.
    """
    n_states: int
    n_params: int
    epsilon: float = 0.01
    gamma: float = 1.0
    k_fast: float = 10.0


class LISADynamicalSystem:
    """
    Core LISA dynamics implementing Eq. (1), Eq. (6), and Eq. (11)–(13).

    Fast inference: ż = −k_fast η pushes z toward Ψ(u, Θ).
    Slow adaptation: Θ̇ = −ε Γ vec(η ϕᵀ) follows the Lyapunov redesign.
    """

    def __init__(
        self,
        config: LISAConfig,
        manifold_map: ManifoldMap,
        regressor: Regressor | None = None,
    ) -> None:
        self.cfg = config
        self.psi = manifold_map
        """Deprecated module. Use lisa.dynamics instead."""
    def compute_derivatives(self, t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Coupled ODE system for numerical integration.
        State vector chi = [z^T, Theta^T]^T[cite: 44].
        """
        n, p = self.cfg.n_states, self.cfg.p_params
        z = state[:n]
        theta = state[n:]
        dz = self.fast_dynamics(z, u, theta)
        dtheta = self.slow_dynamics(z, u, theta)
        return np.concatenate([dz, dtheta])

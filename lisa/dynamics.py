from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import LISAConfig
from .manifolds import ManifoldMap


@dataclass
class LISAState:
    """Composite state χ = [z^T, Θ^T]^T."""
    z: np.ndarray
    theta: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.z, self.theta])

    @classmethod
    def from_vector(cls, vec: np.ndarray, n_states: int) -> "LISAState":
        return cls(z=vec[:n_states], theta=vec[n_states:])


class LISADynamicalSystem:
    """
    Latent Invariant Space Adaptation (LISA) dynamics.

    Fast  subsystem:
        ż = -k_fast (z - Ψ(u, Θ))

    Slow subsystem (Lyapunov-driven structural adaptation):
        Θ̇ = -ε Γ φ(z, u) η^T

    where η = z - Ψ(u, Θ) and φ is a regressor.
    """

    def __init__(self, config: LISAConfig, manifold_map: ManifoldMap, k_fast: float = 10.0):
        self.cfg = config
        self.psi = manifold_map
        self.k_fast = float(k_fast)
        self.Gamma = config.make_gain_matrix()

    def manifold_error(self, z: np.ndarray, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """η = z - Ψ(u, Θ)."""
        return z - self.psi(u, theta)

    def regressor(self, z: np.ndarray, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Regressor φ(z, u) in Θ̇ = -ε Γ φ η^T."""
        return u

    def fast_dynamics(self, z: np.ndarray, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Fast state: ż = -k_fast * η."""
        eta = self.manifold_error(z, u, theta)
        return -self.k_fast * eta

    def slow_dynamics(self, z: np.ndarray, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Slow structural dynamics Θ̇ = -ε Γ φ η^T."""
        eta = self.manifold_error(z, u, theta)
        phi = self.regressor(z, u, theta)

        outer = np.outer(phi, eta)
        grad = outer.ravel()

        p = self.cfg.n_params
        if grad.size < p:
            grad = np.pad(grad, (0, p - grad.size))
        elif grad.size > p:
            grad = grad[:p]

        theta_dot = -self.cfg.epsilon * (self.Gamma @ grad)
        return theta_dot

    def rhs(self, t: float, state_vec: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Coupled RHS for ODE solvers."""
        state = LISAState.from_vector(state_vec, n_states=self.cfg.n_states)
        dz = self.fast_dynamics(state.z, u, state.theta)
        dtheta = self.slow_dynamics(state.z, u, state.theta)
        return np.concatenate([dz, dtheta])

    def lyapunov_energy(
        self,
        z: np.ndarray,
        u: np.ndarray,
        theta: np.ndarray,
        theta_star: Optional[np.ndarray] = None,
    ) -> float:
        """Composite Lyapunov: V = 0.5 ||η||^2 + 0.5 Θ̃^T Γ^{-1} Θ̃."""
        eta = self.manifold_error(z, u, theta)
        v_state = 0.5 * float(eta.T @ eta)

        if theta_star is None:
            return v_state

        theta_tilde = theta - theta_star
        gamma_inv = np.linalg.inv(self.Gamma)
        v_param = 0.5 * float(theta_tilde.T @ gamma_inv @ theta_tilde)
        return v_state + v_param

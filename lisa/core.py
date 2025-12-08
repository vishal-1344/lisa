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
        self.regressor = regressor or self._default_regressor
        self.Gamma: Array = self.cfg.gamma * np.eye(self.cfg.n_params)

    # ------------------------------------------------------------------
    # Manifold tracking helpers
    # ------------------------------------------------------------------
    def manifold_error(self, z: Array, u: Array, theta: Array) -> Array:
        """η = z − Ψ(u, Θ) as in Eq. (8)."""

        return z - self.psi(u, theta)

    def lyapunov_energy(
        self,
        z: Array,
        u: Array,
        theta: Array,
        theta_star: Array | None = None,
    ) -> float:
        """Composite Lyapunov energy (Eq. (9))."""

        eta = self.manifold_error(z, u, theta)

        theta_tilde = theta if theta_star is None else theta - theta_star

        v_eta = 0.5 * float(eta @ eta)
        gamma_inv = np.linalg.inv(self.Gamma)
        v_theta = 0.5 * float(theta_tilde.T @ gamma_inv @ theta_tilde)

        return v_eta + v_theta

    # ------------------------------------------------------------------
    # Fast and slow subsystems
    # ------------------------------------------------------------------
    def fast_dynamics(self, z: Array, u: Array, theta: Array) -> Array:
        """Fast attractor ż = −k_fast η (Eq. (6))."""

        eta = self.manifold_error(z, u, theta)
        return -self.cfg.k_fast * eta

    def slow_dynamics(self, z: Array, u: Array, theta: Array) -> Array:
        """Slow structural update Θ̇ = −ε Γ vec(η ϕᵀ) (Eq. (11)–(13))."""

        eta = self.manifold_error(z, u, theta)
        phi = self.regressor(z, u, theta)
        update_tensor = np.outer(eta, phi).reshape(-1)

        if update_tensor.shape[0] != self.cfg.n_params:
            raise ValueError(
                f"Regressor dimension mismatch: expected {self.cfg.n_params}, got {update_tensor.shape[0]}"
            )

        theta_dot = -self.cfg.epsilon * (self.Gamma @ update_tensor)
        return theta_dot

    # ------------------------------------------------------------------
    # Coupled derivatives for integration
    # ------------------------------------------------------------------
    def compute_derivatives(self, t: float, state: Array, u: Array) -> Array:
        """Composite derivative for χ = [zᵀ, Θᵀ]ᵀ as used by ODE solvers."""

        n, p = self.cfg.n_states, self.cfg.n_params
        z = state[:n]
        theta = state[n:]

        dz = self.fast_dynamics(z, u, theta)
        dtheta = self.slow_dynamics(z, u, theta)

        return np.concatenate([dz, dtheta])

    # ------------------------------------------------------------------
    # Default regressor
    # ------------------------------------------------------------------
    @staticmethod
    def _default_regressor(z: Array, u: Array, theta: Array) -> Array:  # noqa: ARG002
        """Default φ(z, u, Θ) = u for linear Ψ(u, Θ) = W u."""

        return uimport numpy as np
from typing import Callable
from dataclasses import dataclass

@dataclass
class LISAConfig:
    """
    Configuration for the Latent Invariant Space Adaptation (LISA) System.
    
    Parameters:
        n_states (int): Dimension of the fast latent state z[cite: 45].
        p_params (int): Dimension of the structural parameters Theta[cite: 46].
        epsilon (float): Perturbation parameter defining timescale separation (0 < epsilon << 1)[cite: 54].
        gamma (float): Adaptive Gain Matrix scale for structural updates[cite: 121].
    """
    n_states: int          
    p_params: int          
    epsilon: float = 0.01  
    gamma: float = 0.1     

class LISADynamicalSystem:
    """
    Implementation of the LISA Framework.
    
    Models the agent as a singularly perturbed dynamical system[cite: 41]:
        dz/dt      = f(z, u, Theta)      (Fast Inference) [cite: 49]
        dTheta/dt  = epsilon * g(z, u)   (Slow Adaptation) [cite: 49]
    """
    
    def __init__(self, config: LISAConfig, manifold_map: Callable):
        self.cfg = config
        self.psi = manifold_map  # The current estimate of the manifold Psi(u, Theta) [cite: 111]
        
        # Initialize Gain Matrix Gamma (Positive Definite) [cite: 121]
        self.Gamma = np.eye(config.p_params) * config.gamma

    def fast_dynamics(self, z: np.ndarray, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Calculates the Fast State Dynamics.
        Ref: Eq (6) in Technical Report[cite: 88].
        
        The fast subsystem must be contractional, ensuring z(t) rapidly settles 
        onto the local region of the manifold M_theta[cite: 95].
        """
        eta = z - self.psi(u, theta)
        k_fast = 10.0 
        z_dot = -k_fast * eta
        return z_dot

    def slow_dynamics(self, z: np.ndarray, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Calculates the Structural Update Law (LISA Adaptation).
        Ref: Eq (11) and Eq (13) in Technical Report[cite: 128, 148].
        
        This mechanism treats structural reconfiguration as a steepest-descent trajectory 
        on the time derivative of the Lyapunov energy function[cite: 13].
        """
        eta = z - self.psi(u, theta) 
        phi = u 
        grad_v = np.outer(eta, phi).flatten() 
        theta_dot = -self.cfg.epsilon * (self.Gamma @ grad_v[:self.cfg.p_params])
        return theta_dot

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

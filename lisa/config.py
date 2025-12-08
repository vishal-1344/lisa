from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LISAConfig:
    """Configuration for the LISA dual-timescale dynamical system."""

    n_states: int
    n_params: int
    epsilon: float = 0.01
    gamma: float = 1.0

    def make_gain_matrix(self) -> np.ndarray:
        """Construct the positive-definite adaptation gain matrix Γ = γ I."""
        return self.gamma * np.eye(self.n_params, dtype=float)

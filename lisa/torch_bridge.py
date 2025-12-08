from __future__ import annotations

import torch
import torch.nn as nn


class LISALayer(nn.Module):
    """PyTorch wrapper for a simple LISA-style adaptive layer."""

    def __init__(self, input_dim: int, state_dim: int, epsilon: float = 0.01, gamma: float = 1.0):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.theta = nn.Parameter(torch.eye(state_dim, input_dim))
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.k_fast = 10.0

    def forward(self, u: torch.Tensor, z_prev: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        z_hat = u @ self.theta.t()
        eta = z_prev - z_hat

        dz = -self.k_fast * eta
        z_next = z_prev + dz * dt

        eta_exp = eta.unsqueeze(2)
        u_exp = u.unsqueeze(1)
        outer = torch.bmm(eta_exp, u_exp)
        grad_est = outer.mean(dim=0)

        dtheta = -self.epsilon * self.gamma * grad_est

        with torch.no_grad():
            self.theta += dtheta * dt

        return z_next

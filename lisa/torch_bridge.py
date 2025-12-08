import torch
import torch.nn as nn
from typing import Optional


class LISALayer(nn.Module):
    """PyTorch wrapper exposing LISA as a recurrent layer."""

    def __init__(self, input_dim: int, state_dim: int, epsilon: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.theta = nn.Parameter(torch.eye(state_dim, input_dim))
        self.gamma = 0.1

    def forward(self, u: torch.Tensor, z_prev: Optional[torch.Tensor] = None, dt: float = 0.1):
        batch_size = u.shape[0]
        if z_prev is None:
            z_prev = torch.zeros(batch_size, self.state_dim, device=u.device)

        z_target = torch.matmul(u, self.theta.T)

        k_fast = 10.0
        eta = z_prev - z_target
        dz = -k_fast * eta
        z_next = z_prev + dz * dt

        grad_energy = -torch.bmm(eta.unsqueeze(2), u.unsqueeze(1))
        avg_grad = grad_energy.mean(dim=0)

        d_theta = -self.epsilon * self.gamma * avg_grad
        self.theta.data += d_theta * dt

        return z_next, self.theta

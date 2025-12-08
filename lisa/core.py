import numpy as np
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

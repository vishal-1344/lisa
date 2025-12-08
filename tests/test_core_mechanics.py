import unittest
import numpy as np
import sys
import os

# Add parent dir to path to import lisa
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lisa.core import LISADynamicalSystem, LISAConfig


class TestLISAMechanics(unittest.TestCase):
    
    def setUp(self):
        self.manifold_map = lambda u, theta: theta.reshape(2, 2) @ u
        self.config = LISAConfig(n_states=2, n_params=4, epsilon=0.01, gamma=1.0)
        self.system = LISADynamicalSystem(self.config, self.manifold_map)
        
    def test_shapes_integrity(self):
        state = np.random.randn(6)
        u = np.random.randn(2)
        t = 0.0
        
        d_chi = self.system.compute_derivatives(t, state, u)
        
        self.assertEqual(d_chi.shape, (6,))
        self.assertFalse(np.isnan(d_chi).any(), "Derivatives contain NaNs")

    def test_lyapunov_stability_condition(self):
        u = np.array([1.0, 0.0])
        z = np.array([10.0, 10.0])
        theta = np.eye(2).reshape(-1)
        
        eta = z - self.manifold_map(u, theta)
        
        state_0 = np.concatenate([z, theta])
        
        V_0 = 0.5 * np.linalg.norm(eta) ** 2
        
        dt = 1e-4
        d_chi = self.system.compute_derivatives(0, state_0, u)
        state_1 = state_0 + d_chi * dt
        
        z_1 = state_1[:2]
        theta_1 = state_1[2:]
        eta_1 = z_1 - self.manifold_map(u, theta_1)
        V_1 = 0.5 * np.linalg.norm(eta_1) ** 2
        
        self.assertLessEqual(V_1, V_0 + 1e-9, "Lyapunov energy increased! System unstable.")


if __name__ == '__main__':
    unittest.main()

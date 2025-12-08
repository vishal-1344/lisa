import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lisa import LISADynamicalSystem, LISAConfig, linear_manifold


class TestLISA(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = LISAConfig(n_states=2, n_params=4, epsilon=0.05, gamma=1.0)
        self.sys = LISADynamicalSystem(self.cfg, linear_manifold)

    def test_dimensions(self) -> None:
        state = np.zeros(self.cfg.n_states + self.cfg.n_params, dtype=float)
        u = np.zeros(2, dtype=float)
        dstate = self.sys.rhs(0.0, state, u)
        self.assertEqual(dstate.shape, state.shape)

    def test_lyapunov_decrease_small_step(self) -> None:
        u = np.array([1.0, 0.5], dtype=float)
        z = np.array([2.0, -1.0], dtype=float)
        theta = np.eye(2, dtype=float).flatten()

        V0 = self.sys.lyapunov_energy(z, u, theta)

        dt = 1e-3
        dz = self.sys.fast_dynamics(z, u, theta)
        dtheta = self.sys.slow_dynamics(z, u, theta)

        z1 = z + dt * dz
        theta1 = theta + dt * dtheta

        V1 = self.sys.lyapunov_energy(z1, u, theta1)

        self.assertLess(V1, V0, "Lyapunov energy did not decrease for small step")


if __name__ == '__main__':
    unittest.main()

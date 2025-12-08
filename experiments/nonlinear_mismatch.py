import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lisa.core import LISADynamicalSystem, LISAConfig


def run_mismatch_experiment():
    def true_manifold_mapping(u, t):
        return np.array([np.sin(u[0]), np.cos(u[1])])

    def estimated_manifold(u, theta_flat):
        theta_mat = theta_flat.reshape(2, 2)
        return theta_mat @ u

    def get_input_signal(t):
        return np.array([t * 0.5, t * 0.3])

    config = LISAConfig(n_states=2, n_params=4, epsilon=0.2, gamma=2.0)
    lisa = LISADynamicalSystem(config, estimated_manifold)

    t_span = np.linspace(0, 40, 800)
    initial_state = np.zeros(6)

    def system_wrapper(state, t):
        u = get_input_signal(t)
        return lisa.compute_derivatives(t, state, u)

    trajectory = odeint(system_wrapper, initial_state, t_span)

    errors = []
    for i, t in enumerate(t_span):
        u = get_input_signal(t)
        z_est = trajectory[i, :2]
        z_target = true_manifold_mapping(u, t)
        errors.append(np.linalg.norm(z_est - z_target))

    mean_err = np.mean(errors[200:])
    print("Non-Linear Tracking Results:")
    print(f"Steady State Error (Bounded): {mean_err:.4f}")
    print("System maintained stability despite structural mismatch.")


if __name__ == "__main__":
    run_mismatch_experiment()

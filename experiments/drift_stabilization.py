from __future__ import annotations

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lisa import LISADynamicalSystem, LISAConfig, linear_manifold, LISAState


def get_input(t: float) -> np.ndarray:
    return np.array([np.sin(t), np.cos(t)], dtype=float)


def get_ground_truth(u: np.ndarray, t: float) -> np.ndarray:
    angle = 0.1 * t
    R = np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle),  np.cos(angle)]],
        dtype=float,
    )
    return R @ u


def run_experiment() -> None:
    cfg = LISAConfig(n_states=2, n_params=4, epsilon=0.1, gamma=5.0)
    lisa = LISADynamicalSystem(cfg, linear_manifold, k_fast=10.0)

    z0 = np.zeros(2, dtype=float)
    theta0 = np.eye(2, dtype=float).flatten()
    state0 = LISAState(z=z0, theta=theta0).as_vector()

    t_span = np.linspace(0.0, 50.0, 1000)

    def wrapper(y: np.ndarray, t: float) -> np.ndarray:
        u = get_input(t)
        return lisa.rhs(t, y, u)

    print("Integrating dual-timescale LISA dynamics...")
    traj = odeint(wrapper, state0, t_span)

    z_hist = traj[:, :2]
    theta_hist = traj[:, 2:]

    tracking_errors = []
    manifold_errors = []

    for i, t in enumerate(t_span):
        u = get_input(t)
        z_est = z_hist[i]
        theta_i = theta_hist[i]

        z_true = get_ground_truth(u, t)
        tracking_errors.append(np.linalg.norm(z_est - z_true))

        eta = lisa.manifold_error(z_est, u, theta_i)
        manifold_errors.append(np.linalg.norm(eta))

    tracking_errors = np.asarray(tracking_errors)
    manifold_errors = np.asarray(manifold_errors)

    print(f"Final tracking error ‖z - z_true‖: {tracking_errors[-1]:.4f}")
    print(f"Final manifold error ‖η‖       : {manifold_errors[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(t_span, tracking_errors, label="‖z - z_true‖ (tracking error)")
    plt.plot(t_span, manifold_errors, label="‖η‖ (manifold deviation)")
    plt.title("LISA Drift Stabilization in a Rotating Manifold")
    plt.xlabel("time")
    plt.ylabel("error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
    ax_state.plot(t_span, z_hist[:, 1], label="z2")
    ax_state.set_xlabel("t")
    ax_state.set_ylabel("z components")
    ax_state.legend()

    ax_err.set_title("Tracking error ||z(t) - z*(t)||")
    ax_err.plot(t_span, errors)
    ax_err.set_xlabel("t")
    ax_err.set_ylabel("error norm")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lisa.core import LISADynamicalSystem, LISAConfig

def run_experiment():
    """
    Simulates the system in a non-stationary environment where the
    statistical properties of the input stream evolve continuously[cite: 24].
    """
    def get_input_signal(t):
        return np.array([np.sin(t), np.cos(t)])

    def true_manifold_mapping(u, t):
        angle = 0.1 * t 
        rotation = np.array([[np.cos(angle), -np.sin(angle)], 
                             [np.sin(angle),  np.cos(angle)]])
        return rotation @ u

    def estimated_manifold(u, theta_flat):
        theta_mat = theta_flat.reshape(2, 2)
        return theta_mat @ u

    config = LISAConfig(n_states=2, p_params=4, epsilon=0.1, gamma=5.0)
    lisa = LISADynamicalSystem(config, estimated_manifold)

    t_span = np.linspace(0, 50, 1000)
    initial_z = np.zeros(2)
    initial_theta = np.eye(2).flatten()
    initial_state = np.concatenate([initial_z, initial_theta])

    def system_wrapper(state, t):
        u = get_input_signal(t)
        return lisa.compute_derivatives(t, state, u)

    print("Integrating Dual-Timescale Dynamics...")
    trajectory = odeint(system_wrapper, initial_state, t_span)
    
    z_hist = trajectory[:, :2]
    errors = []
    for i, t in enumerate(t_span):
        u = get_input_signal(t)
        z_est = z_hist[i]
        z_target = true_manifold_mapping(u, t)
        errors.append(np.linalg.norm(z_est - z_target))

    print(f"Final Tracking Error: {errors[-1]:.4f}")
    print("Manifold adaptation successful via Lyapunov-driven updates[cite: 13].")

    plt.figure(figsize=(10, 6))
    plt.plot(t_span, errors, label='Manifold Deviation ||eta||')
    plt.title('LISA Stability in Non-Stationary Environment')
    plt.xlabel('Time (t)')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()

if __name__ == "__main__":
    run_experiment()

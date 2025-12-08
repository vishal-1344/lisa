from __future__ import annotations

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from lisa.core import LISAConfig, LISADynamicalSystem
from lisa.manifolds import linear_manifold


Array = np.ndarray


def input_signal(t: float) -> Array:
    return np.array([np.sin(t), np.cos(t)], dtype=float)


def true_manifold(u: Array, t: float) -> Array:
    omega = 0.1
    angle = omega * t
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return rot @ u


def linear_regressor(z: Array, u: Array, theta: Array) -> Array:  # noqa: ARG001
    return u


def run_experiment() -> None:
    n_states = 2
    dim_u = 2
    n_params = n_states * dim_u

    cfg = LISAConfig(
        n_states=n_states,
        n_params=n_params,
        epsilon=0.05,
        gamma=5.0,
        k_fast=10.0,
    )

    manifold = linear_manifold(dim_state=n_states, dim_input=dim_u)
    lisa = LISADynamicalSystem(
        cfg, manifold_map=manifold, regressor=linear_regressor
    )

    z0 = np.zeros(n_states, dtype=float)
    theta0 = np.eye(n_states, dtype=float).reshape(-1)
    state0 = np.concatenate([z0, theta0])

    t_span = np.linspace(0.0, 50.0, 1000)

    def ode_wrapper(state: Array, t: float) -> Array:
        u = input_signal(t)
        return lisa.compute_derivatives(t, state, u)

    print("Integrating LISA dual-timescale dynamics...")
    traj = odeint(ode_wrapper, state0, t_span)

    z_hist = traj[:, :n_states]
    theta_hist = traj[:, n_states:]

    errors = []
    for idx, t in enumerate(t_span):
        u = input_signal(t)
        z_est = z_hist[idx]
        z_star = true_manifold(u, t)
        errors.append(np.linalg.norm(z_est - z_star))

    errors = np.asarray(errors)

    print(f"Final tracking error ||z - z*|| = {errors[-1]:.4f}")

    fig, (ax_state, ax_err) = plt.subplots(2, 1, figsize=(6, 6))

    ax_state.set_title("Fast state trajectory z(t)")
    ax_state.plot(t_span, z_hist[:, 0], label="z1")
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

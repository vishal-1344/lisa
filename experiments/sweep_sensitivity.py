import numpy as np
from scipy.integrate import odeint
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lisa import LISADynamicalSystem, LISAConfig


def run_sweep():
    print(f"{'Epsilon':<10} | {'Gamma':<10} | {'Final Error':<15} | {'Status'}")
    print("-" * 55)

    epsilons = [0.01, 0.1, 0.5, 1.0]
    gammas = [1.0, 5.0, 10.0]

    def get_input(t):
        return np.array([1.0, 0.5])

    def manifold_map(u, theta):
        return theta.reshape(2, 2) @ u

    for eps in epsilons:
        for gam in gammas:
            try:
                config = LISAConfig(n_states=2, n_params=4, epsilon=eps, gamma=gam)
                lisa = LISADynamicalSystem(config, manifold_map)

                t_span = np.linspace(0, 10, 100)
                init = np.concatenate([np.zeros(2), np.eye(2).flatten()])

                def wrapper(state, t):
                    target = np.array([1.0 + 0.1 * t, 0.5])
                    d_chi = lisa.compute_derivatives(t, state, get_input(t))
                    return d_chi

                traj = odeint(wrapper, init, t_span)
                final_z = traj[-1, :2]
                target_z = np.array([1.0 + 0.1 * 10, 0.5])
                err = np.linalg.norm(final_z - target_z)

                status = "Stable" if err < 1.0 else "Diverged"
                print(f"{eps:<10} | {gam:<10} | {err:.4f}          | {status}")

            except Exception:
                print(f"{eps:<10} | {gam:<10} | NaN              | Failed")


if __name__ == "__main__":
    run_sweep()

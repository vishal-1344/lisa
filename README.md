LISA: Latent Invariant Space Adaptation

A Dual-Timescale Framework for Robust Adaptive Control

“Robust system performance is not merely a function of error minimization,
but rather the result of maintaining a low-dimensional attracting manifold.”
— LISA Technical Report

0. Overview

LISA is a control-theoretic architecture for high-dimensional, non-stationary environments.
Instead of treating a model as a static function trained once and frozen, LISA models the agent as a
singularly perturbed dynamical system with:

fast behavioral dynamics (what the system is doing right now), and

slow structural dynamics (how the underlying representation and geometry are adapting).

The core idea:

Fast latent states z(t) evolve continuously under current structure Theta(t) and input u(t).

Slow structural parameters Theta(t) evolve under a small perturbation parameter epsilon,
driven by violation of a Lyapunov-style energy function V(z, Theta).

The system is designed so that structural updates reduce energy and reconstruct an invariant manifold,
yielding robustness even under distribution drift.

LISA is intended as a general template for adaptive controllers, structured representation learning,
and long-horizon agents operating in changing environments.

1. Mathematical Formulation

The system evolves on two explicitly separated time scales (t, tau).

1.1 Fast State Dynamics (behavioral layer)

Fast latent state z evolves according to:

dz/dt = f(z, u, Theta)


where:

z = fast latent state (behavior, beliefs, or internal representation)

u = control input or external signal

Theta = structural parameters (geometry, invariants, slow weights)

f = vector field defining the fast dynamics

1.2 Slow Structural Dynamics (structural layer)

Structural parameters Theta evolve on a slower time scale:

dTheta/dt = epsilon * g(z, u, Theta)


with:

epsilon > 0 small (time-scale separation parameter)

g = structural update field (plasticity / adaptation rule)

The small parameter epsilon enforces a dual-timescale separation:
z reacts quickly, Theta adapts slowly.

2. Invariance and Lyapunov Energy

LISA assumes the existence of a Lyapunov-style energy function:

V(z, Theta) >= 0


interpreted as an invariant manifold reconstruction error or “distance to a stable regime.”

A common abstract form of the structural update law is a steepest-descent type rule on this energy landscape.
One illustrative form is:

dTheta/dt = -Gamma * phi(z, u) * eta^T


where:

Gamma = positive-definite gain matrix (learning / adaptation gain)

phi(z, u) = feature or regressor vector derived from state and input

eta = manifold reconstruction error (e.g., deviation from an invariant relationship)

Under appropriate conditions, this update drives:

eta -> 0  (as t -> infinity)


meaning the system converges to a low-dimensional attracting manifold
even as the environment drifts.

Intuitively:

The fast system z responds to incoming signals.

The slow system Theta reconfigures the latent geometry so that trajectories are attracted to a stable set.

The energy V(z, Theta) provides a certificate of stability during adaptation.

3. Key Features

Dual-timescale separation
Explicit modeling of epsilon allows singular perturbation analysis and separates fast behavior from slow structural change.

Lyapunov-driven adaptation
Structural updates are derived to reduce an energy function, providing a principled alternative to heuristic learning rates.
Adaptation is aligned with stability guarantees, not just gradient descent on a static loss.

Continuous-time learning
LISA is formulated as a continuous-time flow:

no “epochs” or “batches” are required conceptually,

learning proceeds in parallel with operation,

suitable for streaming signals and online control.

Robustness under drift
By continuously reshaping the latent manifold (through Theta) in response to reconstruction error eta,
LISA aims to maintain performance and stability across regime changes and distribution shift.

4. Installation

If you are using the LISA Python implementation:

git clone https://github.com/vishal-1344/lisa.git
cd lisa
pip install -e .
# or
pip install -r requirements.txt


Then, in Python:

import lisa

5. Quickstart: Toy Dual-Timescale System

A minimal example illustrating fast–slow dynamics and a decreasing energy function.

Create examples/quickstart_toy_system.py:

"""
Quickstart: LISA-style dual-timescale dynamics on a toy system.

This example illustrates:
- fast state dynamics dz/dt = f(z, u, Theta)
- slow structural dynamics dTheta/dt = epsilon * g(z, u, Theta)
- decreasing Lyapunov-like energy V(z, Theta)
"""

import numpy as np


def f(z, u, Theta):
    # Simple linear fast dynamics: z_dot = A z + B u, where A depends on Theta
    A = np.array([[Theta[0], 0.0],
                  [0.0, Theta[1]]])
    B = np.eye(2)
    return A @ z + B @ u


def g(z, u, Theta):
    # Simple structural update: move Theta toward |z| statistics (illustrative)
    target = np.abs(z)
    return target - Theta


def V(z, Theta):
    # Example Lyapunov-like energy: norm of mismatch between Theta and |z|
    return 0.5 * np.linalg.norm(np.abs(z) - Theta) ** 2


def main() -> None:
    dt = 0.01
    T = 5.0
    steps = int(T / dt)

    # Fast state (2D) and structural parameters (2D)
    z = np.array([1.0, -0.5])
    Theta = np.array([0.0, 0.0])
    u = np.array([0.0, 0.0])

    epsilon = 0.05

    energies = []

    for _ in range(steps):
        # Fast dynamics
        z = z + dt * f(z, u, Theta)

        # Slow structural dynamics
        Theta = Theta + dt * epsilon * g(z, u, Theta)

        energies.append(V(z, Theta))

    print("Initial energy:", energies[0])
    print("Final energy:", energies[-1])
    print("First 10 energies:", energies[:10])


if __name__ == "__main__":
    main()


Run:

python examples/quickstart_toy_system.py


In a full LISA deployment, f, g, and V will be problem-specific and derived from
your control / modeling setup, but the pattern remains:

explicit fast–slow dynamics,

structurally meaningful updates,

energy decreasing over time.

6. Repository Structure (example)

A typical layout for the LISA codebase might be:

lisa/
  __init__.py
  dynamics.py      # fast dynamics f(z, u, Theta)
  adaptation.py    # slow structural updates g(z, u, Theta)
  energy.py        # Lyapunov / energy functions V(z, Theta)
  simulation.py    # utilities for simulating fast–slow systems

examples/
  quickstart_toy_system.py
  ...

experiments/
  ...              # experiment scripts, configs, logs

tests/
  test_energy.py
  test_dual_timescale.py

pyproject.toml or setup.py
LICENSE
README.md


Adjust this to reflect your actual repository layout.

7. Testing

If tests are provided, they can be run via:

pytest


Recommended tests include:

Verifying that the energy V(z, Theta) decreases along simulated trajectories for simple systems.

Checking numerical stability under small perturbations in z, u, and Theta.

Validating that epsilon correctly controls timescale separation (fast vs slow dynamics).

8. Citation

If you use LISA or build on this framework, please cite the technical report:

Latent Invariant Space Adaptation (LISA): A Dual-Timescale Framework for Robust Adaptive Control, December 2025.

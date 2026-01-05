# LISA: Latent Invariant Space Adaptation

### A Relativistic, Self-Governing Dynamical Architecture  
*(built on a dual-timescale framework for manifold-stable adaptive control)*

> "Robust system performance is not merely a function of error minimization, but rather the result of maintaining a low-dimensional attracting manifold."  
> — **LISA Technical Report**

## Overview

**LISA** is a control-theoretic architecture for high-dimensional, non-stationary environments. Rather than treating a model as a static function trained once and frozen, LISA models an agent as a **singularly perturbed dynamical system** with:

1. **Fast behavioral dynamics** — what the system is doing right now.
2. **Slow structural dynamics** — how the underlying representation/manifold adapts to remain stable under drift.

The central idea is **manifold stability**: maintain a low-dimensional *attracting* latent manifold embedded in a high-dimensional state space, even when the environment changes.

---

## Core Idea (Dual Timescales + Manifold Stability)

- **Fast latent state** $z(t)$ evolves continuously under current structure $\Theta(t)$ and input $u(t)$.
- **Slow structural parameters** $\Theta(t)$ adapt under a small timescale parameter $\epsilon$, derived from (or aligned with) a **Lyapunov-style energy** $V(z,\Theta)$.

Structural updates are designed to reduce a global energy / tension metric and restore *invariance* (or attraction) of the latent manifold, yielding robustness under distribution drift.

---

## Mathematical Formulation

LISA evolves on two explicitly separated timescales $(t,\tau)$.

### 1) Fast State Dynamics (Behavioral Layer)

$$\dot{z} = f(z, u, \Theta)$$

- $z$: fast latent state (behavior, beliefs, internal representation)
- $u$: external input / control signal
- $\Theta$: structural parameters (geometry, invariants, slow weights)
- $f$: vector field for fast dynamics

### 2) Slow Structural Dynamics (Structural Layer)

$$\dot{\Theta} = \epsilon \, g(z, u, \Theta)$$

- $\epsilon>0$: small timescale separation parameter
- $g$: structural update field (plasticity / adaptation rule)

The small $\epsilon$ enforces **fast reaction, slow adaptation**.

### 3) Invariance + Manifold Error

A common way to express off-manifold deviation is:

$$\eta = z - \Psi(u,\Theta)$$

where $\Psi(u,\Theta)$ estimates the manifold location for input $u$.

### 4) Lyapunov-Driven Structural Update (Canonical Form)

LISA derives slow adaptation from a Lyapunov argument so that a composite energy decreases:

$$V_{\text{total}}(z,\Theta)=\frac{1}{2} \eta^T\eta+\frac{1}{2}\mathrm{tr}(\tilde{\Theta}^T\Gamma^{-1}\tilde{\Theta})$$

A canonical Lyapunov-aligned update is:

$$\dot{\Theta} = -\Gamma\,\phi(z,u)\,\eta^T$$

- $\Gamma$: positive-definite adaptation gain
- $\phi(z,u)$: regressor/features
- $\eta$: manifold reconstruction error

Under appropriate conditions, this yields boundedness (and often **UUB** in non-ideal settings).

---

## Optional Self-Governing Extensions (v2.x)

In highly stochastic environments, always-on adaptation can waste plasticity on noise, while overly slow adaptation can fail under genuine drift. LISA optionally adds two *bounded modulators* that regulate **how much** and **when** the slow update runs — without changing the underlying Lyapunov direction.

### A) Perceptual Gravity (State-Dependent Timescale Dilation)

A bounded "stress gain" increases adaptation intensity when system energy rises:

$$\gamma_t = 1+\alpha\tanh(\beta\,\mathcal{S}(t)), \quad \gamma_t \ge 1$$

$$\epsilon(t)=\epsilon_{\text{base}}\gamma_t$$

Typical choices: $\mathcal{S}(t)=V_{\text{total}}(t)$ or $\mathcal{S}(t)=\|\eta(t)\|^2$.

### B) Synthetic Dopamine (Epistemic Plasticity Gating)

A bounded gate suppresses learning under likely noise, and enables learning under reliable novelty:

$$\mathcal{D}_t=\sigma\!\left(\frac{\delta(t)}{\Sigma(t)+\xi}-\tau_{\text{th}}\right), \quad \mathcal{D}_t\in[0,1]$$

- $\delta(t)$: surprise proxy (often $\|\eta(t)\|$ or a prediction residual)
- $\Sigma(t)$: uncertainty proxy (running variance, learned uncertainty head, etc.)
- $\xi>0$: numerical stabilizer

### Unified Modulated Slow Law

The self-governing slow update becomes:

$$\dot{\Theta} = -\epsilon_{\text{base}}\,\gamma_t\,\mathcal{D}_t\,\Gamma\,\phi(z,u)\,\eta^T$$

Interpretation: same stability-driven update direction, with bounded scheduling of **intensity** ($\gamma_t$) and **validity** ($\mathcal{D}_t$).

---

## Key Features

- **Dual-timescale separation:** explicit $\epsilon$ enables singular perturbation analysis and cleanly separates fast behavior from slow structure.
- **Manifold-stable adaptation:** updates aim to maintain (or restore) an attracting invariant manifold under drift.
- **Lyapunov-driven learning:** structural updates are aligned with energy decrease and boundedness, not heuristic learning rates.
- **Continuous-time operation:** conceptually no epochs/batches required; learning runs in parallel with behavior.
- **(Optional) Self-governance:** Perceptual Gravity + Synthetic Dopamine provide bounded gain scheduling to avoid learning-on-noise while responding to genuine regime change.

---

## Installation

```bash
git clone https://github.com/vishal-1344/lisa.git
cd lisa
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

Then:

```python
import lisa
```

## Quickstart: Toy Dual-Timescale System (with Optional Modulators)

Create `examples/quickstart_toy_system.py`:

```python
"""
Quickstart: LISA-style dual-timescale dynamics on a toy system.

Illustrates:
1) fast state dynamics:    dz/dt = f(z, u, Theta)
2) slow structural dynamics dTheta/dt = epsilon * g(z, u, Theta)
3) Lyapunov-like energy    V(z, Theta)
4) optional modulators:    gamma_t (Perceptual Gravity), D_t (Synthetic Dopamine)
"""

from __future__ import annotations
import numpy as np


def f(z: np.ndarray, u: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    # Simple linear fast dynamics: z_dot = A z + B u, where A depends on Theta
    A = np.array([[Theta[0], 0.0], [0.0, Theta[1]]])
    B = np.eye(2)
    return A @ z + B @ u


def g(z: np.ndarray, u: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    # Illustrative structural drift target: move Theta toward |z| statistics
    target = np.abs(z)
    return target - Theta


def V(z: np.ndarray, Theta: np.ndarray) -> float:
    # Example Lyapunov-like energy: mismatch between Theta and |z|
    return 0.5 * float(np.linalg.norm(np.abs(z) - Theta) ** 2)


def perceptual_gravity(S: float, alpha: float = 1.0, beta: float = 2.0) -> float:
    # gamma_t = 1 + alpha * tanh(beta * S), bounded >= 1
    return 1.0 + alpha * float(np.tanh(beta * S))


def synthetic_dopamine(delta: float, Sigma: float, xi: float = 1e-6, tau_th: float = 0.5) -> float:
    # D_t = sigmoid(delta/(Sigma+xi) - tau_th), bounded in [0,1]
    x = delta / (Sigma + xi) - tau_th
    return 1.0 / (1.0 + float(np.exp(-x)))


def main() -> None:
    dt = 0.01
    T = 5.0
    steps = int(T / dt)

    z = np.array([1.0, -0.5], dtype=float)
    Theta = np.array([0.0, 0.0], dtype=float)
    u = np.array([0.0, 0.0], dtype=float)

    epsilon_base = 0.05

    energies = []
    residuals = []  # for a crude uncertainty proxy

    for _ in range(steps):
        # --- fast dynamics
        z = z + dt * f(z, u, Theta)

        # --- compute energy + proxies
        E = V(z, Theta)
        energies.append(E)

        delta = float(np.linalg.norm(np.abs(z) - Theta))  # "surprise" proxy
        residuals.append(delta)
        Sigma = float(np.var(residuals[-200:])) if len(residuals) >= 10 else 1.0  # crude uncertainty proxy

        # --- optional modulators
        gamma_t = perceptual_gravity(S=E, alpha=1.0, beta=2.0)
        D_t = synthetic_dopamine(delta=delta, Sigma=Sigma, tau_th=0.5)

        epsilon = epsilon_base * gamma_t

        # --- slow dynamics (modulated)
        Theta = Theta + dt * (epsilon * D_t) * g(z, u, Theta)

    print("Initial energy:", energies[0])
    print("Final energy:", energies[-1])
    print("First 10 energies:", energies[:10])


if __name__ == "__main__":
    main()
```

Run:

```bash
python examples/quickstart_toy_system.py
```

In a real LISA deployment, $f$, $g$, $\Psi$, and $V$ are problem-specific. The pattern remains: explicit fast–slow dynamics, manifold error, and stability-aligned adaptation, optionally regulated by bounded self-governing gains.

## Repository Structure

```
lisa/
├── __init__.py
├── dynamics.py      # fast dynamics f(z, u, Theta)
├── adaptation.py    # slow updates g(z, u, Theta) (+ modulators)
├── energy.py        # Lyapunov / energy functions V(z, Theta)
└── simulation.py    # utilities for simulating fast–slow systems

examples/
└── quickstart_toy_system.py

experiments/         # scripts, configs, logs
tests/
├── test_energy.py
└── test_dual_timescale.py

pyproject.toml
LICENSE
README.md
```

## Testing

```bash
pytest
```

Recommended tests include:

- Verify $V$ decreases (or remains bounded) on simple systems.
- Check numerical stability under small perturbations in $z$, $u$, $\Theta$.
- Validate that $\epsilon_{\text{base}}$ controls timescale separation, and that $\gamma_t$, $\mathcal{D}_t$ remain bounded.

## Citation

If you use LISA or build on this framework, please cite the technical report:

```
Latent Invariant Space Adaptation (LISA): A Dual-Timescale Framework for Robust Adaptive Control 
(with self-governing modulators), Technical Report, 2025–2026.
```

---

**Note:** If you want, I can also write:
- a tight **"Design Philosophy"** section (3–5 bullets) that matches your voice, and/or
- a **README "Roadmap"** that maps planned repo modules (e.g., `psi.py`, `regressors.py`, `gsp_theory.md`) to the paper sections.

# LISA: Latent Invariant Space Adaptation
### A Dual-Timescale Framework for Robust Adaptive Control

> “Robust system performance is not merely a function of error minimization, but rather the result of maintaining a low-dimensional attracting manifold.” 
> — **LISA Technical Report**

## Overview

**LISA** is a control-theoretic architecture for high-dimensional, non-stationary environments. Instead of treating a model as a static function trained once and frozen, LISA models the agent as a **singularly perturbed dynamical system** with:

1.  **Fast behavioral dynamics** (what the system is doing right now).
2.  **Slow structural dynamics** (how the underlying representation and geometry are adapting).

### The Core Idea
* **Fast latent states** $z(t)$ evolve continuously under current structure $\Theta(t)$ and input $u(t)$.
* **Slow structural parameters** $\Theta(t)$ evolve under a small perturbation parameter $\epsilon$, driven by violation of a **Lyapunov-style energy function** $V(z, \Theta)$.

The system is designed so that structural updates reduce energy and reconstruct an invariant manifold, yielding robustness even under distribution drift. LISA is intended as a general template for adaptive controllers, structured representation learning, and long-horizon agents operating in changing environments.

---

## Mathematical Formulation

The system evolves on two explicitly separated time scales $(t, \tau)$.

### 1. Fast State Dynamics (Behavioral Layer)
Fast latent state $z$ evolves according to:

$$
\frac{dz}{dt} = f(z, u, \Theta)
$$

Where:
* $z$: Fast latent state (behavior, beliefs, or internal representation)
* $u$: Control input or external signal
* $\Theta$: Structural parameters (geometry, invariants, slow weights)
* $f$: Vector field defining the fast dynamics

### 2. Slow Structural Dynamics (Structural Layer)
Structural parameters $\Theta$ evolve on a slower time scale:

$$
\frac{d\Theta}{dt} = \epsilon \cdot g(z, u, \Theta)
$$

With:
* $\epsilon > 0$: Small (time-scale separation parameter)
* $g$: Structural update field (plasticity / adaptation rule)

The small parameter $\epsilon$ enforces a **dual-timescale separation**: $z$ reacts quickly, $\Theta$ adapts slowly.

### 3. Invariance and Lyapunov Energy
LISA assumes the existence of a Lyapunov-style energy function:

$$
V(z, \Theta) \ge 0
$$

This is interpreted as an invariant manifold reconstruction error or “distance to a stable regime.” A common abstract form of the structural update law is a steepest-descent type rule on this energy landscape:

$$
\frac{d\Theta}{dt} = -\Gamma \phi(z, u) \eta^T
$$

Where:
* $\Gamma$: Positive-definite gain matrix (learning / adaptation gain)
* $\phi(z, u)$: Feature or regressor vector derived from state and input
* $\eta$: Manifold reconstruction error (e.g., deviation from an invariant relationship)

Under appropriate conditions, this update drives $\eta \to 0$ (as $t \to \infty$), meaning the system converges to a low-dimensional attracting manifold even as the environment drifts.

---

## Key Features

* **Dual-timescale separation:** Explicit modeling of $\epsilon$ allows singular perturbation analysis and separates fast behavior from slow structural change.
* **Lyapunov-driven adaptation:** Structural updates are derived to reduce an energy function, providing a principled alternative to heuristic learning rates. Adaptation is aligned with stability guarantees, not just gradient descent on a static loss.
* **Continuous-time learning:** LISA is formulated as a continuous-time flow:
    * No “epochs” or “batches” are required conceptually.
    * Learning proceeds in parallel with operation.
    * Suitable for streaming signals and online control.
* **Robustness under drift:** By continuously reshaping the latent manifold (through $\Theta$) in response to reconstruction error $\eta$, LISA aims to maintain performance and stability across regime changes and distribution shift.

---

## Installation

If you are using the LISA Python implementation:

```bash
git clone [https://github.com/vishal-1344/lisa.git](https://github.com/vishal-1344/lisa.git)
cd lisa
pip install -e .
````

or

```bash
pip install -r requirements.txt
```

Then, in Python:

```python
import lisa
```

-----

## Quickstart: Toy Dual-Timescale System

A minimal example illustrating fast–slow dynamics and a decreasing energy function.

Create `examples/quickstart_toy_system.py`:

```python
""" 
Quickstart: LISA-style dual-timescale dynamics on a toy system.
This example illustrates:
1. fast state dynamics dz/dt = f(z, u, Theta)
2. slow structural dynamics dTheta/dt = epsilon * g(z, u, Theta)
3. decreasing Lyapunov-like energy V(z, Theta) 
"""

import numpy as np

def f(z, u, Theta): 
    # Simple linear fast dynamics: z_dot = A z + B u, where A depends on Theta 
    A = np.array([[Theta[0], 0.0], [0.0, Theta[1]]]) 
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
```

Run:

```bash
python examples/quickstart_toy_system.py
```

In a full LISA deployment, $f$, $g$, and $V$ will be problem-specific and derived from your control / modeling setup, but the pattern remains: explicit fast–slow dynamics, structurally meaningful updates, and energy decreasing over time.

-----

## Repository Structure

A typical layout for the LISA codebase might be:

```text
lisa/ 
├── __init__.py 
├── dynamics.py      # fast dynamics f(z, u, Theta) 
├── adaptation.py    # slow structural updates g(z, u, Theta) 
├── energy.py        # Lyapunov / energy functions V(z, Theta) 
└── simulation.py    # utilities for simulating fast–slow systems

examples/ 
└── quickstart_toy_system.py 

experiments/         # experiment scripts, configs, logs

tests/ 
├── test_energy.py 
└── test_dual_timescale.py

pyproject.toml 
LICENSE 
README.md
```

-----

## Testing

If tests are provided, they can be run via:

```bash
pytest
```

Recommended tests include:

1.  Verifying that the energy $V(z, \Theta)$ decreases along simulated trajectories for simple systems.
2.  Checking numerical stability under small perturbations in $z$, $u$, and $\Theta$.
3.  Validating that $\epsilon$ correctly controls timescale separation (fast vs slow dynamics).

-----

## Citation

If you use LISA or build on this framework, please cite the technical report:

> **Latent Invariant Space Adaptation (LISA): A Dual-Timescale Framework for Robust Adaptive Control**, December 2025.

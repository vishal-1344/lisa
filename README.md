# LISA: Latent Invariant Space Adaptation

### A Relativistic, Self-Governing Dynamical Architecture  
*Built on a dual-timescale framework for manifold-stable adaptive control*

> "Robust system performance is not merely a function of error minimization, but rather the result of maintaining a low-dimensional attracting manifold."  
> **LISA Technical Report**

## Overview

**LISA** is a control-theoretic architecture for high-dimensional, non-stationary environments. Rather than treating a model as a static function trained once and frozen, LISA models an agent as a **singularly perturbed dynamical system** with:

1. **Fast behavioral dynamics**: what the system is doing right now.
2. **Slow structural dynamics**: how the underlying representation/manifold adapts to remain stable under drift.

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

In highly stochastic environments, always on adaptation can waste plasticity on noise, while overly slow adaptation can fail under genuine drift. LISA optionally adds two *bounded modulators* that regulate **how much** and **when** the slow update runs without changing the underlying Lyapunov direction.

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

B) Synthetic Dopamine (Epistemic Plasticity Gating)

Synthetic dopamine is a directional, epistemic plasticity gate that governs when learning is legitimate, rather than serving as a reward signal.

A bounded gate suppresses learning under likely noise and enables learning under reliable novelty that coherently advances the system toward its internal objective boundary:

𝐷
𝑡
=
𝜎
 ⁣
(
𝛿
(
𝑡
)
Σ
(
𝑡
)
+
𝜉
−
𝜏
th
)
,
𝐷
𝑡
∈
[
0
,
1
]
D
t
	​

=σ(
Σ(t)+ξ
δ(t)
	​

−τ
th
	​

),D
t
	​

∈[0,1]

Where:

𝛿
(
𝑡
)
δ(t) — surprise proxy
(e.g., 
∥
𝜂
(
𝑡
)
∥
∥η(t)∥, prediction residual, or manifold deviation)

Σ
(
𝑡
)
Σ(t) — uncertainty proxy
(running variance, learned uncertainty head, etc.)

𝜉
>
0
ξ>0 — numerical stabilizer

𝜏
th
τ
th
	​

 — novelty threshold

This gate establishes eligibility for adaptation, suppressing noise-driven plasticity while permitting learning under structured, reliable surprise.

Crucially, synthetic dopamine is interpreted not as novelty alone, but as a carrier of purpose:
it activates most strongly when multiple, independently ambiguous internal factors (e.g., features, residual components, predictive cues) converge directionally toward the same objective trajectory, and when such convergence persists over time as causal progress.

Unified Modulated Slow Law

The self-governing slow update retains the same stability-driven direction, with modulation applied only to when and how strongly adaptation occurs:

Θ
˙
=
−
𝜖
base
  
𝛾
𝑡
  
𝐷
𝑡
  
Γ
  
𝜙
(
𝑧
,
𝑢
)
  
𝜂
𝑇
Θ
˙
=−ϵ
base
	​

γ
t
	​

D
t
	​

Γϕ(z,u)η
T

Where:

𝛾
𝑡
γ
t
	​

 — intensity / timescale modulation (perceptual gravity)

𝐷
𝑡
D
t
	​

 — epistemic validity and directional coherence gate

Interpretation:
the update direction remains Lyapunov-aligned and stability-preserving, while synthetic dopamine schedules adaptation legitimacy, enabling learning only when novelty is reliable, coherent across signals, and aligned with sustained progress toward the system’s objective boundary.
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

Synthetic dopamine is a directional, epistemic plasticity gate that governs when learning is legitimate, rather than serving as a reward signal.

A bounded gate suppresses learning under likely noise and enables learning under reliable novelty that coherently advances the system toward its internal objective boundary:

$\mathcal{D}_t = \sigma\!\left( \frac{\delta(t)}{\Sigma(t)+\xi} - \tau_{\text{th}} \right), \qquad \mathcal{D}_t \in [0,1]$

- $\delta(t)$: surprise proxy (e.g., $\|\eta(t)\|$, prediction residual, or manifold deviation)
- $\Sigma(t)$: uncertainty proxy (running variance, learned uncertainty head, etc.)
- $\xi>0$: numerical stabilizer
- $\tau_{\text{th}}$: novelty threshold

This gate establishes eligibility for adaptation, suppressing noise-driven plasticity while permitting learning under structured, reliable surprise.

Crucially, synthetic dopamine is interpreted not as novelty alone, but as a carrier of purpose: it activates most strongly when multiple, independently ambiguous internal factors (e.g., features, residual components, predictive cues) converge directionally toward the same objective trajectory, and when such convergence persists over time as causal progress.

### Unified Modulated Slow Law

The self-governing slow update retains the same stability-driven direction, with modulation applied only to when and how strongly adaptation occurs:

$\dot{\Theta} = - \epsilon_{\text{base}} \; \gamma_t \; \mathcal{D}_t \; \Gamma \; \phi(z,u) \; \eta^{\mathsf{T}}$

- $\gamma_t$: intensity / timescale modulation (perceptual gravity)
- $\mathcal{D}_t$: epistemic validity and directional coherence gate

Interpretation: the update direction remains Lyapunov-aligned and stability-preserving, while synthetic dopamine schedules adaptation legitimacy, enabling learning only when novelty is reliable, coherent across signals, and aligned with sustained progress toward the system's objective boundary.

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


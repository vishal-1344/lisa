# *LISA: *Latent Invariant Space Adaptation**

## Overview

**LISA** is a control-theoretic architecture for high-dimensional, non-stationary environments. The *controller* treats inference as a dynamical process and introduces feedback mechanisms that regulate how internal trajectories evolve under uncertainty and distribution shift. In this sense, *LISA* can be viewed as an **inference-time control** and **meta-optimization** framework, looking over optimizing task loss to regulating how computation and adaptation unfold so the system maintains stability and remains coherent as conditions change.

*LISA* models an agent as a **singularly perturbed** dynamical system with:

1. **Fast behavioral dynamics**: what the system does in the present.
2. **Slow structural dynamics**: how the underlying structure shaping the behavior is perceived and regulated to remain stable under drift.

The central idea is **manifold stability**: maintain a low-dimensional *attracting* latent manifold embedded in a high-dimensional state space so that inference trajectories remain coherent when inputs, statistics, or regimes shift.

Crucially, it is not a training-time method, but a geometrically structured adaptation of inference and regulation using parallel signal analysis, enabling robust behavior in non-stationary settings.

Moreover, *LISA* uses an **interpretive layer** that treats invariants as provisional hypotheses and derives meaning from their mutual geometric coherence over time. This layer prevents **causal fixation**: repeated regularities are not automatically treated as permanent truths. Instead, the influence of learned structure is continuously modulated by coherence signals, enabling flexible bias alteration under abrupt regime change. Operationally, coherence functions as a trust signal that modulates commitment strength, rather than introducing symbolic reasoning or explicit causal graphs.

---

## Core Idea - Dual Timescales + Manifold Stability

- **Fast latent state** $z(t)$ evolves continuously under current structure $\Theta(t)$ and input $u(t)$.
- **Slow structural parameters** $\Theta(t)$ adapt under a small timescale parameter $\epsilon$, derived from (or aligned with) a **Lyapunov-style energy** $V(z,\Theta)$.

Structural updates are designed to reduce a global energy or tension metric and restore *invariance* or *attraction* of the latent manifold, yielding robustness under distribution drift.

### Interpretive Layer: Meaning as Invariant Coherence

Stability alone does not guarantee rational behavior under non-stationarity. In practice, a system can observe a pattern for a long time and incorrectly treat it as universal. *LISA* therefore treats invariants as monitored hypotheses. **Meaning is defined operationally as the degree to which multiple invariant candidates remain mutually coherent as the latent dynamics evolve.**

Mutual coherence is evaluated geometrically (rather than symbolically) through persistent agreement across dynamical signatures such as frequency structure, oscillatory coupling, phase alignment, temporal consistency, and shared response under perturbation. When coherence is high, the system can commit strongly to the induced structure. When coherence degrades, the system relaxes its commitment without requiring retraining or overwriting parameters, yielding flexible bias alteration rather than permanent fixation.

---

## Mathematical Formulation

*LISA* evolves on two explicitly separated timescales $(t,\tau)$.

### 1) Fast State Dynamics (Behavioral Layer)

$$\dot{z} = f(z, u, \Theta)$$

- $z$: fast latent state (behavior, beliefs, internal representation)
- $u$: external input or control signal
- $\Theta$: structural parameters (geometry, invariants, slow weights)
- $f$: vector field for fast dynamics

### 2) Slow Structural Dynamics (Structural Layer)

$$\dot{\Theta} = \epsilon \, g(z, u, \Theta)$$

- $\epsilon>0$: small timescale separation parameter
- $g$: structural update field (plasticity or adaptation rule)

The small $\epsilon$ enforces **fast reaction, slow adaptation**.

### 3) Invariance + Manifold Error

A common way to express off-manifold deviation is:

$$\eta = z - \Psi(u,\Theta)$$

where $\Psi(u,\Theta)$ estimates the manifold location for input $u$.

### 4) Lyapunov-Driven Structural Update (Canonical Form)

*LISA* derives slow adaptation from a Lyapunov argument so that a composite energy decreases:

$$V_{\text{total}}(z,\Theta)=\frac{1}{2} \eta^T\eta+\frac{1}{2}\mathrm{tr}(\tilde{\Theta}^T\Gamma^{-1}\tilde{\Theta})$$

A canonical Lyapunov-aligned update is:

$$\dot{\Theta} = -\Gamma\,\phi(z,u)\,\eta^T$$

- $\Gamma$: positive-definite adaptation gain
- $\phi(z,u)$: regressor or features
- $\eta$: manifold reconstruction error

Under appropriate conditions, this yields boundedness (and often **UUB** in non-ideal settings).

### 5) Invariant Set and Coherence Signal

Let

$$\mathcal{I}(t) = \{I_k(t)\}_{k=1}^K$$

denote a set of latent invariant candidates derived from the dynamics of $z(t)$ and $\Theta(t)$ (e.g., mode-wise manifold deviations, spectral structure of latent trajectories, energy partitions, cross-timescale consistency checks, or other invariant-like signals).

Define a coherence functional:

$$\mathcal{C}(t) \in [0,1]$$

that measures mutual agreement among these invariant signals. Coherence increases when invariant candidates remain compatible across time and frequency (oscillatory structure, phase coupling, persistence), and decreases when they conflict or drift apart. **Coherence collapse is treated as evidence for regime shift, anomaly, or structural mismatch**, even when individual invariants appear locally stable.

---

## Optional Self-Governing Extensions

In highly stochastic environments, always-on adaptation can waste plasticity on noise, while overly slow adaptation can fail under genuine drift. *LISA* optionally adds bounded modulators that regulate **how much** and **when** the slow update runs without changing the underlying Lyapunov direction.

### A) Perceptual Gravity (State-Dependent Timescale Dilation)

A bounded "stress gain" increases adaptation intensity when system energy rises:

$$\gamma_t = 1+\alpha\tanh(\beta\,\mathcal{S}(t)), \quad \gamma_t \ge 1$$

$$\epsilon(t)=\epsilon_{\text{base}}\gamma_t$$

Typical choices: $\mathcal{S}(t)=V_{\text{total}}(t)$ or $\mathcal{S}(t)=\|\eta(t)\|^2$.

### B) Synthetic Dopamine (Epistemic Plasticity Gating)

Synthetic dopamine is a directional, epistemic plasticity gate that governs when learning is legitimate, rather than serving as a reward signal.

A bounded gate suppresses learning under likely noise and enables learning under reliable novelty that coherently advances the system toward its internal objective boundary:

$$\mathcal{D}_t = \sigma\!\left( \frac{\delta(t)}{\Sigma(t)+\xi} - \tau_{\text{th}} \right), \qquad \mathcal{D}_t \in [0,1]$$

- $\delta(t)$: surprise proxy (e.g., $\|\eta(t)\|$, prediction residual, or manifold deviation)
- $\Sigma(t)$: uncertainty proxy (running variance, learned uncertainty head, etc.)
- $\xi > 0$: numerical stabilizer
- $\tau_{\text{th}}$: novelty threshold

This gate establishes eligibility for adaptation, suppressing noise-driven plasticity while permitting learning under structured, reliable surprise.

Crucially, synthetic dopamine is interpreted not as novelty alone, but as a carrier of purpose: it activates most strongly when multiple, independently ambiguous internal factors (e.g., features, residual components, predictive cues) converge directionally toward the same objective trajectory, and when such convergence persists over time as causal progress.

### C) ERROR-360 (Multi-Perspective Geometric Diagnostics)

**ERROR-360** is a fast diagnostic layer that monitors the latent dynamics from multiple independent geometric perspectives. Instead of treating error as a single scalar, it exposes structured deviation signals that detect drift, oscillatory inharmony, instability, and incoherence early.

Let ERROR-360 produce a diagnostic vector:

$$e(t) = [e_1(t), \ldots, e_M(t)]$$

from which coherence can be derived:

$$\mathcal{C}(t) = h(e(t)) \in [0,1]$$

This prevents uncontrolled drift while allowing *LISA*'s slow dynamics to continue consolidating stable latent structure. In effect, ERROR-360 guards the trajectory while *LISA* shapes the structure.

### Unified Modulated Slow Law (with Coherence)

The self-governing slow update retains the same stability-driven direction, with modulation applied only to when and how strongly adaptation occurs:

$$\dot{\Theta} = - \epsilon_{\text{base}} \; \gamma_t \; \mathcal{D}_t \; \mathcal{C}(t) \; \Gamma \; \phi(z,u) \; \eta^{\mathsf{T}}$$

- $\gamma_t$: intensity or timescale modulation (perceptual gravity)
- $\mathcal{D}_t$: epistemic validity and directional coherence gate (synthetic dopamine)
- $\mathcal{C}(t)$: coherence-based trust modulation (invariant compatibility or ERROR-360 aggregation)

**Interpretation**: the update direction remains Lyapunov-aligned and stability-preserving, while the gates schedule adaptation legitimacy and trust. When coherence is high, the system can commit strongly. When coherence degrades, the system relaxes bias and avoids causal fixation, enabling flexible bias alteration under abrupt change.

---

## Key Features

- **Dual-timescale separation**: explicit $\epsilon$ enables singular perturbation analysis and cleanly separates fast behavior from slow structure.
- **Manifold-stable adaptation**: updates aim to maintain (or restore) an attracting invariant manifold under drift.
- **Lyapunov-driven learning**: structural updates are aligned with energy decrease and boundedness, not heuristic learning rates.
- **Continuous-time operation**: conceptually no epochs or batches required.
- **(Optional) Self-governance**: Perceptual Gravity and Synthetic Dopamine provide bounded gain scheduling to avoid learning-on-noise while responding to genuine regime change.
- **Flexible bias alteration**: strong latent structure is permitted, but its influence is continuously modulated by coherence, preventing permanent fixation while preserving adaptive meaning.

---

## Installation

```bash
git clone https://github.com/vishal-1344/*LISA*.git
cd *LISA*
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

Then:

```python
import *LISA*
```

## Quickstart: Toy Dual-Timescale System (with Optional Modulators)

Create `examples/quickstart_toy_system.py`:

```python
"""
Quickstart: *LISA*-style dual-timescale dynamics on a toy system.

Illustrates:
1) fast state dynamics:     dz/dt = f(z, u, Theta)
2) slow structural dynamics dTheta/dt = epsilon * g(z, u, Theta)
3) Lyapunov-like energy     V(z, Theta)
4) optional modulators:     gamma_t (Perceptual Gravity), D_t (Synthetic Dopamine)
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
        # fast dynamics
        z = z + dt * f(z, u, Theta)

        # compute energy + proxies
        E = V(z, Theta)
        energies.append(E)

        delta = float(np.linalg.norm(np.abs(z) - Theta))  # "surprise" proxy
        residuals.append(delta)
        Sigma = float(np.var(residuals[-200:])) if len(residuals) >= 10 else 1.0  # crude uncertainty proxy

        # optional modulators
        gamma_t = perceptual_gravity(S=E, alpha=1.0, beta=2.0)
        D_t = synthetic_dopamine(delta=delta, Sigma=Sigma, tau_th=0.5)

        epsilon = epsilon_base * gamma_t

        # slow dynamics (modulated)
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

In a real *LISA* deployment, $f$, $g$, $\Psi$, $V$, $\mathcal{I}(t)$, and $\mathcal{C}(t)$ are problem-specific. The pattern remains: explicit fast-slow dynamics, manifold error, stability-aligned adaptation, and a coherence-modulated legitimacy schedule that prevents drift while preserving meaningful structure.

## Repository Structure

```
*LISA*/
├── __init__.py
├── dynamics.py      # fast dynamics f(z, u, Theta)
├── adaptation.py    # slow updates g(z, u, Theta) (+ modulators)
├── energy.py        # Lyapunov / energy functions V(z, Theta)
├── coherence.py     # invariant coherence C(t) from I(t) / ERROR-360 diagnostics
└── simulation.py    # utilities for simulating fast-slow systems

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
- Validate that $\epsilon_{\text{base}}$ controls timescale separation, and that $\gamma_t$, $\mathcal{D}_t$, and $\mathcal{C}(t)$ remain bounded.
- Stress-test regime shifts: confirm coherence collapse precedes unstable adaptation and that bias is relaxed rather than fixated.

## Citation

If you use *LISA* or build on this framework, please cite the technical report:

```
Latent Invariant Space Adaptation (*LISA*): A Dual-Timescale Framework for Robust Adaptive Control 
(with self-governing modulators and coherence-based interpretation), Technical Report, 2025-2026.
```

































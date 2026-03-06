# LISA: Latent Invariant Space Adaptation

LISA converts static neural inference into a closed-loop, dual-timescale dynamical system that guarantees latent manifold stability under environmental non-stationarity.

---

## Overview

**LISA** is a control-theoretic architecture designed for high-dimensional, non-stationary environments. In this framework, neural inference is interpreted not as a static feedforward computation but as a trajectory evolving through a high-dimensional latent state space. The latent state $z_t \in \mathbb{R}^n$ denotes the internal representation vector produced by the base model at inference step $t$. LISA regulates the evolution of $z_t$ as a non-autonomous dynamical process, introducing closed-loop feedback mechanisms that enforce stability and structural coherence.

Contemporary foundation models typically execute open-loop inference governed by static boundary constraints. Because these architectures lack internal mechanisms to estimate and compensate for latent state momentum induced by environmental dynamics, their structural guarantees routinely degrade under domain drift. The **environment** refers to any external process influencing inference dynamics, including data distribution shifts, interaction loops, physical systems, or simulation dynamics. LISA addresses this by functioning as an **inference-time control and meta-optimization framework**. Operating above the task loss, the system actively probes the environment to synthesize dynamic safety boundaries while continuously warping the latent representation to maintain structural homeostasis.

LISA models the agent as a **singularly perturbed dynamical system**. This separation induces a two-timescale dynamical system in which the fast inference dynamics evolve on the operational timescale $t$, while structural adaptation evolves on the slower meta-timescale $\tau = \epsilon t$, where $0 < \epsilon \ll 1$. Biological terminology used herein serves as a functional analogy for formally defined control mechanisms governing latent trajectory regulation.

---

## Architecture Flow

LISA does not replace the base model; it regulates the geometry of the latent trajectory produced during inference. The framework requires read-access to the intermediate latent states $z_t$ and nominal outputs $u_t$ at each step but does not require modification of the underlying weights.

```
flowchart TD
    x[Observation (x_t)] --> BM[Base Model (Read-Access Only)]
    BM --> z_nom[u_nominal, z_t]
    z_nom --> LISA[LISA: Closed-Loop Inference Controller]
    LISA --> UF[u_filtered (Safe, Coherent Action)]
    UF --> ENV[Environment]

    subgraph LISA_module["LISA Internals (Fast t / Slow τ)"]
        LISA --> P[1. probers: Reachability & Momentum (BRT)]
        LISA --> E[2. energy: Meta-Proprioception V(z)]
        LISA --> CBF[3. cbf: Dynamic Safety Filter h(z)]
        LISA --> COH[4. coherence: ERROR-360 & Epistemic Gating]
        LISA --> ADAP[5. adaptation: Slow Manifold Warping (Theta)]
    end
```

---

## Notation

| Symbol | Definition | Timescale | Role |
|--------|------------|-----------|------|
| $z_t \in \mathbb{R}^n$ | Latent state vector | Fast ($t$) | Observed |
| $u_t$ | Nominal output or action vector | Fast ($t$) | Observed |
| $\Theta$ | Slow structural parameters | Slow ($\tau$) | Adapted |
| $\eta$ | Manifold reconstruction error | - | Diagnostic |
| $V$ | Lyapunov energy functional | - | Diagnostic |
| $Z_{crit}$* | Safe latent region boundary | - | Invariant |
| $\lambda$ | Momentum coefficient | - | Parameter |
| $\Psi(u, \Theta)$ | Slow manifold reconstruction mapping | Slow ($\tau$) | Model |

*Note: $Z_{crit}$ is empirically estimated using Backward Reachable Tube (BRT) probes.

---

## Concept Summary

LISA transforms neural inference into a feedback-regulated dynamical process. The base model produces a latent trajectory $z_t$. LISA monitors this trajectory, estimates environmental momentum, enforces safety via control barrier functions, and slowly adapts the latent coordinate system to maintain invariant structure. The resulting system couples fast inference dynamics with slow structural adaptation under provable boundedness guarantees.

---

## Core Mechanics: The Four Pillars of Inference-Time Control

### I. Online Reachability Estimation (Active System Identification)

To operate without the assumption of a fixed latent topology, the system utilizes **active system identification**. Prior to macro-execution, or upon detecting a regime shift, the controller executes an empirical micro-probe to map the boundary of the environment's **Backward Reachable Tube (BRT)**. By observing the passive evolution of latent state velocities after control shutdown, the architecture calculates the empirical supremum of environmental momentum via a multidimensional norm bound:

$$\Delta z_{coast}^{max} = \sup_{\tau \ge t_{off}} \left\| \int_{t_{off}}^\tau \dot{z}(\xi)d\xi \right\|$$

The latent velocity $\dot{z}_t$ is estimated via finite differences across inference steps, such that $\dot{z}_t \approx (z_t - z_{t-1}) / \Delta t$.

### II. Inference-Time Regulation and Architectural Reflex Arcs

Utilizing the probed supremum, the controller extracts $\lambda$ to synthesize a **Velocity-Aware Control Barrier Function (CBF)**. This formulation acts as a first-order empirical bound on the reachable latent manifold:

$$h(z_t) = Z_{crit} - \|z_t + \lambda \dot{z}_t\| \ge 0$$

The **Architectural Reflex Arc** is an event-triggered control mechanism that intervenes when latent trajectories approach unsafe regions, enforcing a hard optimal control projection:

$$u_t = \arg\min_u \|u - u_{nominal}\| \quad \text{subject to} \quad h(z_t) \ge 0$$

### III. Latent-Space Meta-Proprioception (Manifold Stability)

**Latent-Space Meta-Proprioception** is an intrinsic monitoring mechanism that evaluates the geometric integrity of latent trajectories through low-overhead proxy energy functionals $V(z, \Theta)$. While the reflex arc operates on the fast timescale $t$, the architecture simultaneously executes slow parametric updates $\Theta$ on the timescale $\tau$. Rather than directly updating core network weights, the architecture structurally adapts the latent coordinate frame to minimize the manifold reconstruction error $\eta = z - \Psi(u, \Theta)$ via a Lyapunov-aligned structural update.

### IV. The Interpretive Layer (Coherence and Epistemic Gating)

The architecture introduces an **interpretive layer** that treats structural invariants as provisional hypotheses that must be continuously validated through trajectory coherence. This layer derives operational meaning from mutual geometric coherence over time via autonomous, bounded modulators:

- **Perceptual Gravity** ($\gamma_t$): A stress-amplified adaptation gain derived from the Lyapunov energy $V(z)$. It increases structural plasticity monotonically with latent trajectory instability.

- **Synthetic Dopamine** ($D_t$): An epistemic plasticity gate that modulates the adaptation rate $\epsilon$ by evaluating the signal-to-noise ratio of latent trajectory discrepancies.

- **ERROR-360** (Coherence $C(t)$): A geometric diagnostic that evaluates the multi-perspective consistency of latent trajectories through frequency analysis, phase-alignment, and oscillatory coupling.

---

## Theorem of Latent Manifold Stability (UUB Guarantee)

To formally guarantee that LISA prevents catastrophic drift, the architecture relies on **Lyapunov's Direct Method** to prove **Uniform Ultimate Boundedness (UUB)**. Let $\eta(t)$ represent the manifold deviation and $\tilde{\Theta}(t) = \Theta(t) - \Theta^*$ represent the structural parameter error. We define the composite Lyapunov energy functional:

$$V(\eta, \tilde{\Theta}) = \frac{1}{2}\eta^T P \eta + \frac{1}{2}\mathrm{tr}(\tilde{\Theta}^T \Gamma^{-1} \tilde{\Theta})$$

Where $P$ and $\Gamma$ are positive-definite weighting matrices controlling energy scaling. Assuming fast inference error dynamics $\dot{\eta} = A \eta + \phi(z,u) \tilde{\Theta} + d(t)$ where $A$ is Hurwitz such that $A^T P + P A = -Q$ for some $Q > 0$ (defining the Lyapunov decay rate), LISA executes the unmodulated adaptation law $\dot{\Theta} = - \Gamma \phi(z,u)^T P \eta$.

Energy dissipation ($\dot{V} < 0$) is strictly guaranteed whenever $\|\eta\| > \frac{2 \|P\| d_{max}}{\lambda_{min}(Q)}$, where $d_{max}$ is the bound on unmodeled environmental chaos. This proves the latent trajectory is mathematically trapped within a rigorously bounded geometric envelope. The deployed, modulated structural dynamics are governed by:

$$\dot{\Theta} = -\epsilon_{base} \gamma_t D_t C(t) \Gamma \phi(z,u)^T P \eta$$

---

## Installation

```bash
git clone https://github.com/vishal-1344/LISA.git
cd LISA
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

## Quickstart: Numerically Stable Toy Dual-Timescale System

This script illustrates the core singular perturbation mechanics of LISA. This toy example uses a 2D latent state and linear dynamics for demonstration purposes.

```python
"""
Quickstart: Numerically stable LISA dual-timescale dynamics.
Features:
1) Fast/Slow Separation (z: fast, Theta: slow)
2) Perceptual Gravity (gamma_t): Stress-Amplified Gain
3) Synthetic Dopamine (D_t): Epistemic Plasticity Gate
"""
from __future__ import annotations
import numpy as np

def f(z: np.ndarray, u: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    # dz/dt = A(Theta)z + Bu
    A = np.array([[Theta[0], 0.0], [0.0, Theta[1]]])
    B = np.eye(2)
    return A @ z + B @ u

def g(z: np.ndarray, u: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    # dTheta/dt target: Alignment with absolute latent magnitudes
    target = np.abs(z)
    return target - Theta

def V(z: np.ndarray, Theta: np.ndarray) -> float:
    # Lyapunov energy: Geometric mismatch between state and structure
    return 0.5 * float(np.linalg.norm(np.abs(z) - Theta) ** 2)

def main() -> None:
    dt, T = 0.01, 5.0
    steps = int(T / dt)
    z, Theta, u = np.array([1.0, -0.5]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    epsilon_base, Sigma, beta_ema = 0.05, 1.0, 0.95
    energies = []

    for step in range(steps):
        # Fast behavioral dynamics (Operational timescale t)
        z = z + dt * f(z, u, Theta)

        # Meta-proprioceptive stability audit (Energy V)
        E = V(z, Theta)
        energies.append(E)
        delta = float(np.linalg.norm(np.abs(z) - Theta)) # Manifold deviation (error)

        # Epistemic signal-to-noise estimation (EMA)
        Sigma = beta_ema * Sigma + (1 - beta_ema) * (delta ** 2)

        # Modulators: gamma (stress-gain) and D (plasticity gate)
        gamma_t = 1.0 + 1.0 * float(np.tanh(2.0 * E))
        D_t = 1.0 / (1.0 + float(np.exp(-(delta / (Sigma + 1e-6) - 0.5))))

        # Slow structural dynamics (Meta-timescale tau)
        epsilon = epsilon_base * min(gamma_t, 10.0)
        Theta = Theta + dt * (epsilon * D_t) * g(z, u, Theta)

        if step % 100 == 0:
            print(f"Step {step:4}: V={E:.4f} | gamma={gamma_t:.2f} | D={D_t:.2f}")

    print(f"\nFinal manifold error: {energies[-1]:.4f}")
    
    # Optional Visualization:
    # import matplotlib.pyplot as plt
    # plt.plot(energies)
    # plt.xlabel("Time step")
    # plt.ylabel("Lyapunov Energy V(z, Theta)")
    # plt.title("Meta-Proprioceptive Energy Decay")
    # plt.show()

if __name__ == "__main__":
    main()
```

---

## Repository Structure

```
LISA/
├── __init__.py
├── probers/         # Active System Identification & BRT reachability estimation
├── cbf/             # Dynamic Control Barrier Function synthesis & optimal projection
├── adaptation.py    # Slow structural updates g(z, u, Theta) & Lyapunov dynamics
├── dynamics.py      # Fast behavioral dynamics f(z, u, Theta)
├── energy.py        # Meta-proprioceptive proxy energy functionals V(z, Theta)
├── coherence.py     # ERROR-360 diagnostics, Perceptual Gravity, Synthetic Dopamine
└── simulation.py    # ODE integrators and singular perturbation utilities
```

---

## Citation

```
Latent Invariant Space Adaptation (LISA): Empirical Synthesis of Velocity-Aware Control Barrier Functions 
and Manifold Stability via Active Environmental Probing, Technical Report, 2025-2026.
```

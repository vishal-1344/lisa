# LISA: Latent Invariant Space Adaptation

*A Dual-Timescale Framework for Robust Adaptive Control*

This repository contains a minimal, research-grade implementation of the
**Latent Invariant Space Adaptation (LISA)** framework.

LISA models an agent as a singularly perturbed dynamical system that maintains
a **slowly evolving invariant manifold** in a high-dimensional latent space.
Fast dynamics perform inference/control on the manifold; slow dynamics adapt
the manifold geometry using Lyapunov-based structural updates.

## Mathematical Model

Let the composite state be χ = [zᵀ, Θᵀ]ᵀ, where

- z(t) ∈ ℝⁿ is the fast latent state,
- Θ(t) ∈ ℝᵖ are the structural parameters.

LISA is defined by the dual-timescale system

```math
\dot{z} = F(z, u; \Theta)
```

```math
\dot{\Theta} = \varepsilon\, G(z, u, \Theta, E), \qquad 0 < \varepsilon \ll 1.
```

The target invariant manifold is represented implicitly by Ψ(u, Θ), and the
**manifold reconstruction error** is

```math
\eta(t) = z(t) - \Psi(u(t), \Theta(t)).
```

A composite Lyapunov function couples manifold deviation and parameter error:

```math
V_{\text{total}}(z, \Theta)
= \frac{1}{2} \|\eta\|^2 + \frac{1}{2} \tilde{\Theta}^\top \Gamma^{-1} \tilde{\Theta},
```

where Γ ≻ 0 is a gain matrix and \tilde{\Theta} is the structural error.

Imposing (\dot{V}_{\text{total}} \le 0) yields the LISA structural update law

```math
\dot{\Theta} = - \varepsilon\, \Gamma\, \mathrm{vec}(\eta\, \phi(z, u)^\top),
```

which corresponds to a steepest descent on the Lyapunov energy while preserving
boundedness of (z, Θ).

## This Repository

* `lisa/core.py`
  Implements the coupled ODE system, including fast dynamics, slow dynamics,
  manifold error, and Lyapunov energy.

* `lisa/manifolds.py`
  Provides a simple linear manifold Ψ(u, Θ) = W u as a minimal working example.

* `experiments/drift_stabilization.py`
  Simulates a non-stationary environment where the **true manifold rotates
  over time**, and demonstrates that LISA adapts Θ online to track the drift.

The code is intentionally compact and heavily typed so that the mapping from
equations to implementation is easy to inspect at a glance.

## Dependencies

* `numpy`
* `scipy`
* `matplotlib`

Install via:

```bash
pip install -r requirements.txt
```

## Reference

If you use this code, please cite the associated technical report:

> *Latent Invariant Space Adaptation (LISA): A Dual-Timescale Framework for
> Robust Adaptive Control*, Technical Report, December 2025.# LISA: Latent Invariant Space Adaptation

**A Dual-Timescale Framework for Robust Adaptive Control**

> [cite_start]"Robust system performance is not merely a function of error minimization, but rather the result of maintaining a low-dimensional attracting manifold." — *LISA Technical Report* [cite: 10]

## Overview
[cite_start]LISA is a control-theoretic architecture designed for high-dimensional, non-stationary environments[cite: 7]. [cite_start]Unlike standard deep learning which relies on static optimization of fixed architectures [cite: 5][cite_start], LISA models the agent as a **singularly perturbed dynamical system**[cite: 11]. It decouples fast inference dynamics from slow structural adaptation dynamics.

## Mathematical Formulation
[cite_start]The system evolves on two distinct, mathematically separated timescales $(t, \tau)$[cite: 35]:

$$\dot{z} = f(z, u, \Theta) \quad \text{(Fast State Dynamics)}$$
$$\dot{\Theta} = \epsilon g(z, u, \Theta) \quad \text{(Slow Structural Dynamics)}$$

### The Invariance Condition
[cite_start]Adaptation is driven by the violation of the **Lyapunov Energy** function $V(z, \Theta)$[cite: 77]. [cite_start]The structural update law is derived as a steepest descent on the energy landscape[cite: 13]:

$$\dot{\Theta} = -\Gamma \phi(z,u)\eta^{T}$$

[cite_start]This ensures that the manifold reconstruction error $\eta \to 0$ asymptotically, providing a provable stability margin even during distribution drift[cite: 137].

## Key Features
- [cite_start]**Dual-Timescale Separation:** Explicit modeling of the $\epsilon$ perturbation parameter[cite: 54].
- [cite_start]**Lyapunov-Driven Updates:** Structural plasticity is mathematically guaranteed to reduce system energy[cite: 80].
- [cite_start]**Continuous Adaptation:** No "epochs" or "batches"; learning is a continuous-time flow[cite: 150].

## Citation
If you use this codebase, please cite the technical report:
[cite_start]*Latent Invariant Space Adaptation (LISA): A Dual-Timescale Framework for Robust Adaptive Control*, December 2025[cite: 1, 2].

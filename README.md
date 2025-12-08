# LISA: Latent Invariant Space Adaptation

**A Dual-Timescale Framework for Robust Adaptive Control**

> "Robust system performance is not merely a function of error minimization, but rather the result of maintaining a low-dimensional attracting manifold." — *LISA Technical Report*

## Overview
LISA is a control-theoretic architecture designed for high-dimensional, non-stationary environments. Unlike standard deep learning, which relies on the static optimization of fixed architectures, LISA models the agent as a singularly perturbed dynamical system. It explicitly decouples fast inference dynamics from slow structural adaptation dynamics.

## Mathematical Formulation
The system evolves on two distinct, mathematically separated timescales $(t, \tau)$:

$$
\dot{z} = f(z, u, \Theta) \quad \text{(Fast State Dynamics)}
$$
$$
\dot{\Theta} = \epsilon g(z, u, \Theta) \quad \text{(Slow Structural Dynamics)}
$$

### The Invariance Condition
Adaptation is driven by the violation of the Lyapunov Energy function $V(z, \Theta)$. The structural update law is derived as a steepest descent on the system's energy landscape:

$$
\dot{\Theta} = -\Gamma \phi(z,u)\eta^{T}
$$

This ensures that the manifold reconstruction error $\eta \to 0$ asymptotically, providing a provable stability margin even during distribution drift.

## Key Features

* **Dual-Timescale Separation:** Explicit modeling of the $\epsilon$ perturbation parameter allows for rigorous stability analysis.
* **Lyapunov-Driven Updates:** Structural plasticity is mathematically guaranteed to reduce system energy, replacing heuristic learning rates with stability guarantees.
* **Continuous Adaptation:** There are no "epochs" or "batches"; learning is a parallel, continuous-time flow.

## Citation
If you use this codebase, please cite the technical report:

> *Latent Invariant Space Adaptation (LISA): A Dual-Timescale Framework for Robust Adaptive Control*, December 2025.

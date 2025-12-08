# LISA: Latent Invariant Space Adaptation  

*A Dual-Timescale Framework for Robust Adaptive Control*

This repository contains a minimal, research-oriented implementation of the **Latent Invariant Space Adaptation (LISA)** framework.

LISA models an agent as a **singularly perturbed dynamical system** that maintains a slowly evolving invariant manifold in a high-dimensional latent space. Fast dynamics perform inference/control on the manifold; slow dynamics adapt the manifold geometry using Lyapunov-based structural updates.

---

## Mathematical Model

Let the composite state be
\[
\chi = \begin{bmatrix} z^\top & \Theta^\top \end{bmatrix}^\top
\]
where

- \( z(t) \in \mathbb{R}^n \) is the **fast latent state**  
- \( \Theta(t) \in \mathbb{R}^p \) are the **structural parameters**

LISA is defined by the dual-timescale system
\[
\dot{z} = F(z, u; \Theta), \qquad
\dot{\Theta} = \varepsilon\, G(z, u, \Theta, E), \qquad 0 < \varepsilon \ll 1.
\]

The target invariant manifold is represented implicitly by a mapping
\[
\Psi(u, \Theta)
\]
and the **manifold reconstruction error** is
\[
\eta(t) = z(t) - \Psi\big(u(t), \Theta(t)\big).
\]

A composite Lyapunov function couples manifold deviation and parameter error:
\[
V_{\text{total}}(z,\Theta)
=
\frac{1}{2}\,\|\eta\|^2
+
\frac{1}{2}\,\tilde{\Theta}^\top \Gamma^{-1} \tilde{\Theta},
\]
where \( \Gamma \succ 0 \) is a gain matrix and \( \tilde{\Theta} \) is the structural error.

Imposing \( \dot{V}_{\text{total}} \le 0 \) yields the LISA structural update law
\[
\dot{\Theta}
=
-\,\varepsilon\, \Gamma\, \text{vec}\!\big( \eta\,\phi(z,u)^\top \big),
\]
which corresponds to a **steepest descent** on the Lyapunov energy while preserving boundedness of \((z,\Theta)\) and enforcing manifold invariance.

---

## Conceptual Overview

LISA is a **control-theoretic architecture** designed for high-dimensional, non-stationary environments.

- It treats the agent as a **singularly perturbed** system with explicitly separated timescales \((t,\tau)\).
- The **fast subsystem** performs inference or control along a low-dimensional attracting manifold.
- The **slow subsystem** adapts the manifold geometry itself, driven by Lyapunov-based discrepancy signals.

Key differences from standard deep learning:

- Standard deep learning typically optimizes a fixed architecture via batch/epoch updates.
- LISA instead performs **continuous-time adaptation** of structural parameters so that:
  - Fast dynamics remain stable on the manifold.
  - The manifold itself deforms to track non-stationary structure (e.g., distribution drift, changing dynamics).

---

## This Repository

The goal of this repo is to expose the **mathematical mechanism** in a compact, readable codebase.

- `lisa/core.py`  
  Implements the coupled ODE system, including:
  - fast dynamics \( \dot{z} \),
  - slow structural dynamics \( \dot{\Theta} \),
  - manifold error \( \eta \),
  - Lyapunov energy \( V_{\text{total}} \).

- `lisa/manifolds.py`  
  Provides a simple **linear manifold**
  \[
  \Psi(u, \Theta) = W u
  \]
  (with \( W \) encoded in \( \Theta \)) as a minimal working example.

- `experiments/drift_stabilization.py`  
  Simulates a **non-stationary environment** where the true manifold **rotates over time**, and demonstrates that LISA adapts \( \Theta \) online to track the drift.

The code is intentionally compact and heavily typed so that the mapping from equations to implementation is easy to inspect at a glance.

---

## Dependencies

- `numpy`  
- `scipy`  
- `matplotlib`

Install via:

```bash
pip install -r requirements.txt


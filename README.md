# lisa
Latent Invariant Spaces Adaptation
# Latent Invariant Space Adaptation (LISA)

**Latent Invariant Space Adaptation (LISA)** is a dual-timescale dynamical framework for robust adaptive control in high-dimensional learned systems.

LISA treats an intelligent system not as a static function approximator, but as a **singularly perturbed dynamical system** with:

- **Fast latent states** \(z(t)\) governing inference and control.
- **Slow structural parameters** \(\Theta(t)\) governing latent geometry and invariants.

The goal is not only to minimize prediction error, but to maintain a **low-dimensional attracting manifold** – the *latent invariant space* – on which the fast dynamics remain stable even under non-stationary inputs and distribution shift.

---

## 1. Motivation

Modern learned systems face two structural limitations:

- Neural networks are typically trained in a **fixed representation space** and then deployed in non-stationary environments, leading to instability and interference.
- Classical adaptive control provides stability guarantees but assumes **hand-designed state spaces** and relatively simple dynamics.

LISA aims to bridge these worlds:

> **Control-theoretic adaptation of latent manifolds for learned systems.**

Instead of learning a representation once and hoping it generalizes, LISA continuously **adapts the latent structure itself** while preserving stability margins.

---

## 2. Mathematical formulation

We model an agent as a singularly perturbed system:

\[
\begin{aligned}
\dot{z} &= F(z, u; \Theta), \\
\dot{\Theta} &= \varepsilon\,G(z, u, \Theta), \quad 0 < \varepsilon \ll 1,
\end{aligned}
\]

where:

- \(z(t) \in \mathbb{R}^n\): fast latent state (inference, control, prediction),
- \(\Theta(t) \in \mathbb{R}^p\): slow structural parameters (weights, basis vectors, geometry),
- \(u(t)\): exogenous input / environment,
- \(F\): fast dynamics, \(G\): structural adaptation law.

### 2.1 Latent invariant manifold

For a fixed structure \(\Theta\), LISA assumes a target invariant manifold

\[
M_\Theta := \{ z \in \mathbb{R}^n \mid \Phi(z, \Theta) = 0 \},
\]

with invariance condition

\[
\nabla_z \Phi \cdot \dot{z} + \nabla_\Theta \Phi \cdot \dot{\Theta} = 0.
\]

When the environment shifts and trajectories drift off-manifold, the adaptation law \(\dot{\Theta}\) is designed so that the manifold deforms and **restores invariance** rather than simply chasing instantaneous error.

### 2.2 Lyapunov-driven structural updates

We use a composite Lyapunov function of the form

\[
V_{\text{total}}(z,\Theta) 
= \frac{1}{2}\,\| \eta \|^2 
+ \frac{1}{2}\,\mathrm{tr}\big(\tilde{\Theta}^\top \Gamma^{-1} \tilde{\Theta}\big),
\]

where:

- \(\eta\): manifold tracking error,
- \(\tilde{\Theta} = \Theta - \Theta^\ast\): structural deviation from a target geometry,
- \(\Gamma \succ 0\): adaptation gain matrix.

A typical LISA adaptation law is

\[
\dot{\Theta} = -\varepsilon\,\Gamma\,\nabla_\Theta L_{\text{recon}}(z,u,\Theta),
\]

chosen such that \(\dot{V}_{\text{total}} \le 0\). This turns structural learning into steepest descent on Lyapunov energy and yields:

- bounded state and parameter errors,
- uniform ultimate boundedness under approximation and perturbation.

---

## 3. Relation to existing frameworks

LISA is related to, but distinct from:

- **Predictive Coding / Active Inference**  
  Focus on minimizing prediction error / free energy via state updates and local plasticity rules.  
  LISA focuses on **manifold stability** and the geometry of the latent state space.

- **Immersion & Invariance (I&I) control**  
  Designs an invariant manifold and control law to render it attractive.  
  LISA extends this to **adaptive I&I**, where the manifold itself is learned and continuously deformed.

- **Meta-learning (e.g. MAML)**  
  Optimizes initial weights for fast task adaptation.  
  LISA emphasizes **continuous online adaptation** with explicit stability guarantees during adaptation.

---

## 4. Repository structure

Planned structure:

```text
Latent-Invariant-Spaces/
├─ paper/
│  ├─ lisa.tex      # LaTeX source of the technical report
│  ├─ lisa.pdf      # Compiled report
│  └─ figures/      # TikZ or image assets (optional)
├─ src/
│  ├─ lisa/
│  │  ├─ dynamics.py     # fast subsystem F(z,u;Θ)
│  │  ├─ adaptation.py   # slow subsystem Θ̇ = -Γ∇Θ V
│  │  └─ lyapunov.py     # Lyapunov utilities and analysis helpers
├─ examples/
│  ├─ simple_spsystem.ipynb       # toy singularly perturbed system
│  └─ manifold_tracking_demo.ipynb
├─ LICENSE
└─ README.md

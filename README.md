# Latent Invariant Space Adaptation (LISA)

Latent Invariant Space Adaptation (LISA) is a dual-timescale dynamical framework for robust adaptive control in high-dimensional learned systems.

LISA treats an intelligent system not as a static function approximator, but as a **singularly perturbed dynamical system** with:

- fast latent states z(t) governing inference and control, and  
- slow structural parameters Theta(t) governing latent geometry and invariants.

The goal is not only to minimize prediction error, but to maintain a **low-dimensional attracting manifold** – the “latent invariant space” – on which the fast dynamics remain stable even under non-stationary inputs and distribution shift.

---

## 1. Motivation

Modern learned systems face two structural limitations:

- Neural networks are typically trained in a fixed representation space and then deployed in non-stationary environments, leading to instability and interference.
- Classical adaptive control provides stability guarantees but assumes hand-designed state spaces and relatively simple dynamics.

LISA aims to bridge these worlds:

> Control-theoretic adaptation of latent manifolds for learned systems.

Instead of learning a representation once and hoping it generalizes, LISA continuously **adapts the latent structure itself** while preserving stability margins.

---

## 2. Mathematical formulation (ASCII form)

We model an agent as a singularly perturbed system:

```text
dot_z      = F(z, u; Theta)
dot_Theta  = eps * G(z, u, Theta),   with 0 < eps << 1



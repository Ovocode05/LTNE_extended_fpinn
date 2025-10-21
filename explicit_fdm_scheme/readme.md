# Explicit FDM Scheme for Coupled Fractional Diffusion

Implementation of an **explicit Finite Difference Method (FDM)** for a 2D **coupled fractional-in-time diffusion system**, commonly used to represent the **Local Thermal Non-Equilibrium (LTNE)** model.

The discretization uses the **L1 scheme** for the Caputo fractional time derivative and **second-order central differences** for the spatial Laplacian.  
As an explicit method, it is simple to implement but **conditionally stable** and computationally expensive due to the non-local history term.  

---

## Numerical Discretization

The explicit scheme is derived by discretizing the PDEs at time $t_n$ and solving for the unknown state at $t_{n+1}$.

### L1 Time Discretization

The Caputo fractional derivative $\frac{\partial^\alpha \theta}{\partial t^\alpha}$ at time $t_{n+1}$ is approximated by the L1 scheme.  
Rearranging the L1 formula to solve for $\theta^{n+1}$ explicitly gives the update structure:

$$
\frac{\partial^\alpha \theta}{\partial t^\alpha} \Big|_{n+1}
\approx
\frac{1}{\text{cofactor}}
\left[ (\theta^{n+1} - \theta^n) + \text{History}(n) \right]
$$

where:

$$
\text{cofactor} = \Gamma(2-\alpha)\* (\Delta t)^\alpha
$$

$$
\text{History}(n) = \sum_{k=1}^{n} c_k \* (\theta^{n-k+1} - \theta^{n-k})
$$

$$
c_k = (k+1)^{1-\alpha} - k^{1-\alpha}
$$

---

### Explicit Update Equations

Substituting the L1 scheme into the governing PDEs and solving for $\theta^{n+1}$ gives the update equations at each grid point $(i,j)$:

$$
\theta_{s,i,j}^{n+1} = \theta_{s,i,j}^n - \text{History}_s(n) + \frac{\text{cofactor}}{Fh_s} \left[ \nabla^2 \theta_{s,i,j}^n - \text{InterfaceTerm}_{i,j}^n \right]
$$

$$
\theta_{f,i,j}^{n+1} = \theta_{f,i,j}^n - \text{History}_f(n) + \frac{\text{cofactor}}{Fh_f} \left[ \nabla^2 \theta_{f,i,j}^n + \text{InterfaceTerm}_{i,j}^n \right]
$$

where  
$\nabla^2 \theta$ is approximated using **second-order central differences**, and  
the **interface term** models heat exchange between phases.

---

## Stability Condition

The explicit scheme is **conditionally stable**.  
The time step $\Delta t$ must satisfy the following conditions for both solid and fluid phases:

$$
\Delta t^\alpha
\leq
\frac{Fh_s}{
\Gamma(2-\alpha)
\left[
D_s \left(\frac{4}{\Delta x^2} + \frac{4}{\Delta y^2}\right) + 1
\right]}
$$

$$
\Delta t^\alpha
\leq
\frac{Fh_f}{
\Gamma(2-\alpha)
\left[
D_f \left(\frac{4}{\Delta x^2} + \frac{4}{\Delta y^2}\right) + 1
\right]}
$$

Violation of these conditions leads to numerical instability (manifested as `NaN` or overflow values).

> **Note:** $D_s$ and $D_f$ are the effective diffusivities for the solid and fluid domains.

> **Note:** Stability defined assuming linear relation (delta=0) and initial time step (n=0)

---

## Planned Improvements

1. **Fast L1 (SOE):**s 
   Replace the direct history sum with a *Sum-of-Exponentials* (SOE) approximation for $t^{-\alpha}$, reducing the per-step cost from $O(N_t)$ to $O(\log N_t)$ or $O(1)$.

3. **Implicit Scheme:**  
   Develop an implicit variant (planned under `fdm_solver/`) that is **unconditionally stable**, allowing larger time steps without violating stability constraints.
---

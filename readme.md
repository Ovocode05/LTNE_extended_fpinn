# FDM Solver for Coupled Fractional-in-Time Diffusion (LTNE Model)

This repository contains a Python-based Finite Difference Method (FDM) solver for a coupled, fractional-in-time diffusion system. It is designed to model Local Thermal Non-Equilibrium (LTNE) conduction, where the solid and fluid phases have distinct temperature fields.

The numerical scheme uses the **L1 finite-difference scheme** for the Caputo fractional time derivative and a **second-order central difference** scheme for the integer-order spatial diffusion (Laplacian).

## 1. Introduction and Motivation

Coupled diffusion models containing both integer-order and fractional-order time derivatives are widely used to describe multi-physics transport where classical diffusion fails. The integer (classical) diffusion part captures local, Markovian transport, while the fractional time derivative models nonlocal-in-time memory and anomalous diffusion.

Using a fractional derivative in time allows the model to:

- **Capture "Memory" Effects:** It accounts for the system's entire history, which is crucial for materials with complex microstructures.
- **Represent Anomalous Transport:** It can model sub-diffusion, where the mean-square displacement grows slower than linearly in time, often seen in porous media or trapping-dominant systems.
- **Interpolate Dynamics:** It provides a continuous bridge between purely local (integer-order, $\alpha=1$) and strongly nonlocal temporal dynamics.

This solver provides a robust tool for simulating these complex phenomena and will serve as a validation baseline for future work with Fractional Physics-Informed Neural Networks (FPINNs).

## 2. Model Equations

The solver implements the following coupled fractional-in-time diffusion system:

$$
Fh_s \frac{\partial^\alpha \theta_s}{\partial t^\alpha} = \left(\frac{1+\delta\theta_s}{N_{is}}\right) \left(\frac{\partial^2 \theta_s}{\partial x^2} + \frac{\partial^2 \theta_s}{\partial y^2}\right) - (\theta_s - \theta_f)
$$

$$
Fh_f \frac{\partial^\alpha \theta_f}{\partial t^\alpha} = \left(\frac{1+\delta\theta_f}{N_{if}}\right) \left(\frac{\partial^2 \theta_f}{\partial x^2} + \frac{\partial^2 \theta_f}{\partial y^2}\right) + (\theta_s - \theta_f)
$$

where $0 < \alpha \le 1$ is the fractional order of the Caputo time derivative.

### Boundary and Initial Conditions

- **Dirichlet (in $x$):**
  - At $x=0$: $\theta_s = \theta_f = 0$
  - At $x=1$: $\theta_s = \theta_f = 1$
- **Neumann (in $y$):**
  - At $y=0$: $\frac{\partial \theta_s}{\partial y} = \frac{\partial \theta_f}{\partial y} = 0$
  - At $y=H$: $\frac{\partial \theta_s}{\partial y} = \frac{\partial \theta_f}{\partial y} = 0$
- **Initial (at $t=0$):**
  - $\theta_s = \theta_f = \theta_0$ (constant)

## 3. Numerical Approach

### Time Discretization (L1 Scheme)

The Caputo fractional derivative ($0 < \alpha < 1$) at time $t_{k+1}$ is approximated using the L1 scheme, which is first-order accurate:

$$
^C D_t^\alpha u(t_{k+1}) \approx \frac{(\Delta t)^{-\alpha}}{\Gamma(2-\alpha)} \sum_{m=0}^{k} c_m (u^{k+1-m} - u^{k-m})
$$

where the coefficients are $c_m = (m+1)^{1-\alpha} - m^{1-\alpha}$.

### Space Discretization (Central Difference)

The integer-order spatial derivatives (Laplacian $\nabla^2$) are approximated using a standard second-order central difference 5-point stencil:

$$
\nabla^2 u \Big|_{i,j} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}
$$

### Time Stepping Strategy

An **implicit** time-stepping scheme is used. At each time step $t_{k+1}$, we solve for all unknown temperatures $\theta_s^{k+1}$ and $\theta_f^{k+1}$ simultaneously. This involves:

1.  Discretizing the PDEs, which results in a large, sparse, coupled system of linear equations.
2.  Building the system matrix $A$ and the right-hand-side vector $b$. The $b$ vector contains all known information from previous time steps ($k, k-1, \dots, 0$).
3.  Solving the linear system $A \cdot x = b$ using `scipy.sparse.linalg.spsolve` to find the solution vector $x$ (which contains all $\theta_s^{k+1}$ and $\theta_f^{k+1}$).

## 4. Project Structure

The code is organized as a Python package for clarity and maintainability.

```
LTNE_extended_fpinn/
├── readme.md
├── requirements.txt
├── .gitignore                
├── explicit_fdm_scheme/
│   ├── explicit.py
│   └── readme.md
└── implicit_fdm_scheme/
    ├── implicit.py
    ├── alpha_eql_0.4
    ├── alpha_eql_0.8
    └── alpha_eql_1
    |__ readme.md

```

## 5. How to Run

1.  Ensure you have Python and the required packages installed:
    ```bash
    pip install -r requirements.txt
    ```
2.  Adjust parameters in `fdm_solver/config.py` as needed (e.g., $Nx$, $Ny$, $\alpha$).
3.  Run the simulation from the root directory:
    ```bash
    python run_fdm.py
    ```
4.  Plots and results will be saved in the `results/` directory..

## Future Plans

- Implement Fractional Physics-Informed Neural Networks (FPINN) for surrogate modeling and parameter inference.
- Reduce the computational complexity of the explicit L1 scheme for the Caputo derivative by implementing the Sum of Exponentials (SOE) method, which lowers the history term evaluation from O(Nt²) to O(Nt) per time step.

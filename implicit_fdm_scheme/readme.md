# Implicit FDM Solver for Coupled Fractional-in-Time Diffusion (LTNE Model)

This repository contains a Python script (`implicit_fdm.py`) that solves a 2D, coupled, fractional-in-time partial differential equation (PDE) system. This system models Local Thermal Non-Equilibrium (LTNE) heat transfer.

This solver uses a **fully implicit** finite difference method. This approach is chosen for its **unconditional stability**, which allows for much larger time steps ($\Delta t$) than an explicit method, overcoming the $O(\Delta t \le \Delta x^2)$ stability constraint.

## 1. Model Equations

The solver finds the solid ($\theta_s$) and fluid ($\theta_f$) temperature fields, which are governed by the following coupled PDEs:

$$
Fh_s \frac{\partial^\alpha \theta_s}{\partial t^\alpha} = \left(\frac{1+\delta\theta_s}{N_{is}}\right) \left(\frac{\partial^2 \theta_s}{\partial x^2} + \frac{\partial^2 \theta_s}{\partial y^2}\right) - (\theta_s - \theta_f)
$$

$$
Fh_f \frac{\partial^\alpha \theta_f}{\partial t^\alpha} = \left(\frac{1+\delta\theta_f}{N_{if}}\right) \left(\frac{\partial^2 \theta_f}{\partial x^2} + \frac{\partial^2 \theta_f}{\partial y^2}\right) + (\theta_s - \theta_f)
$$

### Boundary and Initial Conditions:
* **Initial (t=0):** $\theta_s = \theta_f = \theta_0$
* **Dirichlet (x=0, x=1):** $\theta_s(0,y,t) = 0$, $\theta_s(1,y,t) = 1$ (and similarly for $\theta_f$).
* **Neumann (y=0, y=H):** $\frac{\partial \theta_s}{\partial y} = 0$, $\frac{\partial \theta_f}{\partial y} = 0$.

---

## 2. Numerical Method: The Implicit Scheme

Instead of solving for $\theta^{k+1}$ directly, an implicit scheme sets up a large system of linear equations to be solved at *every* time step. All spatial derivatives and coupling terms are evaluated at the **unknown** time $k+1$.

This transforms the PDE problem into a linear algebra problem: **$A \cdot x = b$**.

### Time Discretization (L1 Scheme)

The Caputo fractional derivative $\frac{\partial^\alpha \theta}{\partial t^\alpha}$ at time $t_{k+1}$ is approximated by the L1 scheme:

$$
\frac{\partial^\alpha \theta}{\partial t^\alpha} \Big|_{k+1} \approx C_\alpha \left[ (\theta^{k+1} - \theta^k) - \text{History}(k) \right]
$$

where:
* **$C_\alpha$:** The L1 constant, $\frac{(\Delta t)^{-\alpha}}{\Gamma(2-\alpha)}$.
* **$\text{History}(k)$:** The non-local sum over all past time steps, $\sum_{m=1}^{k} c_m (\theta^{k-m+1} - \theta^{k-m})$.

### Space Discretization (Central Difference)

The 2D Laplacian $\nabla^2 \theta = (\frac{\partial^2 \theta}{\partial x^2} + \frac{\partial^2 \theta}{\partial y^2})$ is approximated with a standard 5-point central difference stencil at the **unknown** time $k+1$:

$$
\nabla^2 \theta \Big|_{i,j}^{k+1} \approx \frac{\theta_{i+1,j}^{k+1} - 2\theta_{i,j}^{k+1} + \theta_{i-1,j}^{k+1}}{\Delta x^2} + \frac{\theta_{i,j+1}^{k+1} - 2\theta_{i,j}^{k+1} + \theta_{i,j-1}^{k+1}}{\Delta y^2}
$$

---

## 3. The $A \cdot x = b$ System

This is the core of the solver. We must build the matrix $A$ and vectors $x$ and $b$ for all unknown points at each time step.

### The Unknowns Vector ($x$)
The solver only computes the **internal** grid points (boundaries are known).
* $N_{internal} = (N_x - 2) \times (N_y - 2)$
* $N_{unknowns} = 2 \times N_{internal}$ (since we have two fields, $\theta_s$ and $\theta_f$)

The vector $x$ is a single, "flattened" 1D vector containing all $N_{unknowns}$. We structure it by stacking the two fields together:

* **Block 1 (Indices `0` to `N_internal - 1`):** Contains all $N_{internal}$ values for the solid phase, $\theta_s^{k+1}$.
* **Block 2 (Indices `N_internal` to `N_unknowns - 1`):** Contains all $N_{internal}$ values for the fluid phase, $\theta_f^{k+1}$.

### The Grid-to-Vector Mapping
To map a 2D grid point $(i, j)$ to its 1D position in the vector $x$, we use a row-major mapping function `map_idx(i, j)`.
* A point $(i, j)$ (using 1-based indexing from $1..N_x-2$) maps to $p = (j-1) \times (N_x - 2) + (i-1)$.
* The **solid** equation for $(i, j)$ corresponds to **Row $p$**.
* The **fluid** equation for $(i, j)$ corresponds to **Row $q = p + N_{internal}$**.

### The "Knowns" Vector ($b$)
The vector $b$ contains all terms from the **known** previous time step, $k$. For a solid equation at row $p$ (point $i,j$):

$b[p] = (Fh_s C_\alpha \cdot \theta_{s,i,j}^k) - (Fh_s C_\alpha \cdot \text{History}_s(k)) + (\text{Dirichlet Boundary Terms})$

### The System Matrix ($A$)
The matrix $A$ stores the coefficients of all the **unknown** $\theta^{k+1}$ terms. It has a $2 \times 2$ block structure:

$$
\mathbf{A} =
\begin{bmatrix}
\mathbf{A_{ss}} & \mathbf{A_{sf}} \\
\mathbf{A_{fs}} & \mathbf{A_{ff}}
\end{bmatrix}
$$

* **$A_{ss}$ and $A_{ff}$ (Diagonals):** These blocks represent the spatial Laplacian for each field. They are sparse matrices containing the 5-point stencil coefficients.
* **$A_{sf}$ and $A_{fs}$ (Off-Diagonals):** These blocks represent the coupling. They are diagonal matrices that link $\theta_s$ at point $(i,j)$ to $\theta_f$ at the *same* point $(i,j)$.

---

## 4. Implementation Details

### Handling Nonlinearity (Lagging)
The diffusivity $D_s(\theta_s) = (1+\delta\theta_s)/N_{is}$ is nonlinear. To keep the system linear ($A \cdot x = b$), we **lag** this term: we calculate $D_s$ using the *known* temperature from the previous step, $\theta_s^k$.

$$
A[p, p] \text{ term includes } \rightarrow \left(\frac{1+\delta\theta_{s,i,j}^{\mathbf{k}}}{N_{is}}\right) \left( \frac{2}{\Delta x^2} + \frac{2}{\Delta y^2} \right)
$$

### Handling Neumann Boundaries (Ghost Points)
The $\frac{\partial \theta}{\partial y} = 0$ condition is handled by modifying the Laplacian at the $j=1$ (bottom) and $j=N_y-2$ (top) boundaries.

* At the **bottom boundary ($j=1$)**, the standard Laplacian is replaced:
    $\frac{\theta_{i,2}^{k+1} - 2\theta_{i,1}^{k+1} + \theta_{i,0}^{k+1}}{\Delta y^2} \rightarrow \frac{\theta_{i,2}^{k+1} - 2\theta_{i,1}^{k+1} + \theta_{i,2}^{k+1}}{\Delta y^2} = \frac{2\theta_{i,2}^{k+1} - 2\theta_{i,1}^{k+1}}{\Delta y^2}$
* This changes the matrix $A$: the coefficient for the $j+1$ neighbor is doubled (e.g., $-2 K_y$) and the $j-1$ neighbor coefficient becomes zero.

## 5. How to Run
1.  **Set Parameters:** Adjust grid size (`Nx`, `Ny`), time steps (`Nt_steps`), and physical parameters (`alpha`, `Fhs`, `Nis`, etc.) at the top of the script.
2.  **Run Script:** Execute the Python script.
    ```bash
    python implicit_fdm.py
    ```
3.  **View Results:** The script will print its progress for each time step and save several validation plots as `.png` files.

## 6. Validation
The solver's output is validated by generating plots that can be compared to analytical solutions or published results. The script automatically generates:
* A 2D contour plot of the final temperature.
* A time-evolution plot of the temperature at the domain's center.
* A 1D spatial slice of the final temperature.
* A validation plot that replicates Figure 2a from the reference paper, showing $\theta_s$ and $\theta_f$ at specific x-locations over time.

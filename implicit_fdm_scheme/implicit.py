import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import math

#define the parameters
Y=1.0
X=1.0
T=0.2
Nx=10
Ny=10
Nt_steps = 10 
Nt_points = Nt_steps + 1 
alpha=0.4

#define the pde parameters
dx=X/(Nx-1)
dy=Y/(Ny-1)
dt=T/Nt_steps 
Fhs = 1.5
Nis = 0.5
Fhf = 1.5
Nif = 1.0
delta = 0.0
theta_0 = 0.5

# Total number of internal points
N_internal = (Nx-2)*(Ny-2)
# Total number of unknowns (solid + fluid)
N_unknowns = 2 * N_internal

# L1 scheme constants
C_alpha = (dt**(-alpha)) / math.gamma(2 - alpha)
dx2_inv = 1.0 / (dx * dx)
dy2_inv = 1.0 / (dy * dy)


def map_idx(i, j):
    i_vec = i - 1
    j_vec = j - 1
    return j_vec * (Nx-2) + i_vec

def calculate_l1_history_term(theta_history, l1_coeffs, k, i, j):
    history_sum = 0.0
    for m in range(1, k + 1):  
        term_k_minus_m = theta_history[i, j, k - m]
        history_sum += l1_coeffs[m] * (theta_history[i, j, k + 1 - m] - term_k_minus_m)

    return history_sum

def initialize_grid():
    theta_s = np.zeros((Nx, Ny, Nt_points))
    theta_f = np.zeros((Nx, Ny, Nt_points))

    theta_s[1:-1, 1:-1, 0] = theta_0
    theta_f[1:-1, 1:-1, 0] = theta_0

    theta_s[0, :, :] = 0.0  # left boundary
    theta_s[-1, :, :] = 1.0 # right boundary
    theta_f[0, :, :] = 0.0  # left boundary
    theta_f[-1, :, :] = 1.0 # right boundary

    return theta_s, theta_f

print("Starting Implicit FDM Simulation...")

theta_s, theta_f = initialize_grid()

# Pre-calculate all possible L1 coefficients
k_vals = np.arange(Nt_points)
l_coeffs = (k_vals + 1)**(1 - alpha) - k_vals**(1 - alpha)

# Time-stepping loop
for k in range(Nt_steps):
    
    A = lil_matrix((N_unknowns, N_unknowns))
    b = np.zeros(N_unknowns)

    theta_s_k = theta_s[:, :, k]
    theta_f_k = theta_f[:, :, k]

    # Spatial loops to fill matrix A and vector b
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            #mapping indices
            p = map_idx(i, j)     
            q = p + N_internal 
            
            history_s = calculate_l1_history_term(theta_s, l_coeffs, k, i, j)
            history_f = calculate_l1_history_term(theta_f, l_coeffs, k, i, j)
            
            D_s = (1.0 + delta * theta_s_k[i, j]) / Nis
            D_f = (1.0 + delta * theta_f_k[i, j]) / Nif
            
            Kx_s = D_s * dx2_inv
            Ky_s = D_s * dy2_inv
            Kx_f = D_f * dx2_inv
            Ky_f = D_f * dy2_inv

            #! BUILD ROW p (Solid Equation)
            b[p] = Fhs * C_alpha * theta_s_k[i, j] - Fhs * C_alpha * history_s
            A[p, p] = Fhs * C_alpha + 2*Kx_s + 2*Ky_s + 1.0
            A[p, q] = -1.0 # Coupling to fluid
            

            if i == 1: # Left Dirichlet boundary
                b[p] += Kx_s * theta_s[0, j, k+1]
            else:
                A[p, map_idx(i-1, j)] = -Kx_s
                
            if i == Nx - 2: # Right Dirichlet boundary
                b[p] += Kx_s * theta_s[Nx-1, j, k+1] 
            else:
                A[p, map_idx(i+1, j)] = -Kx_s

            if j == 1: # Bottom Neumann boundary
                A[p, map_idx(i, j+1)] = -2*Ky_s 
            elif j == Ny - 2: # Top Neumann boundary
                A[p, map_idx(i, j-1)] = -2*Ky_s
            else: # Standard internal point
                A[p, map_idx(i, j-1)] = -Ky_s
                A[p, map_idx(i, j+1)] = -Ky_s

            #! BUILD ROW q (Fluid Equation)
            b[q] = Fhf * C_alpha * theta_f_k[i, j] - Fhf * C_alpha * history_f
            A[q, q] = Fhf * C_alpha + 2*Kx_f + 2*Ky_f + 1.0
            A[q, p] = -1.0 # Coupling to solid

            if i == 1: # Left Dirichlet boundary
                b[q] += Kx_f * theta_f[0, j, k+1]
            else:
                A[q, map_idx(i-1, j) + N_internal] = -Kx_f
                
            if i == Nx - 2: # Right Dirichlet boundary
                b[q] += Kx_f * theta_f[Nx-1, j, k+1]
            else:
                A[q, map_idx(i+1, j) + N_internal] = -Kx_f

            if j == 1: # Bottom Neumann boundary
                A[q, map_idx(i, j+1) + N_internal] = -2*Ky_f
            elif j == Ny - 2: # Top Neumann boundary
                A[q, map_idx(i, j-1) + N_internal] = -2*Ky_f
            else: # Standard internal point
                A[q, map_idx(i, j-1) + N_internal] = -Ky_f
                A[q, map_idx(i, j+1) + N_internal] = -Ky_f

    #  Solve the Linear System 
    A_csc = A.tocsc()
    
    try:
        solution_vector = spsolve(A_csc, b)
    except Exception as e:
        print(f"Error solving system at time step {k}: {e}")
        break

    # Unpack the Solution Vector -> 2D Grid
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            p = map_idx(i, j)
            theta_s[i, j, k+1] = solution_vector[p]
            theta_f[i, j, k+1] = solution_vector[p + N_internal]

    #Apply Neumann Boundary Conditions
    theta_s[:, 0, k+1] = theta_s[:, 1, k+1]   # bottom boundary
    theta_s[:, -1, k+1] = theta_s[:, -2, k+1] # top boundary
    theta_f[:, 0, k+1] = theta_f[:, 1, k+1]   # bottom boundary
    theta_f[:, -1, k+1] = theta_f[:, -2, k+1] # top boundary

    if (k+1) % 1 == 0:
        print(f"Time step {k+1}/{Nt_steps} complete.")

print("End of Implicit FDM Simulation.")

# Plotting the results of the simulation
plt.figure(figsize=(12, 5))
final_time_index = 5

# contour plots for solid and fluid phases
plt.subplot(1, 2, 1)
plt.contourf(np.linspace(0, X, Nx), np.linspace(0, Y, Ny), theta_s[:, :, final_time_index].T, levels=20, cmap='inferno')
plt.colorbar(label='Solid Temp $\\theta_s$')
plt.title(f'Solid Phase at t = {final_time_index*dt:.2f}s')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.contourf(np.linspace(0, X, Nx), np.linspace(0, Y, Ny), theta_f[:, :, final_time_index].T, levels=20, cmap='inferno')
plt.colorbar(label='Fluid Temp $\\theta_f$')
plt.title(f'Fluid Phase at t = {final_time_index*dt:.2f}s')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig('implicit_fdm_result.png')
print("Result plot saved to 'implicit_fdm_result.png'")


# time evolution at center point
center_x_idx = Nx // 2
center_y_idx = Ny // 2
time_array = np.linspace(0, T, Nt_points)

plt.figure(figsize=(10, 6))
plt.plot(time_array, theta_s[center_x_idx, center_y_idx, :], label='Solid $\\theta_s$ at center', color='r', marker='o', markersize=4, linestyle='-')
plt.plot(time_array, theta_f[center_x_idx, center_y_idx, :], label='Fluid $\\theta_f$ at center', color='b', marker='x', markersize=4, linestyle='--')
plt.title(f'Temperature Evolution at Center ({X/2}, {Y/2})')
plt.xlabel('Time (s)')
plt.ylabel('Temperature ($\\theta$)')
plt.legend()
plt.grid(True)
plt.savefig('validation_time_evolution.png')
print("Validation plot saved to 'validation_time_evolution.png'")

#1D spatial profile at final time (slice at y=Y/2)
center_y_idx = Ny // 2
x_array = np.linspace(0, X, Nx)

plt.figure(figsize=(10, 6))
plt.plot(x_array, theta_s[:, center_y_idx, -1], label='Solid $\\theta_s$ at y=Y/2', color='r', marker='o')
plt.plot(x_array, theta_f[:, center_y_idx, -1], label='Fluid $\\theta_f$ at y=Y/2', color='b', marker='x')
plt.title(f'Final Temperature Profile at t={T}s (Slice at y={Y/2})')
plt.xlabel('x-coordinate')
plt.ylabel('Temperature ($\\theta$)')
plt.legend()
plt.grid(True)
plt.savefig('validation_spatial_slice.png')
print("Validation plot saved to 'validation_spatial_slice.png'")

# 2D Temperature Difference at Final Time
temp_difference = theta_s[:, :, -1] - theta_f[:, :, -1]

plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, X, Nx), np.linspace(0, Y, Ny), temp_difference.T, levels=20, cmap='RdBu')
plt.colorbar(label='Difference ($\\theta_s - \\theta_f$)')
plt.title(f'Final Temperature Difference at t = {T:.2f}s')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('validation_temp_difference.png')
print("Validation plot saved to 'validation_temp_difference.png'")


# Replicating Paper Figure for Validation
center_y_idx = Ny // 2
time_array = np.linspace(0, T, Nt_points)
x_locations_to_plot = [0.1, 0.2, 0.3, 0.4]
x_indices = [int(round(x_val / dx)) for x_val in x_locations_to_plot]
plt.figure(figsize=(10, 7))
ax = plt.gca()
colors = ['C0', 'C1', 'C2', 'C3']
linestyles = ['-', '--'] 
markers = ['o', 's', 'D', '^']

for i, x_idx in enumerate(x_indices):
    x_val = x_locations_to_plot[i]
    
    solid_temp_series = theta_s[x_idx, center_y_idx, :]
    fluid_temp_series = theta_f[x_idx, center_y_idx, :]
    
    ax.plot(time_array, solid_temp_series, 
            label=f'$\\theta_s$ at x={x_val}', 
            color=colors[i], 
            linestyle=linestyles[0], 
            marker=markers[i], 
            markevery=4,
            fillstyle='none')
            
    ax.plot(time_array, fluid_temp_series, 
            label=f'$\\theta_f$ at x={x_val}', 
            color=colors[i], 
            linestyle=linestyles[1], 
            marker=markers[i], 
            markevery=4)

ax.set_ylim(0.05, 0.55) 
ax.set_xlim(0, 0.2)  
ax.set_title(f'Solver Validation vs. Paper (Fig. 2a)\n(y = {Y/2})')
ax.set_xlabel('Time, t [dim-less]')
ax.set_ylabel('$\\theta_s$ & $\\theta_f$')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.4)

plt.savefig('validation_paper_replication.png')
print("Paper replication plot saved to 'validation_paper_replication.png'")

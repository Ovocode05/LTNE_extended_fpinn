from math import gamma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration parameters for the simulation
Fhs = 1.5
Nis = 0.5
Fhf = 1.5
Nif = 1.0
delta = 0.0
alpha = 0.8
X = 1.0
Y = 1.0
T = 1.0
Nx = 3
Ny = 3
dt = 0.015
theta_0 = 0.5

cofactor = gamma(2 - alpha) * (dt**alpha)

#function to initialize the grid
def initialize_grid():
    Nt = int(T / dt)
    theta_s = np.zeros((Nx, Ny, Nt+1))
    theta_f = np.zeros((Nx, Ny, Nt+1))

    #boundary conditions x-direction
    theta_s[0, :, :] = 0.0  # left boundary
    theta_s[-1, :, :] = 1.0  # right boundary
    theta_f[0, :, :] = 0.0  # left boundary
    theta_f[-1, :, :] = 1.0  # right boundary

    #initial conditions
    theta_s[1:-1, :, 0] = theta_0
    theta_f[1:-1, :, 0] = theta_0

    return theta_s, theta_f  

#perform the explicit finite difference method
print("Starting Explicit FDM Simulation...")
theta_s, theta_f = initialize_grid()
Nt = int(T / dt)
dx = X / (Nx - 1)
dy = Y / (Ny - 1)
k_vals = np.arange(0, Nt+1)
l_coeff = (k_vals + 1)**(1 - alpha) - k_vals**(1 - alpha)

for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            history_s = 0.0
            history_f = 0.0
            for k in range(1, n+1):
                history_s += l_coeff[k]*(theta_s[i,j,n-k+1] - theta_s[i,j,n-k])
                history_f += l_coeff[k]*(theta_f[i,j,n-k+1] - theta_f[i,j,n-k])

            history_s /=  cofactor
            history_f /=  cofactor

            laplacian_s = ((1 + delta*theta_s[i,j,n])/Nis)*(
                    (theta_s[i+1,j,n] - 2*theta_s[i,j,n] + theta_s[i-1,j,n])/(dx**2) +
                    (theta_s[i,j+1,n] - 2*theta_s[i,j,n] + theta_s[i,j-1,n])/(dy**2)
                ) 
            interface_s = (theta_s[i, j, n] - theta_f[i, j, n])

            laplacian_f = ((1 + delta*theta_f[i,j,n])/Nif)*(
                    (theta_f[i+1,j,n] - 2*theta_f[i,j,n] + theta_f[i-1,j,n])/(dx**2) +
                    (theta_f[i,j+1,n] - 2*theta_f[i,j,n] + theta_f[i,j-1,n])/(dy**2)
                ) 
            interface_f = (theta_s[i, j, n] - theta_f[i, j, n])

            #main equations
            theta_s[i,j, n+1] = (theta_s[i, j, n] - history_s) + (cofactor/Fhs)*(laplacian_s - interface_s)
            theta_f[i,j, n+1] = (theta_f[i, j, n] - history_f) + (cofactor/Fhf)*(laplacian_f + interface_f)

            if n>1:
                #after every step, enforce boundary conditions
                theta_s[:, 0, n+1] = theta_s[:, 1, n+1]  # bottom boundary   
                theta_s[:, -1, n+1] = theta_s[:, -2, n+1]  # top boundary
                theta_f[:, 0, n+1] = theta_f[:, 1, n+1]  # bottom boundary
                theta_f[:, -1, n+1] = theta_f[:, -2, n+1]  # top boundary


print("Explicit FDM Simulation Completed.")
import numpy as np

H= 1.0
L= 1.0
T= 1.0
alpha = 0.8

#model coefficients
Fhs = 1.0
Fhf = 1.0
Nis = 1.0
Nif = 1.0
theta_0 = 0.5

#numerical parameters
nx = 10
ny = 10
nt = 10 
dx = L/(nx-1)
dt = T/(nt-1)
dt = T/(nt)

x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
t = np.linspace(0, T, nt)

#intializing solution arrays
theta_s = np.zeros((nt+1,nx, ny))
theta_f = np.zeros((nt+1,nx, ny))
theta_s[0,:,:] = theta_0
theta_f[0,:,:] = theta_0
theta_s[:,0,:] = 0.0
theta_f[:,0,:] = 0.0
theta_s[:,1,:] = 1.0
theta_f[:,1,:] = 1.0

print("Configuration parameters set.")
print(f'Solution arrays initialized with shape: {theta_s.shape}, {theta_f.shape}')

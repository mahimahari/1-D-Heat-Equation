import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = 1.0         # Domain length
T = 1.0          # Total simulation time
alpha = 0.01     # Thermal diffusivity
Nx = 100         # Number of spatial points
Nt = 1000        # Number of time steps

# Grid
x = np.linspace(0, Lx, Nx)
dx = x[1] - x[0]
dt = T / Nt

# Initial condition: u(x,0) = sin(πx/Lx)
u = np.sin(np.pi * x / Lx)

# Boundary conditions: u(0,t) = u(Lx,t) = 0
u[0] = 0
u[-1] = 0

# Solution matrix
U = np.zeros((Nt, Nx))
U[0, :] = u

# Time stepping (explicit method)
for n in range(1, Nt):
    un = U[n-1, :]
    
    # Compute new values (excluding boundaries)
    u[1:-1] = un[1:-1] + alpha * dt / dx**2 * (un[2:] - 2*un[1:-1] + un[:-2])
    
    # Apply boundary conditions
    u[0] = 0
    u[-1] = 0
    
    U[n, :] = u

# Plotting
t = np.linspace(0, T, Nt)
X, T_grid = np.meshgrid(x, t)

plt.figure(figsize=(12, 6))
plt.pcolormesh(X, T_grid, U, cmap='hot', shading='auto')
plt.colorbar(label='Temperature (u)')
plt.xlabel("Position (x)")
plt.ylabel("Time (t)")
plt.title("1D Heat Equation (Numerical Solution)")
plt.tight_layout()
plt.show()
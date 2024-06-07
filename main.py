import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider

# Grid size
N = 50
dt = 0.1

# Initialize parameters
diffusion = 0.01
viscosity = 0.01

def initialize_grid(N):
    return np.zeros((N, N))

# Initialize velocity and density fields
u = initialize_grid(N)
v = initialize_grid(N)
u_prev = initialize_grid(N)
v_prev = initialize_grid(N)
dens = initialize_grid(N)
dens_prev = initialize_grid(N)

def set_boundary(b, x):
    for i in range(1, N-1):
        x[0, i] = -x[1, i] if b == 1 else x[1, i]
        x[N-1, i] = -x[N-2, i] if b == 1 else x[N-2, i]
        x[i, 0] = -x[i, 1] if b == 2 else x[i, 1]
        x[i, N-1] = -x[i, N-2] if b == 2 else x[i, N-2]
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N-1] = 0.5 * (x[1, N-1] + x[0, N-2])
    x[N-1, 0] = 0.5 * (x[N-2, 0] + x[N-1, 1])
    x[N-1, N-1] = 0.5 * (x[N-2, N-1] + x[N-1, N-2])

def lin_solve(b, x, x0, a, c):
    for k in range(20):
        x[1:N-1, 1:N-1] = (x0[1:N-1, 1:N-1] + a * (x[0:N-2, 1:N-1] + x[2:N, 1:N-1] + x[1:N-1, 0:N-2] + x[1:N-1, 2:N])) / c
        set_boundary(b, x)

def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 4 * a)

def advect(b, d, d0, u, v, dt):
    dt0 = dt * (N - 2)
    for i in range(1, N-1):
        for j in range(1, N-1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            if x < 0.5: x = 0.5
            if x > N - 1.5: x = N - 1.5
            i0 = int(x)
            i1 = i0 + 1
            if y < 0.5: y = 0.5
            if y > N - 1.5: y = N - 1.5
            j0 = int(y)
            j1 = j0 + 1
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    set_boundary(b, d)

def project(u, v, p, div):
    h = 1.0 / N
    div[1:N-1, 1:N-1] = -0.5 * h * (u[2:N, 1:N-1] - u[0:N-2, 1:N-1] + v[1:N-1, 2:N] - v[1:N-1, 0:N-2])
    p[1:N-1, 1:N-1] = 0
    set_boundary(0, div)
    set_boundary(0, p)
    lin_solve(0, p, div, 1, 4)
    u[1:N-1, 1:N-1] -= 0.5 * (p[2:N, 1:N-1] - p[0:N-2, 1:N-1]) / h
    v[1:N-1, 1:N-1] -= 0.5 * (p[1:N-1, 2:N] - p[1:N-1, 0:N-2]) / h
    set_boundary(1, u)
    set_boundary(2, v)

def density_step(x, x0, u, v, diff, dt):
    add_source(x, x0, dt)
    x0, x = x, x0
    diffuse(0, x, x0, diff, dt)
    x0, x = x, x0
    advect(0, x, x0, u, v, dt)

def velocity_step(u, v, u0, v0, visc, dt):
    add_source(u, u0, dt)
    add_source(v, v0, dt)
    u0, u = u, u0
    v0, v = v, v0
    diffuse(1, u, u0, visc, dt)
    diffuse(2, v, v0, visc, dt)
    project(u, v, u0, v0)
    u0, u = u, u0
    v0, v = v, v0
    advect(1, u, u0, u0, v0, dt)
    advect(2, v, v0, u0, v0, dt)
    project(u, v, u0, v0)

def add_source(x, s, dt):
    x += dt * s

def add_initial_conditions():
    # Adding some initial conditions for testing
    dens[N//2, N//2] = 100.0
    u[N//2, N//2] = 1.0
    v[N//2, N//2] = 1.0

add_initial_conditions()

def update(frame, viscosity, diffusion):
    global u, v, u_prev, v_prev, dens, dens_prev
    velocity_step(u, v, u_prev, v_prev, viscosity, dt)
    density_step(dens, dens_prev, u, v, diffusion, dt)
    
    # Update density plot
    im.set_array(dens)
    
    # Update velocity quiver plot
    quiver.set_UVC(u, v)
    
    # Update density grid
    ax_density_grid.clear()
    ax_density_grid.imshow(dens, cmap='gray', origin='lower')
    ax_density_grid.set_title('Density Grid')
    
    # Update velocity grid
    velocity_magnitude = np.sqrt(u**2 + v**2)
    ax_velocity_grid.clear()
    ax_velocity_grid.imshow(velocity_magnitude, cmap='viridis', origin='lower')
    ax_velocity_grid.set_title('Velocity Grid')
    
    return [im, quiver]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle("Fluid Simulation by Jos Stam", fontsize=16)

ax_density = axs[0, 0]
ax_density.set_title('Simulation of Fluid')

ax_velocity = axs[0, 1]
ax_velocity.set_title('Velocity Movement')

# Define density grid subplot
ax_density_grid = axs[1, 0]
ax_density_grid.set_title('Density Grid')

# Define velocity grid subplot
ax_velocity_grid = axs[1, 1]
ax_velocity_grid.set_title('Velocity Grid')

# Plot initial density and velocity
im = ax_density.imshow(dens, interpolation='bilinear', cmap='gray', origin='lower')
quiver = ax_velocity.quiver(np.arange(N), np.arange(N), u, v)



def run_simulation(viscosity, diffusion):
    ani = animation.FuncAnimation(fig, update, fargs=(viscosity, diffusion), frames=200, interval=50, blit=True)
    plt.show()



interact(run_simulation, 
         viscosity=FloatSlider(min=0.0, max=0.1, step=0.001, value=0.0001), 
         diffusion=FloatSlider(min=0.0, max=0.1, step=0.001, value=0.0001))

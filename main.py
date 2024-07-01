import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider
from matplotlib.animation import FuncAnimation,PillowWriter

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

# Function to set boundary conditions
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

# Function to solve linear system via Gauss-Seidel method
def lin_solve(b, x, x0, a, c):
    for k in range(20):
        x[1:N-1, 1:N-1] = (x0[1:N-1, 1:N-1] + a * (x[0:N-2, 1:N-1] + x[2:N, 1:N-1] + x[1:N-1, 0:N-2] + x[1:N-1, 2:N])) / c
        set_boundary(b, x)
# Function to diffuse density and velocity fields
def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 4 * a)

# Function to advect density and velocity fields
def advect(b, d, d0, u, v, dt):
    dt0 = dt * (N - 2)
    for i in range(1, N-1):
        for j in range(1, N-1):
            x = i - dt0 * u[i, j]
            y = j - dt0 * v[i, j]
            # finding the closest grid cell to the particle position (x, y)
            if x < 0.5: x = 0.5
            if x > N - 1.5: x = N - 1.5
            i0 = int(x)
            i1 = i0 + 1
            if y < 0.5: y = 0.5
            if y > N - 1.5: y = N - 1.5
            j0 = int(y)
            j1 = j0 + 1

            # interpolation coefficients
            # vertices of the cell are (i0, j0), (i1, j0), (i0, j1), (i1, j1) where i0, j0 are lowest coordinates
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            # interpolate values
            d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    set_boundary(b, d)

# Function to project the velocity field
def project(u, v, p, div):
    # divergence of the velocity field
    h = 1.0 / N
    div[1:N-1, 1:N-1] = -0.5 * h * (u[2:N, 1:N-1] - u[0:N-2, 1:N-1] + v[1:N-1, 2:N] - v[1:N-1, 0:N-2])
    p[1:N-1, 1:N-1] = 0
    set_boundary(0, div)
    set_boundary(0, p)
    lin_solve(0, p, div, 1, 4)
    # subtract the gradient of the pressure field
    u[1:N-1, 1:N-1] -= 0.5 * (p[2:N, 1:N-1] - p[0:N-2, 1:N-1]) / h
    v[1:N-1, 1:N-1] -= 0.5 * (p[1:N-1, 2:N] - p[1:N-1, 0:N-2]) / h
    set_boundary(1, u)
    set_boundary(2, v)

# Function to update the density field
def density_step(x, x0, u, v, diff, dt):
    add_source(x, x0, dt)
    x0, x = x, x0
    diffuse(0, x, x0, diff, dt)
    x0, x = x, x0
    advect(0, x, x0, u, v, dt)

# Function to update the velocity field
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

# Function to add source to the density field
def add_source(x, s, dt):
    x += dt * s


def modify_values(index, density_value, velocity_v_value,velocity_u_value):
    dens[index, index] = density_value
    v[index, index] = velocity_v_value  # x-component of velocity
    u[index, index] = velocity_u_value  # y-component of velocity


def add_initial_conditions():
    # Adding some initial conditions for testing
    modify_values(3 * N // 4, 1.0, -1.0, -1.0)
    modify_values(N // 4, 1.0, 1.0, 1.0)

add_initial_conditions()

def update(frame, viscosity, diffusion):
    global u, v, u_prev, v_prev, dens, dens_prev
    add_initial_conditions()
    velocity_step(u, v, u_prev, v_prev, viscosity, dt)
    density_step(dens, dens_prev, u, v, diffusion, dt)
    
    # Update density plot
    im.set_array(dens)
    
    # Update velocity quiver plot
    quiver.set_UVC(u, v)    

    # Update frame text
    frame_text.set_text(f'Frame: {frame}')
    return [im, quiver]

fig, axs = plt.subplots(2, 1, figsize=(14, 10))
plt.suptitle("Fluid Simulation by Jos Stam", fontsize=16)

ax_density = axs[0]
ax_density.set_title('Simulation of Fluid')

ax_velocity = axs[1]
ax_velocity.set_title('Velocity Movement')



# Plot initial density and velocity
im = ax_density.imshow(dens, interpolation='gaussian', cmap='Blues', origin='lower')
quiver = ax_velocity.quiver(np.arange(N), np.arange(N), u, v)

frame = int(0)
frame_text = fig.text(0.5, 0.94, '', ha='center', fontsize=12)


def run_simulation(viscosity, diffusion):
    ani = animation.FuncAnimation(fig, update, fargs=(viscosity, diffusion), frames=300, interval=20, blit=True)
    # # Save the animation as an MP4 file
    ani.save('Fluid_Simulation.mp4', writer='ffmpeg', fps=10)

    # Save the animation as a GIF file (optional)
    # ani.save('Fluid_Simulation.gif', writer=PillowWriter(fps=10))
    plt.show()



interact(run_simulation, 
         viscosity=FloatSlider(min=0.0, max=0.1, step=0.001, value=0.0001), 
         diffusion=FloatSlider(min=0.0, max=0.1, step=0.001, value=0.0001))

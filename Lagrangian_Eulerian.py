import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fluid flow parameters
n_particles = 100  # Number of particles for Lagrangian approach
grid_size = 20     # Size of the grid for Eulerian approach
time_steps = 100    # Number of time steps for the simulation
dt = 0.1           # Time step size

# Velocity field (example: circular flow)
def velocity_field(x, y, t):
    u = -y  # x-component of velocity
    v = x   # y-component of velocity
    return u, v

# Initialize particles (Lagrangian)
particles = np.random.rand(n_particles, 2) * 2 - 1  # Random positions in [-1, 1] x [-1, 1]

# Initialize grid (Eulerian)
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
plt.suptitle("Lagrangian versus Eulerian", fontsize=16)

# Lagrangian plot
particles_scatter = ax1.scatter(particles[:, 0], particles[:, 1])
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_title('Lagrangian (Particle Tracking)')

# Eulerian plot
quiver = ax2.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y))
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_title('Eulerian (Fixed Grid)')

def update(frame):
    global particles
    t = frame * dt

    # Update Lagrangian particles
    for i in range(n_particles):
        u, v = velocity_field(particles[i, 0], particles[i, 1], t)
        particles[i, 0] += u * dt
        particles[i, 1] += v * dt

    # Update Eulerian grid velocities
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(grid_size):
        for j in range(grid_size):
            U[i, j], V[i, j] = velocity_field(X[i, j], Y[i, j], t)

    # Update plots
    particles_scatter.set_offsets(particles)
    quiver.set_UVC(U, V)
    return particles_scatter, quiver

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100, blit=True)
ani.save('Lagrangian versus Eulerian 2.mp4', writer='ffmpeg', fps=10)

# Show the plot
plt.tight_layout()
plt.show()

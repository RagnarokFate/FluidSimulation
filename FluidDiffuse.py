import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
grid_size = 100  # Size of the grid
diffusion_rate = 0.1  # Diffusion rate
num_iterations = 100  # Number of iterations
gauss_seidel_iterations = 20  # Number of Gauss-Seidel iterations for relaxation

# Initialize the grids
grid_good = np.zeros((grid_size, grid_size))
grid_bad = np.zeros((grid_size, grid_size))
grid_fluid = np.zeros((grid_size, grid_size))

# Set an initial concentration in the center

def inital_condition():
    initial_concentration = 1
    grid_good[grid_size // 2, grid_size // 2] = initial_concentration
    grid_bad[grid_size // 2, grid_size // 2] = initial_concentration
    grid_fluid[grid_size // 2, grid_size // 2] = initial_concentration


# Function to perform diffusion
def diffuse_good(grid, rate):
    new_grid = grid.copy()
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            new_grid[i, j] += rate * (
                grid[i+1, j] +
                grid[i-1, j] +
                grid[i, j+1] +
                grid[i, j-1])/(1+4*rate)
    return new_grid

def diffuse_bad(grid, rate):
    new_grid = grid.copy()
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            new_grid[i, j] += rate * (
                grid[i+1, j] +
                grid[i-1, j] +
                grid[i, j+1] +
                grid[i, j-1] -
                4 * grid[i, j]
            )
    return new_grid

def diffuse_fluid(grid, rate):
    new_grid = grid.copy()
    for _ in range(gauss_seidel_iterations):
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                new_grid[i, j] = (grid[i, j] + rate * (
                    new_grid[i+1, j] +
                    new_grid[i-1, j] +
                    new_grid[i, j+1] +
                    new_grid[i, j-1])) / (1 + 4 * rate)
    return new_grid

def diffuse_weighted_average(grid, rate):
    new_grid = grid.copy()
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            new_grid[i, j] += rate * (
                0.25 * grid[i+1, j] +
                0.25 * grid[i-1, j] +
                0.25 * grid[i, j+1] +
                0.25 * grid[i, j-1])
    return new_grid

# Setup the figure and axes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.suptitle("Fluid Diffusion", fontsize=16)

cax_good = axs[0].imshow(grid_good, cmap='hot', interpolation='nearest')
cax_bad = axs[1].imshow(grid_bad, cmap='hot', interpolation='nearest')
cax_fluid = axs[2].imshow(grid_fluid, cmap='hot', interpolation='nearest')

fig.colorbar(cax_good, ax=axs[0])
fig.colorbar(cax_bad, ax=axs[1])
fig.colorbar(cax_fluid, ax=axs[2])

axs[0].set_title('Weight Average Diffusion')
axs[1].set_title('SOD Diffusion')
axs[2].set_title('Smooth Diffusion (Jos Stam)')

# Text annotation for frame number
text_good = axs[0].text(0.02, 0.95, '', transform=axs[0].transAxes, color='white')
text_bad = axs[1].text(0.02, 0.95, '', transform=axs[1].transAxes, color='white')
text_fluid = axs[2].text(0.02, 0.95, '', transform=axs[2].transAxes, color='white')

inital_condition()

# Update function for animation
def update(frame):
    global grid_good, grid_bad, grid_fluid
    grid_good = diffuse_good(grid_good, diffusion_rate)
    grid_bad = diffuse_bad(grid_bad, diffusion_rate)
    grid_fluid = diffuse_fluid(grid_fluid, diffusion_rate)
    cax_good.set_array(grid_good)
    cax_bad.set_array(grid_bad)
    cax_fluid.set_array(grid_fluid)
    text_good.set_text(f'Frame: {frame}')
    text_bad.set_text(f'Frame: {frame}')
    text_fluid.set_text(f'Frame: {frame}')
    return cax_good, cax_bad, cax_fluid, text_good, text_bad, text_fluid

# Create the animation
ani = FuncAnimation(fig, update, frames=num_iterations, blit=True)

# Save the animation as an MP4 file
ani.save('diffusion_simulation.mp4', writer='ffmpeg', fps=10)

# Save the animation as a GIF file (optional)
# ani.save('diffusion_simulation.gif', writer=PillowWriter(fps=10))

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import time as Time
# Parameters
grid_size = 50  # Size of the grid
diffusion_rate = 0.1  # Diffusion rate
num_iterations = 100  # Number of iterations

# Initialize the grid
grid = np.zeros((grid_size, grid_size))

# Set an initial concentration in the center
initial_concentration = 100.0
grid[grid_size // 2, grid_size // 2] = initial_concentration

# Function to perform diffusion
def diffuse(grid, rate):
    new_grid = grid.copy()
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            new_grid[i, j] += rate * (
                grid[i+1, j] +
                grid[i-1, j] +
                grid[i, j+1] +
                grid[i, j-1])/(1+4*rate)
            
    return new_grid

# Setup the figure and axis
fig, ax = plt.subplots()
cax = ax.imshow(grid, cmap='hot', interpolation='nearest')
fig.colorbar(cax)
ax.set_title('Diffusion Simulation - good Implementation')

# Text annotation for frame number
text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

# Update function for animation
def update(frame):
    global grid
    grid = diffuse(grid, diffusion_rate)
    cax.set_array(grid)
    text.set_text(f'Frame: {frame}')
    return cax, text

# Create the animation
ani = FuncAnimation(fig, update, frames=num_iterations, blit=True)

# Save the animation as an MP4 file
# ani.save('diffusion_simulation_better.mp4', writer='ffmpeg', fps=10)

# Save the animation as a GIF file (optional)
ani.save('diffusion_simulation_better.gif', writer=PillowWriter(fps=10))

plt.show()

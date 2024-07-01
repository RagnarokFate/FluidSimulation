import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define the function
def func(x, y):
    return np.sin(x) * np.cos(y)

# Define the grid
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

# Compute the gradient
grad_x, grad_y = np.gradient(Z, x, y)

# Compute the divergence (for this 2D example, it's the sum of the partial derivatives of the gradient)
div = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)

# Compute the Laplacian (sum of second partial derivatives)
laplacian = np.gradient(np.gradient(Z, axis=0), axis=0) + np.gradient(np.gradient(Z, axis=1), axis=1)

# Create the figure and axes
fig = plt.figure(figsize=(18, 12))

# 3D plot for the function
ax0 = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax0.plot_surface(X, Y, Z, cmap='viridis')
ax0.set_title('Function $f(x, y)$')
fig.colorbar(surf, ax=ax0, shrink=0.5, aspect=5)

# 2D plot for the gradient
ax1 = fig.add_subplot(2, 2, 2)
quiver1 = ax1.quiver(X, Y, grad_x, grad_y, color='r')
ax1.set_title('Gradient of $f(x, y)$')

# 2D plot for the divergence
ax2 = fig.add_subplot(2, 2, 3)
contour2 = ax2.contourf(X, Y, div, cmap='viridis')
ax2.set_title('Divergence of Gradient')
fig.colorbar(contour2, ax=ax2)

# 2D plot for the Laplacian
ax3 = fig.add_subplot(2, 2, 4)
contour3 = ax3.contourf(X, Y, laplacian, cmap='viridis')
ax3.set_title('Laplacian of $f(x, y)$')
fig.colorbar(contour3, ax=ax3)

# Function to update the plots
def update(frame):
    Z = func(X + frame * 0.1, Y + frame * 0.1)
    grad_x, grad_y = np.gradient(Z, x, y)
    div = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)
    laplacian = np.gradient(np.gradient(Z, axis=0), axis=0) + np.gradient(np.gradient(Z, axis=1), axis=1)
    
    # Update the function plot
    ax0.clear()
    surf = ax0.plot_surface(X, Y, Z, cmap='viridis')
    ax0.set_title('Function $f(x, y)$')
    
    # Update the gradient plot
    ax1.clear()
    quiver1 = ax1.quiver(X, Y, grad_x, grad_y, color='r')
    ax1.set_title('Gradient of $f(x, y)$')
    
    # Update the divergence plot
    ax2.clear()
    contour2 = ax2.contourf(X, Y, div, cmap='viridis')
    ax2.set_title('Divergence of Gradient')
    
    # Update the Laplacian plot
    ax3.clear()
    contour3 = ax3.contourf(X, Y, laplacian, cmap='viridis')
    ax3.set_title('Laplacian of $f(x, y)$')
    
    return surf, quiver1, contour2, contour3

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=50, blit=False, interval=10)

# Display the plot
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the scalar function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Create a grid of x and y values
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Compute the gradient of f (partial derivatives)
Fx, Fy = np.gradient(Z, x, y)

# Compute the divergence of the vector field (Fx, Fy)
div_F = np.gradient(Fx, x, axis=0) + np.gradient(Fy, y, axis=1)

# Compute the Laplacian of f (second partial derivatives)
Laplacian_f = np.gradient(np.gradient(Z, x, axis=0), x, axis=0) + np.gradient(np.gradient(Z, y, axis=1), y, axis=1)

# Create the plots
fig = plt.figure(figsize=(20, 15))

# Plot the scalar function f(x, y)
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax1.set_title('Scalar function $f(x, y) = x^2 + y^2$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

# Plot the gradient of f
ax2 = fig.add_subplot(222, projection='3d')
ax2.quiver(X, Y, Z, Fx, Fy, np.zeros_like(Z), length=0.1, color='r')
ax2.set_title('Gradient of $f$, $\\nabla f(x, y)$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')

# Plot the divergence of the vector field (Fx, Fy)
ax3 = fig.add_subplot(223, projection='3d')
surf3 = ax3.plot_surface(X, Y, div_F, cmap='coolwarm', edgecolor='none')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
ax3.set_title('Divergence of $\\mathbf{F}$, $\\nabla \\cdot \\mathbf{F}$')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('div($\\mathbf{F}$)')

# Plot the Laplacian of f
ax4 = fig.add_subplot(224, projection='3d')
surf4 = ax4.plot_surface(X, Y, Laplacian_f, cmap='plasma', edgecolor='none')
fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
ax4.set_title('Laplacian of $f$, $\\nabla^2 f$')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('Laplacian($f$)')

plt.tight_layout()
plt.show()

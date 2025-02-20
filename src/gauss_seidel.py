"""
Gauss-Seidel Iterative Method for Solving Laplace's Equation

This script simulates the steady-state solution of Laplace's equation using the Gauss-Seidel iterative method.
It iteratively updates a 2D grid until convergence is achieved within a specified tolerance.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def gauss_seidel_simulation(grid: np.ndarray, max_iter: int, N: int, tol: float):
    """
    Runs the Gauss-Seidel iterative method for solving Laplace's equation.

    Parameters:
    grid (np.ndarray): 2D NumPy array representing the simulation grid (N+1, N+1).
    max_iter (int): Maximum number of iterations before stopping.
    tol (float): Convergence tolerance.
    N (int): Grid size (N x N domain with additional boundary layer).

    Returns:
    tuple: (history, t) where history is a list of grid states and t is the iteration count.
    """
    history = [grid.copy()]
    for t in range(1, max_iter + 1):
        for j in range(1, N):
            for i in range(N + 1):
                left = grid[(i - 1) % (N + 1), j]
                right = grid[(i + 1) % (N + 1), j]
                up = grid[i, j + 1]
                down = grid[i, j - 1]

                # Gauss-Seidel update
                grid[i, j] = 0.25 * (up + down + left + right)


        # Check for convergence
        if np.allclose(grid, history[-1], atol=tol):
            print(f"Converged at t = {t}")
            break
        history.append(grid.copy())

    return history, t

if __name__ == '__main__':
    # Parameters
    N = 50  # Grid size
    max_iter = 1_000_000  # Maximum iterations
    tol = 1e-5  # Convergence tolerance
    print(f"max_iter: {max_iter}")

    # Create a 2D grid
    grid = np.zeros((N + 1, N + 1))
    grid[:, N] = 1.0  # Top boundary condition c(x, y=1) = 1
    print("Grid shape:", grid.shape)
    print(grid)

    # Run the simulation
    history, t = gauss_seidel_simulation(grid, max_iter, N, tol)
    history = np.array(history)
    np.save(f'data/gauss-seidel_({N}x{N})_{t}.npy', history)

    # Plot the final result
    plt.imshow(history[-1].T, cmap='inferno', origin='lower')
    plt.colorbar()
    plt.title(f'Gauss-Seidel Convergence at t = {t}')
    plt.show()

"""
Jacobi Iterative Method for Solving Laplace's Equation

This script simulates the steady-state solution of Laplace's equation using the Jacobi iterative method.
It iteratively updates a 2D grid until convergence is achieved within a specified tolerance.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def jacobi_simulation(grid: np.ndarray, max_iter: int, N: int, tol: float):
    """
    Runs the Jacobi iterative method for solving Laplace's equation.

    Parameters:
    grid (np.ndarray): 3D NumPy array representing the simulation grid (2, N+1, N+1).
    max_iter (int): Maximum number of iterations before stopping.
    N (int): Grid size (N x N domain with additional boundary layer).
    tol (float): Convergence tolerance.

    Returns:
    tuple: (history, t) where history is a list of grid states and t is the iteration count.
    """
    history = [grid[0].copy()]
    for t in range(1, max_iter + 1):
        for i in range(N + 1):
            for j in range(1, N):
                grid[1, i, j] = 0.25 * (
                    grid[0, (i + 1) % (N + 1), j] +
                    grid[0, (i - 1) % (N + 1), j] +
                    grid[0, i, j + 1] +
                    grid[0, i, j - 1]
                )

        # Check for convergence
        if np.allclose(grid[0], grid[1], atol=tol):
            print(f"Converged at t = {t}")
            break

        grid[0] = grid[1].copy()  # Update grid[0] with grid[1]
        history.append(grid[0].copy())  # Store the current state in history
        print(f"t: {t}")

    return history, t


if __name__ == '__main__':
    # Parameters
    N = 100  # Grid size
    max_iter = 1_000_000  # Maximum iterations
    tol = 1e-5  # Convergence tolerance
    print(f"max_iter: {max_iter}")

    grid = np.zeros((2, N + 1, N + 1))
    grid[:, :, N] = 1.0  # Set top boundary condition c(x, y=1) = 1
    print("Grid shape:", grid.shape)

    # Run the simulation
    history, t = jacobi_simulation(grid, max_iter, N, tol)
    history = np.array(history)
    np.save(f'data/jacobi_({N}x{N})_{t}.npy', history)

    # Plot the final result
    print("Final history shape:", history.shape)
    plt.imshow(history[-1].T, cmap='inferno', origin='lower')
    plt.title(f'Jacobi Convergence at t = {t}')
    plt.colorbar()
    plt.show()

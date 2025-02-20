"""
Successive Over-Relaxation (SOR) Method for Solving Laplace's Equation

This script simulates the steady-state solution of Laplace's equation using the Successive Over-Relaxation (SOR) method.
It iteratively updates a 2D grid until convergence is achieved within a specified tolerance.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def sor_simulation(omega: float, grid: np.ndarray, max_iter: int, N: int, tol: float):
    """
    Runs the Successive Over-Relaxation (SOR) iterative method for solving Laplace's equation.

    Parameters:
    omega (float): Relaxation factor for SOR.
    grid (np.ndarray): 2D NumPy array representing the simulation grid (N+1, N+1).
    max_iter (int): Maximum number of iterations before stopping.
    N (int): Grid size (N x N domain with additional boundary layer).
    tol (float): Convergence tolerance.

    Returns:
    tuple: (history, t) where history is a list of grid states and t is the iteration count.
    """
    history = [grid.copy()]
    for t in range(1, max_iter + 1):
        for j in range(1, N):
            for i in range(N+1):
                old = grid[i, j]
                left = grid[(i - 1) % (N + 1), j]
                right = grid[(i + 1) % (N + 1), j]
                up = grid[i, j + 1]
                down = grid[i, j - 1]

                # SOR update
                grid[i, j] = (1 - omega) * old + (omega / 4) * (up + down + left + right)
                # diff = max(diff, abs(grid[i, j] - old))  # Track max change

        #     # Ensure periodicity for the x-boundary
        #     grid[i, -1] = grid[i, 0]

        # # Reapply fixed boundary conditions
        # grid[0, :] = 0.0   # Bottom: c(x, y=0) = 0
        # grid[N, :] = 1.0   # Top: c(x, y=1) = 1

        # # Enforce periodicity for the y boundaries
        # grid[0, N] = grid[0, 0]
        # grid[N, N] = grid[N, 0]

        if np.allclose(grid, history[-1], atol=tol):
            print(f"Converged at t = {t}")
            break
        history.append(grid.copy())
        # if t % 1000 == 0:
        #     print(f"Iteration {t}: max change = {diff}")

    return history, t

if __name__ == '__main__':
    # Parameters
    N = 50            # Grid size (N+1) x (N+1)
    max_iter = 10000
    tol = 1e-5
    omega = 1.9  # Relaxation factor
    print(f"max_iter: {max_iter}")

    grid = np.zeros((N + 1, N + 1))
    grid[:, N] = 1.0  # Top boundary condition c(x, y=1) = 1
    print("Grid shape:", grid.shape)
    print(grid)

    # Run the simulation
    history, t = sor_simulation(omega, grid, max_iter, N, tol)
    history = np.array(history)
    np.save(f'data/SOR_({N}x{N})_{t}.npy', history)

    # Plot the final result
    plt.imshow(history[-1].T, cmap='inferno', origin='lower')
    plt.colorbar()
    plt.title(f'SOR Convergence at t = {t}')
    plt.show()

"""
Gauss-Seidel Iterative Method for Solving Laplace's Equation

This script simulates the steady-state solution of Laplace's equation using the Gauss-Seidel iterative method.
It iteratively updates a 2D grid until convergence is achieved within a specified tolerance.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def gauss_seidel_simulation(grid: np.ndarray, max_iter: int, tol: float, N: int):
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
        diff = 0.0

        for i in range(1, N):  
            for j in range(N + 1):  
                old = grid[i, j]
                left = grid[i, (j - 1) % (N + 1)]  
                right = grid[i, (j + 1) % (N + 1)]  
                up = grid[i + 1, j]  
                down = grid[i - 1, j]  

                # Gauss-Seidel update
                grid[i, j] = 0.25 * (up + down + left + right)
                # diff = max(diff, abs(grid[i, j] - old))  # Track max change

            # Ensure periodicity for the x-boundary (rightmost column)
            grid[i, N] = grid[i, 0]

        # Apply fixed y-boundary conditions
        grid[0, :] = 0.0       
        grid[N, :] = 1.0

        if np.allclose(grid, history[-1], atol=tol):
            print(f"Converged at t = {t}")
            break   

        history.append(grid.copy())
        # if diff < tol:
        #     print(f"Converged at t = {t} with diff = {diff}")
        #     break
        # if t % 1000 == 0:
        #     print(f"Iteration {t}: max change = {diff}")

    return history, t

if __name__ == '__main__':
    # Parameters
    N = 50  # Grid size
    max_iter = 1_000_000  # Maximum iterations
    tol = 1e-5  # Convergence tolerance
    print(f"max_iter: {max_iter}")

    # Create a 2D grid
    grid = np.zeros((N + 1, N + 1))
    grid[N, :] = 1.0  # Top boundary condition c(x, y=1) = 1
    print("Grid shape:", grid.shape)

    # Run the simulation
    history, t = gauss_seidel_simulation(grid, max_iter, tol, N)
    history = np.array(history)
    np.save(f'data/gauss-seidel_({N}x{N})_{t}.npy', history)

    # Plot the final result
    plt.imshow(history[-1], cmap='inferno', origin='lower')
    plt.colorbar()
    plt.title(f'Gauss-Seidel Convergence at t = {t}')
    plt.show()

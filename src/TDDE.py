"""
Time-Dependent Diffusion Equation (TDDE) Simulation

This script simulates the time-dependent diffusion equation (TDDE) using the finite
difference method. The simulation is performed on a 2D grid with periodic boundary
conditions on the left and right, and a fixed boundary condition at the top. The
results are stored at regular time intervals and visualized using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def tdde_simulation(grid: np.ndarray, const: float, tsteps: int, N: int) -> np.ndarray:
    """
    Simulates the time-dependent diffusion equation (TDDE) using the finite difference method.

    Parameters:
    grid (np.ndarray): 3D NumPy array representing the simulation grid (2, N+1, N+1).
    const (float): Diffusion constant calculated as D * dt / dx**2.
    tsteps (int): Number of time steps to simulate.
    N (int): Grid size (N x N domain with additional boundary layer).

    Returns:
    np.ndarray: Simulation history at recorded intervals.
    """
    history = np.zeros((tsteps + 1, N + 1, N + 1))
    history[0] = grid[0]

    for t in range(1, tsteps + 1):
        for i in range(N + 1):
            for j in range(1, N):
                grid[1, i, j] = grid[0, i, j] + const * (
                    grid[0, (i + 1) % (N + 1), j] +
                    grid[0, (i - 1) % (N + 1), j] +
                    grid[0, i, j + 1] +
                    grid[0, i, j - 1] -
                    4 * grid[0, i, j]
                )

        grid[0], grid[1] = grid[1], grid[0]
        # if t % 1000 == 0:
        history[t] = grid[0]
        print(f"t: {t}")

    return history


if __name__ == '__main__':
    # Parameters
    N = 50  # Grid size
    D = 1  # Diffusion coefficient
    dx = dy = 1.0 / N  # Spatial step size
    tsteps = 10_000  # Total time steps
    dt = 0.0001  # Time step size (1 / tsteps)
    const = D * dt / dx**2  # Diffusion constant

    print(f"const: {const}")
    print(f"tsteps: {tsteps}")

    grid = np.zeros((2, N + 1, N + 1))
    grid[:, :, N] = 1.0  # Set top boundary condition
    print("Grid shape:", grid.shape)

    # Run the simulation
    history = tdde_simulation(grid, const, tsteps, N)
    np.save(f'data/TDDE_({N}x{N})_{tsteps}.npy', history)

    # Plot the final result
    plt.imshow(history[-1], cmap='inferno')
    plt.title(f'Time-dependent Diffusion at tsteps = {tsteps}')
    plt.colorbar()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def run_simulation(grid, const, tsteps, N):
    history = np.zeros((int(tsteps / 1000) + 1, N + 1, N + 1))
    history[0] = grid[0]

    for t in range(1, tsteps + 1):
        # Update the grid using the finite difference method
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
        if t % 1000 == 0:
            history[int(t / 1000)] = grid[0]
        print(f"t: {t}")

    return history


if __name__ == '__main__':
    # Parameters
    N = 100
    D = 1
    dx = dy = 1.0 / N
    dt = 0.000001
    const = D * dt / dx**2
    tsteps = 1000_000

    print(f"const: {const}")
    print(f"tsteps: {tsteps}")

    grid = np.zeros((2, N + 1, N + 1))
    grid[:, :, N] = 1.0  # Set top boundary condition
    print("Grid shape:", grid.shape)

    # Run the simulation
    history = run_simulation(grid, const, tsteps, N)
    np.save(f'data/TDDE_({N}x{N})_{tsteps}.npy', history)

    # Plot the final result
    plt.imshow(history[-1], cmap='inferno')
    plt.title(f'Time-dependent Diffusion at tsteps = {tsteps}')
    plt.colorbar()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def run_simulation(grid, max_iter, N, tol):
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
    N = 100
    max_iter = 1_000_000
    tol = 1e-5
    print(f"max_iter: {max_iter}")

    grid = np.zeros((2, N + 1, N + 1))
    grid[:, :, N] = 1.0  # Set top boundary condition c(x, y=1) = 1
    print("Grid shape:", grid.shape)

    # Run the simulation
    history, t = run_simulation(grid, max_iter, N, tol)
    history = np.array(history)
    np.save(f'data/jacobi_({N}x{N})_{t}.npy', history)

    # Plot the final result
    print("Final history shape:", history.shape)
    plt.imshow(history[-1].T, cmap='inferno', origin='lower')
    plt.title(f'Jacobi Convergence at t = {t}')
    plt.colorbar()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Parameters
N = 100
max_iter = 1_000_000
print(f"max_iter: {max_iter}")

# Create a grid with layout [[c, y, x], [c, y, x]]
grid = np.zeros((N + 1, N + 1))
print(grid.shape)
grid[:, N] = 1.0
print(grid)

@jit(nopython=True)
def run_simulation(grid):
    history = []
    history.append(grid)
    t = 0
    for t in range(1, max_iter + 1):
        for i in range(0, N+1):
            for j in range(1, N):
                grid[1, i, j] = 0.25 * (grid[0, (i + 1) % N, j] + \
                                        grid[0, (i - 1) % N, j] + \
                                        grid[0, i, j + 1] + \
                                        grid[0, i, j - 1])
        # Check if the grid has converged
        if np.allclose(grid[0], grid[1], atol=1e-5):
            print(f"Converged at t = {t}")
            break
        grid[0] = grid[1]
        history.append(grid[0])
        print(f"t: {t}")
    return history, t


if __name__ == '__main__':
    history, t = run_simulation(grid)
    history = np.array(history)
    print(history.shape)
    np.save(f'data/gauss-seidel_({N}x{N})_{t}.npy', history)
    print(history[-1].shape)
    plt.imshow(history[-1], cmap='inferno')
    plt.show()
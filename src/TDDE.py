import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
import scipy as sp
# Parameters

N = 100
D = 1
dx = dy = 1.0 / N
dt = 0.00001

const = D * dt / dx**2
print(f"const: {const}")

tmax = 1
tsteps = 100_000
print(f"tsteps: {tsteps}")

# Create a grid with layout [[c, y, x], [c, y, x]]
grid = np.zeros((2, N + 1, N + 1))
print(grid.shape)
print(grid[0].shape)
grid[:, :, N] = 1.0
print(grid[0])

@jit(nopython=True)
def run_simulation(grid):
    history = np.zeros((int(tsteps / 1000) + 1, N + 1, N + 1))
    print(history.shape)
    history[0] = grid[0]
    t = 0
    for t in range(1, tsteps + 1):
        # t += dt
        for i in range(0, N+1):
            for j in range(1, N):
                grid[1, i, j] = grid[0, i, j] + const * \
                                (grid[0, (i + 1) % N, j] + grid[0, (i - 1) % N, j] + \
                                 grid[0, i, j + 1] + grid[0, i, j - 1] - \
                                 4 * grid[0, i, j])
        grid[0] = grid[1]
        if t % 1000 == 0:
            history[int(t / 1000)] = grid[0]
        print(f"t: {t}")
    return history


if __name__ == '__main__':
    history = run_simulation(grid)
    np.save(f'data/TDDE_({N}x{N})_{tsteps}.npy', history)
    # print(history[-1].shape)
    plt.imshow(history[-1], cmap='inferno')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

from SOR import sor_simulation

def init_grid():
    grid = np.zeros((N + 1, N + 1))
    grid[N, :] = 1.0  # Top boundary condition c(x, y=1) = 1

    return grid

if __name__ == '__main__':
    max_iter = 1_000_000  # Maximum iterations


    # Test various omega and N values
    omega_values = np.linspace(1.60, 1.95, 10)
    N_values = np.linspace(10, 100, 10, dtype=int)

    sor_results = {omega: [] for omega in omega_values}

    for N in N_values:
        print(f"Current N: {N}")

        tol = 1e-5
        mask = np.zeros((N + 1, N + 1))

        for omega in omega_values:
            print(f"Current omega: {omega}")
            grid = init_grid()
            h_sor, t_sor = sor_simulation(omega, grid, max_iter, N, tol, mask)
            sor_results[omega].append(t_sor)

    # Plot a heatmap, where the x-axis is N, the y-axis is omega, and the color represents the number of iterations.
    plt.figure()
    plt.imshow(np.array([sor_results[omega] for omega in omega_values]), aspect='auto', cmap='viridis')
    plt.colorbar(label='Iterations to Converge')
    plt.xlabel('N')
    plt.ylabel('Omega')
    plt.title('Optimal Omega for SOR Method')
    plt.xticks(range(len(N_values)), N_values)
    plt.yticks(range(len(omega_values)), [f'{omega:.2f}' for omega in omega_values])
    plt.savefig('fig/optimal_omega.png')

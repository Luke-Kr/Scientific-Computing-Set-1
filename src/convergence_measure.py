import numpy as np
import matplotlib.pyplot as plt

from jacobi import jacobi_simulation
from gauss_seidel import gauss_seidel_simulation
from SOR import sor_simulation


def init_grid():
    grid = np.zeros((N + 1, N + 1))
    grid[N, :] = 1.0  # Top boundary condition c(x, y=1) = 1
    print("Grid shape:", grid.shape)

    return grid


if __name__ == '__main__':
    # Parameters
    N = 50  # Grid size
    max_iter = 1_000_000  # Maximum iterations
    omega_values = [1.7, 1.8, 1.9]  # Relaxation factors for SOR

    jacobi = []
    gauss = []
    sor_results = {omega: [] for omega in omega_values}

    for p in range(2, 9):
        print(f"p: {p}")  # Check

        tol = 10**-p  # Convergence tolerance

        grid = init_grid()
        h_jacobi, t_jacobi = jacobi_simulation(grid, max_iter, N, tol)
        jacobi.append(t_jacobi)

        grid = init_grid()
        h_gauss, t_gauss = gauss_seidel_simulation(grid, max_iter, N, tol)
        gauss.append(t_gauss)

        for omega in omega_values:
            grid = init_grid()
            h_sor, t_sor = sor_simulation(omega, grid, max_iter, N, tol)
            sor_results[omega].append(t_sor)

    # Plot the final results with the y-axis as a log-lin scale
    plt.plot(range(2, 9), jacobi, label='Jacobi')
    plt.plot(range(2, 9), gauss, label='Gauss-Seidel')
    for omega in omega_values:
        plt.plot(range(2, 9), sor_results[omega], label=f'SOR (omega={omega})')
    # plt.yscale('log')
    plt.xlabel('p')
    plt.ylabel('Iterations to Converge')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.savefig('fig/convergence_comparison.png')

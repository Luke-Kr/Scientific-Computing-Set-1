import numpy as np
import matplotlib.pyplot as plt

from SOR import sor_simulation


def init_grid():
    grid = np.zeros((N + 1, N + 1))
    grid[N, :] = 1.0  # Top boundary condition c(x, y=1) = 1

    return grid


if __name__ == '__main__':
    max_iter = 1_000_000  # Maximum iterations

    # In the SOR method, find the optimal omega. How does it depend on N?
    omega_values = np.linspace(1.0, 1.9, 4)
    N_values = [10, 50, 100]

    sor_results = {omega: [] for omega in omega_values}

    for N in N_values:
        print(f"Current N: {N}")

        tol = 1e-5
        # print(f"max_iter: {max_iter}")

        for omega in omega_values:
            print(f"Current omega: {omega}")
            grid = init_grid()
            h_sor, t_sor = sor_simulation(omega, grid, max_iter, N, tol)
            sor_results[omega].append(t_sor)

    # Plot the final results with the y-axis as a log-lin scale
    plt.figure()
    for omega in omega_values:
        plt.plot(N_values, sor_results[omega], label=f'omega={omega}')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Iterations to Converge')
    plt.title('Optimal Omega for SOR Method')
    plt.legend()
    plt.savefig('fig/optimal_omega.png')

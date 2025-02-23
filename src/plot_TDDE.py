import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special as sp

N = 50
D = 1


def analytical_solution(x, t, max_i=1000):
    sqrt_dt = np.sqrt(D * t) if t > 0 else 1e-10
    i_vals = np.arange(max_i)[:, None]

    term1 = sp.erfc((1 - x + 2 * i_vals) / (2 * sqrt_dt))
    term2 = sp.erfc((1 + x + 2 * i_vals) / (2 * sqrt_dt))

    c = np.sum(term1 - term2, axis=0)
    return c


if __name__ == '__main__':

    history = np.load("data/TDDE_(50x50)_10000.npy")  # Shape: (101, 101, 101)

    # Rotate the inner grid -90 degrees
    history = np.rot90(history, k=-1, axes=(1, 2))
    x = np.linspace(0, 1, N + 1)

    print(history.shape)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Plot the initial condition

    axs[0].plot(x, history[10_000, :, 0], label="t = 1")
    axs[0].plot(x, history[1_000, :, 0], label="t = 0.1")
    axs[0].plot(x, history[100, :, 0], label="t = 0.01")
    axs[0].plot(x, history[10, :, 0], label="t = 0.001")
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title("Numerical Solution")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("c")

    # Analytical solution (right) - Initialize with zeros
    analyticals = []
    for t in [1, 0.1, 0.01, 0.001]:
        analytical = analytical_solution(x, t)
        analyticals.append(analytical)
        axs[1].plot(x, analytical, label=f"t = {t}")
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title("Analytical Solution")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("c")

    axs[2].plot(x, (history[10_000, :, 0] - analyticals[0]), label=f"t = 1")
    print(history[10_000, :, 0])
    print(analytical[0])
    axs[2].plot(x, history[1_000, :, 0] - analyticals[1], label=f"t = 0.1")
    axs[2].plot(x, history[100, :, 0] - analyticals[2], label=f"t = 0.01")
    axs[2].plot(x, history[10, :, 0] - analyticals[3], label=f"t = 0.001")
    # print(analytical)
    axs[2].grid()
    axs[2].legend()
    # axs[2].set_ylim(-0.1, 1)
    axs[2].set_title("Difference between numerical and analytical solution")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("c")

    # Difference

    plt.show()

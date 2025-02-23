import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special as sp

N = 100
D = 1


def analytical_solution(x, t, max_i=1000):
    sqrt_dt = np.sqrt(D * t) if t > 0 else 1e-10
    i_vals = np.arange(max_i)[:, None]

    term1 = sp.erfc((1 - x + 2 * i_vals) / (2 * sqrt_dt))
    term2 = sp.erfc((1 + x + 2 * i_vals) / (2 * sqrt_dt))

    c = np.sum(term1 - term2, axis=0)
    return c


if __name__ == '__main__':

    # Shape: (101, 101, 101)
    history = np.load("data/TDDE_(100x100)_100000.npy")

    # Rotate the inner grid -90 degrees
    history = np.rot90(history, k=-1, axes=(1, 2))

    # Create side-by-side subplots
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    # Numerical solution (left)
    im1 = axs[0].imshow(history[0], cmap="inferno", origin="lower")
    axs[0].set_title("Numerical Solution")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # Analytical solution (right) - Initialize with zeros
    analytical_grid = (history[0])
    im2 = axs[1].imshow(analytical_grid, cmap="inferno", origin="lower")
    axs[1].set_title("Analytical Solution")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    diff = np.zeros(101)
    # trange = np.linspace(0, 1, 101)
    line = axs[2].plot(diff)
    # axs[2].set_xticks(np.arange(0, 101, 1), labels=[f"{t:.2f}" for t in trange])
    axs[2].set_title("Difference")
    axs[2].set_xlabel("Time step")
    axs[2].set_ylabel("Sum of differences")
    axs[2].set_ylim(0, 50)
    # axs[2].set_xlim(0, 1)

    # im3 = axs[2].imshow(history[0] - analytical_grid, cmap="inferno", origin="lower")

    def update(frame):
        # Update numerical solution
        im1.set_array(history[frame])
        axs[0].set_title(f"Numerical Solution (t={frame/100:.2f})")

        analytical = analytical_solution(np.linspace(0, 1, 101), frame / 100)
        print(analytical)
        analytical_grid = np.tile(analytical, (history.shape[1], 1)).T

        # Update analytical solution plot
        im2.set_array(analytical_grid)
        axs[1].set_title(f"Analytical Solution (t={frame/100:.2f})")

        diff[frame] = np.sum((history[frame, :, 0] - analytical))
        line[0].set_ydata(diff)

        return [im1, im2]

    ani = animation.FuncAnimation(
        fig, update, frames=history.shape[0], interval=50, blit=False)

    plt.show()

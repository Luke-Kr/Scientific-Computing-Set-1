import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special as sp

N = 100
D = 1


def animate(history):
    fig, ax = plt.subplots()
    im = ax.imshow(history[0], cmap="inferno", origin="lower")

    def update(frame):
        im.set_array(history[frame])
        ax.set_title(f"Time step: {frame/10000:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=history.shape[0], interval=50, blit=False)

    plt.show()


def show_times(history):
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))

    fig.suptitle("Diffusion spread of heat in a 2D grid")
    im = axs[0, 0].imshow(history[0], cmap="inferno",
                          origin="lower", extent=[0, 1, 0, 1])
    axs[0, 0].set_title("t = 0")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")

    axs[0, 1].imshow(history[10], cmap="inferno", origin="lower", extent=[0, 1, 0, 1])
    axs[0, 1].set_title("t = 0.001")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")

    axs[0, 2].imshow(history[100], cmap="inferno", origin="lower", extent=[0, 1, 0, 1])
    axs[0, 2].set_title("t = 0.01")
    axs[0, 2].set_xlabel("x")
    axs[0, 2].set_ylabel("y")

    axs[1, 0].imshow(history[1000], cmap="inferno", origin="lower", extent=[0, 1, 0, 1])
    axs[1, 0].set_title("t = 0.1")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")

    axs[1, 1].imshow(history[-1], cmap="inferno",
                     origin="lower", extent=[0, 1, 0, 1])
    axs[1, 1].set_title("t = 1")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("y")
    print(history[-1])

    axs[1][2].set_visible(False)
    axs[1][0].set_position([0.24, 0.125, 0.228, 0.343])
    axs[1][1].set_position([0.55, 0.125, 0.228, 0.343])

    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, pad=0.05)

    plt.show()


if __name__ == '__main__':

    history = np.load("data/TDDE_(50x50)_10000.npy")  # Shape: (101, 101, 101)

    # Rotate the inner grid -90 degrees
    history = np.rot90(history, k=-1, axes=(1, 2))

    # animate(history)
    show_times(history)

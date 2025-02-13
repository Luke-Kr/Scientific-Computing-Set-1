import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special as sp

N = 100
D = 1

def analyitical_solution(x, t, max_i=100):
    sqrt_dt = np.sqrt(D * t)
    i_vals = np.arange(max_i)[:, None]  # Create a column vector for broadcasting

    term1 = sp.erfc((1 - x + 2 * i_vals) / (2 * sqrt_dt))
    term2 = sp.erfc((1 + x + 2 * i_vals) / (2 * sqrt_dt))

    c = np.sum(term1 - term2, axis=0) 

    return c

if __name__ == '__main__':

    history = np.load("data/TDDE_(100x100)_100000.npy")  # Shape: (101, 101, 101)

    # Rotate the inner grid -90 degrees
    history = np.rot90(history, k=-1, axes=(1, 2))


    fig, ax = plt.subplots()
    im = ax.imshow(history[0], cmap="inferno", origin="lower")


    def update(frame):
        im.set_array(history[frame])
        ax.set_title(f"Time step: {frame/100:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return [im]

    # ani = animation.FuncAnimation(fig, update, frames=history.shape[0], interval=50, blit=False)

    # plt.show()

    print(history[10, :, 0])
    print(analyitical_solution(history[10, :, 0], 10/100))
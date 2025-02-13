import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special as sp

N = 100
D = 1

if __name__ == '__main__':

    history = np.load("data/TDDE_(100x100)_100000.npy")  # Shape: (101, 101, 101)

    # Rotate the inner grid -90 degrees
    history = np.rot90(history, k=-1, axes=(1, 2))
    x = np.linspace(0, 1, N + 1)
    
    plt.plot(x, history[100, :, 0] , label="1")
    plt.plot(x, history[10, :, 0], label="0.1")
    plt.plot(x, history[1, :, 0], label="0.01")
    # plt.plot(history[0, :, 0], label="0")
    plt.grid()
    plt.legend()
    plt.show()
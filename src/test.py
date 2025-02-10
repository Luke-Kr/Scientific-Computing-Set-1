import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

history = np.load("data/TDDE_(100x100)_100000.npy")  # Shape: (101, 101, 101)

# Rotate the inner grid -90 degrees
history = np.rot90(history, k=-1, axes=(1, 2))


fig, ax = plt.subplots()
im = ax.imshow(history[0], cmap="inferno", origin="lower")


def update(frame):
    im.set_array(history[frame])
    ax.set_title(f"Time step: {frame}") 
    return [im]

ani = animation.FuncAnimation(fig, update, frames=history.shape[0], interval=50, blit=False)

plt.show()
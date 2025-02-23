import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data
history = np.load("data/TDDE_(100x100)_100000.npy")  # Shape: (101, 101, 101)

# Rotate the inner grid -90 degrees
history = np.rot90(history, k=-1, axes=(1, 2))

# Set up figure and axis
fig, ax = plt.subplots()
im1 = ax.imshow(history[0], cmap="inferno",
                origin="lower", extent=[0, 1, 0, 1])
title = ax.set_title("Numerical Solution")

ax.set_xlabel("x")
ax.set_ylabel("y")


# Update function
def update(frame):
    im1.set_array(history[frame])
    title.set_text(f"Numerical Solution (t={frame/100:.2f})")
    return im1, title


# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=100, interval=100, blit=False)
ani.save("fig/TDDE_animation.gif", writer="imagemagick", fps=100)

plt.show()

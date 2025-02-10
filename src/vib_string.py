import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

# Parameters
L = 1.0  # Length of string
N = 100  # Number of spatial points
c = 1.0  # Wave speed
T = 2.0  # Total simulation time

# Discretization
dx = L / N
dt = 0.01
num_steps = int(T / dt)

# Initialize solution arrays
u = np.zeros((N + 1, 3))  # Three time layers (n-1, n, n+1)
x = np.linspace(0, L, N + 1)


def init_con_1(x):
    """Initial condition 1: Sine wave with frequency 1."""
    return np.sin(2 * np.pi * x)


def init_con_2(x):
    """Initial condition 2: Sine wave with frequency 2.5."""
    return np.sin(5 * np.pi * x)


def init_con_3(x):
    """Initial condition 3: Sine wave in the range [1/5, 2/5]."""
    a = np.zeros_like(x)
    mask = (1 / 5 <= x) & (x <= 2 / 5)  # Boolean mask for valid x values
    a[mask] = np.sin(5 * np.pi * x[mask])  # Apply sine only where mask is True
    return a


@jit(nopython=True)
def time_stepping(u, c, dt, dx, initial_condition, num_steps):
    """Perform time-stepping for the wave equation."""
    u[:, 0] = initial_condition
    history = np.zeros((N + 1, num_steps))  # Shape: (N + 1, num_steps)
    history[:, 0] = u[:, 0]

    for step in range(1, num_steps):
        for j in range(1, N):  # Start from 1 to N-1 to avoid boundary issues
            u[j, 2] = (2 * u[j, 1] - u[j, 0] +
                       (c * dt / dx) ** 2 * (u[j + 1, 1] - 2 * u[j, 1] + u[j - 1, 1]))

        # Apply boundary conditions
        u[0, 2] = 0  # Boundary condition at x=0
        u[N, 2] = 0  # Boundary condition at x=L

        u[:, 0] = u[:, 1]
        u[:, 1] = u[:, 2]

        history[:, step] = u[:, 1]

    return history  # Return the complete history of states


def run_simulation(u, x, c, dt, dx, init_cons, num_steps):
    """Run the simulation with different initial conditions."""
    results = []
    for init_con in init_cons:
        initial_condition = init_con(x)  # Evaluate the initial condition function
        result = time_stepping(u, c, dt, dx, initial_condition, num_steps)  # Pass the evaluated initial condition
        results.append(result)

    results = np.array(results)
    print(results.shape)
    return results


def plot_wave_states(results, x, num_time_points=5):
    """Plot wave states for each initial condition at evenly spaced time steps."""
    num_conditions = results.shape[0]
    time_indices = np.linspace(0.0, results.shape[2] - 1, num_time_points, dtype=int)  # Evenly spaced indices

    fig, axs = plt.subplots(num_conditions, 1, figsize=(10, 8))

    if num_conditions == 1:
        axs = [axs]  # Ensure axs is iterable for a single subplot

    for i in range(num_conditions):
        for t_idx in time_indices:
            axs[i].plot(x, results[i, :, t_idx], label=f'Time = {t_idx * dt:.2f}s')

        axs[i].set_title(f'Wave Evolution for Initial Condition {i + 1}')
        axs[i].set_xlabel('Position along the string (x)')
        axs[i].set_ylabel('Displacement (u)')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig('fig/wave_evolution.png')
    plt.show()


def animate_wave(results, x, init_condition_index, filename):
    """Animate the wave evolution for a given initial condition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, x[-1])
    ax.set_ylim(-20, 20)  # Adjust y-limits based on expected wave heights
    ax.set_title(
        f'Wave Evolution for Initial Condition {init_condition_index + 1}')
    ax.set_xlabel('Position along the string (x)')
    ax.set_ylabel('Displacement (u)')
    line, = ax.plot(x, results[init_condition_index, :, 0], color='b')  # Initial plot

    def update(frame):
        # Update the line with the new data
        line.set_ydata(results[init_condition_index, :, frame])
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=results.shape[2], blit=True)

    # Save the animation as a GIF
    ani.save(filename.replace('.mp4', '.gif'),
             writer='pillow', fps=30)  # Use Pillow writer


if __name__ == "__main__":
    # Run simulation
    init_cons = [init_con_1, init_con_2, init_con_3]
    results = run_simulation(u, x, c, dt, dx, init_cons, num_steps)
    plot_wave_states(results, x)
    for i in range(len(init_cons)):
        animate_wave(results, x, i, f'fig/wave_animation_condition_{i + 1}.mp4')
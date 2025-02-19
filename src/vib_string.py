import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

# Parameters
L = 1.0  # Length of string
N = 200  # Number of spatial points
c = 1.0  # Wave speed
T = 1  # Total simulation time
dx = L / N
dt = 0.001
num_steps = int(T / dt)

u = np.zeros((N + 1, 3))  # Three time layers (n-1, n, n+1)
x = np.linspace(0, L, N + 1)

def init_con_1(x):
    return np.sin(2 * np.pi * x)

def init_con_2(x):
    return np.sin(5 * np.pi * x)

def init_con_3(x):
    a = np.zeros_like(x)
    mask = (1 / 5 <= x) & (x <= 2 / 5)
    a[mask] = np.sin(5 * np.pi * x[mask])
    return a

@jit(nopython=True)
def time_stepping(u, c, dt, dx, initial_condition, num_steps, boundary='dirichlet'):
    """Perform time-stepping with proper initialization."""
    u[:, 0] = initial_condition
    u[:, 1] = initial_condition.copy()

    # Compute first step explicitly
    for j in range(1, N):
        u[j, 1] = u[j, 0] + 0.5 * ((c * dt / dx) ** 2) * (u[j + 1, 0] - 2 * u[j, 0] + u[j - 1, 0])

    history = np.zeros((N + 1, num_steps))
    history[:, 0] = u[:, 0]
    history[:, 1] = u[:, 1]

    for step in range(2, num_steps):
        for j in range(1, N):
            u[j, 2] = (2 * u[j, 1] - u[j, 0] + (c * dt / dx) ** 2 * (u[j + 1, 1] - 2 * u[j, 1] + u[j - 1, 1]))

        if boundary == 'dirichlet':
            u[0, 2] = 0
            u[N, 2] = 0
        elif boundary == 'neumann':
            u[0, 2] = u[1, 2]  # Free end at x=0
            u[N, 2] = u[N - 1, 2]  # Free end at x=L

        u[:, 0] = u[:, 1]
        u[:, 1] = u[:, 2]
        history[:, step] = u[:, 1]

    return history

def run_simulation(u, x, c, dt, dx, init_cons, num_steps):
    results = []
    for init_con in init_cons:
        initial_condition = init_con(x)
        result = time_stepping(u, c, dt, dx, initial_condition, num_steps)
        results.append(result)
    return np.array(results)

def plot_wave_states(results, x, num_time_points=3):
    num_conditions = results.shape[0]
    time_indices = np.linspace(0, results.shape[2] - 1, num_time_points, dtype=int)
    fig, axs = plt.subplots(num_conditions, 1, figsize=(10, 8))
    if num_conditions == 1:
        axs = [axs]

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
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, x[-1])
    y_min, y_max = results.min(), results.max()
    ax.set_ylim(y_min * 1.2, y_max * 1.2)
    ax.set_title(f'Wave Evolution for Initial Condition {init_condition_index + 1}')
    ax.set_xlabel('Position along the string (x)')
    ax.set_ylabel('Displacement (u)')
    line, = ax.plot(x, results[init_condition_index, :, 0], color='b')

    def update(frame):
        line.set_ydata(results[init_condition_index, :, frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=results.shape[2], blit=True, interval=1)
    ani.save(filename.replace('.mp4', '.gif'), writer='pillow')

if __name__ == "__main__":
    init_cons = [init_con_1, init_con_2, init_con_3]
    results = run_simulation(u, x, c, dt, dx, init_cons, num_steps)
    plot_wave_states(results, x)
    for i in range(len(init_cons)):
        animate_wave(results, x, i, f'fig/wave_animation_condition_{i + 1}.mp4')

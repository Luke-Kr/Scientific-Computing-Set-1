# Scientific-Computing-Set-1
Repository for Set 1 of the assignments for Scientific Computing [5284SCCO6Y].

Question 1.1 "Vibrating String":
    Run vib_string.py
    A plot showing the time development is saved under "fig/wave_evolution.png"
    Three animations are saved under "wave_animation_condition_x.gif", where x indicates the initial condition used.

Question 1.2 "The Time Dependent Diffusion Equation":
    Run TDDE.py
    This will plot the state of the grid at t=1, and timesteps in between will be saved in "data/TDDE_(`N`x`N`)_`timesteps`.npy.
    An animation can be generated from this timestep by running animate_TDDE.py.
    plot_TDDE.py will plot lines representing single collumns at a certain timestep of the grid, which are then compared to the analytical solution.
    show_diffusion.py will generated several heatmaps of different timesteps.

Question 1.4 "The Jacobi Iteration":
    jacobi.py will perform the jacobi iteration to solve the 2D Laplace Equation.

Question 1.5 "The Gauss-Seidel Iteration:
    gauss_seidel.py will perform the gaus-seidel iteration to solve the 2D Laplace Equation.

Question 1.6 "Successive Over Relaxation"
   SOR.py will perform the gauss-seidel iteration to solve the 2D Laplace Equation. Objects can be placed in the domain either as sinks or isolated objects using a mask.
   convergence_measure.py makes use of jacobi.py, gauss_seidel.py and SOR.py to measure and plot rates of convergence for the different algorithms.
   optimal_omega.py generates a heatmap of SOR that can be used to find the optimal omega value for this problem.
   

## Dependencies
The following libraries are used in this project:
- `numpy`
- `matplotlib`
- `numba`

All dependencies can be installed via `pip`.

```
pip install requirements.txt
```
## Compatibility
This code has been tested with **Python 3.12.0** on **Windows 10/11**.


Particle Swarm Optimization (PSO) Visual GUI
An interactive Python GUI application for experimenting with Particle Swarm Optimization (PSO). This tool allows users to select benchmark functions, set variable bounds,
 configure PSO parameters, visualize particle behavior in real time, and save the optimization as an animated GIF.

Features
-Objective Function Selection:
Choose from standard benchmark functions (Ackley, Beale, Rastrigin, Rosenbrock, Sphere, etc.) with hover previews.
-Custom Bounds:
Set the variable search space interactively.
-PSO Parameter Configuration:
Adjust parameters such as particle count, inertia, cognitive/social coefficients, and iteration count.
-Real-Time Visualization:
Animated 3D surface plot of the objective function and particle positions over iterations.
-Save Animations:
Optionally save the PSO optimization process as a .gif with a descriptive filename.
-Interactive GUI:
Multiple GUI windows using Tkinter for a smooth user experience.

Usage
1. Run the program PSO_Intro.py
2. Select an Objective Function:
Hover over buttons to see a preview, then click a function to choose it.
3. Set Variable Bounds:
Enter the min/max values for each dimension.
4. Configure PSO Parameters:
Adjust parameters such as N_p (particles), c1, c2, w, k, p, v_min, v_max, and iters.
5. View Optimization Animation:
The program runs PSO and displays particle movement on the function surface.
6. Save Animation (optional):
After the animation finishes, you can save it as a .gif.

Supported Objective Functions
-Ackley
-Beale
-Booth
-Bukin6
-Crossintray
-Easom
-Eggholder
-Goldstein
-Himmelblau
-Holdertable
-Levi
-Matyas
-Rastrigin
-Rosenbrock
-Schaffer2
-Sphere
-Threehump
-Custom (disabled by default)

References
-PySwarms Documentation
https://pyswarms.readthedocs.io/en/latest/index.html
-Inspired by
https://www.youtube.com/watch?v=E-tBOEoFLXs
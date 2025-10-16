"""
Particle Swarm Optimization (PSO) Visual GUI with Function Selection, Bound Setup, 
Parameter Configuration, and Animated Visualization.
Author:     Marvin Kösters
Based on:   https://www.youtube.com/watch?v=E-tBOEoFLXs
Date:       2025-10-16
"""


# -------------------------------------------------------------------------
# Import Modules
# -------------------------------------------------------------------------
import os
import webbrowser
from datetime import datetime

import numpy as np                                  # Math toolbox
import tkinter as tk                                # GUIs
from tkinter import messagebox, filedialog as fd
from PIL import Image, ImageTk

import matplotlib.pyplot as plt                     # Visualization with Python
from matplotlib.animation import FuncAnimation, PillowWriter
from pyswarms.utils.functions.single_obj import rosenbrock, ackley, sphere, rastrigin   # Objective functions
import pyswarms as ps                               # PSO toolkit

#from mpl_toolkits.mplot3d import Axes3D
#import time                                                 # OS time toolkit
#from screeninfo import get_monitors                         # Like get biggest monitor
#import platform
#import numpy as np
#import tkinter.filedialog as fd




# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def open_link(event):   # Open PySwarms documentation
    webbrowser.open_new("https://pyswarms.readthedocs.io/en/latest/")


# -------------------------------------------------------------------------
# Objective Functions
# -------------------------------------------------------------------------
def beale(x):
    x1 = x[:,0]; x2 = x[:,1]
    return ((1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2)

def custom(x):
    return (x**2).sum(axis=1)


# -------------------------------------------------------------------------
# GUI 1 — Objective Function Selection
# -------------------------------------------------------------------------
button_labels = [   # Button labels
    "Ackley","Beale","Booth","Bukin6","Crossintray","Easom",
    "Eggholder","Goldstein","Himmelblau","Holdertable","Levi",
    "Matyas","Rastrigin","Rosenbrock","Schaffer2","Sphere",
    "Threehump","Custom"
]

function_map = {    # Function mapping to actual functions
    "Ackley": ackley,
    "Beale": beale,
    "Booth": lambda x: (x[:,0] + 2*x[:,1] - 7)**2 + (2*x[:,0] + x[:,1] - 5)**2,
    "Bukin6": lambda x: 100*np.sqrt(np.abs(x[:,1]-0.01*x[:,0]**2)) + 0.01*np.abs(x[:,0]+10),
    "Crossintray": lambda x: -0.0001*(np.abs(np.sin(x[:,0])*np.sin(x[:,1])*np.exp(np.abs(100 - np.sqrt(x[:,0]**2 + x[:,1]**2)/np.pi)))+1)**0.1,
    "Easom": lambda x: -np.cos(x[:,0])*np.cos(x[:,1])*np.exp(-((x[:,0]-np.pi)**2 + (x[:,1]-np.pi)**2)),
    "Eggholder": lambda x: -(x[:,1]+47)*np.sin(np.sqrt(np.abs(x[:,1]+x[:,0]/2+47))) - x[:,0]*np.sin(np.sqrt(np.abs(x[:,0]-(x[:,1]+47)))),
    "Goldstein": lambda x: (1+(x[:,0]+x[:,1]+1)**2*(19-14*x[:,0]+3*x[:,0]**2-14*x[:,1]+6*x[:,0]*x[:,1]+3*x[:,1]**2))*\
                           (30+(2*x[:,0]-3*x[:,1])**2*(18-32*x[:,0]+12*x[:,0]**2+48*x[:,1]-36*x[:,0]*x[:,1]+27*x[:,1]**2)),
    "Himmelblau": lambda x: (x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2,
    "Holdertable": lambda x: -np.abs(np.sin(x[:,0])*np.cos(x[:,1])*np.exp(np.abs(1 - np.sqrt(x[:,0]**2 + x[:,1]**2)/np.pi))),
    "Levi": lambda x: np.sin(3*np.pi*x[:,0])**2 + (x[:,0]-1)**2*(1+np.sin(3*np.pi*x[:,1])**2) + (x[:,1]-1)**2*(1+np.sin(2*np.pi*x[:,1])**2),
    "Matyas": lambda x: 0.26*(x[:,0]**2 + x[:,1]**2) - 0.48*x[:,0]*x[:,1],
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Schaffer2": lambda x: 0.5 + (np.sin(x[:,0]**2 - x[:,1]**2)**2 - 0.5)/((1 + 0.001*(x[:,0]**2 + x[:,1]**2))**2),
    "Sphere": sphere,
    "Threehump": lambda x: 2*x[:,0]**2 - 1.05*x[:,0]**4 + (x[:,0]**6)/6 + x[:,0]*x[:,1] + x[:,1]**2,
    "Custom": custom
}

root = tk.Tk()  # Create main window
root.title("Pyswarms Reference and Objective Selection")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

top_frame = tk.Frame(root, height=200, bg="white")          # Frames in window
top_frame.pack(side="top", fill="x")
bottom_frame = tk.Frame(root, height=screen_height-200)
bottom_frame.pack(side="top", fill="both", expand=True)

left_frame = tk.Frame(bottom_frame, width=200, bg="white") 
left_frame.pack(side="left", fill="y")
right_frame = tk.Frame(bottom_frame, bg="white")
right_frame.pack(side="left", fill="both", expand=True)

try:    # Reference image on top, PySwarms documentation
    img_ref = Image.open("images/pyswarms.png")
    img_ref = img_ref.resize((screen_width, 200))
    img_ref_tk = ImageTk.PhotoImage(img_ref)
    label_ref = tk.Label(top_frame, image=img_ref_tk, cursor="hand2")
    label_ref.pack()
    label_ref.bind("<Button-1>", open_link)
    label_ref.image = img_ref_tk
except:
    label_ref = tk.Label(top_frame, text="Reference Image Not Found", bg="white")
    label_ref.pack()

images = []     # Function loading images
for label in button_labels:
    try:
        img = Image.open(f"images/{label}.png").resize((400, 300))
        images.append(ImageTk.PhotoImage(img))
    except:
        images.append(None)
global_images = images  # Keep copy of image

img_label = tk.Label(right_frame, bg="white")   # Preview of right image
img_label.pack(expand=True, padx=10, pady=10)

global_images = {}              # Loading image based on label
max_w, max_h = 400, 300
for label in button_labels:
    try:
        img = Image.open(f"images/{label}.png")
        w, h = img.size
        ratio = min(max_w / w, max_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        global_images[label] = ImageTk.PhotoImage(img.resize((new_w, new_h), Image.LANCZOS))
    except:
        global_images[label] = None

def on_hover(event, label):             # Hover function based on label
    img = global_images.get(label)
    if img:
        img_label.config(image=img)
        img_label.image = img
    else:
        img_label.config(image="")
        img_label.image = None

for lbl in button_labels:               # Creation of buttons
    btn = tk.Button(left_frame, text=lbl, width=15)
    btn.pack(pady=5)
    btn.bind("<Enter>", lambda e, l=lbl: on_hover(e, l))
    if lbl == "Custom":
        btn.config(state="disabled")
    else:
        btn.config(command=lambda l=lbl: on_click(l))

chosen_obj = None                       # Objective buttons
chosen_label = None
def on_click(label):
    global chosen_obj, chosen_label
    chosen_obj = function_map[label]
    chosen_label = label
    root.destroy()                      # Close window after selection

root.mainloop()                         # End of 1st window

if chosen_obj is None:                  # Feedback after closing window
    raise SystemExit("No objective function selected!")

objective_function = chosen_obj         # Assign to a consistent variable
obj = objective_function                # For visualization titles
print("Objective function selected:", chosen_label)


# -------------------------------------------------------------------------
# Setting Defaults
# -------------------------------------------------------------------------
function_bounds = {     # Function bounds
    "Ackley": (-5, 5, -5, 5), "Beale": (-4.5, 4.5, -4.5, 4.5), "Booth": (-10, 10, -10, 10),
    "Bukin6": (-15, -5, -3, 3), "Crossintray": (-10, 10, -10, 10), "Easom": (-100, 100, -100, 100),
    "Eggholder": (-512, 512, -512, 512), "Goldstein": (-2, 2, -2, 2), "Himmelblau": (-5, 5, -5, 5),
    "Holdertable": (-10, 10, -10, 10), "Levi": (-10, 10, -10, 10), "Matyas": (-10, 10, -10, 10),
    "Rastrigin": (-5.12, 5.12, -5.12, 5.12), "Rosenbrock": (-5, 10, -5, 10), "Schaffer2": (-100, 100, -100, 100),
    "Sphere": (-5.12, 5.12, -5.12, 5.12), "Threehump": (-5, 5, -5, 5), "Custom": (-5, 5, -5, 5)
}

a, b, c, d = function_bounds.get(chosen_label, (-5.12, 5.12, -5.12, 5.12))
default_vals = {"a": a, "b": b, "c": c, "d": d}
bounds = None

ps_params = {           # Set parameters
    "n_dim": 2,
    "N_p": 30,
    "c1": 0.5,
    "c2": 0.3,
    "w": 0.9,
    "k": 9,
    "p": 2,
    "v_min": 0,
    "v_max": 1,
    "iters": 100       
}


# -------------------------------------------------------------------------
# GUI 2 — Bounds Input Window
# -------------------------------------------------------------------------
root = tk.Tk()
root.title("Set Bounds")
window_width, window_height = 600, 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

left_frame = tk.Frame(root, width=window_width//2, bg="white")
left_frame.pack(side="left", fill="both", expand=True)
right_frame = tk.Frame(root, width=window_width//2, bg="white")
right_frame.pack(side="right", fill="both", expand=True)

img_label = tk.Label(left_frame, bg="white")
img_label.pack(expand=True)
try:
    img = Image.open(f"images/{chosen_label}.png").resize((window_width//2-40, window_height-40))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk
except:
    img_label.config(text=f"Image not found for {chosen_label}")

main_frame = tk.Frame(right_frame, bg="white")
main_frame.place(relx=0.5, rely=0.5, anchor="center")

def submit():
    global bounds
    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        c = float(entry_c.get())
        d = float(entry_d.get())
        bounds = (np.array([a, c]), np.array([b, d]))
        root.destroy()
    except ValueError:
        error_label.config(text="Please enter valid numbers!")

key_map = {"p1_min": "a", "p1_max": "b", "p2_min": "c", "p2_max": "d"}
for idx, gui_key in enumerate(["p1_min", "p1_max", "p2_min", "p2_max"]):
    row_frame = tk.Frame(main_frame, bg="white")
    row_frame.pack(pady=5)
    tk.Label(row_frame, text=f"{gui_key}:", bg="white").pack(side="left", padx=5)
    entry = tk.Entry(row_frame, width=10)
    entry.insert(0, str(default_vals[key_map[gui_key]]))
    entry.pack(side="left")
    if gui_key == "p1_min": entry_a = entry
    elif gui_key == "p1_max": entry_b = entry
    elif gui_key == "p2_min": entry_c = entry
    elif gui_key == "p2_max": entry_d = entry

error_label = tk.Label(main_frame, text="", fg="red", bg="white")
error_label.pack(pady=5)
tk.Button(main_frame, text="Set", command=submit).pack(pady=10)
root.bind("<Escape>", lambda e: root.destroy())
root.mainloop()

if bounds is None:
    raise SystemExit("Bounds not set!")


# -------------------------------------------------------------------------
# GUI 3 — PSO Parameter Configuration
# -------------------------------------------------------------------------
root = tk.Tk()
root.title("PSO Parameter Configuration")
window_width, window_height = 1000, 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

left_frame = tk.Frame(root, width=200, bg="white")          # Frames
left_frame.pack(side="left", fill="y")
middle_frame = tk.Frame(root, width=400, bg="white")
middle_frame.pack(side="left", fill="both", expand=True)
right_frame = tk.Frame(root, width=400, bg="white")
right_frame.pack(side="left", fill="both", expand=True)

try:        # Top PSO image
    img_pso = Image.open("images/PSO/PSO.png")
    w, h = img_pso.size
    ratio = min(380 / w, 200 / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_pso_tk = ImageTk.PhotoImage(img_pso.resize((new_w, new_h), Image.LANCZOS))
    top_img_label = tk.Label(middle_frame, image=img_pso_tk, bg="white", cursor="hand2")
    top_img_label.pack(pady=10)
    top_img_label.image = img_pso_tk
    
    def open_full_image(event):                 # Click event to open full-size image
        top = tk.Toplevel(root)
        top.title("PSO Full Image")
        full_img = ImageTk.PhotoImage(img_pso)  # Keep original size
        lbl = tk.Label(top, image=full_img)
        lbl.image = full_img                    # Keep reference
        lbl.pack()
    
    top_img_label.bind("<Button-1>", open_full_image)

except:
    top_img_label = tk.Label(middle_frame, text="PSO.png not found", bg="white")
    top_img_label.pack(pady=10)

img_label = tk.Label(middle_frame, bg="white")  # Middle image label for parameter hover 
img_label.pack(expand=True, padx=10, pady=10)

param_labels = list(ps_params.keys())           # Load hover images 
global_param_images = []

def load_and_resize_keep_ratio(path, max_w, max_h):
    img = Image.open(path)
    w, h = img.size
    ratio = min(max_w / w, max_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    return ImageTk.PhotoImage(img.resize((new_w, new_h), Image.LANCZOS))

for label in param_labels:
    try:
        img = load_and_resize_keep_ratio(f"images/PSO/{label}.png", 380, 460)
        global_param_images.append(img)
    except:
        global_param_images.append(None)

def on_hover(idx):                              # Load hover images 
    if global_param_images[idx]:
        img_label.config(image=global_param_images[idx], cursor="hand2")  # Cursor icon
        img_label.image = global_param_images[idx]
        
        def open_full_image(event, path=f"images/PSO/{param_labels[idx]}.png"): # Bind click event to open full-size image
            try:
                img = Image.open(path)                          # Original full-size
                top = tk.Toplevel(root)
                top.title(param_labels[idx] + " Full Image")
                full_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(top, image=full_img)
                lbl.image = full_img                            # Keep reference
                lbl.pack()
            except:
                tk.messagebox.showerror("Error", f"Cannot open image {path}")

        img_label.bind("<Button-1>", open_full_image)

    else:
        img_label.config(image="", cursor="")
        img_label.image = None
        img_label.unbind("<Button-1>")                          # Remove click if no image

def on_leave(event):
    img_label.config(image="", cursor="")
    img_label.image = None
    img_label.unbind("<Button-1>")

for idx, key in enumerate(param_labels):                        # Left buttons with parameter name
    btn = tk.Label(left_frame, text=key, width=15, bg="lightgrey", relief="raised")
    if key in ["N_p", "c1"]:
        btn.pack(pady=15)       # Extra spacing
    else:
        btn.pack(pady=5)
    btn.bind("<Enter>", lambda e, i=idx: on_hover(i))
    
entries = {}                # Right side entries
for key in param_labels:
    row = tk.Frame(right_frame, bg="lightgrey")
    if key in ["N_p", "c1"]:
        row.pack(pady=15, anchor="w")  # Extra spacing
    else:
        row.pack(pady=5, anchor="w")
    tk.Label(row, text=key + ":", width=10, anchor="w", bg="lightgrey").pack(side="left")
    entry = tk.Entry(row, width=10)
    entry.insert(0, str(ps_params[key]))
    entry.pack(side="left")
    if key == "n_dim":
        entry.config(state="disabled", disabledbackground="white", disabledforeground="black")
    entries[key] = entry

def submit_params():        # Submit button
    global ps_params
    try:
        for key, entry in entries.items():
            if key == "n_dim":
                continue
            if key in ["N_p", "k", "p", "iters"]:
                val = int(entry.get())
            else:
                val = float(entry.get())
            ps_params[key] = val
        root.destroy()
    except ValueError:
        error_label.config(text="Please enter valid numbers!")

error_label = tk.Label(right_frame, text="", fg="red", bg="white")
error_label.pack(pady=5)
tk.Button(right_frame, text="Set Parameters", command=submit_params).pack(pady=10)
root.bind("<Escape>", lambda e: root.destroy())

root.mainloop()


# -------------------------------------------------------------------------
# PSO Optimization 
# -------------------------------------------------------------------------
n_dim = 2
N_p = ps_params["N_p"]
c1 = ps_params["c1"]; c2 = ps_params["c2"]; w = ps_params["w"]
k = ps_params["k"]; p = ps_params["p"]
v_min = ps_params["v_min"]; v_max = ps_params["v_max"]
options = {'c1':c1,'c2':c2,'w':w,'k':k,'p':p}

optimizer = ps.single.GlobalBestPSO(n_particles=N_p, dimensions=n_dim,
                                    bounds=bounds, options=options,
                                    velocity_clamp=(v_min,v_max))

def safe_objective(x):
    lower, upper = bounds
    x = np.clip(x, lower, upper)
    return objective_function(x)

cost, pos = optimizer.optimize(safe_objective, iters=ps_params["iters"])

pos_history = np.array(optimizer.pos_history)


# -------------------------------------------------------------------------
# PSO Visualization
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title(f"Objective Function - {obj.__name__.capitalize()}")
ax1.set_xlabel(r'p1'); ax1.set_ylabel(r'p2')

X = np.linspace(bounds[0][0], bounds[1][0], 100)
Y = np.linspace(bounds[0][1], bounds[1][1], 100)
X, Y = np.meshgrid(X, Y)
Z = np.array([safe_objective(np.array([[x,y]])).item() for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
ax1.plot_surface(X,Y,Z, cmap='viridis', alpha=0.8)
fig.colorbar(ax1.plot_surface(X,Y,Z, cmap='viridis', alpha=0.8), ax=ax1, orientation='horizontal', shrink=0.6, aspect=40, pad=0.2)

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Particle Distribution")
ax2.set_xlabel(r'p1'); ax2.set_ylabel(r'p2')
ax2.set_xlim(bounds[0][0], bounds[1][0]); ax2.set_ylim(bounds[0][1], bounds[1][1])
z_min, z_max = np.nanmin(Z), np.nanmax(Z)
ax1.set_zlim(z_min, z_max); ax2.set_zlim(z_min, z_max)
for ax in [ax1, ax2]: ax.set_box_aspect([1,1,0.5])

x = pos_history[0,:,0]; y = pos_history[0,:,1]
z = [safe_objective(np.array([p])).item() for p in pos_history[0]]
particles = ax2.scatter(x,y,z, c='b', marker='o', label='Particles')

def animate(i):
    x = pos_history[i,:,0]
    y = pos_history[i,:,1]
    z = [safe_objective(np.array([p])).item() for p in pos_history[i]]
    particles._offsets3d = (x, y, z)
    iter_num = i + 1  
    ax2.set_title(f'Particle Distribution - Iteration {iter_num}')
    return (particles,)

plt.subplots_adjust(wspace=0.3)
ax2.legend(loc='upper right')
ani = FuncAnimation(fig, animate, frames=int(pos_history.shape[0]), interval=200, blit=False, repeat=False)

plt.show()

# -------------------------------------------------------------------------
# Saving Animation
# -------------------------------------------------------------------------
root = tk.Tk()      # Winow, Save animation?
root.withdraw()     # Hide main window

save_answer = messagebox.askyesno("", "Do you want to save the PSO animation?")  # Ask user

if save_answer:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")   # Timestamp

    p1_min, p1_max = bounds[0][0], bounds[1][0]             # Extract bounds
    p2_min, p2_max = bounds[0][1], bounds[1][1]
    
    filename_parts = [                                      # Create descriptive filename
        timestamp,
        f"PSO_{chosen_label}",
        f"p1_min_{p1_min}",
        f"p1_max_{p1_max}",
        f"p2_min_{p2_min}",
        f"p2_max_{p2_max}",
        f"n_dim_{ps_params['n_dim']}",
        f"Np_{ps_params['N_p']}",
        f"c1_{ps_params['c1']}",
        f"c2_{ps_params['c2']}",
        f"w_{ps_params['w']}",
        f"k_{ps_params['k']}",
        f"p_{ps_params['p']}",
        f"v_min_{ps_params['v_min']}",
        f"v_max_{ps_params['v_max']}",
        f"iters_{ps_params['iters']}",
    ]

    safe_name = "_".join(map(str, filename_parts)).replace("-", "m").replace(".", "p")
    default_filename = safe_name + ".gif"

    file_path = fd.asksaveasfilename(               # Open Save dialog 
        title="Save PSO animation as GIF",
        defaultextension=".gif",
        initialfile=default_filename,
        filetypes=[("GIF Image", "*.gif"), ("All Files", "*.*")]
    )
    
    if file_path:       # Save the animation 
        try:
            print(f"Saving animation to {file_path} ...")
            writer = PillowWriter(fps=10)
            ani.save(file_path, writer=writer)
            print("Animation saved successfully as GIF!")
        except Exception as e:
            print("Failed to save animation:", e)
    else:
        print("No file selected — animation not saved.")
else:
    print("User chose not to save the animation.")


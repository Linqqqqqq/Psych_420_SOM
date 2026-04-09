import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from som_1D_model import (
    n_nodes, eta, sigma, find_bmu,
    neighbourhood, update_weights
)

# Inputs
inputs = np.array([0.1, 0.4, 0.9, 0.7])

# Training parameters
eta_initial = 0.5
sigma_initial = 3.0
n_iterations = 200
tau = 200

# Initialize weights
np.random.seed(10)
weights = np.random.rand(n_nodes)

# Store all frames
weight_history = []

# TRAINING LOOP (record frames)
for t in range(n_iterations):
    for x in inputs:
        c = find_bmu(weights, x)
        alpha = eta(t, eta_initial, tau)
        sig = max(sigma(t, sigma_initial, tau), 0.1)
        h = neighbourhood(n_nodes, c, sig)
        weights = update_weights(weights, x, alpha, h)

    weight_history.append(weights.copy())

# ANIMATION SETUP
fig, ax = plt.subplots(figsize=(8, 4))

def update(frame):
    ax.clear()
    ax.plot(weight_history[frame], marker='o')
    ax.set_ylim(0, 1)
    ax.set_title(f"Weights at iteration {frame}")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Weight value")

ani = FuncAnimation(fig, update, frames=len(weight_history), interval=50)

# SAVE GIF
ani.save("som_weights.gif", writer=PillowWriter(fps=20))

plt.show()

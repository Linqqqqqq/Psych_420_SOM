import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(10)

n_nodes = 9
input_dim = 2
weights = np.random.rand(n_nodes, input_dim)

def eta(t, eta_initial, tau):
    return eta_initial * np.exp(-t / tau)

def sigma(t, sigma_initial, tau):
    return sigma_initial * np.exp(-t / tau)

def find_bmu(weights, x):
    distances = np.sum((weights - x)**2, axis=1)
    return np.argmin(distances)

def neighbourhood(n_nodes, c, sig):
    h = np.zeros(n_nodes)
    for i in range(n_nodes):
        dist_sq = (i - c)**2
        h[i] = np.exp(-dist_sq / (2 * sig**2))
    return h

def update_weights(weights, x, eta, h):
    weights += eta * h[:, np.newaxis] * (x - weights)
    return weights

inputs = np.array([
    [0.1, 0.2],
    [0.4, 0.3],
    [0.9, 0.8],
    [0.7, 0.6]
])
grid_size = int(np.sqrt(n_nodes))

eta_initial = 0.3
sigma_initial = grid_size
n_iterations = 100
tau = n_iterations / np.log(sigma_initial + 1e-8)

# U-Matrix Computation

def compute_u_matrix(weights):
    weights_grid = weights.reshape(grid_size, grid_size, input_dim)
    u_matrix = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            neighbours = []
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = i+dx, j+dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    neighbours.append(weights_grid[nx, ny])

            if neighbours:
                distances = [np.linalg.norm(weights_grid[i,j] - n) for n in neighbours]
                u_matrix[i,j] = np.mean(distances)

    return u_matrix


u_matrices = []
weights_history = []

for t in range(n_iterations):
    for x in inputs:
        c = find_bmu(weights, x)
        alpha = eta(t, eta_initial, tau)
        sig = max(sigma(t, sigma_initial, tau), 1e-8)
        h = neighbourhood(n_nodes, c, sig)
        weights = update_weights(weights, x, alpha, h)

    # store every iteration
    u_matrices.append(compute_u_matrix(weights))
    weights_history.append(weights.copy())

print("\nFinal weights:\n", weights)

# Animation - U-Matrix & Weights

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

u_max = np.max(u_matrices)

def update(frame):
    ax1.clear()
    ax2.clear()

    # Left Side: U-Matrix
    ax1.set_title(f"U-Matrix (iteration {frame})")

    im = ax1.imshow(
        u_matrices[frame],
        cmap='viridis',
        vmin=0,
        vmax=u_max
    )

    # Draw grid lines
    for i in range(grid_size + 1):
        ax1.axhline(i - 0.5, color='white', linewidth=1)
        ax1.axvline(i - 0.5, color='white', linewidth=1)

    # Node centers
    for i in range(grid_size):
        for j in range(grid_size):
            ax1.plot(j, i, 'o', color='red', markersize=6)

    ax1.set_xticks([])
    ax1.set_yticks([])

    # Right Side: Weight Updates
    ax2.set_title("Weight vectors in input space")

    # Plot training samples
    ax2.scatter(inputs[:,0], inputs[:,1], c='blue', s=60, label="Inputs")

    # Plot weight vectors
    w = weights_history[frame]
    ax2.scatter(w[:,0], w[:,1], c='red', s=60, label="Weights")

    # Connect weights in SOM grid order
    for i in range(grid_size):
        for j in range(grid_size - 1):
            idx1 = i * grid_size + j
            idx2 = i * grid_size + (j + 1)
            ax2.plot([w[idx1,0], w[idx2,0]], [w[idx1,1], w[idx2,1]], 'k-', linewidth=1)

    for j in range(grid_size):
        for i in range(grid_size - 1):
            idx1 = i * grid_size + j
            idx2 = (i + 1) * grid_size + j
            ax2.plot([w[idx1,0], w[idx2,0]], [w[idx1,1], w[idx2,1]], 'k-', linewidth=1)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.legend(loc="upper left")
    ax2.text(0.02, 0.95, f"Iteration {frame}", transform=ax2.transAxes)

    return [im]

anim = FuncAnimation(fig, update, frames=len(u_matrices), interval=120, blit=False)

writer = PillowWriter(fps=5)
anim.save("som_side_by_side.gif", writer=writer)

print("GIF saved as som_side_by_side.gif")

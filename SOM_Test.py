import numpy as np

np.random.seed(10)
weights = np.random.rand(3, 3)
print(weights)

def eta(t, eta_initial, tau):
    return eta_initial * np.exp(-t / tau)

def sigma(t, sigma_initial, tau):
    return sigma_initial * np.exp(-t / tau)

def find_bmu(weights, x):
    distances = (weights - x)**2
    return np.unravel_index(np.argmin(distances), weights.shape)
    
def neighborhood(weights_shape, c, sig):
    rows, cols = weights_shape
    h = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            dist_sq = (i - c[0])**2 + (j - c[1])**2
            h[i, j] = np.exp(-dist_sq / (2 * sig**2))
    return h

def update_weights(weights, x, c, eta, h):
    weights += eta * h * (x - weights)
    return weights
   
# Parameters
weights = np.random.rand(3,3)
inputs = [0.1, 0.4, 0.9, 0.7]
eta_initial = 0.5
sigma_initial = 1.0
tau = 50
n_iterations = 100

for t in range(n_iterations):
    for x in inputs:
        c = find_bmu(weights, x)
        alpha = eta(t, eta_initial, tau)
        sig = sigma(t, sigma_initial, tau)
        h = neighborhood(weights.shape, c, sig)
        weights = update_weights(weights, x, c, alpha, h)

print(f"Weights after iteration {t}:\n{weights}\n")

import matplotlib.pyplot as plt

def plot_som(weights, title="SOM Grid"):
    plt.imshow(weights, cmap="viridis")
    plt.colorbar(label="Weight value")
    plt.title(title)
    plt.show()
plot_som(weights, title=f"After iteration {t}")
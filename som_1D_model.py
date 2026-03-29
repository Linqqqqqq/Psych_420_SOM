import numpy as np

# 1D SOM: weights are a 1D array
n_nodes = 10

def eta(t, eta_initial, tau):
    return eta_initial * np.exp(-t / tau)

def sigma(t, sigma_initial, tau):
    return sigma_initial * np.exp(-t / tau)

def find_bmu(weights, x):
    distances = (weights - x)**2
    return np.argmin(distances)

def neighbourhood(n_nodes, c, sig):
    h = np.zeros(n_nodes)
    for i in range(n_nodes):
        dist_sq = (i - c)**2
        h[i] = np.exp(-dist_sq / (2 * sig**2))
    return h

def update_weights(weights, x, eta, h):
    weights += eta * h * (x - weights)
    return weights


def train_som(inputs, eta_initial=0.5, sigma_initial=1.0, n_iterations=1000):
    np.random.seed(10)
    weights = np.random.rand(n_nodes)
    tau = n_iterations / np.log(sigma_initial + 1e-8)

    bmu_history = []

    for t in range(n_iterations):
        for x in inputs:
            c = find_bmu(weights, x)
            bmu_history.append(c)   # record BEFORE updating

            alpha = eta(t, eta_initial, tau)
            sig = max(sigma(t, sigma_initial, tau), 1e-8)
            h = neighbourhood(n_nodes, c, sig)
            weights = update_weights(weights, x, alpha, h)

    return weights, bmu_history
    


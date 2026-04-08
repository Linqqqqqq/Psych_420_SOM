import numpy as np
import matplotlib.pyplot as plt
 
data = np.vstack([
    np.random.normal([0, 0], 0.5, (100, 2)),
    np.random.normal([3, 3], 0.5, (100, 2)),
    np.random.normal([0, 4], 0.5, (100, 2))
])
 
labels = np.array([0]*100 + [1]*100 + [2]*100)
 
grid_size = 10
input_dim = 2
 
weights = np.random.rand(grid_size, grid_size, input_dim)
 
def find_bmu(sample):
    distances = np.linalg.norm(weights - sample, axis=2)
    return np.unravel_index(np.argmin(distances), distances.shape)
 
def update_weights(sample, bmu, lr, radius):
    for x in range(grid_size):
        for y in range(grid_size):
            dist_to_bmu = np.linalg.norm(np.array([x, y]) - np.array(bmu))
            
            if dist_to_bmu < radius:
                influence = np.exp(-dist_to_bmu**2 / (2*(radius**2)))
                weights[x, y] += lr * influence * (sample - weights[x, y])
 
epochs = 100
 
for epoch in range(epochs):
    lr = 0.5 * (1 - epoch / epochs)     
    radius = grid_size/2 * (1 - epoch / epochs)  
    
    for sample in data:
        bmu = find_bmu(sample)
        update_weights(sample, bmu, lr, radius)
 
plt.figure(figsize=(6,6))
plt.title("SOM Grid")
 
for x in range(grid_size):
    for y in range(grid_size):
        plt.scatter(x, y, c='gray')
 
plt.grid()
plt.xlim(-1, grid_size)
plt.ylim(-1, grid_size)
plt.show()
 
plt.figure(figsize=(6,6))
plt.title("Data mapped to SOM")
 
colors = ['red', 'blue', 'green']
 
for i, sample in enumerate(data):
    bmu = find_bmu(sample)
    plt.scatter(bmu[0], bmu[1], c=colors[labels[i]], s=20)
 
plt.grid()
plt.xlim(-1, grid_size)
plt.ylim(-1, grid_size)
plt.show()
 
u_matrix = np.zeros((grid_size, grid_size))
 
for x in range(grid_size):
    for y in range(grid_size):
        neighbors = []
        
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbors.append(weights[nx, ny])
        
        if neighbors:
            distances = [np.linalg.norm(weights[x,y] - n) for n in neighbors]
            u_matrix[x,y] = np.mean(distances)
 
plt.figure(figsize=(6,6))
plt.title("U-Matrix")
 
plt.imshow(u_matrix)
plt.colorbar()
 
plt.show()
 
 
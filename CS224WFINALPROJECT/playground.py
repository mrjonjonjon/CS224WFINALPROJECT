import numpy as np
import matplotlib.pyplot as plt

# example weighted adjacency matrix
adj_matrix = np.array([[0.0, 0.2, 0.4, 0.8],
                      [0.2, 0.0, 0.6, 0.3],
                      [0.4, 0.6, 0.0, 0.7],
                      [0.8, 0.3, 0.7, 0.0]])

# plot heatmap of adjacency matrix
plt.imshow(adj_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

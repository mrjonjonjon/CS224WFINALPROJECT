import numpy as np
import matplotlib.pyplot as plt
from utils import *

if __name__=='__main__':
    # example weighted adjacency matrix
    adj_matrix = np.random.randn(1000,1000)

    visualize_adj_grid(adj_matrix)

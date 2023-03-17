import numpy as np
import matplotlib.pyplot as plt
from utils import *

if __name__=='__main__':
    # example weighted adjacency matrix
    adj_matrix = torch.Tensor(np.random.randn(1000,1000))

    print(adj_matrix.shape[0])

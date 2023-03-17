import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import matmul
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random


def save_model(model,name='default_name'):
    torch.save(model,f'./saved_models/{name}.pth')
    
def load_model(name='default_name'):
    return torch.load(f'./saved_models/{name}.pth')


    
#visualizes adjacency matrix as a heatmap grid
def visualize_adj_grid(adj_matrix):
    plt.imshow(adj_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()
    
#generate data using a linear SEM model (ie the value of a node is a linear combination of its parents)
def generate_data_linear_sem(var_dim,w_adj,num_samples):
    
    num_vars  = w_adj.shape[0]
    #X = (I − A^T)^-1 Z 
    w_adj2 = inv(np.eye(num_vars) - w_adj.T)
    all_samples=[]
    for i in range(num_samples): 
        noise = np.random.standard_normal(size=(num_vars,var_dim))
        #print("NOISE IS ",noise)
        sample = matmul(w_adj2,noise)
        all_samples.append(torch.Tensor(sample))

    return all_samples

#generate a random weighted acyclic graph
def generate_weighted_acyclic_graph(n, p):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # Generate random weights for the edges
    weights = np.random.rand(n, n)
    
    # Add edges with probability p and check for acyclicity
    for i in range(n):
        for j in range(i+1, n):
            if np.random.rand() < p:
                G.add_edge(i, j, weight=weights[i][j])
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(i, j)

    return G
#get the adjacency matrix of a weighted graph
def get_adjacency_matrix(G):
    A = nx.to_numpy_array(G, weight='weight')
    
    return A

def random_acyclic_binary_matrix(n):

    T = nx.DiGraph()

    root = 0
    T.add_node(root)

    nodes = list(range(1, n))

    while nodes:
        node = random.choice(nodes)
        parent = random.choice(list(T.nodes()))
        T.add_node(node)
        T.add_edge(parent, node)
        nodes.remove(node)

    return nx.to_numpy_array(T, nodelist=range(n), dtype=int)


def visualize_digraph_nice_layout(G):
    start=(0,0)
    pos={}
    start_node=0
    visited = set()
    pos=assign_coordinates_proper(G)
    print(pos)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G,pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    plt.axis('off')

    plt.show()
def assign_coordinates_proper(G):
    visited=set()
    pos={}
    for i in range(len(G.nodes)):
        if i not in visited:
            
            assign_coordinates(G,i,0,0,pos,visited)
            print("COMPONENT DONE")
    return pos
def assign_coordinates(G, node, x, y,pos,visited):
    print(node,x,y)
    visited.add(node)
    pos[node]=(x,y)
    children = list(G.successors(node))
    if not children:
        return # base case
    num_children = len(children)
    for i, child in enumerate(children):
        x_offset = (i - num_children/2) # adjust x coordinate based on number of children
        pos[child] = (x + x_offset, y - 1)
        assign_coordinates(G, child, x + x_offset, y - 1,pos,visited)
    
#visualize a weighted digraph
def visualize_weighted_digraph(G):
    sorted_nodes = list(nx.topological_sort(G))

    # apply the Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(G)

    #pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G,pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    plt.axis('off')
    plt.show()
    
    
def weight_matrix_to_digraph(adj_matrix):
    

    # create a new directed graph
    G = nx.DiGraph()

    # add nodes to the graph
    n = len(adj_matrix)
    for i in range(n):
        G.add_node(i)

    # add weighted edges to the graph
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] != 0:
                G.add_weighted_edges_from([(i, j, adj_matrix[i][j])])
    return G
class CustomDataset(Dataset):
    def __init__(self, var_dim,weight_matrix,num_samples, transform=None, target_transform=None):
        self.data = generate_data_linear_sem(var_dim,weight_matrix,num_samples)
        self.adj = weight_matrix
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    
if __name__=='__main__':
    g = generate_weighted_acyclic_graph(50,1)
    #g.add_edge(0,1,weight = 1)
    a = get_adjacency_matrix(g)
    samples = generate_data_linear_sem(2,a,1)


    print(a,'\n')
    print("SAMPLES ARE ",samples)
    #visualize_weighted_digraph(g)
    visualize_adj_grid(a)
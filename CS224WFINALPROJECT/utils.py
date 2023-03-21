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
from sklearn.preprocessing import MinMaxScaler
from os.path import dirname, abspath, join
from pandas import read_stata


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
            #print("COMPONENT DONE")
    return pos
def assign_coordinates(G, node, x, y,pos,visited):
    #print(node,x,y)
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
   # sorted_nodes = list(nx.topological_sort(G))

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

def generate_cyclic_digraph(n):
    G = nx.DiGraph()

    G.add_nodes_from(range(n))

    # Add edges to form a cycle
    for i in range(n):
        G.add_edge(i, (i+1)%n)
    return G

def h(A):
    #h(A) ≡ tr[(I + αA ◦ A)^m] − m = 0,
    #alpha = A.shape[0]
    m = A.shape[0]
    alpha = 1/m
    id = torch.eye(m)
    operand = id + alpha*A*A
    #print(operand)
    operand = torch.matrix_power(operand,m)
    #print(operand)
    trace = torch.trace(operand)-m
    return trace


def neg_log_likelihood_loss(mean1,mean2,log_variance):
    
    neg_log_p = log_variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * log_variance))
    return neg_log_p.sum() / mean2.size(0)

#paper uses sem version instead of nonlinear version?
def kl_gaussian_sem(z):
    #output of the encoder is the mean
    mu = z
    kl_div = mu*mu
    kl_sum = kl_div.sum()
    return (kl_sum/z.size(0))*0.5

    
class CustomDataset(Dataset):
    def __init__(self, var_dim,weight_matrix,num_samples, transform=None, target_transform=None):
        self.data = generate_data_linear_sem(var_dim,weight_matrix,num_samples)
        self.adj = weight_matrix
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class CollegeDataset(Dataset):
    def __init__(self, weight_matrix):
        path = dirname(abspath(__file__))
        data = read_stata(join(path, 'close_college.dta'))
        data['lwage'] = data['lwage'].apply(lambda x: int(x)) # Change living wage to discrete variable (Num Figures)
        data = torch.tensor(data.values.astype('float32'))
        scaler = MinMaxScaler()
        for i in range(data.shape[1]):
            column = data[:, i].reshape(-1, 1)
            scaler.fit(column)
            column_normalized = scaler.transform(column)
            data[:, i] = torch.tensor(column_normalized.reshape(-1), dtype=torch.float32)
        self.data = torch.unsqueeze(data, 2)
        self.adj = weight_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    
if __name__=='__main__':
    g = generate_cyclic_digraph(2)
    g = generate_weighted_acyclic_graph(10,0.5)
    a = get_adjacency_matrix(g)
    
    visualize_weighted_digraph(g)
    visualize_adj_grid(a)
    print(h(a,0.9))

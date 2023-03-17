import torch;
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; 
from torch.autograd import Variable
from torch import matmul

plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self,num_vars,var_dim,hidden_dim=512,latent_dim=2): 
        super(Encoder, self).__init__()
        '''Newer versions of PyTorch allows nn.Linear to accept N-D input tensor, the only constraint is that the last dimension of the input tensor will equal in_features of the linear layer. The linear transformation is then applied on the last dimension of the tensor.
For instance, if in_features=5 and out_features=10 and the input tensor x has dimensions 2-3-5, then the output tensor will have dimensions 2-3-10'''
        #input is a single sample [num_vars,var_dim] where n is the total number of variables and d is the dimension of each variable
        #input_dim is 
        self.num_vars = num_vars
        self.var_dim=var_dim
        self.hidden_dim=hidden_dim
        self.latent_dim = latent_dim
       
       #========WEIGHTS========
        #the adjacency matrix
        self.adj_A = nn.Parameter(torch.zeros((num_vars,num_vars)), requires_grad=True)
        #last dimension of input(var dim) matches in_features(var_dim)
        self.w1= nn.Linear(var_dim, hidden_dim,dtype=torch.double)
        self.relu = nn.ReLU()
        #self.mu_w2 = nn.Linear(hidden_dim,latent_dim,dtype=torch.double)
        #self.sig_w2 = nn.Linear(hidden_dim,latent_dim,dtype=torch.double)
        self.w2 = nn.Linear(hidden_dim,latent_dim,dtype = torch.double)
        
        self.bias = nn.Parameter(torch.zeros(latent_dim),requires_grad=True)
        #====norm dist======
        #self.N = torch.distributions.Normal(0,1)
        
        
        #MZ| log SZ] = (I − A^T) MLP(X, W1, W2)
        
    def forward(self, x):
        #I-A^T
        adj_2 = (torch.eye(self.adj_A.shape[0]).double() - (self.adj_A.transpose(0,1)))
        #supposedly accelerates convergence
        amped_adj = torch.sinh(3.*self.adj_A)
        amped_adj2 = (torch.eye(amped_adj.shape[0]).double() - (amped_adj.transpose(0,1)))
        x=x.double()
        x=self.w1(x).double()
        x=self.relu(x).double()
    
        x = self.w2(x).double()
        z_mu = matmul(amped_adj2,x+self.bias) - self.bias
        #pre_mu=self.mu_w2(x).double()
        #pre_sigma=self.sig_w2(x).double()
        #mu = matmul(adj_2,pre_mu).double()
        #sigma = matmul(adj_2,pre_sigma).double()
        #sampling. equivalent to sampling from normal distribution with mean mu, covariance matrix sigma*Id
        #z = mu.double() + sigma*self.N.sample(mu.shape).double()
        
        #penalize latent distribution for being far from standard normal
        from utils import kl_gaussian_sem
        self.kl = kl_gaussian_sem(z_mu)
        return z_mu
    
    
class Decoder(nn.Module):
    def __init__(self,num_vars,latent_dim,hidden_dim,output_dim):
        super(Decoder, self).__init__()
        #self.adj_A = #nn.Parameter(torch.eye(num_vars), requires_grad=True)
        self.w3 = nn.Linear(latent_dim, hidden_dim,dtype=torch.double)
        self.relu = nn.ReLU()
        self.w4 = nn.Linear(hidden_dim, output_dim,dtype=torch.double)

    def forward(self, z,encoder:Encoder):
        adj_A=encoder.adj_A
        bias = encoder.bias
        #(I-A^T)^-1
        #print(torch.eye(adj_A.shape[0]).double()-adj_A.transpose(0,1),torch.eye(adj_A.shape[0]))
        amped_adj = torch.sinh(3.*adj_A)
        amped_adj_inv = torch.inverse(torch.eye(amped_adj.shape[0]).double()-amped_adj.transpose(0,1))
        
        x = matmul(amped_adj_inv,z+bias) - bias
        x = self.w3(x)
        x=self.relu(x)
        x=self.w4(x)
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, num_vars,var_dim,enc_hidden_dim,dec_hidden_dim,latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(num_vars,var_dim,enc_hidden_dim,latent_dim)
        self.decoder = Decoder(num_vars,latent_dim,dec_hidden_dim,var_dim)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z,self.encoder)
        return x
    

def save_model(model):
    torch.save(model,'vae.pth')
    
def load_model():
    return torch.load('vae.pth')


    
if __name__=='__main__':
        
    #TRAINING
    '''latent_dims = 2
    autoencoder = Autoencoder(latent_dims).to(device) # GPU

    data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data',
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=128,
            shuffle=True)
    print(data)
    #autoencoder = train(autoencoder, data)
    for x,y in data:
        print(x.shape,y.shape)
    #plot_reconstructed(autoencoder)
    plt.show()'''
    encoder = VariationalAutoencoder()
    print(encoder)
    for i in encoder.named_parameters():
        print(i)
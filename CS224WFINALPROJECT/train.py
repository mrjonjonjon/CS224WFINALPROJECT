
from torch.utils.data import DataLoader
from utils import *
from codec import VariationalAutoencoder
import copy
from torch.optim import lr_scheduler
#============================GLOBALS========================#
var_dim=1
num_vars=10
num_samples=5000
#===========================================================



custom_weight_matrix = random_acyclic_binary_matrix(num_vars)

training_data = CustomDataset(var_dim=var_dim,weight_matrix=custom_weight_matrix,num_samples=num_samples)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

#fixed variance of 1
log_variance =0.

#augmented lagrangian hyperparameters
eta = 10
gamma = 1/4


def train_one_epoch(vae:VariationalAutoencoder, data, c,lamb,opt):
    
    #opt = torch.optim.Adam(vae.parameters())
    for x in data:
        #x = x.to(device) # GPU
        opt.zero_grad()
        x_hat = vae(x)
        _h = h(vae.encoder.adj_A)
        loss = vae.encoder.kl + neg_log_likelihood_loss(x,x_hat,log_variance)+lamb*_h + (1/2)*c*_h*_h
        #could add huber norm regularization later
        #loss +=100. * torch.trace(vae.encoder.adj_A*vae.encoder.adj_A) + args.tau_A * torch.sum(torch.abs(one_adj_A))
        loss.backward()
        opt.step()
    #print(f'{100*epoch/epochs}%')
    return vae

def train(vae:VariationalAutoencoder,data,num_epochs,optimizer):
    old_adj=np.inf
    c=1.
    lamb = 0.
    for i in range(1):
        while c<1e+20:
            for cur_epoch in range(num_epochs):
                
                vae = train_one_epoch(vae,data,c,lamb,optimizer)
                print(f"PROGRESS: {round(100*cur_epoch/num_epochs)}%")
            new_adj = vae.encoder.adj_A.clone().detach()
            if h(new_adj)>gamma*old_adj:
                c *= eta
            else:
                print("BREAKING OUT OF c LOOP")
                break
        old_adj = new_adj.clone().detach()
        return vae
    
untrained_vae = VariationalAutoencoder(num_vars=num_vars,var_dim=var_dim,enc_hidden_dim=64,dec_hidden_dim=64,latent_dim=1)
optimizer = torch.optim.Adam(untrained_vae.parameters())
#learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=200,gamma=1)





g = weight_matrix_to_digraph(custom_weight_matrix)
#visualize_digraph_nice_layout(g)
visualize_adj_grid(custom_weight_matrix)
save_model(untrained_vae,'untrained_vae')
trained_vae = train(untrained_vae,training_data,num_epochs=300,optimizer=optimizer)
print("TRAINING COMPLETE")
save_model(trained_vae,'trained_vae')
print(f"SAVED TRAINED MODEL")
learned_adj = copy.deepcopy(trained_vae.encoder.adj_A)
visualize_adj_grid(learned_adj.detach())
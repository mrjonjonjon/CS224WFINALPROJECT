
from torch.utils.data import DataLoader
from utils import *
from codec import VariationalAutoencoder
custom_weight_matrix = random_acyclic_binary_matrix(50)
training_data = CustomDataset(var_dim=5,weight_matrix=custom_weight_matrix,num_samples=10)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

def train(vae:VariationalAutoencoder, data, epochs=20):
    opt = torch.optim.Adam(vae.parameters())
    for epoch in range(epochs):
        for x in data:
            #x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss.backward()
            opt.step()
    return vae

untrained_vae = VariationalAutoencoder(num_vars=50,var_dim=5,enc_hidden_dim=512,dec_hidden_dim=512,latent_dim=2)
g = weight_matrix_to_digraph(custom_weight_matrix)
visualize_digraph_nice_layout(g)

save_model(untrained_vae,'untrained_vae')
trained_vae = train(untrained_vae,training_data,epochs=20)
print("TRAINING COMPLETE")
save_model(trained_vae,'trained_vae')
print(f"SAVED TRAINED MODEL")
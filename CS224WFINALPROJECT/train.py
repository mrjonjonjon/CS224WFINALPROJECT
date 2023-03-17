
from torch.utils.data import DataLoader
from utils import *
from codec import VariationalAutoencoder
custom_weight_matrix = np.array([[0., 1., 1., 0],
                                [0., 0., 0., 1.],
                                [0., 0., 0.0, 0.],
                                [0., 0., 0., 0.0]])
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

untrained_vae = VariationalAutoencoder(num_vars=4,var_dim=5,enc_hidden_dim=512,dec_hidden_dim=512,latent_dim=2)

save_model(untrained_vae,'untrained_vae')
trained_vae = train(untrained_vae,training_data,epochs=20)
print("TRAINING COMPLETE")
save_model(trained_vae,'trained_vae')
print(f"SAVED TRAINED MODEL")
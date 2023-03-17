
from torch.utils.data import DataLoader
from utils import *
from codec import VariationalAutoencoder
custom_weight_matrix = np.array([[0.0, 0.2, 0.4, 0.8],
                                [0.2, 0.0, 0.6, 0.3],
                                [0.4, 0.6, 0.0, 0.7],
                                [0.8, 0.3, 0.7, 0.0]])
training_data = CustomDataset(var_dim=2,weight_matrix=custom_weight_matrix,num_samples=10)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

def train(vae:VariationalAutoencoder, data, epochs=20):
    opt = torch.optim.Adam(vae.parameters())
    for epoch in range(epochs):
        for x in data:
            print(x)
    return vae

for x in training_data:
    print(f'SINGLE SAMPLE IS \n{x}')
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Dataset Class
class WineDataset(Dataset):
    def __init__(self):
        # Data Loading
        # skiprows = 1 --> Skips the header row
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # size would be n_samples, 1
        self.n_samples = xy.shape[0]


    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
# Create dataset
dataset = WineDataset()

# DataLoader
# num_workers --> number of subprocess to load the data
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# Iterator
# dataiter = iter(dataloader)
# data = next(dataiter) # WIthout the loop, this will output 4 data-label pairs as batch size = 4
# features, labels = data
# print(features, labels)

# Training Loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass, # Backward pass
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784 #28*28 --> 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_data = torchvision.datasets.MNIST(root='data', train=True, 
                                        transform=transforms.ToTensor(), download=True)

test_data = torchvision.datasets.MNIST(root='data', train=False,
                                        transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Data visualization
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)




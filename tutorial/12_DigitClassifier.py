import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784  # 28*28 --> 784
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
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=False)

# Data visualization
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

# Visualization
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output


model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
n_total_step = len(train_loader)
for epoch in range(num_epochs):
    for batch, (images, labels) in enumerate(train_loader):
        # 100, 1 , 28, 28 --> 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        output = model(images)

        # Loss
        loss = criterion(output, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) % 100 == 0:
            print(f"Epoch: {epoch+1} / {num_epochs}, step {batch+1}/{n_total_step}, loss = {loss.item():.4f}")

# Testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # Value and index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    accuracy = 100.0 * n_correct / n_samples
    print(f"Test accuracy: {accuracy}")

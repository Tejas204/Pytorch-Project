# ********************************************************************************************
# TRAINING PIPELINE
# 1. Design model
# 2. Construct loss and optimizer
# 3. Training loop
#   - Forward pass: compute prediction
#   - Backward pass: gradients
#   - Update weights
# ********************************************************************************************

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Dataset preparation
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)   # Convert to column vector

n_samples, n_features = X.shape

# Define model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss function
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X)

    # Loss
    loss = criterion(predictions, y)

    # Backward pass
    loss.backward()

    # Update
    optimizer.step()

    # Empty gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}; Loss = {loss.item():.4f}")

# plot
# detach - prevent from going into computation graph
predicted = model(X).detach()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
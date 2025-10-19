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

# Define inputs and weights
# Difference between [1, 2, 3, 4] and [[1], [2], [3], [4]] is:
# - the first is a vector in single dimension like a number line with shape (4)
# - the second is a column vector with shape (4, 1)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

# n_sampels = 4, n_features = 1 ==> 4 samples, each with one feature
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# Define model
# Linear performs the operation y = xW^T + b
model = nn.Linear(input_size, output_size)

# Define loss function
loss = nn.MSELoss()

# Define learning rate
learning_rate = 0.01

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
print(f"Predction before training for f(5): {model(X_test).item():.3f}")

for epoch in range(100):
    # Forware pass
    predictions = model(X)

    # Compute loss
    loss_value = loss(predictions, Y)

    # Backward pass
    loss_value.backward()

    # Update weights
    optimizer.step()

    # Empty gradients
    optimizer.zero_grad()

    [w, b] = model.parameters()
    print(f"Epoch {epoch + 1}: W = {w[0][0].item():.3f}, Loss = {loss_value:.8f}")


print(f"Predction after training for f(5): {model(X_test).item():.3f}")


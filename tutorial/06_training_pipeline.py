# ********************************************************************************************
# TRAINING PIPELINE
# 1. Design model
# 2. COnstruct loss and optimizer
# 3. Training loop
#   - Forward pass: compute prediction
#   - Backward pass: gradients
#   - Update weights
# ********************************************************************************************

import torch
import torch.nn as nn

# Define inputs and weights
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

W = torch.tensor(0.0, requires_grad=True)

# Define loss function
def loss(y_hat, y):
    return ((y - y_hat)**2).mean()

# Define forward pass
def forward(x):
    return x * W

learning_rate = 0.01

# Training loop
print(f"Predction before training for f(5): {forward(5):.3f}")

for epoch in range(20):
    predictions = forward(X)

    loss_value = loss(predictions, Y)

    loss_value.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad

    W.grad.zero_()

    print(f"Epoch {epoch + 1}: W = {W:.3f}, Loss = {loss_value:.8f}")


print(f"Predction after training for f(5): {forward(5):.3f}")


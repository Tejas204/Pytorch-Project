import torch

# f(x) = w * x
# f(x) = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=float)
Y = torch.tensor([2, 4, 6, 8], dtype=float)

w = torch.tensor(0.0, requires_grad=True)

# forward pass
def forward(x):
    return w * x

# loss: MSE loss
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f"Predction before training for f(5): {forward(5):.3f}")

# Training
epochs = 20
learning_rate = 0.01


for epoch in range(epochs):
    # Make predictions
    y_pred = forward(X)

    # Calculate loss
    error = loss(Y, y_pred)

    # Calculate gradient
    error.backward()

    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Zero gradients
    w.grad.zero_()

    print(f"Epoch {epoch + 1}: W = {w:.3f}, Loss = {error:.8f}")

print(f"Predction after training for f(5): {forward(5):.3f}")
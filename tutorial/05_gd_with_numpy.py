import numpy as np

# f(x) = w * x
# f(x) = 2 * x

X = np.array([1, 2, 3, 4], dtype=float)
Y = np.array([2, 4, 6, 8], dtype=float)

w = 0.0

# forward pass
def forward(x):
    return w * x

# loss: MSE loss
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradients
# MSE: 1/N * (y_predicted - y)**2 := 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2 * (w*x - y) * x
def gradients(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

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
    dw = gradients(X, Y, y_pred)

    # Update parameters
    w -= learning_rate * dw

    print(f"Epoch {epoch + 1}: W = {w:.3f}, Loss = {error:.8f}")

print(f"Predction after training for f(5): {forward(5):.3f}")
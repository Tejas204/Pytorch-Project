import torch

# ********************************************************************************************
# INITIALIZATION

x = torch.randn(3, requires_grad=True)
y = x + 2
z = y*y*2
z = z.mean()

# ############################################################################################
# NOTE: When we use required_grad = True, pytorch creates a computational graph
# Since we used required_grad, y and z will have a grad_fn attribute
# OP: y --> tensor([2.4578, 3.3414, 1.7199], grad_fn=<AddBackward0>)
# OP: z --> tensor([0.0281, 2.4600, 9.2711], grad_fn=<MulBackward0>)

# Due to the use of addition and multiplication in y and z respectively, we see AddBackward
# and MulBackward in the output
# ############################################################################################

# COMPUTING BACK-PROPAGATION
z.backward() # dz/dx

# STOPPING BACKWARD FLOW OF GRADIENTS
# Option 1: x.requires_grad_(False)
# Option 2: y = x.detach() --> Creates a new tensor without gradients
# Option 3: with torch.no_grad():

# TRAINING EXAMPLE
weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    # Important step
    weights.grad.zero_()

# ############################################################################################
# NOTE: The grad attribute contains the accumulation of all previous gradients
# Output: For 2 loops, output is as below.
# tensor([0.9784, 0.6461, 0.3201], requires_grad=True)
# tensor([3., 3., 3., 3.])
# tensor([6., 6., 6., 6.])

# This is incorrect as we require fresh gradients for each iteration.
# We need to zero out the gradients
# ############################################################################################

# BACKPROPAGATION
input = torch.tensor(1.0)
weight = torch.tensor(2.0, requires_grad=True)
actual = torch.tensor(4.0)

# Perform forward pass, get the loss
y_hat = weight * input
print(f"Input: {input}")
print(f"Weight: {weight}")
print(f"y hat: {y_hat}")
loss = (y_hat - actual)**2

print(f"Loss is: {loss}")

# Perform backward pass
loss.backward()
print(weight.grad)

# Update weight
weight = weight - 1*(weight.grad)
print(f"Updated weight: {weight}")
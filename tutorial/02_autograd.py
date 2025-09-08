import torch

# ********************************************************************************************
# INITIALIZATION

x = torch.randn(3, requires_grad=True)
y = x + 2
z = y*y*2
z = z.mean()
print(y)
print(z)

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
print(f"Gradients of z: {x.grad}")
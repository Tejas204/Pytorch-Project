import torch
import numpy as np

# ********************************************************************************************
# INITIALIZATION
x = torch.rand(2, 2)
y = torch.rand(2,2)
z = x + y


# We can also use the below function for element wise addition
# z = torch.add(x, y)



# ############################################################################################
# NOTE: By using add_, we do in place addition
# Ex: y.add_(x)
# All functions with '_' will do in place operations
# ############################################################################################


# SLICING OPERATIONS
a = torch.rand(5, 3)
b = a[0, :]



# RESHAPING
c = torch.rand(4,4)
d = c.view(16)
e = c.view(-1, 8) # Automatically determines the number of dims required based on 2nd arg


# NUMPY TO TORCH AND VICE VERSA
f = torch.ones(5)
g = f.numpy()

h = np.ones(5)
i = torch.from_numpy(h)

# ############################################################################################
# NOTE: If we are on a CPU, pytorch and nunpy arrays will share the same memory location
# Modification to one modifies the other
# ############################################################################################

# SETTING UP GPU ENV
if torch.cuda.is_available():
    device = torch.device("cuda")
    j = torch.ones(5,5, device=device)
    k = torch.ones(5,5).to(device)
    l = j + k                       # This is on GPU
    l = l.to("CPU")                 # Brought back to GPU


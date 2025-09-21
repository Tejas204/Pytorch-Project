import numpy as np
import torch
import matplotlib.pyplot as plt

# Compute similarity
# Sentence is: The cat sat on the mat
# Each word will be converted into an embedding
# the = torch.tensor([0.1,0.0,0.2], dtype=float)
# cat = torch.tensor([0.9,0.1,0.4], dtype=float)
# sat = torch.tensor([0.3,0.8,0.2], dtype=float)
# on = torch.tensor([0.2,0.3,0.7], dtype=float)
# mat = torch.tensor([0.7,0.2,0.9], dtype=float)

matrix = torch.tensor([[0.1,0.0,0.2], [0.9,0.1,0.4], [0.3,0.8,0.2], [0.2,0.3,0.7], [0.7,0.2,0.9]], dtype=float)


# Plotting
z = np.linspace(0, 1, 100)
x = z * np.sin(25 * z)
y = z * np.cos(25 * z)


fig = plt.figure()
ax = plt.axes(projection='3d')
plots = ax.plot3D(y, z)
ax.set_title('Vectors')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
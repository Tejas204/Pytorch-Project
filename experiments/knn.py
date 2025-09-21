# Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ********************************************************************************************
# TRAIN DATA POINTS
# A dictionary with the format: {point: class}

# Class variables: 0 --> A, 1 --> B, 2 --> C
# ********************************************************************************************

train_class_variable = np.random.randint(3, size=300)
print(f"--> Class variables:\n{train_class_variable}\n")


train_data_variable = np.random.random_sample((300, ))
print(f"--> Data points:\n{train_data_variable.size}\n")
# plt.plot(train_data_variable, 'bo')
# plt.show()

train_data_dictionary = {key:value for key, value in zip(train_data_variable, train_class_variable)}

# ********************************************************************************************
# TEST DATA POINTS
# A list of test data points

# ********************************************************************************************
test_data_variable = np.random.random_sample((10,))

test_data_dictionary = dict.fromkeys(test_data_variable)
print(f"--> Test data set:\n{test_data_dictionary}\n")

# ********************************************************************************************
# PLOTTING
# Plot test data and train data
# ********************************************************************************************

fig, ax = plt.subplots()
for data_class in [0, 1, 2]:
    if data_class == 0:
        color = 'r+'
    elif data_class == 1:
        color = 'bo'
    else:
        color = 'g*'

    plt.plot({key: value for key, value in train_data_dictionary.items() if value == data_class}.keys(), color, label="Class "+str(data_class))
    ax.legend()

plt.show()
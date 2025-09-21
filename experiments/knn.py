# Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from collections import Counter

# ********************************************************************************************
# TRAIN DATA POINTS
# A dictionary with the format: {point: class}

# Class variables: 0 --> A, 1 --> B, 2 --> C
# ********************************************************************************************

train_class_variable = np.random.randint(3, size=30)
print(f"--> Class variables:\n{train_class_variable}\n")


train_data_variable = np.random.random_sample((30, ))
print(f"--> Data points:\n{train_data_variable.size}\n")

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
        color = 'ro'
    elif data_class == 1:
        color = 'bo'
    else:
        color = 'go'

    plt.plot({key: value for key, value in train_data_dictionary.items() if value == data_class}.keys(), color, label="Class "+str(data_class))
    ax.legend()

plt.show()

# ********************************************************************************************
# ALGORITHM

# ********************************************************************************************

k = 5

for test_data_point in test_data_variable:
    # Calculate distance to each point
    distance = []
    for train_data_point in list(train_data_dictionary.keys()):
        distance.append(np.square(test_data_point - train_data_point))

    # Find top k points and their classes
    distance.sort()
    top_k = distance[:k]
    top_k_dictionary = {key:value for key, value in train_data_dictionary.items() if key in top_k}
    print(top_k_dictionary)

    # Calculate majority vote
    votes = Counter(top_k_dictionary.values())
    classes = list(votes.keys())
    counts = list(votes.values())
    class_allotment = classes.index(counts.index(max(counts)))

    # Update dictionary
    test_data_dictionary.update({test_data_point:class_allotment})





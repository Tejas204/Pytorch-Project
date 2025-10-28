'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

# Dataset Class
class WineDataset(Dataset):
    def __init__(self, transform=None):
        # Data Loading
        # skiprows = 1 --> Skips the header row
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # size would be n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform


    def __getitem__(self, index):
        # dataset[0]
        sample =  self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
# Custom Transform classes
class ToTensor():
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)
    
class MulTransform():
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

# Dataset with single transform
dataset_1 = WineDataset(transform=ToTensor())
first_data = dataset_1[0]
features, labels = first_data
print(features)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset_2 = WineDataset(transform=composed)
first_data = dataset_2[0]
features, labels = first_data
print(features)

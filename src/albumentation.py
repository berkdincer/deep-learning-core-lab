import torch
import torchvision.datasets as datasets
from torchvision.transforms import v2
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

# Methods for dealing with imbalanced dataset

# 1. Oversampling # Data augmentation.
# 2. Class weighting # Multiply loss function via number.

def get_loader(root_dir,batch_size):
    my_transforms = v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
    ])
    dataset = datasets.ImageFolder(root=root_dir,transform=my_transforms)
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),replacement=True)
    loader = DataLoader(dataset= dataset, batch_size = batch_size, sampler = sampler)
    return loader

def main():
    loader = get_loader(root_dir='imbalanced_dataset',batch_size=32)
    for data,label in loader:
        print(label)
if __name__ == '__main__':
    main()
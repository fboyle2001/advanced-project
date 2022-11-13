
import torch
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image

# https://stackoverflow.com/a/59661024
class CustomImageDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))#.transpose(1,2,0))
            x = self.transform(x)
        
        return x, y

    def get_transformed_data(self):
        assert self.transform is not None

        xs = []

        for index in range(len(self.data)):
            x = Image.fromarray(self.data[index].astype(np.uint8))#.transpose(1,2,0))
            x = self.transform(x)
            xs.append(x.unsqueeze(0))

        return torch.cat(xs)
    
    def __len__(self):
        return len(self.data)

    def get_sample(self, sample_size):
        return random.sample(self.data, k=sample_size)
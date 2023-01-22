import torch
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image

# Full credit: https://stackoverflow.com/a/59661024
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
    
    def __len__(self):
        return len(self.data)

    def get_sample(self, sample_size):
        return random.sample(self.data, k=sample_size)
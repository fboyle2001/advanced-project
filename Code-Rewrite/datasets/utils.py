
import torch
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image

# https://stackoverflow.com/a/59661024
class CustomImageDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

    def get_sample(self, sample_size):
        return random.sample(self.data, k=sample_size)
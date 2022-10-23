import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def get_gpu(fatal_on_error=True):
    cuda_available = torch.cuda.is_available()
    if not cuda_available and fatal_on_error:
        print(cuda_available, fatal_on_error)
        raise RuntimeError("No GPU available")
    
    return torch.device("cuda" if cuda_available else "cpu")

def create_iterator(dataloader): 
    # helper function to make getting another batch of data easier
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    return iter(cycle(dataloader))

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
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
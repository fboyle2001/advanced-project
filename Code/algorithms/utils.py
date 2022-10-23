import random
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

# https://stackoverflow.com/a/59661024
class ReplayBufferDataset(Dataset):
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

class HashReplayBuffer:
    VALID_STRATEGIES = ["random_from_largest_class"]

    def __init__(self, max_size, max_strategy):
        assert max_strategy in self.VALID_STRATEGIES, f"{max_strategy} is not a valid strategy"
        
        self.max_size = max_size
        self.max_strategy = max_strategy

        """
        Structure: 
        {
            "class_name": [
                {
                    "hash": "tensor_hash",
                    "sample": Tensor
                }, ...
            ], ...
        }
        """
        self.memory = {}
        self.hashes = set()
        self.count = 0

    def _remove_sample(self):
        if self.count == 0:
            return

        match self.max_strategy:
            case "random_from_largest_class":
                largest_class = max(self.memory.keys(), key=lambda c: len(self.memory[c]))
                remove_index = random.randrange(0, len(self.memory[largest_class]))
                item = self.memory[largest_class][remove_index]
                item_hash = item["hash"]
                self.hashes.remove(item_hash)
                self.memory[largest_class].remove(self.memory[largest_class][remove_index])
            case _:
                assert False
        
        self.count -= 1

    def add_to_buffer(self, img, label):
        img = img.detach().cpu()

        # If the label is not in the keys then add it
        if label not in self.memory.keys():
            self.memory[label] = []
        
        assert self.count <= self.max_size

        # Need to remove a sample according to the strategy selected
        if self.count == self.max_size:
            self._remove_sample()
        
        # Prevent duplicates
        tensor_hash = hash(pickle.dumps(img))

        if tensor_hash in self.hashes:
            return

        self.hashes.add(tensor_hash)
        self.memory[label].append({
            "hash": tensor_hash,
            "sample": img
        })

        self.count += 1

    def to_torch_dataset(self):
        data = []
        targets = []

        for clazz in self.memory.keys():
            clazz_items = self.memory[clazz]

            for item in clazz_items:
                data.append(item["sample"])
                targets.append(clazz)
        
        return ReplayBufferDataset(data, targets)


    

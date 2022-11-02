from typing import Any

import random
import pickle

import torch
from datasets.utils import CustomImageDataset

import torchvision

# TODO: Generate docstrings
class HashReplayBuffer:
    VALID_STRATEGIES = ["random_from_largest_class"]

    def __init__(
        self, 
        max_size: int, 
        max_strategy: str
    ):
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

    def _remove_sample(self) -> None:
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

    def add_to_buffer(self, img: Any, label: Any) -> None:
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

    def to_torch_dataset(self) -> CustomImageDataset:
        data = []
        targets = []

        for clazz in self.memory.keys():
            clazz_items = self.memory[clazz]

            for item in clazz_items:
                data.append(item["sample"])
                targets.append(clazz)
        
        return CustomImageDataset(data, targets, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))


    

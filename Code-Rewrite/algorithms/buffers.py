from typing import Dict, Any, List
from loguru import logger

import random
import pickle

import torch
import numpy as np
from datasets.utils import CustomImageDataset

from queue import Queue

class BalancedReplayBuffer:
    def __init__(self, max_memory_size: int, disable_warnings: bool = False):
        self.max_memory_size = max_memory_size
        self.disable_warnings = disable_warnings

        self.class_hash_pointers: Dict[Any, List[int]] = {}
        self.hash_map: Dict[int, Any] = {}

    @property
    def count(self):
        return len(self.hash_map.keys())

    @property
    def known_classes(self):
        return self.class_hash_pointers.keys()

    def _remove_sample(self) -> None:
        if self.count == 0:
            return
        
        largest_class = max(self.class_hash_pointers.keys(), key=lambda c: len(self.class_hash_pointers[c]))
        remove_index = random.randrange(0, len(self.class_hash_pointers[largest_class]))

        # Remove from the hash map
        data_hash = self.class_hash_pointers[largest_class][remove_index]
        del self.hash_map[data_hash]

        # Remove from the class map
        self.class_hash_pointers[largest_class].remove(data_hash)

    def add_sample(self, data: np.ndarray, target: Any) -> None:
        if not self.disable_warnings and type(data) is torch.Tensor:
            logger.warning(f"Received data of type tensor")

        if not self.disable_warnings and type(target) is torch.Tensor:
            logger.warning(f"Received target of type tensor ({target})")

        data_hash = hash(pickle.dumps(data))

        self.hash_map[data_hash] = data

        if target not in self.class_hash_pointers.keys():
            self.class_hash_pointers[target] = []

        self.class_hash_pointers[target].append(data_hash)

        # Need to delete a sample now
        if self.count > self.max_memory_size:
            self._remove_sample()

    def to_torch_dataset(self, transform=None) -> CustomImageDataset:
        data = []
        targets = []

        for target in self.class_hash_pointers.keys():
            for hash_pointer in self.class_hash_pointers[target]:
                data.append(self.hash_map[hash_pointer])
                targets.append(target)

        return CustomImageDataset(data, targets, transform)

class FIFORingReplayBuffer:
    def __init__(self, max_memory_per_class: int, disable_warnings: bool = False):
        self.max_memory_per_class = max_memory_per_class
        self.disable_warnings = disable_warnings

        self.class_hash_pointers: Dict[Any, Queue[int]] = {}
        self.hash_map: Dict[int, Any] = {}

    @property
    def count(self):
        return len(self.hash_map.keys())

    @property
    def known_classes(self):
        return self.class_hash_pointers.keys()

    def _remove_sample_from_class(self, class_name: Any) -> None:
        if not self.class_hash_pointers[class_name].full():
            return

        # Remove from the hash map and the queue
        data_hash = self.class_hash_pointers[class_name].get()
        del self.hash_map[data_hash]

    def add_sample(self, data: np.ndarray, target: Any) -> None:
        if not self.disable_warnings and type(data) is torch.Tensor:
            logger.warning(f"Received data of type tensor")

        if not self.disable_warnings and type(target) is torch.Tensor:
            logger.warning(f"Received target of type tensor ({target})")

        data_hash = hash(pickle.dumps(data))
        self.hash_map[data_hash] = data

        if target not in self.class_hash_pointers.keys():
            self.class_hash_pointers[target] = Queue(maxsize=self.max_memory_per_class)

        self._remove_sample_from_class(target)
        self.class_hash_pointers[target].put(data_hash)

    def to_torch_dataset(self, transform=None) -> CustomImageDataset:
        data = []
        targets = []

        for target in self.class_hash_pointers.keys():
            for hash_pointer in self.class_hash_pointers[target].queue:
                data.append(self.hash_map[hash_pointer])
                targets.append(target)

        return CustomImageDataset(data, targets, transform)

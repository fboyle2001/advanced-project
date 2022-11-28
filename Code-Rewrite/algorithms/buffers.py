from typing import Dict, Any, List, Tuple
from loguru import logger

import random
import pickle

import torch
import numpy as np
from datasets.utils import CustomImageDataset
from PIL import Image

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

        if data_hash in self.hash_map.keys():
            logger.warning(f"Duplicate hash: {data_hash}")
            return

        self.hash_map[data_hash] = data

        if target not in self.class_hash_pointers.keys():
            self.class_hash_pointers[target] = []

        self.class_hash_pointers[target].append(data_hash)

        # Need to delete a sample now
        if self.count > self.max_memory_size:
            self._remove_sample()

    def draw_sample(self, batch_size, device, transform=None):
        indices = sorted(random.sample(range(self.count), k=batch_size))
        # logger.debug(f"{self.count}, {indices}")

        seen = 0
        indice_index = 0
        next_index = indices[indice_index]
        apply_transform = lambda x: x

        if transform is not None:
            apply_transform = lambda x: transform(Image.fromarray(x.astype(np.uint8)))

        samples = []
        targets = []

        for target in self.class_hash_pointers.keys():
            length = len(self.class_hash_pointers[target])

            while next_index < seen + length:
                # print(next_index, seen, length, seen + length, next_index < seen + length)
                # print(seen + length - next_index)
                samples.append(self.hash_map[self.class_hash_pointers[target][next_index - seen]])
                targets.append(target)

                indice_index += 1
                if indice_index >= len(indices): 
                    next_index = None
                    break
                else:
                    next_index = indices[indice_index]

            seen += length

            if next_index is None:
                break
        
        return torch.stack([apply_transform(x) for x in samples]).to(device), torch.LongTensor(targets).to(device)

    def to_torch_dataset(self, transform=None) -> CustomImageDataset:
        data = []
        targets = []

        for target in self.class_hash_pointers.keys():
            for hash_pointer in self.class_hash_pointers[target]:
                data.append(self.hash_map[hash_pointer])
                targets.append(target)

        return CustomImageDataset(data, targets, transform)

class BalancedReplayBufferWithLogits:
    def __init__(self, max_memory_size: int, disable_warnings: bool = False):
        self.max_memory_size = max_memory_size
        self.disable_warnings = disable_warnings

        self.class_hash_pointers: Dict[Any, List[int]] = {}
        # Image at index 0, Logits at index 1
        self.hash_map: Dict[int, Tuple[np.ndarray, torch.Tensor]] = {}

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

    def add_sample(self, data: np.ndarray, target: Any, z: torch.Tensor) -> None:
        if not self.disable_warnings and type(data) is torch.Tensor:
            logger.warning(f"Received data of type tensor")

        if not self.disable_warnings and type(target) is torch.Tensor:
            logger.warning(f"Received target of type tensor ({target})")

        data_hash = hash(pickle.dumps(data))

        if data_hash in self.hash_map.keys():
            logger.warning(f"Duplicate hash: {data_hash}")
            return

        self.hash_map[data_hash] = (data, z)

        if target not in self.class_hash_pointers.keys():
            self.class_hash_pointers[target] = []

        self.class_hash_pointers[target].append(data_hash)

        # Need to delete a sample now
        if self.count > self.max_memory_size:
            self._remove_sample()

    def draw_sample(self, batch_size, device, transform=None):
        indices = sorted(random.sample(range(self.count), k=batch_size))
        # logger.debug(f"{self.count}, {indices}")

        seen = 0
        indice_index = 0
        next_index = indices[indice_index]
        apply_transform = lambda x: x

        if transform is not None:
            apply_transform = lambda x: transform(Image.fromarray(x.astype(np.uint8)))

        samples = []
        targets = []
        logits = []

        for target in self.class_hash_pointers.keys():
            length = len(self.class_hash_pointers[target])

            while next_index < seen + length:
                # print(next_index, seen, length, seen + length, next_index < seen + length)
                # print(seen + length - next_index)
                sample, logit = self.hash_map[self.class_hash_pointers[target][next_index - seen]]
                samples.append(sample)
                targets.append(target)
                logits.append(logit)

                indice_index += 1
                if indice_index >= len(indices): 
                    next_index = None
                    break
                else:
                    next_index = indices[indice_index]

            seen += length

            if next_index is None:
                break
        
        return torch.stack([apply_transform(x) for x in samples]).to(device), torch.LongTensor(targets).to(device), torch.stack(logits).to(device)

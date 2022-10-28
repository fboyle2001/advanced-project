from typing import List, Union
from .dataset_base import BaseCLDataset

import torchvision

cifar_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_classes: List[Union[str, int]] = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class CIFAR10(BaseCLDataset):
    def __init__(
        self,
        disjoint: bool,
        classes_per_task: Union[int, None] = None
    ):
        super().__init__(
            dataset_class=torchvision.datasets.CIFAR10,
            training_dataset_parameters={ "train": True, "download": True },
            testing_dataset_parameters={ "train": False, "download": True },
            transform=cifar_transform,
            classes=cifar_classes,
            disjoint=disjoint,
            classes_per_task=classes_per_task
        )
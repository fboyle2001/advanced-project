from typing import List, Union
from .dataset_base import BaseCLDataset

import torchvision

# MNIST is greyscale
mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

mnist_classes: List[Union[str, int]] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

class MNIST(BaseCLDataset):
    def __init__(
        self,
        disjoint: bool,
        classes_per_task: Union[int, None] = None
    ):
        super().__init__(
            dataset_class=torchvision.datasets.MNIST,
            training_dataset_parameters={ "train": True, "download": True },
            testing_dataset_parameters={ "train": False, "download": True },
            training_transform=mnist_transform,
            testing_transform=mnist_transform,
            classes=mnist_classes,
            disjoint=disjoint,
            classes_per_task=classes_per_task
        )
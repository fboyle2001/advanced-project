from typing import List, Union
from .dataset_base import BaseCLDataset

import torchvision

training_cifar_transform = [
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
]

base_cifar_transform = [
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]

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
            training_transform=torchvision.transforms.Compose(training_cifar_transform + base_cifar_transform),
            testing_transform=torchvision.transforms.Compose(base_cifar_transform),
            classes=cifar_classes,
            disjoint=disjoint,
            classes_per_task=classes_per_task
        )
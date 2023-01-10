from typing import List, Union
from .dataset_base import BaseCLDataset

import torchvision

training_cifar_transform = lambda x: [
    torchvision.transforms.RandomCrop(x, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
]

# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2627261#gistcomment-2627261 
base_cifar_transform = lambda x: [
    torchvision.transforms.Resize((x, x)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) # changed from (0.2023, 0.1994, 0.2010)
]

cifar_classes: List[Union[str, int]] = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class CIFAR10(BaseCLDataset):
    def __init__(
        self,
        disjoint: bool,
        size: int,
        classes_per_task: Union[int, None] = None
    ):
        super().__init__(
            dataset_class=torchvision.datasets.CIFAR10,
            training_dataset_parameters={ "train": True, "download": True },
            testing_dataset_parameters={ "train": False, "download": True },
            training_transform=torchvision.transforms.Compose(training_cifar_transform(size) + base_cifar_transform(size)),
            testing_transform=torchvision.transforms.Compose(base_cifar_transform(size)),
            classes=cifar_classes,
            disjoint=disjoint,
            classes_per_task=classes_per_task
        )
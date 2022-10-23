from .dataset_base import BaseCLDataset

import torchvision
from torchvision import transforms

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class CIFAR10(BaseCLDataset):
    def __init__(self, batch_size, per_split=None, randomised_split=True):
        super().__init__(
            dataset=torchvision.datasets.CIFAR10,
            training_dataset_parameters={ "train": True, "download": True },
            testing_dataset_parameters={ "train": False, "download": True },
            transform=cifar_transform,
            classes=cifar_classes,
            per_split=per_split,
            randomised_split=randomised_split,
            batch_size=batch_size
        )
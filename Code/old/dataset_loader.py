import torch
import abc
from torchvision import transforms
import torchvision
import torch.utils.data

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LoadedImageDataset(abc.ABC):
    REGISTERED_DATASETS = {}

    @staticmethod
    def register_dataset(clazz, overwrite=False):
        name = clazz.__qualname__
        if name in LoadedImageDataset.REGISTERED_DATASETS.keys() and not overwrite:
            raise RuntimeError(f"Dataset {name} is already registered")

        LoadedImageDataset.REGISTERED_DATASETS[name] = clazz

    @staticmethod
    def init_dataset(name, batch_size):
        if name not in LoadedImageDataset.REGISTERED_DATASETS.keys():
            raise RuntimeError(f"Unknown dataset {name}")
        
        return LoadedImageDataset.REGISTERED_DATASETS[name](batch_size)

    def __init__(self, name, training_dataset, testing_dataset, classes, default_transform, batch_size):
        self.name = name
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.classes = classes
        self.default_transform = default_transform
        self.batch_size = batch_size

        self.training_dataloader = None
        self.testing_dataloader = None
    
    def get_training_dataloader(self):
        if self.training_dataloader is None:
            self.prepare_training_dataloader()
        
        return self.training_dataloader
    
    def get_testing_dataloader(self):
        if self.testing_dataloader is None:
            self.prepare_testing_dataloader()
        
        return self.testing_dataloader

    def display_batch(self):
        self.training_dataloader

    @abc.abstractmethod
    def prepare_training_dataloader(self, shuffle=True):
        pass

    @abc.abstractmethod
    def prepare_testing_dataloader(self, shuffle=False):
        pass

class CIFAR10(LoadedImageDataset):
    def __init__(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        training_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testing_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        super().__init__("CIFAR10", training_dataset, testing_dataset, classes, transform, batch_size)

    def prepare_training_dataloader(self, shuffle=True):
        self.training_dataloader = torch.utils.data.DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def prepare_testing_dataloader(self, shuffle=False):
        self.testing_dataloader = torch.utils.data.DataLoader(self.testing_dataset, batch_size=self.batch_size, shuffle=shuffle)

def register_datasets():
    LoadedImageDataset.register_dataset(CIFAR10)



register_datasets()
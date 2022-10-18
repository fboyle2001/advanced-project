from typing import Type

import torchvision
import torchvision.transforms as transforms
import torch.utils.data

class BaseDataset():
    def __init__(
        self,
        dataset: Type[torchvision.datasets.VisionDataset], 
        training_dataset_parameters: dict,
        testing_dataset_parameters: dict,
        transform: transforms.Compose,
        batch_size: int = 64,
        folder: str = "./data",
    ):
        self._dataset_class = dataset
        self._transform_composition = transform
        self._batch_size = batch_size
        self._folder = folder

        self.training_data = dataset(root=folder, transform=transform, **training_dataset_parameters)
        self.training_loader = torch.utils.data.DataLoader(self.training_data, batch_size=batch_size, shuffle=True, num_workers=0)
        
        self.testing_data = dataset(root=folder, transform=transform, **testing_dataset_parameters)
        self.testing_loader = torch.utils.data.DataLoader(self.testing_data, batch_size=batch_size, shuffle=False, num_workers=0)
import random
from typing import Type, List, Any, Tuple, Union

import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from torch_utils import CustomImageDataset
        
class BaseCLDataset():
    def __init__(
        self,
        dataset: Type[torchvision.datasets.VisionDataset], 
        training_dataset_parameters: dict,
        testing_dataset_parameters: dict,
        transform: transforms.Compose,
        classes: List[Any],
        per_split: Union[int, None],
        randomised_split: bool = True,
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

        self.classes = classes

        if per_split is not None:
            assert len(classes) % per_split == 0, f"Cannot split evenly into {per_split} tasks as there are {len(classes)} classes"

        self.per_split = per_split
        self.randomised_split = randomised_split
        
        if per_split is None:
            self.task_splits = [i for i in range(len(self.classes))]
            self.task_datasets = [self.training_data]
        else:
            self.task_splits, self.task_datasets = self._setup_task_datasets()

    def get_metadata(self) -> str:
        return f"{self._dataset_class.__qualname__} with {self._batch_size} batch size, is split: {self.per_split is not None}"

    def _setup_task_datasets(self) -> Tuple[List[List[str]], List[torch.utils.data.Dataset]]:
        assert self.per_split is not None, "Cannot setup task datasets as per_split is not set"
        divided = {}

        for img, label in self.training_data: 
            if label not in divided.keys():
                divided[label] = []
            
            divided[label].append(img)

        indexes = [i for i in range(len(self.classes))]

        if self.randomised_split:
            random.shuffle(indexes)

        task_datasets = []
        task_split = []

        for i in range(len(self.classes) // self.per_split):
            data = []
            targets = []
            split = []

            for j in range(self.per_split):
                label = indexes[self.per_split * i + j]
                split.append(label)
                labelled_targets = [label] * len(divided[label])

                data += divided[label]
                targets += labelled_targets
            
            task_dataset = CustomImageDataset(data, targets)
            task_datasets.append(task_dataset)
            task_split.append(split)

        return task_split, task_datasets

    def iterate_task_dataloaders(self):
        for i in range(0, len(self.task_datasets)):
            yield self.task_splits[i], torch.utils.data.DataLoader(self.task_datasets[i], batch_size=self._batch_size, shuffle=True, num_workers=0)

# class ExampleDataset(BaseDataset):
#     def __init__(self, batch_size):
#         super().__init__(
#             dataset=None,
#             training_dataset_parameters={},
#             testing_dataset_parameters={},
#             transform=transforms.Compose(...),
#             classes=["a", "b", "c"],
#             batch_size=batch_size
#         )
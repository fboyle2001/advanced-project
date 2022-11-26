import abc
from typing import Type, List, Union, Tuple
from collections.abc import Iterator
import json

from loguru import logger

import torch
import torch.utils.data
import torchvision
import random
import datasets.utils as utils

class BaseCLDataset(abc.ABC):
    """
    Base dataset class to represent a dataset to be used for Continual Learning scenarios.
    Supports disjoint and non-disjoint tasks with minimal effort needed to switch between.
    """

    def __init__(
        self,
        dataset_class: Type[torchvision.datasets.VisionDataset], 
        training_dataset_parameters: dict,
        testing_dataset_parameters: dict,
        training_transform: torchvision.transforms.Compose,
        testing_transform: torchvision.transforms.Compose,
        classes: List[Union[int, str]],
        disjoint: bool,
        classes_per_task: Union[int, None],
        folder: str = "./store/data"
    ):
        """
        For each dataset it is important to consider the scenarios that can arise:

        1) Disjoint Tasks: The dataset is split into N tasks consisting of K classes each.
        These tasks are then presented individually for training

        2) Non-Disjoint Tasks: There is no boundary between the tasks, there initially is 1
        class and over time, randomly, the number of classes increases to N*K classes just like
        in regular model training

        Each dataset here has the option to be split into tasks or for the data to simply be streamed

        Args:
            dataset_class (Type[torchvision.datasets.VisionDataset]): The dataset class from torchvision
            training_dataset_parameters (dict): Parameters for the dataset for the training configurations
            testing_dataset_parameters (dict): _description_
            training_transform (torchvision.transforms.Compose): _description_
            testing_transform (torchvision.transforms.Compose): _description_
            classes (List[Union[int, str]]): _description_
            disjoint (bool, optional): _description_. Defaults to False.
            classes_per_task (Union[int, None], optional): _description_. Defaults to 0.
            folder (str, optional): _description_. Defaults to "./store/data".
        """
        self.folder = folder
        self.dataset_class = dataset_class
        self.training_dataset_parameters = training_dataset_parameters
        self.testing_dataset_parameters = testing_dataset_parameters

        self.training_transform = training_transform
        self.testing_transform = testing_transform

        self.training_data = dataset_class(root=folder, transform=training_transform, **training_dataset_parameters)
        self.testing_data = dataset_class(root=folder, transform=testing_transform, **testing_dataset_parameters)

        self.classes = classes

        self.disjoint = disjoint
        self.classes_per_task = classes_per_task
        self.task_count: int = 1

        self.task_splits: List[List[int]] = [[i for i in range(len(self.classes))]]
        self.task_datasets: List[torch.utils.data.Dataset] = [self.training_data]
        self.raw_task_datasets: List[torch.utils.data.Dataset] = [utils.CustomImageDataset(self.training_data.data, self.training_data.targets)] # type: ignore

        if self.disjoint:
            assert classes_per_task is not None, "In disjoint mode, classes_per_task cannot be None"
            assert len(classes) % classes_per_task == 0, "In disjoint mode, classes_per_task must divide the number of classes"
            self.task_count = len(classes) // classes_per_task
            logger.info(f"Using disjoint tasks with {self.classes_per_task} ({self.task_count} tasks) with a randomised split")
            self.task_splits, self.task_datasets, self.raw_task_datasets = self._setup_task_datasets()

        self.current_task: int = 1

        logger.info(self)

    def _setup_task_datasets(self) -> Tuple[List[List[int]], List[torch.utils.data.Dataset], List[torch.utils.data.Dataset]]:
        """
        Internal method to split the dataset into tasks

        Returns:
            Tuple[List[List[int]], List[torch.utils.data.Dataset], List[torch.utils.data.Dataset]]: List of tasks and their datasets
        """
        assert self.classes_per_task is not None
        logger.debug(f"Splitting dataset into {self.task_count} disjoint tasks")

        divided = {}

        for idx in range(len(self.training_data.data)): # type: ignore
            img = self.training_data.data[idx] # type: ignore
            label = self.training_data.targets[idx] # type: ignore

            if label not in divided.keys():
                divided[label] = []
            
            divided[label].append(img)

        indexes = [i for i in range(len(self.classes))]

        # Randomise the split
        random.shuffle(indexes)

        task_datasets = []
        task_split = []
        raw_task_datasets = []

        for i in range(len(self.classes) // self.classes_per_task):
            data = []
            targets = []
            split = []

            for j in range(self.classes_per_task):
                label = indexes[self.classes_per_task * i + j]
                split.append(label)
                labelled_targets = [label] * len(divided[label])

                data += divided[label]
                targets += labelled_targets
            
            task_dataset = utils.CustomImageDataset(data, targets, transform=self.training_transform)
            task_datasets.append(task_dataset)
            raw_task_datasets.append(utils.CustomImageDataset(data, targets))
            task_split.append(split)

        return task_split, task_datasets, raw_task_datasets

    def resolve_class_indexes(self, indexes: List[int]) -> List[str]:
        """
        Convert a list of classes indices to their corresponding class names

        Args:
            indexes (List[int]): List of class indices

        Returns:
            List[str]: List of class names corresponding to the indices
        """
        labels = []

        for i in indexes:
            labels.append(self.classes[i])

        return labels

    def __str__(self) -> str:
        info = {
            "name": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "dataset_class": f"{self.dataset_class.__module__}.{self.dataset_class.__qualname__}",
            "training_dataset_parameters": str(self.training_dataset_parameters),
            "testing_dataset_parameters": str(self.testing_dataset_parameters),
            "training_transform": str(self.training_transform),
            "testing_transform": str(self.testing_transform),
            "classes": ", ".join([str(clazz) for clazz in self.classes]),
            "disjoint": self.disjoint
        }

        if self.disjoint:
            info["task_count"] = self.task_count
            info["classes_per_task"] = self.classes_per_task,
            info["task_split_indices"] = self.task_splits
            info["task_split_classes"] = [self.resolve_class_indexes(split) for split in self.task_splits]

        return json.dumps(info, indent=2)

    def create_dataloader(self, batch_size: int, task_number: int = 0) -> torch.utils.data.DataLoader:
        """
        Creates a torch.utils.data.DataLoader instance for the specified task for training
        If self.disjoint = False then this will return the whole training dataset in a DataLoader

        Args:
            batch_size (int): Size of each batch when iterating over the DataLoader
            task_number (int, optional): The task number starting from 0. Defaults to 0.

        Returns:
            torch.utils.data.DataLoader: A DataLoader containing the shuffled data in the task dataset
        """
        assert task_number < self.task_count
        return torch.utils.data.DataLoader(self.task_datasets[task_number], batch_size=batch_size, shuffle=True, num_workers=0)

    def iterate_task_dataloaders(self, batch_size: int) -> Iterator[Tuple[List[int], torch.utils.data.DataLoader]]:
        """
        Iterate over each tasks corresponding DataLoader in order of the tasks

        Args:
            batch_size (int): Size of each batch when iterating over the individual DataLoader

        Yields:
            Iterator[Tuple[List[int], torch.utils.data.DataLoader]]: Yields the task number and the corresponding shuffled DataLoader
        """
        for i in range(0, len(self.task_datasets)):
            yield self.task_splits[i], self.create_dataloader(batch_size=batch_size, task_number=i)

    def create_evaluation_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Creates a torch.utils.data.DataLoader instance using the evaluation dataset

        Args:
            batch_size (int): Size of each batch when iterating over the DataLoader

        Returns:
            torch.utils.data.DataLoader: A DataLoader containing the non-shuffled evaluation data
        """
        return torch.utils.data.DataLoader(self.testing_data, batch_size=batch_size, shuffle=False, num_workers=0) 
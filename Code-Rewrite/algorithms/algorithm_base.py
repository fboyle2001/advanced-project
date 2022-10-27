import abc
import json
from typing import Dict, Union

from loguru import logger

import torch
import datasets

class BaseCLAlgorithm(abc.ABC):
    """
    Complete training and classification package for a Continual Learning algorithm.
    """
    def __init__(
        self,
        name: str,
        model_instance: torch.nn.Module,
        dataset_instance: datasets.BaseCLDataset,
        optimiser_instance: torch.optim.Optimizer,
        loss_criterion_instance: torch.nn.modules.loss._Loss
    ): 
        """
        Represents a complete training and classification package for a Continual Learning algorithm.

        Args:
            name (str): Name of the algorithm
            model_instance (torch.nn.Module): Instance of the model to train
            dataset_instance (datasets.BaseCLDataset): Instance of the dataset to use for training
            optimiser_instance (torch.optim.Optimizer): Instance of the optimiser to use with parameters set
            loss_criterion_instance (torch.nn.modules.loss._Loss): Instance of the loss criterion instance to use
        """
        self.name = name

        self.model = model_instance
        self.device: torch.device = next(self.model.parameters()).device

        self.dataset = dataset_instance
        self.optimiser = optimiser_instance
        self.loss_criterion = loss_criterion_instance

    @staticmethod
    @abc.abstractmethod
    def get_algorithm_folder() -> str:
        """
        Returns the directory name for this algorithm

        Returns:
            str: Directory name for this algorithm
        """
        pass

    @abc.abstractmethod
    def get_unique_information(self) -> Dict[str, Union[str, int]]:
        """
        Return non-standard parameters that are set for this method.
        If there are none then this should return {}

        Returns:
            Dict[str, Union[str, int]]: _description_
        """
        pass

    def get_information(self) -> Dict[str, Union[str, int]]:
        """
        Generate a dictionary containing key information about the algorithm. 
        Used to generate the string representation of this object.

        Returns:
            Dict[str, Union[str, int]]: A dictionary with key information about the algorithm
        """
        info = {
            "name": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "model_class": f"{self.model.__class__.__module__}.{self.model.__class__.__qualname__}",
            "device": str(self.device),
            "dataset_class": f"{self.dataset.__class__.__module__}.{self.dataset.__class__.__qualname__}",
            "optimiser_class": f"{self.optimiser.__class__.__module__}.{self.optimiser.__class__.__qualname__}",
            "loss_criterion_class": f"{self.loss_criterion.__class__.__module__}.{self.loss_criterion.__class__.__qualname__}",
        }

        info = {**info, **self.get_unique_information()}

        return info
    
    def __str__(self) -> str:
        return json.dumps(self.get_information(), indent=2)

    def _setup_training(self) -> None:
        """
        Internal method called to prepare the model for training
        """
        self.model.to(self.device)
        self.model.train()
        logger.debug(f"Model moved to {self.device} and set to train mode")

    @abc.abstractmethod
    def train(self) -> None:
        """
        Trains the model. This method should be overridden.
        """
        logger.info(self)
        self._setup_training()
        logger.info("Starting training")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Classify samples using max of the outputs of the model

        Args:
            batch (torch.Tensor): The batch of images to classify

        Returns:
            torch.Tensor: The predicted classes of the batch of images
        """
        self.model.eval()
        output = self.model(batch)
        _, predicted = torch.max(output.data, 1)
        return predicted
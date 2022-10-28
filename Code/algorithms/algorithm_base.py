from abc import ABC, abstractmethod
from typing import Type, Union

import logging
import time
import os
import traceback
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from datasets import BaseCLDataset

class BaseTrainingAlgorithm(ABC):
    def __init__(
        self,
        name: str,
        alg_directory: str,
        optimiser_class: Type[optim.Optimizer],
        initial_optimiser_parameters: dict,
        criterion_class: Type[nn.modules.loss._Loss],
        device: Union[str, torch.device, None],
        verbose: bool,
        log_to_file: bool,
        log_to_console: bool
    ):
        self.name = name
        self.device = device
        self.verbose = verbose

        self._optimiser_class = optimiser_class
        self._initial_optimiser_parameters = initial_optimiser_parameters

        self._criterion_class = criterion_class
        self._init_time = time.time()
        self._alg_directory = alg_directory
        self._save_count = 0
        self._log_to_file = log_to_file

        self.logger = logging.getLogger(f"{__name__}-{time.time()}")
        log_handlers = []

        if log_to_file:
            log_handlers.append(logging.FileHandler(f"{self.save_directory}/log.log"))
        
        if log_to_console:
            log_handlers.append(logging.StreamHandler())

        logging.basicConfig(
            handlers=log_handlers,
            format="[{asctime}][{levelname}][{filename}:{funcName}:{lineno}] {message}",
            style="{",
            level=(logging.DEBUG if verbose else logging.INFO)
        )

        def exc_handler(exctype, value, tb):
            self.logger.exception(''.join(traceback.format_exception(exctype, value, tb)))
            
        sys.excepthook = exc_handler

        self.logger.debug(f"Initialised training algorithm {name}")
        self.logger.info(f"Storing files at {self.save_directory}/")

    def _setup_training(
        self, 
        model: nn.Module
    ) -> None:
        self.optimiser = self._optimiser_class(model.parameters(), **self._initial_optimiser_parameters)
        self.logger.debug(f"Initialised optimiser of type {self._optimiser_class}")
        self.criterion = self._criterion_class()
        self.logger.debug(f"Initialised criterion of type {self._criterion_class}")
        
        model.to(self.device)
        model.train()
        self.logger.debug("Model moved to device and set to training mode")

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        dataset: BaseCLDataset
    ) -> None:
        dataset_metadata = dataset.get_metadata()

        for line in dataset_metadata:
            self.logger.info(line)

        with open(f"{self.save_directory}/dataset_metadata.txt", "w+") as fp:
            for line in dataset_metadata:
                fp.write(f"{line}\n")

        self._setup_training(model)

    @property
    def save_directory(self) -> str:
        directory = f"./models/{self._alg_directory}/{self._init_time}"
        os.makedirs(directory, exist_ok=True)
        return directory

    def dump_model(
        self,
        model: nn.Module,
        name: Union[str, None] = None
    ) -> str:
        if name is None:
            name = f"save-{self._save_count}"
            self._save_count += 1

        loc = f"{self.save_directory}/{name}.pth"

        self.logger.debug(f"Initiated dump of model to {loc}")
        torch.save(model.state_dict(), loc)
        return loc

# from .algorithm_base import BaseTrainingAlgorithm
#
# import torch.optim as optim
# import torch.nn as nn
#
# """
# [Name] (Reference)
# 
# [Description]
# 
# Disjoint Task Formulation: Yes / No
# Online CL: Yes / No
# Class Incremental: Yes / No
# """
# class ExampleTrainingAlgorithm(BaseTrainingAlgorithm):
#     def __init__(self, device, verbose=True, log_to_file=True, log_to_console=True):
#         super().__init__(
#             name="",
#             alg_directory="",
#             optimiser_class=optim.Adam, 
#             initial_optimiser_parameters={ "lr": 1e-3 },
#             criterion_class=nn.CrossEntropyLoss,
#             device=device,
#             verbose=verbose,
#             log_to_file=log_to_file,
#             log_to_console=log_to_console
#         )

#     def train(self, model, dataset, max_epochs):
#         super().train(model, dataset)
#         train_loader = dataset.training_loader

#         for epoch in range(max_epochs):
#             self.logger.info(f"Starting epoch {epoch+1} / {max_epochs}")
        
#         self.logger.info("Training completed")
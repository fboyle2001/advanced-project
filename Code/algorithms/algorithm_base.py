from abc import ABC, abstractmethod
from typing import Type, Union

import logging
import time
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data

class BaseTrainingAlgorithm(ABC):
    def __init__(
        self,
        name: str,
        alg_directory: str,
        optimiser_class: Type[optim.Optimizer],
        initial_optimiser_parameters: dict,
        criterion_class: Type[nn.modules.loss._Loss],
        device: Union[str, None],
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
        train_loader: torch.utils.data.DataLoader
    ) -> None:
        self._setup_training(model)

    @property
    def save_directory(self):
        directory = f"./models/{self._alg_directory}/{self._init_time}"
        os.makedirs(directory, exist_ok=True)
        return directory

    def dump_model(
        self,
        model: nn.Module,
        name: Union[str, None] = None
    ):
        if name is None:
            name = f"save-{self._save_count}"
            self._save_count += 1

        self.logger.debug(f"Initiated dump of model to {self.save_directory}/{name}.pth")
        torch.save(model.state_dict(), f"{self.save_directory}/{name}.pth")
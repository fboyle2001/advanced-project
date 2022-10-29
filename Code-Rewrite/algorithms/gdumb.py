from typing import Dict, Union
from loguru import logger

from .algorithm_base import BaseCLAlgorithm
import datasets
from . import utils

import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard

class GDumb(BaseCLAlgorithm):
    """
    GDumb (Prabhu et al. 2020)

    Stores samples in a replay buffer and uses it at inference time to train
    a model from scratch. Challenges the success of existing algorithms

    Disjoint Task Formulation: No
    Online CL: Yes
    Class Incremental: Yes
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        batch_size: int,
        max_memory_samples: int,
        post_population_max_epochs: int
    ):
        super().__init__(
            name="GDumb",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.batch_size = batch_size
        self.max_memory_samples = max_memory_samples
        self.post_population_max_epochs = post_population_max_epochs
        
        self.replay_buffer = utils.HashReplayBuffer(max_memory_samples, "random_from_largest_class")

    @staticmethod
    def get_algorithm_folder() -> str:
        return "gdumb"

    def get_unique_information(self) -> Dict[str, Union[str, int]]:
        info: Dict[str, Union[str, int]] = {
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "post_population_max_epochs": self.post_population_max_epochs
        }

        return info

    def train(self) -> None:
        super().train()
        logger.info("Populating replay buffer")

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            for batch_no, data in enumerate(task_dataloader, 0):
                logger.debug(f"{batch_no + 1} / {len(task_dataloader)}")
                inp, labels = data

                for j in range(0, len(inp)):
                    self.replay_buffer.add_to_buffer(inp[j], labels[j])
        
        logger.info("Replay buffer populated")

        buffer_dataset = self.replay_buffer.to_torch_dataset()
        buffer_dataloader = DataLoader(buffer_dataset, batch_size=self.batch_size, shuffle=True)

        logger.info("Training model for inference from buffer")

        for epoch in range(1, self.post_population_max_epochs + 1):
            logger.info(f"Starting epoch {epoch} / {self.post_population_max_epochs}")
            running_loss = 0

            for batch_no, data in enumerate(buffer_dataloader, 0):
                inp, labels = data
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                self.optimiser.zero_grad()
                predictions = self.model(inp)
                loss = self.loss_criterion(predictions, labels)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

                if batch_no == len(buffer_dataloader) - 1:
                    logger.info(f"{epoch} loss: {running_loss / (len(buffer_dataloader) - 1):.3f}")
                    running_loss = 0
        
        logger.info("Training completed")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
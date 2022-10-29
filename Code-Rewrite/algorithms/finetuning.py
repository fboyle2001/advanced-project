from typing import Dict, Union
from loguru import logger

from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard

class Finetuning(BaseCLAlgorithm):
    """
    Finetuning is a baseline algorithm. It is the same as Offline Training but only a
    single epoch is allowed for each task.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        batch_size: int
    ):
        super().__init__(
            name="Finetuning",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.max_epochs_per_task = 1
        self.batch_size = batch_size

    @staticmethod
    def get_algorithm_folder() -> str:
        return "finetuning"

    def get_unique_information(self) -> Dict[str, Union[str, int]]:
        info: Dict[str, Union[str, int]] = {
            "max_epochs_per_task": self.max_epochs_per_task,
            "batch_size": self.batch_size
        }

        return info

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            for epoch in range(1, self.max_epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.max_epochs_per_task}")
                running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    inp, labels = data
                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    labels = labels.to(self.device)
                    self.optimiser.zero_grad()
                    predictions = self.model(inp)
                    loss = self.loss_criterion(predictions, labels)
                    loss.backward()
                    self.optimiser.step()

                    running_loss += loss.item()

                epoch_offset = self.max_epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)
                
                running_loss = 0
        
            self.run_base_task_metrics(task_no)

        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
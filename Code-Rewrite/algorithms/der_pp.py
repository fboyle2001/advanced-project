from typing import Dict, Union
from loguru import logger

from . import buffers
from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
from torch.utils.data import DataLoader

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip
import numpy as np
from PIL import Image

class DarkExperiencePlusPlus(BaseCLAlgorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        epochs_per_task: int,
        batch_size: int,
        max_memory_samples: int,
        alpha: float,
        beta: float
    ):
        super().__init__(
            name="DER++",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.max_memory_samples = max_memory_samples

        self.buffer = buffers.BalancedReplayBufferWithLogits(self.max_memory_samples)
        self.alpha = alpha # recommended by the paper
        self.beta = beta # recommended by the paper

        # Similar to https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/agents/scr.py
        self.augment = torch.nn.Sequential(
            RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
            RandomHorizontalFlip()
            # ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            # RandomGrayscale(p=0.2)
        ).to(self.device)

        self.memory_batch_size = batch_size
        self.mse_loss = torch.nn.MSELoss()

    @staticmethod
    def get_algorithm_folder() -> str:
        return "der_pp"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "alpha": self.alpha,
            "beta": self.beta
        }

        return info

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            self.model.train()

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(1, self.epochs_per_task + 1):
                self.require_mean_calculation = True

                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0
                short_running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    raw_inp, labels = data
                    labels = labels.to(self.device)
                    # Need to account for the potential that the last batch is smaller than the actual batch size
                    B = raw_inp.shape[0]
                    
                    # Apply the training transform i.e. convert it to a gaussian normalised tensor
                    inp = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore

                    # Augment the samples
                    inp = self.augment(inp.detach().clone())
                    loss = torch.zeros(1)
                    x_logits = None

                    # Wait until the buffer is sufficiently filled
                    if self.buffer.count >= self.memory_batch_size * 2:
                        # Draw the two batches of samples
                        x_p, y_p, z_p = self.buffer.draw_sample(self.memory_batch_size, self.device, transform=self.dataset.training_transform)
                        x_dp, y_dp, z_dp = self.buffer.draw_sample(self.memory_batch_size, self.device, transform=self.dataset.training_transform)

                        # Augment both batches
                        x_p = self.augment(x_p.detach().clone())
                        x_dp = self.augment(x_dp.detach().clone())

                        joint_logits = self.model(torch.cat([inp, x_p, x_dp], dim=0))
                        x_logits = joint_logits[:B]
                        x_p_logits = joint_logits[B : B + self.memory_batch_size]
                        x_dp_logits = joint_logits[B + self.memory_batch_size : B + 2 * self.memory_batch_size]

                        # Compute the standard loss for the batch
                        loss = self.loss_criterion(x_logits, labels)

                        # Compute the DER standard loss on x_p
                        # x_p_logits = self.model(x_p)
                        der_standard_loss = self.mse_loss(z_p, x_p_logits)
                        loss += self.alpha * der_standard_loss

                        # Compute the DER++ loss term on x_dp
                        # x_dp_logits = self.model(x_dp)
                        der_pp_loss = self.loss_criterion(x_dp_logits, y_dp)
                        loss += self.beta * der_pp_loss
                    else:
                        # Compute the standard loss for the batch
                        x_logits = self.model(inp)
                        loss = self.loss_criterion(x_logits, labels)

                    # Update the memory buffer
                    for x, y, z in zip(raw_inp, labels, x_logits):
                        self.buffer.add_sample(x.numpy(), y.item(), z.detach().clone())

                    # Optimise the model
                    ## Might need to negate the batch_loss! Unclear if we want gradient ascent!
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()

                    running_loss += loss.item()
                    short_running_loss += loss.item()
                    
                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.info(f"{task_no}:{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                        short_running_loss = 0
                
                logger.debug("Current sample stats:")

                for class_name in self.buffer.known_classes:
                    logger.debug(f"{class_name} has {len(self.buffer.class_hash_pointers[class_name])} samples")

                epoch_offset = self.epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                running_loss = 0
        
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
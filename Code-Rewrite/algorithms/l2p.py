from typing import Dict, Union, Optional
from loguru import logger

from . import utils
from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim

import torchvision.models
import torch.nn as nn

import random

class LearningToPrompt(BaseCLAlgorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        epochs_per_task: int,
        batch_size: int,
        gradient_clip: Optional[float],
        apply_learning_rate_annealing: bool,
        max_lr: Optional[float],
        min_lr: Optional[float],
        cutmix_probability: float
    ):
        super().__init__(
            name="L2P",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip

        self.apply_learning_rate_annealing = apply_learning_rate_annealing
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cutmix_probability = cutmix_probability

        self.pretrained_vit = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        )

    @staticmethod
    def get_algorithm_folder() -> str:
        return "l2p"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "apply_learning_rate_annealing": self.apply_learning_rate_annealing,
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": str(self.max_lr),
            "min_lr": str(self.min_lr),
            "cutmix_probability": self.cutmix_probability
        }

        return info

    # https://github.com/pytorch/vision/blob/4a310f26049371959617921d0eb9b001f4d262c6/torchvision/models/vision_transformer.py#L268
    def flatten_2d_patch_imgs(self, imgs: torch.Tensor):
        xs_p = self.pretrained_vit._process_input(imgs)
        return xs_p

    """
    imgs in B x H x W x C (or similar)
    -> flatten_2d_patch_imgs becomes x_p in R^{B x L x (S^2 * C)}
    -> f_e I think D = S^2 * C here?
    """

    def f_r(self):
        pass

    # https://github.com/pytorch/vision/blob/4a310f26049371959617921d0eb9b001f4d262c6/torchvision/models/vision_transformer.py#L289
    # https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029
    def f_e(self, embeddings: torch.Tensor):
        n = embeddings.shape[0]
        encoder = self.pretrained_vit.encoder
        # Expand the class token to the full batch
        batch_class_token = self.pretrained_vit.class_token.expand(n, -1, -1)
        encoded = torch.cat([batch_class_token, embeddings], dim=1)
        # This returns batch x L x (S^2 * C) i.e. this will give x_p 
        encoded = self.pretrained_vit.encoder(encoded)
        # Classifier "token" as used by standard language architectures
        # encoded = encoded[:, 0] <- use this to draw the [class] token out
        return encoded # B x L x D (really L + 1?)

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            for epoch in range(1, self.epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0

                

                for batch_no, data in enumerate(task_dataloader, 0):
                    pass

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
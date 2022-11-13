from typing import Dict, Union, Optional, Tuple
from loguru import logger

from . import utils, buffers
from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision

import random
import copy
import numpy as np
import sys
from PIL import Image
import time

class HindsightAnchor(BaseCLAlgorithm):
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
        cutmix_probability: float,
        max_memory_per_class: int
    ):
        super().__init__(
            name="Hindsight Anchor",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        assert self.epochs_per_task == 1
        self.batch_size = 16
        self.gradient_clip = gradient_clip

        self.apply_learning_rate_annealing = apply_learning_rate_annealing
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cutmix_probability = cutmix_probability

        self.max_memory_per_class = 5
        self.fifo_buffer = buffers.FIFORingReplayBuffer(max_memory_per_class=self.max_memory_per_class)
        self.finetuning_epochs = 50

    @staticmethod
    def get_algorithm_folder() -> str:
        return "hindsight_anchor_rew"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "apply_learning_rate_annealing": self.apply_learning_rate_annealing,
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": str(self.max_lr),
            "min_lr": str(self.min_lr),
            "cutmix_probability": self.cutmix_probability,
            "max_memory_per_class": self.max_memory_per_class
        }

        return info

    def _update_fifo_buffer(self, inp: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Pass NON-TRANSFORMED inputs to the buffer
        """
        for idx in range(len(inp)):
            raw_sample = inp[idx].detach().cpu().numpy()
            raw_target = targets[idx].detach().cpu().numpy().item()

            self.fifo_buffer.add_sample(raw_sample, raw_target)

    def _preprocess_data(
        self, 
        B_untransformed: torch.Tensor,
        B_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B_M_untransformed, B_M_targets = self.fifo_buffer.draw_sample(self.batch_size, transform=None)
        A_untransformed = torch.cat((B_untransformed, B_M_untransformed), dim=0)
        A_targets = torch.cat((B_targets, B_M_targets), dim=0).long()

        # logger.debug(f"B_untransformed: {B_untransformed.shape}, B_targets: {B_targets.shape}")
        # logger.debug(f"B_M_untransformed: {B_M_untransformed.shape}, B_M_targets: {B_M_targets.shape})")
        # logger.debug(f"A_untransformed: {A_untransformed.shape}, A_targets: {A_targets.shape})")
        
        B = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in B_untransformed]) # type: ignore
        B_M = None
        
        if B_M_untransformed.shape[0] != 0:
            B_M = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in B_M_untransformed]) # type: ignore
        
        A = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in A_untransformed]) # type: ignore
        
        # logger.debug(f"B_untransformed: {B.shape}")

        # if B_M is not None:
        #     logger.debug(f"B_M_untransformed: {B_M.shape}")

        # logger.debug(f"A_untransformed: {A.shape}")
        
        B_untransformed = B_untransformed.to(self.device)
        B_targets = B_targets.to(self.device)
        B_M_untransformed = B_M_untransformed.to(self.device)
        B_M_targets = B_M_targets.to(self.device)
        A_untransformed = A_untransformed.to(self.device)
        A_targets = A_targets.to(self.device)

        B = B.to(self.device)

        if B_M is not None:
            B_M = B_M.to(self.device)

        A = A.to(self.device)

        return B_untransformed, B_targets, B_M_untransformed, B_M_targets, A_untransformed, A_targets, B, B_M, A

    def _finetune_model(self):
        finetuned_model = copy.deepcopy(self.model).to(self.device)
        finetuned_model_opt = optim.SGD(finetuned_model.parameters(), lr=1e-3)
        M_dataloader = torch.utils.data.DataLoader(
            dataset=self.fifo_buffer.to_torch_dataset(transform=self.dataset.training_transform),
            batch_size=self.batch_size,
            shuffle=True
        )

        for finetuning_epoch in range(self.finetuning_epochs):
            finetuning_running_loss = 0

            for finetuning_batch_no, (B_M, B_M_targets) in enumerate(M_dataloader):
                B_M = B_M.to(self.device)
                B_M_targets = B_M_targets.to(self.device)

                finetuned_model_opt.zero_grad()
                finetuned_predictions = finetuned_model(B_M)
                finetuned_loss = self.loss_criterion(finetuned_predictions, B_M_targets)
                finetuned_loss.backward()

                # Clip gradients
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore
                
                finetuned_model_opt.step()
                finetuning_running_loss += finetuned_loss.item()

            if finetuning_epoch % 10 == 0 and finetuning_epoch != 0:
                logger.debug(f"Finetuning Epoch {finetuning_epoch}, RL: {finetuning_running_loss / 40}")
                finetuning_running_loss = 0
        
        return finetuned_model

    def train(self) -> None:
        super().train()

        ### START OF PARAMETERS 
        # batch_size = 16 (see self.batch_size)
        episodic_memory_size = 5 * 5 * 10 # mem_size * num_tasks * total_classes
        ### END OF PARAMETERS

        # Get the raw, untransformed data and their targets
        untransformed_task_datasets = {
            task_no: { 
                "data": self.dataset.task_datasets[task_no].data, # type: ignore
                "targets": self.dataset.task_datasets[task_no].targets # type: ignore
            } for task_no in range(len(self.dataset.task_datasets))
        }

        for task_no in range(self.dataset.task_count):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(self.dataset.task_splits[task_no])}")

            untransformed_task_dataset = datasets.CustomImageDataset(
                data=untransformed_task_datasets[task_no]["data"],
                targets=untransformed_task_datasets[task_no]["targets"],
                transform=None
            )

            untransformed_task_dataloader = torch.utils.data.DataLoader(
                dataset=untransformed_task_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            running_loss = 0

            for batch_no, (B_untransformed, B_targets) in enumerate(untransformed_task_dataloader):
                # Preprocess the data into B, B_M, and A. Put all on the correct device.
                B_untransformed, B_targets, B_M_untransformed, B_M_targets, A_untransformed, A_targets, B, B_M, A = self._preprocess_data(B_untransformed, B_targets)

                # There are no anchors to train on for the first task!
                if task_no == 0: 
                    self.optimiser.zero_grad()
                    predictions = self.model(A)
                    loss = self.loss_criterion(predictions, A_targets) # / A.shape[0]
                    loss.backward()

                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()
                    running_loss += loss.item()

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.debug(f"Batch {batch_no}, RL: {running_loss / 40}")
                        running_loss = 0
                else:
                    sys.exit(0)

                self._update_fifo_buffer(B_untransformed, B_targets)

            # Once the model has been updated, we finetune on the episodic memory
            finetuned_model = self._finetune_model()

            self.model.eval()
            self.run_base_task_metrics(task_no=task_no)
            self.model.train()

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
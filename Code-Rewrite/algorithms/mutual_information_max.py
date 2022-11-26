from typing import Dict, Union, Optional
from loguru import logger

from . import utils
from . import buffers
from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import copy
import random

class MutualInformationMaximisation(BaseCLAlgorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        epochs_per_task: int,
        batch_size: int,
    ):
        super().__init__(
            name="Mutual Information Maximisation",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size

        self.model.final = torch.nn.Identity()
        self.dim_h: int = self.model.dim_out # type: ignore Feature dimensions
        assert type(self.dim_h) is int
        self.sigma = torch.nn.Linear(self.dim_h, len(self.dataset.classes)) # Classifier
        self.d_1 = 128
        self.phi = torch.nn.Linear(self.dim_h, self.d_1) # Feature project head
        self.r = 0 # Temperature, need to configure

        self.alpha = 0 # InfoNCE alpha, need to configure
        self.beta = 0 # InfoNCE beta, need to configure
        self.g_lambda = 0 # need to configure for g' and g*

        self.previous_feature_extractor = None
        self.previous_sigma = None
        self.previous_phi = None

        self.replay_buffer = buffers.BalancedReplayBuffer(1000)

    @staticmethod
    def get_algorithm_folder() -> str:
        return "mim"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size
        }

        return info

    def F(self, X):
        # F is defined as normalisation(phi(f_theta(X)))
        return torch.nn.functional.normalize(self.phi(self.model(X)))

    def F_old(self, X):
        assert self.previous_feature_extractor is not None and self.previous_phi is not None
        return torch.nn.functional.normalize(self.previous_phi(self.previous_feature_extractor(X)))

    # def g(self, a, b):
    #     return torch.exp(torch.dot(self.F(a), self.F(b)).item() / self.r)

    def g_star(self, A, B):
        z = random.randint(0, self.dim_h - self.d_1)
        F_A = self.F(A)
        F_B = self.F(B)
        truncated_features = self.model(A)[:, z:z + self.d_1]
        left = self.g_lambda * torch.dot(F_A, F_B)
        right = (1 - self.g_lambda) * torch.dot(truncated_features, F_B)
        return torch.exp((left + right) / self.r)

    def g_dash(self, A, B):
        return torch.exp(torch.dot(self.F(A), self.F_old(B)) / self.r)

    def compute_similarity_matrix(self, A, B):
        return torch.exp(torch.mm(self.F(A), self.F(B)) / self.r)

    def augment(self, X):
        pass

    def infoNCE_without_targets(self, X, g):
        similarity_matrix = self.compute_similarity_matrix(X, self.augment(X))
        diagonal = similarity_matrix.diagonal()
        B = X.shape[0]
        infoNCE_sum = 0

        for i in range(B):
            infoNCE_sum += torch.log(diagonal[i] / similarity_matrix[i].sum()).item()

        return infoNCE_sum / B
        
    def infoNCE_with_targets(self, X, Y, g):
        right = 0

        return self.alpha * self.infoNCE_without_targets(X, g) + self.beta * right

    def train(self) -> None:
        super().train()

        # Need to change 
        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.task_datasets)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            exemplar_iter = None
            exemplar_dataloader = None
            batch_size = self.batch_size

            if task_no > 0:
                exemplar_dataset = self.replay_buffer.to_torch_dataset(None)
                batch_size = batch_size // 2
                exemplar_dataloader = DataLoader(exemplar_dataset, batch_size=batch_size, shuffle=True)
                exemplar_iter = iter(exemplar_dataloader)

            task_dataloader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(1, self.epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    exemplar_inp, exemplar_labels = None, None

                    if exemplar_iter is not None and exemplar_dataloader is not None:
                        if batch_no == 0:
                            logger.debug("exemplar_iter available")

                        try:
                            exemplar_inp, exemplar_labels = next(exemplar_iter)
                        except:
                            exemplar_iter = iter(exemplar_dataloader)
                            exemplar_inp, exemplar_labels = next(exemplar_iter)

                        exemplar_inp = exemplar_inp.to(self.device)
                        exemplar_labels = exemplar_labels.to(self.device)

                    inp, labels = data

                    # if exemplar_inp is not None and exemplar_labels is not None:
                    #     if batch_no == 0:
                    #         logger.info(f"Dims: {inp.shape}, {exemplar_inp.shape}")

                    #     inp = torch.cat([inp, exemplar_inp], dim=0)
                    #     labels = torch.cat([labels, exemplar_labels], dim=0)

                    #     if batch_no == 0:
                    #         logger.debug(f"Stacked: {inp.shape}, {labels.shape}")
                    # else:
                    #     if batch_no == 0:
                    #         logger.debug(f"Not Stacked: {inp.shape}, {labels.shape}")

                    inp = inp.to(self.device)
                    labels = labels.to(self.device)
                    
                    ## For NEW samples: rotate, copy, augment the (copied) 
                    loss = None

                    if task_no == 0:
                        pass
                    else:
                        ## For OLD samples: rotate, copy, augment the (copied) 
                        pass

                    assert loss is not None
                    loss *= -1
                    self.optimiser.zero_grad()
                    (-loss).backward() 
                    self.optimiser.step()

                    running_loss += loss.item()

                epoch_offset = self.epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                running_loss = 0
        
            self.run_base_task_metrics(task_no)

            self.previous_feature_extractor = copy.deepcopy(self.model).to(self.device)
            self.previous_phi = copy.deepcopy(self.phi)
            self.previous_sigma = copy.deepcopy(self.sigma)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        assert False, "Needs changing"
        return super().classify(batch)
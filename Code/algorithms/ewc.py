from typing import Dict, Union
from .algorithm_base import BaseTrainingAlgorithm

import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F

from torch.distributions import Categorical

import copy
import random

from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector

"""
Elastic Weight Consolidation (Kirkpatrick et al. 2017)

For each parameter in the model, compute the corresponding Fisher information
and use this in the loss function to constrain the model by applying a quadratic
penalty based on the difference between the current parameter value and the new
parameter value

Reference repositories:
https://github.com/kuc2477/pytorch-ewc/
https://github.com/moskomule/ewc.pytorch
"""
class ElasticWeightConsolidation(BaseTrainingAlgorithm):
    def __init__(self, device, task_importance, verbose=True, log_to_file=True, log_to_console=True):
        super().__init__(
            name="Elastic Weight Consolidation",
            alg_directory="ewc",
            optimiser_class=optim.SGD, 
            initial_optimiser_parameters={ "lr": 1e-3 },
            criterion_class=nn.CrossEntropyLoss,
            device=device,
            verbose=verbose,
            log_to_file=log_to_file,
            log_to_console=log_to_console
        )

        self.task_importance = task_importance
        self.stored_parameters = {}

        self.fim = None
        self.v0 = None

    def train(self, model, dataset, epochs_per_task=1):
        super().train(model, dataset)
        self.logger.info(f"Running EWC with {epochs_per_task} epochs pers task for {len(dataset.task_splits)} tasks")
        self.logger.info(f"Task importance scaling: {self.task_importance}")

        fisher_estimation_sample_size = 1024
        fisher_batch_size = 32
        fim = {}

        running_loss = 0

        # 1. Train on first task normally
        # 2. Store the model parameters after the first task is trained on
        # 3. Compute the FIM??
        # 4. For each subsequent task:
            # a. Train on the task but apply a per parameter cumulative loss penalty
            # b. After training, store the model parameters again
            # c. Recompute the FIM??
        # Not too sure on when to compute the FIM??

        for task_no, (task_split, task_training_loader) in enumerate(dataset.iterate_task_dataloaders()):
            self.logger.info(f"EWC on Task #{task_no + 1} with class split {task_split} (classes: {dataset.resolve_class_indexes(task_split)})")

            for epoch in range(epochs_per_task): 
                self.logger.info(f"Epoch {epoch + 1} / {epochs_per_task} for Task #{task_no + 1}")

                for batch_no, (imgs, labels) in enumerate(task_training_loader, 0):
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimiser.zero_grad()
                    predictions = model(imgs)

                    loss = self.criterion(predictions, labels)

                    if self.fim is not None: 
                        c = 0
                        l = 0
                        for n, p in model.named_parameters():
                            l += ((p - self.stored_parameters[n]) ** 2).sum()
                            c += 1
                        l /= c
                        loss += self.task_importance * l

                    loss.backward()
                    self.optimiser.step()

                    running_loss += loss.item()

                    if batch_no == len(task_training_loader) - 1:
                        self.logger.info(f"Loss: {running_loss / len(task_training_loader):.3f}")
                        running_loss = 0

            if task_no != len(dataset.task_datasets):
                self.logger.debug("Consolidating EWC parameters")
                self.consolidate(model, dataset.task_datasets[0])
        
        self.logger.info("Training completed")

    def consolidate(self, model, dataset):
        self.logger.debug("Copying existing model parameters for ")
        self.stored_parameters = copy.deepcopy({n: p for n, p in model.named_parameters()})
        self.fim = {}

def fim_diag(model: nn.Module, dataset: list, device: str = "cuda:0"):
    precision_matrices = {}
    for n, p in copy.deepcopy({n: p for n, p in model.named_parameters() if p.requires_grad}).items():
        p.data.zero_()
        precision_matrices[n] = p.data.to(device)

    model.eval()
    for input in dataset:
        input = torch.tensor(input[None, ...])
        model.zero_grad()
        input = input.to(device)
        output = model(input).view(1, -1)
        label = output.max(1)[1].view(-1)
        loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        loss.backward()

        for n, p in model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2 / len(dataset) # type: ignore
    model.train()

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices
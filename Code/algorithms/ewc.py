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

    def estimate_fim(self, model, dataset, sample_size, batch_size, device="cuda:0"):
        # Compute the log likelihoods of samples
        # Take the gradient of each log likelihood
        # Square and mean the gradients

        loglike = []

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for img, labels in loader:
            img = img.to(device)
            labels = labels.to(device)

            predictions = model(img)
            loglike.append(F.log_softmax(predictions, dim=1)[range(batch_size), labels.data])

            if len(loglike) >= sample_size // batch_size:
                break

        self.logger.debug(f"Have {len(loglike)} LL samples")

        loglike = torch.cat(loglike).unbind()

        self.logger.debug("Trying to autograd")
        fisher_diagonals = []

        parts = []

        for i, l in enumerate(loglike):
            # self.logger.debug(f"Autogradding {i + 1}")
            gs = torch.autograd.grad(l, model.parameters(), retain_graph=True)

            for j in range(len(gs)):
                if i == 0:
                    parts.append((gs[j] ** 2).mean(0))
                else:
                    parts[j] += (gs[j] ** 2).mean(0)

        for i in range(len(parts)):
            parts[i] /= len(parts)
        
        # loglike_grads = zip(*[torch.autograd.grad(l, model.parameters(), retain_graph=(i < len(loglike))) for i, l in enumerate(loglike, 1)])
        # self.logger.debug("Autogradded")
        # loglike_grads = [torch.stack(gs) for gs in parts]
        # fisher_diagonals = [(g ** 2).mean(0) for g in loglike_grads]

        return {n: f.detach() for n, f in zip([n for n, _ in model.named_parameters()], parts)}

    def compute_ewc_loss_component(self, model: nn.Module, fim: Dict):
        if len(self.stored_parameters.keys()) == 0:
            return 0

        losses = []

        for name, parameter in model.named_parameters():
            # Might need to require grad here?
            locked_param = self.stored_parameters[name]
            fisher_information = fim[name] 

            losses.append((fisher_information * (parameter - locked_param) ** 2).sum())

        return self.task_importance * sum(losses)

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
                    loss = self.criterion(predictions, labels) + self.compute_ewc_loss_component(model, fim)
                    loss.backward()
                    self.optimiser.step()

                    running_loss += loss.item()

                    if batch_no == len(task_training_loader) - 1:
                        self.logger.info(f"Loss: {running_loss / len(task_training_loader):.3f}")
                        running_loss = 0

            self.logger.debug("Copying parameters")
            self.stored_parameters = copy.deepcopy({n: p for n, p in model.named_parameters()})
            self.logger.debug("Computing FIM")

            if task_no != len(dataset.task_datasets) - 1:
                fim = self.estimate_fim(model, dataset.task_datasets[task_no], fisher_estimation_sample_size, fisher_batch_size)
        
        self.logger.info("Training completed")

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
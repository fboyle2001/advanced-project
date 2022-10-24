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

Uses Fisher information to calculate the importance of parameters. Fisher information
is the variance of the score 
"""
class ElasticWeightConsolidation(BaseTrainingAlgorithm):
    def __init__(self, device, task_importance, verbose=True, log_to_file=True, log_to_console=True):
        super().__init__(
            name="Elastic Weight Consolidation",
            alg_directory="ewc",
            optimiser_class=optim.Adam, 
            initial_optimiser_parameters={ "lr": 1e-3 },
            criterion_class=nn.CrossEntropyLoss,
            device=device,
            verbose=verbose,
            log_to_file=log_to_file,
            log_to_console=log_to_console
        )

        self.task_importance = task_importance

    def train(self, model, dataset, epochs_per_task=1):
        super().train(model, dataset)
        self.logger.info(f"Running EWC with {epochs_per_task} epochs pers task for {len(dataset.task_splits)} tasks")
        self.logger.info(f"Task importance scaling: {self.task_importance}")

        running_loss = 0
        previous_parameters = None
        fish = {}
        sample_size = 1000

        for task_id, (task_split, task_training_loader) in enumerate(dataset.iterate_task_dataloaders()):
            self.logger.info(f"EWC on Task #{task_id + 1} with class split {task_split} (classes: {dataset.resolve_class_indexes(task_split)})")

            for epoch in range(epochs_per_task): 
                self.logger.info(f"Epoch {epoch + 1} / {epochs_per_task} for Task #{task_id + 1}")

                for i, data in enumerate(task_training_loader, 0):
                    inp, labels = data
                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # First task, no parameters to compare against
                    if previous_parameters is None:
                        self.optimiser.zero_grad()
                        predictions = model(inp)
                        loss = self.criterion(predictions, labels)
                        loss.backward()
                        self.optimiser.step()

                        running_loss += loss.item()
                    else:
                        self.optimiser.zero_grad()
                        predictions = model(inp)
                        loss = self.criterion(predictions, labels)
                        imp = 0

                        for n, p in model.named_parameters():
                            _loss = fish[n] * (p - previous_parameters[n]) ** 2
                            # self.logger.debug(f"n, {n}, l, {_loss.detach().sum()}")
                            imp += _loss.sum()

                        loss += imp * self.task_importance

                        loss.backward()
                        self.optimiser.step()

                        running_loss += loss.item()

                    if i == len(task_training_loader) - 1:
                        self.logger.info(f"Loss: {running_loss / len(task_training_loader):.3f}")
                        running_loss = 0
            
            # Here we need to sample 200 samples from the old tasks and use them for the FIM
            old_task_samples = []

            for j in range(task_id + 1):
                print("Old samples from", j)
                old_task_samples += dataset.task_datasets[j].get_sample(sample_size)

            old_task_samples = random.sample(old_task_samples, k=sample_size)
            fish = fim_diag(model, old_task_samples)
            previous_parameters = copy.deepcopy({n: p for n, p in model.named_parameters()})
        
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
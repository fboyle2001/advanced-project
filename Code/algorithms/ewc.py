from typing import Dict, Union

from fim.fisher_metrics import unit_trace_diag
from .algorithm_base import BaseTrainingAlgorithm

import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.nn.utils

from torch.distributions import Categorical

import copy
import random

from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector

from fim import fim_diag

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

    def train(self, model: nn.Module, dataset, epochs_per_task=1):
        super().train(model, dataset)
        self.logger.info(f"Running EWC with {epochs_per_task} epochs pers task for {len(dataset.task_splits)} tasks")
        self.logger.info(f"Task importance scaling: {self.task_importance}")

        running_loss = 0
        running_ewc = 0

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
                        cum_param_loss = 0
                        # self.logger.debug(f"BATCH {batch_no}")

                        for n, p in model.named_parameters():
                            # self.logger.debug(f"{n}, {((p - self.stored_parameters[n]) ** 2).sum().detach().item()}")
                            # self.logger.debug(f"Stored {n}: {self.stored_parameters[n].sum().detach().item()}")
                            parameter_loss = self.fim[n] * ((p - self.stored_parameters[n]) ** 2) # may need to be made mean if the fim is vector?
                            cum_param_loss += parameter_loss.sum()

                        ewc_loss = self.task_importance * cum_param_loss
                        # self.logger.debug(f"EWC loss {ewc_loss.detach().item()}")
                        loss += ewc_loss
                        running_ewc += ewc_loss.detach().item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    self.optimiser.step()

                    running_loss += loss.item()

                    if batch_no == len(task_training_loader) - 1:
                        self.logger.info(f"Loss: {running_loss / len(task_training_loader):.3f}")

                        if self.fim is not None:
                            self.logger.info(f"EWC Loss Contrib: {running_ewc / len(task_training_loader):.3f}")
                        
                        running_loss = 0
                        running_ewc = 0

            if task_no != len(dataset.task_datasets) - 1:
                self.logger.debug("Consolidating EWC parameters")
                self.consolidate(model, dataset.task_datasets[0])
        
        self.logger.info("Training completed")

    def consolidate(self, model, dataset):
        self.logger.debug("Copying existing model parameters")
        self.stored_parameters = copy.deepcopy({n: p.detach() for n, p in model.named_parameters()})

        self.logger.debug("Computing FIM")
        # self.fim = {n: torch.ones_like(p) for n, p in model.named_parameters()}

        real_fim = fim_diag(model, torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True), samples_no=1024, device=torch.device("cuda:0"), verbose=True)
        self.fim = real_fim

        # for n in self.stored_parameters.keys():
        #    self.logger.debug(f"{n}, {real_fim[n].shape}, {self.stored_parameters[n].shape}")

import time
import sys

def fim_diag(model: nn.Module,
             data_loader: torch.utils.data.DataLoader,
             samples_no: int = None,
             empirical: bool = False,
             device: torch.device = None,
             verbose: bool = False,
             every_n: int = None) -> Dict[str, torch.Tensor]:
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    seen_no = 0
    last = 0
    tic = time.time()

    all_fims = dict({})

    while samples_no is None or seen_no < samples_no:
        data_iterator = iter(data_loader)
        try:
            data, target = next(data_iterator)
        except StopIteration:
            if samples_no is None:
                break
            data_iterator = iter(data_loader)
            data, target = next(data_iterator)

        if device is not None:
            data = data.to(device)
            if empirical:
                target = target.to(device)

        logits = model(data)
        if empirical:
            outdx = target.unsqueeze(1)
        else:
            outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)

        idx, batch_size = 0, data.size(0)
        while idx < batch_size and (samples_no is None or seen_no < samples_no):
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad * param.grad)
                    fim[name].detach_()
            seen_no += 1
            idx += 1

            if verbose and seen_no % 100 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

            if every_n and seen_no % every_n == 0:
                all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                     for (n, f) in fim.items()}

    if verbose:
        if seen_no > last:
            toc = time.time()
            fps = float(seen_no - last) / (toc - tic)
        sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

    for name, grad2 in fim.items():
        grad2 /= float(seen_no)

    # all_fims[seen_no] = fim

    return fim
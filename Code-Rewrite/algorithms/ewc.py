from typing import Dict, Union
from loguru import logger

from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.data
import torch.utils.tensorboard
import copy

import matplotlib.pyplot as plt

class ElasticWeightConsolidation(BaseCLAlgorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        loss_criterion: torch.nn.modules.loss._Loss,
        max_epochs_per_task: int,
        batch_size: int,
        task_importance: int
    ):
        super().__init__(
            name="Elastic Weight Consolidation",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.max_epochs_per_task = max_epochs_per_task
        self.batch_size = batch_size
        self.task_importance = task_importance

        self.fim = None
        self.stored_parameters = None

    @staticmethod
    def get_algorithm_folder() -> str:
        return "ewc"

    def get_unique_information(self) -> Dict[str, Union[str, int]]:
        info: Dict[str, Union[str, int]] = {
            "max_epochs_per_task": self.max_epochs_per_task,
            "batch_size": self.batch_size,
            "task_importance": self.task_importance
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
                running_ewc = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    inp, labels = data
                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    self.optimiser.zero_grad()
                    predictions = self.model(inp)
                    loss = self.loss_criterion(predictions, labels)

                    if self.fim is not None and self.stored_parameters is not None:
                        cum_param_loss = 0
                        # self.logger.debug(f"BATCH {batch_no}")

                        for n, p in self.model.named_parameters():
                            # self.logger.debug(f"{n}, {((p - self.stored_parameters[n]) ** 2).sum().detach().item()}")
                            # self.logger.debug(f"Stored {n}: {self.stored_parameters[n].sum().detach().item()}")
                            parameter_loss = self.fim[n] * ((p - self.stored_parameters[n]) ** 2) # may need to be made mean if the fim is vector?
                            cum_param_loss += parameter_loss.sum()

                        ewc_loss = self.task_importance * cum_param_loss
                        # self.logger.debug(f"EWC loss {ewc_loss.detach().item()}")
                        loss += ewc_loss
                        running_ewc += ewc_loss.detach().item() # type: ignore

                    loss.backward()

                    if self.fim is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # type: ignore

                    self.optimiser.step()

                    running_loss += loss.item()

                epoch_offset = self.max_epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                avg_running_ewc = running_ewc / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, EWC Loss: {avg_running_ewc:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_EWC_avg", avg_running_ewc, epoch)
                self.writer.add_scalar("Loss/Overall_EWC_avg", avg_running_ewc, epoch_offset + epoch)

                running_loss = 0
                running_ewc = 0

            self.run_base_task_metrics(task_no)

            if task_no != len(self.dataset.task_datasets) - 1:
                concat_ds = self.dataset.task_datasets[0]

                for k in range(1, task_no + 1):
                    concat_ds = torch.utils.data.ConcatDataset([concat_ds, self.dataset.task_datasets[k]])

                logger.debug("Consolidating EWC parameters")
                self.consolidate(concat_ds)
        
        logger.info("Training complete")
    
    def consolidate(self, task_datasets):
        logger.debug("Copying existing model parameters")
        self.stored_parameters = copy.deepcopy({n: p.detach() for n, p in self.model.named_parameters()})

        logger.debug("Computing FIM")
        # self.fim = {n: torch.ones_like(p) for n, p in model.named_parameters()}

        real_fim = fim_diag(self.model, torch.utils.data.DataLoader(task_datasets, batch_size=32, shuffle=True), samples_no=4096, device=torch.device("cuda:0"), verbose=True)
        self.fim = real_fim

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)

import sys
import time
from torch.distributions import Categorical

# Need to replace, this is completely copied
def fim_diag(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             samples_no: Union[int, None] = None,
             empirical: Union[bool, None] = False,
             device: Union[torch.device, None] = None,
             verbose: Union[bool, None] = False,
             every_n: Union[int, None] = None) -> Dict[str, torch.Tensor]:
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
                    fim[name] += (param.grad * param.grad) # type: ignore
                    fim[name].detach_()
            seen_no += 1
            idx += 1

            if verbose and seen_no % 100 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                logger.debug(f"Samples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

            if every_n and seen_no % every_n == 0:
                all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                     for (n, f) in fim.items()}

    if verbose:
        if seen_no > last:
            toc = time.time()
            fps = float(seen_no - last) / (toc - tic)
        sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n") # type: ignore

    for name, grad2 in fim.items():
        grad2 /= float(seen_no)

    # all_fims[seen_no] = fim

    return fim
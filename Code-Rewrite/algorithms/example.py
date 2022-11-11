from typing import Dict, Union, Optional
from loguru import logger

from . import utils
from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim

import random

class Example(BaseCLAlgorithm):
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
            name="Example",
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

    @staticmethod
    def get_algorithm_folder() -> str:
        return "example"

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

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            lr_warmer = None

            if self.apply_learning_rate_annealing:
                assert self.max_lr is not None and self.min_lr is not None, "Must set min and max LRs for annealing"
                lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=self.min_lr)
                logger.info("Annealing Scheduler setup")

            for epoch in range(1, self.epochs_per_task + 1):
                # Apply learning rate warmup if turned on
                if lr_warmer is not None and self.min_lr is not None and self.max_lr is not None:
                    if epoch == 0:
                        for param_group in self.optimiser.param_groups:
                            param_group['lr'] = self.max_lr * 0.1

                        self.writer.add_scalar("LR/Current_LR", self.max_lr * 0.1, task_no * self.epochs_per_task + epoch)
                    elif epoch == 1:
                        for param_group in self.optimiser.param_groups:
                            param_group['lr'] = self.max_lr

                        self.writer.add_scalar("LR/Current_LR", self.max_lr, task_no * self.epochs_per_task + epoch)
                    else:
                        lr_warmer.step()
                        self.writer.add_scalar("LR/Current_LR", lr_warmer.get_last_lr()[-1], task_no * self.epochs_per_task + epoch)

                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    inp, labels = data

                    # for ix in inp:
                    #     unique_imgs.add(hash(pickle.dumps(ix.detach().cpu())))

                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # Apply cutmix
                    apply_cutmix = random.uniform(0, 1) < self.cutmix_probability
                    lam, labels_a, labels_b = None, None, None

                    # Cannot merge the two if statements because inp will change causing issues in autograd
                    if apply_cutmix: 
                        inp, labels_a, labels_b, lam = utils.cutmix_data(x=inp, y=labels, alpha=1.0)

                    self.optimiser.zero_grad()
                    predictions = self.model(inp)
                    
                    if apply_cutmix: 
                        assert lam is not None and labels_a is not None and labels_b is not None
                        loss = lam * self.loss_criterion(predictions, labels_a) + (1 - lam) * self.loss_criterion(predictions, labels_b)
                    else:
                        loss = self.loss_criterion(predictions, labels)
                    
                    loss.backward()

                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()

                    running_loss += loss.item()

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
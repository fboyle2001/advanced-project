from typing import Dict, Union
from loguru import logger

from .algorithm_base import BaseCLAlgorithm
import datasets
from . import utils

import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import torch.optim as optim

from simplified_dl import VisionDataset
from dotmap import DotMap

class GDumb(BaseCLAlgorithm):
    """
    GDumb (Prabhu et al. 2020)

    Stores samples in a replay buffer and uses it at inference time to train
    a model from scratch. Challenges the success of existing algorithms

    Disjoint Task Formulation: No
    Online CL: Yes
    Class Incremental: Yes
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        batch_size: int,
        max_memory_samples: int,
        post_population_max_epochs: int,
        gradient_clip: Union[None, int],
        max_lr: float,
        min_lr: float
    ):
        super().__init__(
            name="GDumb",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.batch_size = batch_size
        self.max_memory_samples = max_memory_samples
        self.post_population_max_epochs = post_population_max_epochs
        self.gradient_clip = gradient_clip
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        self.replay_buffer = utils.HashReplayBuffer(max_memory_samples, "random_from_largest_class")

    @staticmethod
    def get_algorithm_folder() -> str:
        return "gdumb"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "post_population_max_epochs": self.post_population_max_epochs,
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr
        }

        return info

    def train(self) -> None:
        super().train()
        logger.info("Populating replay buffer")

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            for batch_no, data in enumerate(task_dataloader, 0):
                logger.debug(f"{batch_no + 1} / {len(task_dataloader)}")
                inp, labels = data

                for j in range(0, len(inp)):
                    self.replay_buffer.add_to_buffer(inp[j], labels[j])
        
        logger.info("Replay buffer populated")
        logger.info(f"Buffer keys: {self.replay_buffer.memory.keys()}")

        for class_name in self.replay_buffer.memory.keys():
            logger.info(f"{class_name} has {len(self.replay_buffer.memory[class_name])} samples")

        buffer_dataset = self.replay_buffer.to_torch_dataset()

        logger.info("Setting up VDS")

        # opt = {
        #     "workers": 0,
        #     "batch_size": 16,
        #     "dataset": "CIFAR10",
        #     "data_dir": "./store/data",
        #     "num_tasks": 5,
        #     "num_classes_per_task": 2,
        #     "memory_size": 1000,
        #     "num_pretrain_classes": 0
        # }

        # opt = DotMap(opt)
        # vds = VisionDataset(opt)
        vds = VisionDataset()
        vds.gen_cl_mapping()

        buffer_dataloader = vds.cltrain_loader

        logger.info("Training model for inference from buffer")

        lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=0.0005)

        for epoch in range(1, self.post_population_max_epochs + 1):
            logger.info("Creating new DL")
            # buffer_dataloader = DataLoader(buffer_dataset, batch_size=self.batch_size, shuffle=True)

            logger.info(f"Starting epoch {epoch} / {self.post_population_max_epochs}")
            running_loss = 0

            if epoch == 0:
                for param_group in self.optimiser.param_groups:
                    param_group['lr'] = self.max_lr * 0.1

                self.writer.add_scalar("LR/Current_LR", self.max_lr * 0.1, epoch)
            elif epoch == 1:
                for param_group in self.optimiser.param_groups:
                    param_group['lr'] = self.max_lr

                self.writer.add_scalar("LR/Current_LR", self.max_lr, epoch)
            else:
                lr_warmer.step()
                self.writer.add_scalar("LR/Current_LR", lr_warmer.get_last_lr()[-1], epoch)

            self.model.train()

            for batch_no, data in enumerate(buffer_dataloader, 0):
                inp, labels = data
                inp = inp.to(self.device)
                labels = labels.to(self.device)

                do_cutmix = np.random.rand(1) < 0.5
                if do_cutmix: inp, labels_a, labels_b, lam = cutmix_data(x=inp, y=labels, alpha=1.0)
                
                predictions = self.model(inp)
                loss = lam * self.loss_criterion(predictions, labels_a) + (1 - lam) * self.loss_criterion(predictions, labels_b) if do_cutmix else self.loss_criterion(predictions, labels)
                self.optimiser.zero_grad()
                loss.backward()

                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                self.optimiser.step()

                running_loss += loss.item()

            avg_running_loss = running_loss / (len(buffer_dataloader) - 1)
            logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
            self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch)
            running_loss = 0

            if epoch > 0 and epoch % 10 == 0:
                self.model.eval()
                self.run_base_task_metrics(task_no=epoch, tl=vds.cltest_loader)
                # self.run_base_task_metrics(task_no=epoch, tl=self.dataset.create_evaluation_dataloader(16))
                self.model.train()
        
        self.run_base_task_metrics(task_no=0, tl=vds.cltest_loader)
        self.run_base_task_metrics(task_no=1, tl=self.dataset.create_evaluation_dataloader(16))
        logger.info("Training completed")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)

# https://github.com/drimpossible/GDumb/blob/ca38afcec332fa523ceff0cc8d3846e2bcf78697/src/utils.py
# Taken from official implementation
import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.intc(W * cut_rat)
    cut_h = np.intc(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam
from typing import Dict, Union, List, Any, Optional, Tuple, Generic, TypeVar, Iterator
from loguru import logger

from algorithms import BaseCLAlgorithm, buffers
from datasets.utils import CustomImageDataset
import datasets

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch.optim as optim
from torch.utils.data import DataLoader

import models.vit.vit_models as vit_models
from models.scr import scr_resnet
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomSolarize, RandomInvert
import torchvision
from algorithms.rainbow_online import Cutout

import numpy as np
from PIL import Image
import random
import pickle
import json

class NovelExperimentSix(BaseCLAlgorithm):
    """
    Experiment 6:
    Use the samples selected from NE5 to train a separate model
    This was not good!
    
    CIFAR-10  (2000 samples) final accuracy: 
    CIFAR-100 (5000 Samples) final accuracy: ( + ~1h20m for memory extraction)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
    ):
        super().__init__(
            name="Novel Experiment: Idea Six",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = 256
        self.batch_size = 32
        
        self.augmentations = [
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(45),
            RandomRotation(90),
            RandomSolarize(thresholds=32, p=1),
            RandomSolarize(thresholds=64, p=1),
            RandomSolarize(thresholds=128, p=1),
            RandomInvert(p=1)
        ]

        # self.buffer = LimitedPriorityBuffer(1000, high_priority=False)
        self.memory: Dict[int, List[np.ndarray]] = {}
        self.memory_dataloader = None
        self.load_memory()

        # Not recommended, generates a ~400mb file
        # Probably going to use this tomorrow to investigate the feature embeddings
        self.dump_memory = False

        self.apply_learning_rate_annealing = True
        self.max_lr =  0.05
        self.min_lr = 0.0005

    def load_memory(self):
        stored = None

        with open("ne5_features.json", "r") as fp:
            stored = json.load(fp)
        
        assert stored is not None

        memory: Dict[int, List[np.ndarray]] = {}

        for target in stored["memory"].keys():
            memory[int(target)] = [np.asarray(sample) for sample in stored["memory"][target]]

        logger.info("Loaded memory samples")

        data = np.concatenate(list(memory.values()), axis=0)
        targets = []

        for target in memory.keys():
            for _ in range(len(memory[target])):
                targets.append(target)

        targets = torch.LongTensor(targets)

        print(data.shape, targets.shape)

        dataset = CustomImageDataset(data, targets, transform=self.dataset.training_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.memory_dataloader = dataloader
        logger.info("Created memory dataloader")

    @staticmethod
    def get_algorithm_folder() -> str:
        return "novel_experiment/idea_six"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        return {}

    def train(self) -> None:
        super().train()

        assert self.memory_dataloader is not None

        lr_warmer = None

        if self.apply_learning_rate_annealing:
            assert self.max_lr is not None and self.min_lr is not None, "Must set min and max LRs for annealing"
            lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=self.min_lr)
            logger.info("Annealing Scheduler setup")

        for epoch in range(self.epochs_per_task):
            # Apply learning rate warmup if turned on
            if lr_warmer is not None and self.min_lr is not None and self.max_lr is not None:
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

            running_loss = 0
            short_running_loss = 0

            for batch_no, data in enumerate(self.memory_dataloader):
                inp, labels = data
                inp = inp.to(self.device)
                labels = labels.to(self.device)

                self.optimiser.zero_grad()
                predictions = self.model(inp)
                loss = self.loss_criterion(predictions, labels)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()
                short_running_loss += loss.item()
                
                if batch_no % 40 == 0 and batch_no != 0:
                    logger.info(f"{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                    short_running_loss = 0
            
            logger.info(f"{epoch}:end, loss: {running_loss / len(self.memory_dataloader):.3f}")
            running_loss = 0

            if epoch > 0 and epoch % 10 == 0:
                self.model.eval()
                self.run_base_task_metrics(task_no=epoch)
                self.model.train()
        
        logger.info("Training complete")
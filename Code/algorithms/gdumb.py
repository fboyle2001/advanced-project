from typing import Dict, Union
from loguru import logger

from .algorithm_base import BaseCLAlgorithm
import datasets
from . import buffers
from . import utils

import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import torch.optim as optim
import random

import copy

class GDumb(BaseCLAlgorithm):
    """
    Greedily stores samples as they arrive while maintaining a balanced memory buffer.
    At inference, trains a model on the memory samples only.

    Reference: Prabhu et al. "GDumb: A simple approach that questions our progress in continual learning." 2020
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
        min_lr: float,
        cutmix_probability: float
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

        self.cutmix_probability = cutmix_probability

        self.raw_model = copy.deepcopy(self.model).to("cpu")
        
        self.replay_buffer = buffers.BalancedReplayBuffer(max_memory_samples)

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
            "min_lr": self.min_lr,
            "cutmix_probability": self.cutmix_probability
        }

        return info

    def train(self) -> None:
        super().train()
        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            logger.info("Populating replay buffer")
            
            # Greedily sample at random
            for batch_no, data in enumerate(task_dataloader, 0):
                raw_inp, raw_labels = data

                for data, target in zip(raw_inp, raw_labels):
                    self.replay_buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())
        
            logger.info("Replay buffer populated")
            logger.info(f"Buffer keys: {self.replay_buffer.known_classes}")

            # Check the replay buffer is balanced
            for class_name in self.replay_buffer.known_classes:
                logger.info(f"{class_name} has {len(self.replay_buffer.class_hash_pointers[class_name])} samples")

            self.model = copy.deepcopy(self.raw_model).to(self.device)
            self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
            logger.info("Copied new raw model")

            # Convert the raw images to a PyTorch dataset with a dataloader
            buffer_dataset = self.replay_buffer.to_torch_dataset(transform=self.dataset.training_transform)
            buffer_dataloader = DataLoader(buffer_dataset, batch_size=self.batch_size, shuffle=True)

            offset = self.post_population_max_epochs * task_no

            logger.info("Training model for inference from buffer")

            lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=self.min_lr)
            # unique_imgs = set()

            for epoch in range(1, self.post_population_max_epochs + 1):
                logger.info(f"Starting epoch {epoch} / {self.post_population_max_epochs}")
                # logger.info(f"Unique images: {len(unique_imgs)}")
                running_loss = 0

                # Apply learning rate warmup
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

                for batch_no, data in enumerate(buffer_dataloader, 0):
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

                # Metrics
                avg_running_loss = running_loss / (len(buffer_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch)
                running_loss = 0

                if epoch > 0 and epoch % 10 == 0:
                    self.model.eval()
                    self.run_base_task_metrics(task_no=offset+epoch)
                    self.model.train()
        
        self.run_base_task_metrics(task_no=6 * self.post_population_max_epochs + 1)
        logger.info("Training completed")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
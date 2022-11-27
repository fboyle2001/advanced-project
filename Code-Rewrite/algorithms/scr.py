from typing import Dict, Union
from loguru import logger

from .algorithm_base import BaseCLAlgorithm
from . import buffers
from models import scr_resnet

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

import numpy as np
from PIL import Image

class SupervisedContrastiveReplay(BaseCLAlgorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        epochs_per_task: int,
        batch_size: int,
        max_memory_samples: int, # 2000
        memory_batch_size: int, # 100
        temperature: float # 0.07
    ):
        super().__init__(
            name="Supervised Contrastive Replay",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.max_memory_samples = max_memory_samples

        self.buffer = buffers.BalancedReplayBuffer(max_memory_samples)
        
        # Use the reduced model from the official repo
        self.model = scr_resnet.SupConResNet().to(self.device)  #self.model.to(self.device)
        self.optimiser = optim.SGD(self.model.parameters(), lr=0.1)

        self.memory_batch_size = memory_batch_size
        self.tau = temperature

        # Taken directly from https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/agents/scr.py
        self.augment = nn.Sequential(
            RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        ).to(self.device)
        
        self.require_mean_calculation = True
        self.mean_embeddings = {}

    @staticmethod
    def get_algorithm_folder() -> str:
        return "scr"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "memory_batch_size": self.memory_batch_size,
            "tau (temperature)": self.tau
        }

        return info

    def scl_loss(self, batch, targets) -> torch.Tensor:
        target_indices = {}
        loss = torch.zeros(1, requires_grad=True).to(self.device)
        dot_prod_mul = torch.exp(torch.mm(batch, batch.T / self.tau))

        for i in range(targets.shape[0]):
            target = targets[i].item()

            if target not in target_indices.keys():
                target_indices[target] = set()
            
            target_indices[target].add(i)

        for i, (z_i, target) in enumerate(zip(batch, targets)):
            P_i = list(target_indices[target.item()] - {i})

            if len(P_i) == 0:
                logger.warning(f"Only one sample total for {target}")
                continue

            shared_class = batch[P_i] / self.tau
            numerator_terms = torch.exp(torch.mm(z_i.unsqueeze(0), shared_class.T))
            denom = dot_prod_mul[i].sum() - dot_prod_mul[i, i]
            summed = torch.log(numerator_terms / denom).sum()
            loss += (-1 / len(P_i)) * summed

        return loss / batch.shape[0]

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            self.model.train()

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(1, self.epochs_per_task + 1):
                self.require_mean_calculation = True

                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0
                short_running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    raw_inp, raw_labels = data

                    if self.buffer.count >= self.memory_batch_size:
                        inp = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
                        labels = raw_labels.to(self.device)
                        
                        buffer_data, buffer_targets = self.buffer.draw_sample(self.memory_batch_size, self.device, transform=self.dataset.training_transform)

                        # tags = {x: [0 if x not in self.buffer.known_classes else len(self.buffer.class_hash_pointers[x]), 0] for x in range(10)}

                        # for t in buffer_targets:
                        #     tags[t.item()][1] += 1 # type: ignore

                        # logger.debug(tags)

                        inp = torch.cat([inp, buffer_data], dim=0)
                        labels = torch.cat([labels, buffer_targets], dim=0)
                        
                        augmented = self.augment(inp.detach().clone())

                        inp = torch.cat([inp, augmented], dim=0)
                        labels = torch.cat([labels, labels], dim=0)

                        features = self.model.forward(inp)
                        loss = self.scl_loss(features, labels) 

                        self.optimiser.zero_grad()
                        loss.backward()
                        self.optimiser.step()

                        running_loss += loss.item()
                        short_running_loss += loss.item()

                    for data, target in zip(raw_inp, raw_labels):
                        self.buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.info(f"{task_no}:{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                        short_running_loss = 0
                
                logger.debug("Current sample stats:")

                for class_name in self.buffer.known_classes:
                    logger.debug(f"{class_name} has {len(self.buffer.class_hash_pointers[class_name])} samples")

                epoch_offset = self.epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                running_loss = 0
        
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        if self.require_mean_calculation:
            logger.info("Generating fresh mean embedding vectors")
            self.mean_embeddings = dict()

            for target in self.buffer.known_classes:
                total_embed = []
                count = 0

                for hash_key in self.buffer.class_hash_pointers[target]:
                    count += 1
                    sample = self.dataset.testing_transform(Image.fromarray(self.buffer.hash_map[hash_key].astype(np.uint8))).unsqueeze(0).to(self.device) # type: ignore
                    encoded = self.model.features(sample).detach().clone().squeeze(0)
                    encoded = encoded / encoded.norm()
                    total_embed.append(encoded)
                
                assert total_embed is not None
                mean_embed = torch.stack(total_embed).mean(0).squeeze()
                mean_embed = mean_embed / mean_embed.norm()
                self.mean_embeddings[target] = mean_embed
            
            self.require_mean_calculation = False
            logger.debug(f"Generated mean embedding vectors for: {self.mean_embeddings.keys()}")

        predictions = []
        batch = batch.to(self.device)

        for sample in batch:
            encoded_sample = self.model.features(sample.unsqueeze(0)).squeeze(0)
            encoded_sample = encoded_sample / encoded_sample.norm()
            closest_target = None
            closest_value = None

            for target in self.mean_embeddings.keys():
                norm = (self.mean_embeddings[target] - encoded_sample).square().sum().sqrt()

                if (closest_target is None and closest_value is None) or norm < closest_value:
                    closest_target = target
                    closest_value = norm

            predictions.append(torch.tensor([closest_target]))

        return torch.cat(predictions, dim=0)
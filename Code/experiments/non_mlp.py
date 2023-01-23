"""
The purpose of this experiment is to test train an additional head using the features
from the ViT using SCL loss to segregate the classes and then classify using NCM classification
"""

from typing import Dict, Union, List, Optional, Tuple
from loguru import logger

from algorithms import BaseCLAlgorithm, buffers
import datasets

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch.optim as optim
from torch.utils.data import DataLoader

import models.vit.vit_models as vit_models
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomSolarize, RandomInvert, ColorJitter, RandomGrayscale

import numpy as np
from PIL import Image

class NonMLPImplementation(BaseCLAlgorithm):
    """
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter
    ):
        super().__init__(
            name="Non MLP Implementation",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = 70
        self.batch_size = 16

        self.pretrained_vit = vit_models.create_model_non_prompt(len(self.dataset.classes)).to(self.device)
        self.D = self.pretrained_vit.feat_dim # number of features
        self.tau = 0.07

        self.max_memory_size = 5000
        self.buffer = buffers.BalancedReplayBuffer(self.max_memory_size)
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
        
        # Use the reduced model from the official repo
        self.model = None
        self.optimiser = None
        
        # Taken directly from https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/agents/scr.py
        self.augment = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        ).to(self.device)
        
        self.memory: Dict[int, List[np.ndarray]] = {}
        self.require_mean_calculation = True
        self.mean_embeddings = torch.zeros(len(self.dataset.classes), self.D).to(self.device)

    @staticmethod
    def get_algorithm_folder() -> str:
        return "non_mlp"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_size
        }

        return info

    def gamma(self, sample, compare):
        query_features, _ = self.pretrained_vit.enc.transformer(sample)
        query_features = query_features[:, 0]
        # print("q", query_features.shape)
        sim_calc = torch.nn.CosineSimilarity()
        return 1 - sim_calc(query_features, compare)

    def f_e(self, img_batch):
        # f_e is the embedding layer
        embeddings = self.pretrained_vit.enc.transformer.embeddings(img_batch)
        return embeddings

    def f_r(self, embeddings): 
        # f_r is the self-attention layers
        # prompts = prompts.reshape(-1, self.D)
        encoded, attn_weights = self.pretrained_vit.enc.transformer.encoder(embeddings)
        return encoded

    ## UNCERTAINTY SAMPLING

    def _closest_mean_embeddings(self, batch: torch.Tensor, k: int):
        batch = batch.to(self.device)
        B = batch.shape[0]
        C = len(self.dataset.classes)
        # Get normalised features
        samples = self.f_r(self.f_e(batch))[:, 0, :]
        samples = samples / torch.linalg.norm(samples, dim=1).reshape(-1, 1)
        # Extend the vectors
        tiled = samples.tile(dims=(1, C)).reshape(B, C, self.D)
        tiled_means = self.mean_embeddings.tile(dims=(B, 1, 1))
        # Compute Euclidean distance between all means
        distances = (tiled - tiled_means).square().sum(dim=2).sqrt()
        # Get the smallest k distances i.e. the closest vectors
        return distances.topk(k, dim=1, largest=False)

    def _augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        augmented = batch

        for augmentation in self.augmentations:
            duped = batch.clone()
            augmented_batch = augmentation(duped)
            augmented = torch.cat([augmented, augmented_batch], dim=0)
        
        return augmented
   
    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            for batch_no, data in enumerate(task_dataloader, 0):
                raw_inp, raw_labels = data

                for data, target in zip(raw_inp, raw_labels):
                    self.buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())
            
            logger.info("Populated buffer")
            self.run_base_task_metrics(task_no)

        logger.info("Training complete")

    ## CLASSIFICATION

    # Generate the mean embedding vectors
    def generate_embeddings(self):
        logger.info("Generating fresh mean embedding vectors")
        means = []
        batch_size = 32

        for target in range(0, len(self.dataset.classes)):
            # Zero mean'd, could set them to be really far away alternatively
            if target not in self.buffer.known_classes:
                means.append(torch.zeros(1, self.D).to("cpu"))
                continue
            
            # Extract and convert to tensors
            buffered = [self.dataset.testing_transform(Image.fromarray(self.buffer.hash_map[k].astype(np.uint8))).unsqueeze(0) for k in self.buffer.class_hash_pointers[target]] # type: ignore
            buffered = torch.cat(buffered, dim=0)
            embeds: Optional[torch.Tensor] = None

            # Get the encodings
            for batch_no in range(0, buffered.shape[0] // batch_size + 1):
                start_idx = batch_no * batch_size
                end_idx = (batch_no + 1) * batch_size
                buffer_batch = buffered[start_idx:end_idx].to(self.device)
                encoded = self.f_r(self.f_e(buffer_batch))[:, 0, :].detach().clone()
                encoded = encoded / torch.linalg.norm(encoded, dim=1).reshape(-1, 1)

                if embeds is None:
                    embeds = encoded.to("cpu")
                else:
                    embeds = torch.cat([embeds, encoded.to("cpu")], dim=0)
            
            assert embeds is not None
            means.append(embeds.mean(dim=0).unsqueeze(0))
    
        self.mean_embeddings = torch.cat(means, dim=0).to(self.device)
        self.require_mean_calculation = False
        logger.debug(f"Generated mean embedding vectors for: {self.buffer.known_classes}")

    def ncm_classify(self, batch: torch.Tensor) -> torch.Tensor:
        if self.require_mean_calculation:
            self.generate_embeddings()

        # Classify according to the closest mean
        classes = self._closest_mean_embeddings(batch, k=1).indices.squeeze()
        return classes
    
    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return self.ncm_classify(batch)

class TestNet(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.linear_one = nn.Linear(768, 768)
        self.relu1 = nn.ReLU()

        self.linear_two = nn.Linear(768, 768)
        self.relu2 = nn.ReLU()

        self.linear_three = nn.Linear(768, 768)
        
        self.classifier = nn.Linear(in_features=768, out_features=classes)

    def features(self, x):
        x = self.linear_one(x)
        x = self.relu1(x)

        x = self.linear_two(x)
        x = self.relu2(x)

        x = self.linear_three(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(dim=1)
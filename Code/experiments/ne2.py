from typing import Dict, Union, List, Any, Optional
from loguru import logger

from algorithms import BaseCLAlgorithm, buffers
import datasets

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch.optim as optim
from torch.utils.data import DataLoader

import models.vit.vit_models as vit_models
from models.scr import scr_resnet
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

import numpy as np
from PIL import Image

class NovelExperimentTwo(BaseCLAlgorithm):
    """
    Experiment 2:
    Just use Nearest Feature Embedding classification with the ViT and a memory buffer
    Use random sampling to draw the samples to store
    
    CIFAR-10  (2000 samples) final accuracy: 90.20%
    CIFAR-100 (5000 Samples) final accuracy: 70.31%
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
            name="Novel Experiment: Idea Two",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )
        
        self.model = None
        self.optimiser = None

        self.pretrained_vit = vit_models.create_model_non_prompt(len(self.dataset.classes)).to(self.device)
        self.D = self.pretrained_vit.feat_dim
        self.buffer = buffers.BalancedReplayBuffer(5000)
        
        self.require_mean_calculation = True
        self.mean_embeddings = torch.zeros(len(self.dataset.classes), self.D).to(self.device)

    @staticmethod
    def get_algorithm_folder() -> str:
        return "novel_experiment/idea_two"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        return {}

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

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            self.require_mean_calculation = True
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            for data, target in zip(task_dataset.data, task_dataset.targets): # type: ignore
                self.buffer.add_sample(data, target.detach().cpu().item())

            logger.debug("Current sample stats:")

            for class_name in self.buffer.known_classes:
                logger.debug(f"{class_name} has {len(self.buffer.class_hash_pointers[class_name])} samples")

            if task_no == self.dataset.task_count - 1 or True:
                self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        if self.require_mean_calculation:
            logger.info("Generating fresh mean embedding vectors")
            means = []
            batch_size = 32

            for target in range(0, len(self.dataset.classes)):
                if target not in self.buffer.known_classes:
                    means.append(torch.zeros(1, self.D).to("cpu"))
                    continue
                
                buffered = [self.dataset.testing_transform(Image.fromarray(self.buffer.hash_map[k].astype(np.uint8))).unsqueeze(0) for k in self.buffer.class_hash_pointers[target]] # type: ignore
                buffered = torch.cat(buffered, dim=0)
                embeds: Optional[torch.Tensor] = None

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

        batch = batch.to(self.device)
        B = batch.shape[0]
        C = len(self.dataset.classes)
        samples = self.f_r(self.f_e(batch))[:, 0, :]
        samples = samples / torch.linalg.norm(samples, dim=1).reshape(-1, 1)
        tiled = samples.tile(dims=(1, C)).reshape(B, C, self.D)
        tiled_means = self.mean_embeddings.tile(dims=(B, 1, 1))
        distances = (tiled - tiled_means).square().sum(dim=2).sqrt()
        classes = distances.topk(1, dim=1, largest=False).indices.squeeze()
        return classes
        
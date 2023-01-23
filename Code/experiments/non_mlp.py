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
    valid_sampling_types: List[str] = ["batch_normalised", "relative_distances", "random"]

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
        self.loss_criterion = None
        
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

        self.uncertainty_type = "relative_distances"

        assert self.uncertainty_type in self.valid_sampling_types, "Invalid sampling type"

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

    def _compute_batch_distance(self, batch: torch.Tensor) -> torch.Tensor:
        # This is approach 1 outlined in the paper
        augmented_input = self._augment_batch(batch)
        distances = self._closest_mean_embeddings(augmented_input, k=1).values
        # Normalise across the batch
        distances = (distances / distances.norm()).squeeze().reshape(batch.shape[0], -1).mean(dim=1).cpu()
        return distances

    def _secondary_uncertainty(self, batch: torch.Tensor) -> torch.Tensor:
        # This is approach 2 outlined in the paper
        augmented_input = self._augment_batch(batch)
        # Get the 2 closest
        distances = self._closest_mean_embeddings(augmented_input, k=2).values
        # Transpose and split the vector
        first, second = distances.T.cpu().unbind()
        uncertainty = first / second

        return uncertainty
    
    def _uncertainity_memory_update(self, new_task: DataLoader):
        logger.info("Starting memory update")
        segmented_uncertainty: Dict[int, List[Tuple[np.ndarray, float]]] = {}

        uncertainty_func = None

        if self.uncertainty_type == "batch_normalised":
            uncertainty_func = self._compute_batch_distance
        elif self.uncertainty_type == "relative_distances":
            uncertainty_func = self._secondary_uncertainty

        assert uncertainty_func is not None

        logger.info(f"Using uncertainty type: {self.uncertainty_type} {uncertainty_func.__name__}")

        logger.debug("Processing new task samples")
        # Handle the new task samples first
        for batch_data in new_task:
            raw_inp, labels = batch_data
            inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
            distances = uncertainty_func(inp)

            for sample, target, uncertainty in zip(raw_inp, labels, distances):
                target = target.item()
                
                if target not in segmented_uncertainty.keys():
                    segmented_uncertainty[target] = []

                segmented_uncertainty[target].append((sample.cpu().numpy(), uncertainty.item())) # type: ignore

        # Now we have the new samples, update the mean embeddings
        self.generate_embeddings()
        
        logger.debug(f"Processing old task samples: total targets = {self.memory.keys()}")
        # Handle the old task samples
        for target in self.memory.keys():
            logger.debug(f"On target {target}")
            if target not in segmented_uncertainty.keys():
                segmented_uncertainty[target] = []

            batches = len(self.memory[target]) // self.batch_size

            # Process batches
            for batch_no in range(batches):
                raw_inp = self.memory[target][batch_no * self.batch_size:(batch_no + 1) * self.batch_size]
                inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
                distances = uncertainty_func(inp)

                for sample, uncertainty in zip(raw_inp, distances):
                    segmented_uncertainty[target].append((sample, uncertainty.item())) # type: ignore

            # Process remains
            raw_inp = self.memory[target][batches * self.batch_size:]
            inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
            distances = uncertainty_func(inp)

            for sample, uncertainty in zip(raw_inp, distances):
                segmented_uncertainty[target].append((sample, uncertainty.item())) # type: ignore
        
        logger.debug("Drawing memory samples")
        # Now draw the samples out for the new buffer
        seen_class_count = len(segmented_uncertainty.keys())
        new_memory: Dict[int, List[np.ndarray]] = {}
        memory_per_class = self.max_memory_size // seen_class_count

        for target in segmented_uncertainty.keys():
            items = segmented_uncertainty[target]
            item_count = len(items)
            uncertainty_sorted_items = sorted(items, key=lambda item: item[1])
            selected_samples: List[np.ndarray] = []

            for j in range(memory_per_class):
                selected_samples.append(uncertainty_sorted_items[(j * item_count) // memory_per_class][0])

            new_memory[target] = selected_samples

        self.memory = new_memory
        self.require_mean_calculation = True

        logger.debug("Creating buffer")
        buffer = buffers.BalancedReplayBuffer(5000)

        for target in new_memory.keys():
            for i in range(len(new_memory[target])):
                buffer.add_sample(new_memory[target][i], target)
        
        self.buffer = buffer

        logger.info("Memory update completed")
   
    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            if self.uncertainty_type == "random":
                for batch_no, data in enumerate(task_dataloader, 0):
                    raw_inp, raw_labels = data

                    for data, target in zip(raw_inp, raw_labels):
                        self.buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())
            else:
                self._uncertainity_memory_update()
            
            logger.info("Populated buffer")
            self.require_mean_calculation = True
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
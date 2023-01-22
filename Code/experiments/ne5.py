from typing import Dict, Union, List, Any, Optional, Tuple, Generic, TypeVar, Iterator
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
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomSolarize, RandomInvert
import torchvision
from algorithms.rainbow_online import Cutout

import numpy as np
from PIL import Image
import random
import pickle
import json

class NovelExperimentFive(BaseCLAlgorithm):
    """
    Experiment 5:
    Use Nearest Feature Embedding classification with the ViT and a memory buffer
    Estimate the uncertainty of the samples using augmentation and mean distance to other mean class embeddings
    
    CIFAR-10  (2000 samples) final accuracy: 
    CIFAR-100 (5000 Samples) final accuracy: 73.17% (~1h20m)
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
            name="Novel Experiment: Idea Five",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )
        
        self.model = None
        self.optimiser = None

        self.pretrained_vit = vit_models.create_model_non_prompt(len(self.dataset.classes)).to(self.device).eval()
        self.D = self.pretrained_vit.feat_dim

        self.require_mean_calculation = True
        self.mean_embeddings = torch.zeros(len(self.dataset.classes), self.D).to(self.device)

        self.epochs_per_task = 1
        self.batch_size = 16
        self.max_memory_size = 5000
        
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

        # Not recommended, generates a ~400mb file
        # Probably going to use this tomorrow to investigate the feature embeddings
        self.dump_memory = True

    @staticmethod
    def get_algorithm_folder() -> str:
        return "novel_experiment/idea_five"

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

    def augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        augmented = batch

        for augmentation in self.augmentations:
            duped = batch.clone()
            augmented_batch = augmentation(duped)
            augmented = torch.cat([augmented, augmented_batch], dim=0)
        
        return augmented

    def compute_batch_distance(self, batch: torch.Tensor) -> torch.Tensor:
        augmented_input = self.augment_batch(batch)
        distances = self.closest_mean_embeddings(augmented_input, k=1).values
        distances = (distances / distances.norm()).squeeze().reshape(batch.shape[0], -1).mean(dim=1).cpu()
        return distances

    def update_memory(self, new_task: DataLoader):
        logger.info("Starting memory update")
        segmented_uncertainty: Dict[int, List[Tuple[np.ndarray, float]]] = {}

        logger.debug("Processing new task samples")
        # Handle the new task samples first
        for batch_data in new_task:
            raw_inp, labels = batch_data
            inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
            distances = self.compute_batch_distance(inp)

            for sample, target, uncertainty in zip(raw_inp, labels, distances):
                target = target.item()
                
                if target not in segmented_uncertainty.keys():
                    segmented_uncertainty[target] = []

                segmented_uncertainty[target].append((sample.cpu().numpy(), uncertainty.item())) # type: ignore
        
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
                distances = self.compute_batch_distance(inp)

                for sample, uncertainty in zip(raw_inp, distances):
                    segmented_uncertainty[target].append((sample, uncertainty.item())) # type: ignore

            # Process remains
            raw_inp = self.memory[target][batches * self.batch_size:]
            inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
            distances = self.compute_batch_distance(inp)

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

        logger.info("Memory update completed")

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            with torch.no_grad():
                self.update_memory(task_dataloader)

            logger.debug("Current buffer stats:")

            for target in self.memory.keys():
                logger.debug(f"{target} has {len(self.memory[target])} samples")

            # Need to update
            with torch.no_grad():
                self.generate_embeddings()

            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def run_base_task_metrics(self, task_no: int, eval_batch_size=32):
        if self.dump_memory:
            with open(f"{self.directory}/task_{task_no}_dumped_memory.json", "w+") as fp:
                json.dump({
                    "memory": {
                        target: [x.tolist() for x in self.memory[target]]
                        for target in self.memory.keys()
                    },
                    "embeddings": {
                        target: self.mean_embeddings[target].detach().cpu().numpy().tolist()
                        for target in range(self.mean_embeddings.shape[0])
                    }
                }, fp, indent=2)
            
        super().run_base_task_metrics(task_no, eval_batch_size)

    def generate_embeddings(self):
        logger.info("Generating fresh mean embedding vectors")
        means = []
        batch_size = 32

        for target in range(0, len(self.dataset.classes)):
            if target not in self.memory.keys():
                means.append(torch.full((1, self.D), 1e8).to("cpu"))
                continue
            
            buffered = [self.dataset.testing_transform(Image.fromarray(v.astype(np.uint8))).unsqueeze(0) for v in self.memory[target]] # type: ignore
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
        logger.debug(f"Generated mean embedding vectors for: {self.memory.keys()}")

    def closest_mean_embeddings(self, batch: torch.Tensor, k: int):
        batch = batch.to(self.device)
        B = batch.shape[0]
        C = len(self.dataset.classes)
        samples = self.f_r(self.f_e(batch))[:, 0, :]
        samples = samples / torch.linalg.norm(samples, dim=1).reshape(-1, 1)
        tiled = samples.tile(dims=(1, C)).reshape(B, C, self.D)
        tiled_means = self.mean_embeddings.tile(dims=(B, 1, 1))
        distances = (tiled - tiled_means).square().sum(dim=2).sqrt()
        return distances.topk(k, dim=1, largest=False)

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        if self.require_mean_calculation:
            self.generate_embeddings()

        classes = self.closest_mean_embeddings(batch, k=1).indices.squeeze()
        return classes

from typing import Dict, List, Tuple, Generic, TypeVar, Iterator, Any

import bisect
import pickle

T = TypeVar("T")

class SingleClassLimitedPriorityBuffer(Generic[T]):
    def __init__(self, max_length: int, high_priority: bool = True):
        self.max_length = max_length
        self.high_priority = high_priority

        # Order: Lowest priority --> Highest priority
        # Search by priority: identifiy index i in self.priorities -> draw self.hash_map[self.ordered_hashes[i]] as sample
        self.priorities: List[float] = []
        self.ordered_hashes: List[int] = []
        self.hash_map: Dict[int, T] = {}

    def __len__(self) -> int:
        return len(self.priorities)

    def __iter__(self) -> Iterator[Tuple[float, T]]:
        for priority, hashed in zip(self.priorities, self.ordered_hashes):
            yield priority if not self.high_priority else -priority, self.hash_map[hashed]

    def __str__(self) -> str:
        return f"[{', '.join([str(x) for x in iter(self)])}]"

    def is_full(self) -> bool:
        return len(self) == self.max_length

    def add_sample(self, item: T, priority: float) -> None:
        if self.high_priority:
            priority *= -1

        # If full and priority is lower than the lowest priority, ignore it
        if self.is_full() and priority >= self.priorities[-1]:
            return

        # Calculate the hash key
        hashed = hash(pickle.dumps(item))
        self.hash_map[hashed] = item

        # Insert into the arrays in sorted position
        insert_idx = bisect.bisect(self.priorities, priority)
        self.priorities.insert(insert_idx, priority)
        self.ordered_hashes.insert(insert_idx, hashed)

        self._remove_one_sample()

    def _remove_one_sample(self) -> None:
        # No need to remove a sample if there is still space
        if len(self) <= self.max_length:
            return
        
        # Remove the hashed value
        delete_hash = self.ordered_hashes[-1]
        del self.hash_map[delete_hash]
        
        # Remove one sample
        old_length = len(self.priorities)
        self.priorities = self.priorities[:old_length - 1]
        self.ordered_hashes = self.ordered_hashes[:old_length - 1]

    def reduce_max_length(self, new_max_length: int) -> None:
        if self.max_length < new_max_length:
            self.max_length = new_max_length
            return

        self.max_length = new_max_length

        while len(self) > self.max_length:
            self._remove_one_sample()

class LimitedPriorityBuffer(Generic[T]):
    def __init__(self, max_samples: int, high_priority: bool = True):
        self.max_samples = max_samples
        self.high_priority = high_priority

        self.target_buffers: Dict[Any, SingleClassLimitedPriorityBuffer[T]] = dict()
    
    @property
    def known_targets(self) -> List[Any]:
        return list(self.target_buffers.keys())
    
    def __len__(self):
        return sum([len(buffer) for buffer in self.target_buffers.values()])

    def __str__(self):
        return str({str(target): str(self.target_buffers[target]) for target in self.known_targets})

    def __getitem__(self, target) -> SingleClassLimitedPriorityBuffer[T]:
        return self.target_buffers[target]

    def add_sample(self, target: Any, item: T, priority: float) -> None:
        # Add the new target if it does not exist
        if target not in self.target_buffers.keys():
            self._add_new_target(target)
        
        # Add the sample to the correct buffer
        self.target_buffers[target].add_sample(item, priority)

    def _add_new_target(self, target: Any):
        # Ignore if it is already in the buffer
        if target in self.target_buffers.keys():
            return

        # Recalculate the max samples per class
        max_samples_per_class = self.max_samples // (len(self.known_targets) + 1)
        self.target_buffers[target] = SingleClassLimitedPriorityBuffer(max_samples_per_class, high_priority=self.high_priority)

        # Reduce the size of existing buffers accordingly w.r.t to the priorities
        for known_target in self.known_targets:
            self.target_buffers[known_target].reduce_max_length(max_samples_per_class)
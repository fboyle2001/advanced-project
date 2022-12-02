from typing import Dict, Union, List, Any, Optional, Tuple
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

class NovelExperimentFour(BaseCLAlgorithm):
    """
    Experiment 4:
    Use Nearest Feature Embedding classification with the ViT and a memory buffer
    Estimate the uncertainty of the samples using augmentation and mean distance to other mean class embeddings
    
    CIFAR-10  (2000 samples) final accuracy:
    CIFAR-100 (5000 Samples) final accuracy:
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
            name="Novel Experiment: Idea Four",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )
        
        self.model = None
        self.optimiser = None

        self.pretrained_vit = vit_models.create_model_non_prompt().to(self.device)
        self.D = self.pretrained_vit.feat_dim
        self.buffer = buffers.BalancedReplayBuffer(5000)

        self.require_mean_calculation = True
        self.mean_embeddings = torch.zeros(len(self.dataset.classes), self.D).to(self.device)

        self.epochs_per_task = 1
        self.batch_size = 16
        
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

    @staticmethod
    def get_algorithm_folder() -> str:
        return "novel_experiment/idea_four"

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

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            self.require_mean_calculation = True

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            if task_no == 0:
                for data, target in zip(task_dataset.data, task_dataset.targets): # type: ignore
                    self.buffer.add_sample(data, target.detach().cpu().item())
                
                self.generate_embeddings()
                #self.run_base_task_metrics(task_no)
                continue

            buf_distances = {}
            buf_samples = {}

            for batch_no, data in enumerate(task_dataloader, 0):
                raw_inp, labels = data
                inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
                
                augmented_input = self.augment_batch(inp)
                # labels = labels.tile(len(self.augmentations) + 1) #.reshape(-1, 1)
                closest = self.closest_mean_embeddings(augmented_input, k=1)
                closest = closest.values.squeeze().reshape(inp.shape[0], -1).mean(dim=1)
                
                for dist, sample, label in zip(closest.to("cpu"), raw_inp.to("cpu"), labels.to("cpu")):
                    label = label.detach().item()

                    if label not in buf_distances.keys():
                        buf_distances[label] = []
                        buf_samples[label] = []

                    buf_distances[label].append(dist.detach().item())
                    buf_samples[label].append(sample.detach().numpy())

                if batch_no % 40 == 0 and batch_no != 0:
                    logger.info(f"{task_no}:{batch_no}")
                    # break
                
                self.require_mean_calculation = True

            logger.info("Populating buffer")
            space = self.buffer.count // (len(self.buffer.known_classes) + len(task_indices))

            for target in buf_samples.keys():
                buf_sorted_indices = sorted(range(0, len(buf_distances[target])), key=lambda i: buf_distances[target][i])
                for i in range(space // 2):
                    if i >= len(buf_sorted_indices) // 2:
                        break

                    self.buffer.add_sample(buf_samples[target][buf_sorted_indices[i]], target)
                    self.buffer.add_sample(buf_samples[target][buf_sorted_indices[-i]], target)

            logger.info("Buffer populated")

            logger.debug("Current buffer stats:")

            for class_name in self.buffer.known_classes:
                logger.debug(f"{class_name} has {len(self.buffer.class_hash_pointers[class_name])} samples")
        
            self.generate_embeddings()
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def generate_embeddings(self):
        logger.info("Generating fresh mean embedding vectors")
        means = []
        batch_size = 32

        for target in range(0, len(self.dataset.classes)):
            if target not in self.buffer.known_classes:
                means.append(torch.full((1, self.D), 1e8).to("cpu"))
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

class BalancedReplayBufferWithPriority:
    def __init__(self, max_memory_size: int, largest: bool, disable_warnings: bool = False):
        self.max_memory_size = max_memory_size
        self.disable_warnings = disable_warnings
        self.largest = largest

        self.class_hash_pointers: Dict[Any, List[Tuple[int, float]]] = {} # hash, score
        self.hash_map: Dict[int, Any] = {}

    @property
    def count(self):
        return len(self.hash_map.keys())

    @property
    def known_classes(self):
        return self.class_hash_pointers.keys()

    def _remove_sample(self) -> None:
        if self.count == 0:
            return
        
        largest_class = max(self.class_hash_pointers.keys(), key=lambda c: len(self.class_hash_pointers[c]))
        remove_index = sorted(enumerate(self.class_hash_pointers[largest_class]), key=lambda x: x[1][1], reverse=self.largest)[-1][0]
        # remove_index = random.randrange(0, len(self.class_hash_pointers[largest_class]))

        # Remove from the hash map
        data = self.class_hash_pointers[largest_class][remove_index]
        del self.hash_map[data[0]]

        # Remove from the class map
        self.class_hash_pointers[largest_class].remove(data)

    def add_sample(self, data: np.ndarray, target: Any, score: float) -> None:
        if not self.disable_warnings and type(data) is torch.Tensor:
            logger.warning(f"Received data of type tensor")

        if not self.disable_warnings and type(target) is torch.Tensor:
            logger.warning(f"Received target of type tensor ({target})")

        data_hash = hash(pickle.dumps(data))
        data_key = (data_hash, score)

        if data_key in self.hash_map.keys():
            logger.warning(f"Duplicate key: {data_key}")
            return

        self.hash_map[data_hash] = data

        if target not in self.class_hash_pointers.keys():
            self.class_hash_pointers[target] = []

        self.class_hash_pointers[target].append(data_key)

        # Need to delete a sample now
        if self.count > self.max_memory_size:
            self._remove_sample()
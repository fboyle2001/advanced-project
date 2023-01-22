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
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomSolarize, RandomInvert, ColorJitter, RandomGrayscale
import torchvision
from algorithms.rainbow_online import Cutout

import numpy as np
from PIL import Image
import random
import pickle
import json

import models.vit.mlp as mlp

class NovelExperimentSeven(BaseCLAlgorithm):
    """
    Experiment 7:
    
    CIFAR-10  (2000 samples) final accuracy: 
    CIFAR-100 (5000 Samples) final accuracy:  (+ ~1h20m)
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
            name="Novel Experiment: Idea Seven",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )
        
        self.model = mlp.MLP(784, [784, 512, 256, 128, 100])
        self.optimiser = optim.SGD(self.model.parameters(), lr=1e-3)
        self.loss_criterion = SupConLoss()

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

        # Taken directly from https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/agents/scr.py
        self.augment = nn.Sequential(
            RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        ).to(self.device)

        # self.buffer = LimitedPriorityBuffer(1000, high_priority=False)
        self.memory: Dict[int, List[np.ndarray]] = {}
        self.buffer = None

        # Not recommended, generates a ~400mb file
        # Probably going to use this tomorrow to investigate the feature embeddings
        self.dump_memory = True
        self.memory_batch_size = 100

    @staticmethod
    def get_algorithm_folder() -> str:
        return "novel_experiment/idea_seven"

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
            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            with torch.no_grad():
                self.update_memory(task_dataloader)
                self.generate_embeddings()

            logger.debug("Current buffer stats:")

            for target in self.memory.keys():
                logger.debug(f"{target} has {len(self.memory[target])} samples")

            running_loss = 0
            short_running_loss = 0

            assert self.buffer is not None

            for batch_no, data in enumerate(task_dataloader, 0):
                raw_inp, raw_labels = data

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
                inp = self.f_r(self.f_e(inp))[:, 0, :].unsqueeze(1)
                labels = torch.cat([labels, labels], dim=0)

                features = self.model.forward(inp)
                loss = self.loss_criterion(features, labels) 

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()
                short_running_loss += loss.item()

                if batch_no % 40 == 0 and batch_no != 0:
                    logger.info(f"{task_no}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                    short_running_loss = 0

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
        vit_features = self.f_r(self.f_e(batch))[:, 0, :].unsqueeze(1)
        output = self.model(vit_features)
        _, predicted = torch.max(output.data, 1)
        return predicted

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

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device) # type: ignore

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
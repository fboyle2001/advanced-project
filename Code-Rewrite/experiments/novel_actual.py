from typing import Dict, Union, List, Any, Optional, Tuple, Generic, TypeVar, Iterator, Literal
from loguru import logger

from algorithms import BaseCLAlgorithm, buffers
import datasets

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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

class NovelActual(BaseCLAlgorithm):
    """

    
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
            name="Novel Actual",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.model = None
        self.loss_criterion = None
        self.optimiser = None

        # setup
        self.classification_mode: Literal["ncm", "max"] = "ncm"
        self.training_loss_mode: Literal["none", "ce", "scl"] = "scl"
        self.sampling_mode: Literal["random", "uncertainty"] = "random"

        # assert self.training_loss_mode != "scl", "SCL not yet implemented"

        self.epochs_per_task = 50
        self.batch_size = 32
        self.max_memory_size = 5000
        self.memory_batch_size = 16

        # basics
        self.pretrained_vit = vit_models.create_model_non_prompt(len(self.dataset.classes)).to(self.device).eval()
        self.D = self.pretrained_vit.feat_dim
        
        self.require_mean_calculation = True
        self.mean_embeddings = torch.zeros(len(self.dataset.classes), self.D).to(self.device)
        
        # sampling
        ## TODO: Remove self.memory in favour of self.buffer
        self.memory: Dict[int, List[np.ndarray]] = {}
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

        # Taken directly from https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/agents/scr.py
        self.augment = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        ).to(self.device)

        # loss and optimisation

        if self.training_loss_mode != "none":
            self.optimiser = optim.SGD(self.pretrained_vit.head.parameters(), lr=0.02)

            if self.training_loss_mode == "ce":
                self.loss_criterion = torch.nn.CrossEntropyLoss().to(self.device)
            elif self.training_loss_mode == "scl":
                self.loss_criterion = SupConLoss().to(self.device)
            else:
                assert 1 == 0
            
            logger.info(f"Assigned {self.loss_criterion}")


    ### CLASS METHODS

    @staticmethod
    def get_algorithm_folder() -> str:
        return "novel_actual/one"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        return {}

    ### VIT METHODS

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

    ### MEMORY UPDATE

    def update_memory(self, task_dataset: Dataset, task_dataloader: DataLoader):
        if self.sampling_mode == "random":
            self._random_memory_update(task_dataset)
        else:
            self._uncertainity_memory_update(task_dataloader)

    ## RANDOM SAMPLING

    def _random_memory_update(self, task_dataset: Dataset):
        for data, target in zip(task_dataset.data, task_dataset.targets): # type: ignore
            self.buffer.add_sample(data, target.detach().cpu().item())
        
        self.require_mean_calculation = True

    ## UNCERTAINTY SAMPLING

    def _closest_mean_embeddings(self, batch: torch.Tensor, k: int):
        batch = batch.to(self.device)
        B = batch.shape[0]
        C = len(self.dataset.classes)
        samples = self.f_r(self.f_e(batch))[:, 0, :]
        samples = samples / torch.linalg.norm(samples, dim=1).reshape(-1, 1)
        tiled = samples.tile(dims=(1, C)).reshape(B, C, self.D)
        tiled_means = self.mean_embeddings.tile(dims=(B, 1, 1))
        # Compute Euclidean distance
        distances = (tiled - tiled_means).square().sum(dim=2).sqrt()
        return distances.topk(k, dim=1, largest=False)

    def _augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        augmented = batch

        for augmentation in self.augmentations:
            duped = batch.clone()
            augmented_batch = augmentation(duped)
            augmented = torch.cat([augmented, augmented_batch], dim=0)
        
        return augmented

    def _compute_batch_distance(self, batch: torch.Tensor) -> torch.Tensor:
        augmented_input = self._augment_batch(batch)
        distances = self._closest_mean_embeddings(augmented_input, k=1).values
        distances = (distances / distances.norm()).squeeze().reshape(batch.shape[0], -1).mean(dim=1).cpu()
        return distances
    
    def _uncertainity_memory_update(self, new_task: DataLoader):
        logger.info("Starting memory update")
        segmented_uncertainty: Dict[int, List[Tuple[np.ndarray, float]]] = {}

        logger.debug("Processing new task samples")
        # Handle the new task samples first
        for batch_data in new_task:
            raw_inp, labels = batch_data
            inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
            distances = self._compute_batch_distance(inp)

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
                distances = self._compute_batch_distance(inp)

                for sample, uncertainty in zip(raw_inp, distances):
                    segmented_uncertainty[target].append((sample, uncertainty.item())) # type: ignore

            # Process remains
            raw_inp = self.memory[target][batches * self.batch_size:]
            inp = torch.stack([self.dataset.testing_transform(Image.fromarray(x.astype(np.uint8))) for x in raw_inp]).to(self.device) # type: ignore
            distances = self._compute_batch_distance(inp)

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

    ## NCM Classification

    def generate_embeddings(self):
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

    def ncm_classify(self, batch: torch.Tensor) -> torch.Tensor:
        if self.require_mean_calculation:
            self.generate_embeddings()

        classes = self._closest_mean_embeddings(batch, k=1).indices.squeeze()
        return classes
    
    ## Max Classification

    def max_classify(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.pretrained_vit(batch)
        _, predicted = torch.max(output.data, 1)
        return predicted

    ## Training

    def train(self) -> None:
        super().train()

        for task_no, (task_indices, task_dataset) in enumerate(zip(self.dataset.task_splits, self.dataset.raw_task_datasets)):
            self.require_mean_calculation = True
            task_dataloader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=True)

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            with torch.no_grad():
                self.update_memory(task_dataset, task_dataloader)
                logger.info(f"Updated memory using mode {self.sampling_mode}")

                if self.classification_mode == "ncm" or self.sampling_mode == "uncertainty":
                    self.generate_embeddings()
                    logger.info(f"Embeddings generated")

            logger.debug("Current buffer stats:")

            for class_name in self.buffer.known_classes:
                logger.debug(f"{class_name} has {len(self.buffer.class_hash_pointers[class_name])} samples")

            if self.training_loss_mode != "none":
                logger.info(f"TLM is {self.training_loss_mode} so entered training")
                assert self.optimiser is not None, "Optimiser is none"
                assert self.loss_criterion is not None, "Loss Criterion is none"
                assert self.buffer is not None, "Buffer is none"
                
                running_loss = 0
                short_running_loss = 0
                
                buffer_ds = self.buffer.to_torch_dataset(transform=self.dataset.training_transform)
                buffer_dl = DataLoader(buffer_ds, batch_size=self.batch_size, shuffle=True)

                for epoch in range(self.epochs_per_task):
                    for batch_no, data in enumerate(buffer_dl, 0):
                        inp, labels = data
                        inp = inp.to(self.device)
                        labels = labels.to(self.device)
                        
                        augmented = self.augment(inp.detach().clone())

                        inp = torch.cat([inp, augmented], dim=0)
                        labels = torch.cat([labels, labels], dim=0)

                        outputs = self.pretrained_vit(inp) 

                        if self.training_loss_mode == "ce":
                            print("In CE")
                            # outputs = self.pretrained_vit(inp)
                        elif self.training_loss_mode == "scl":
                            print("In SCL")
                            outputs = outputs.unsqueeze(1) # self.f_r(self.f_e(inp))[:, 0, :].unsqueeze(1)
                        
                        assert outputs is not None
                        print(outputs.shape, labels.shape)

                        print(self.loss_criterion)
                        loss = self.loss_criterion(outputs, labels)
                        print(loss.shape)
                        print(loss)

                        self.optimiser.zero_grad()
                        loss.backward()
                        self.optimiser.step()

                        running_loss += loss.item()
                        short_running_loss += loss.item()

                        if batch_no % 40 == 0 and batch_no != 0:
                            logger.info(f"{task_no}:{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                            short_running_loss = 0
                    
                    logger.info(f"{task_no}:{epoch}, loss: {running_loss / len(buffer_dl):.3f}")
                    running_loss = 0
                    short_running_loss = 0

                    if self.require_mean_calculation:
                        with torch.no_grad():
                            self.generate_embeddings()
            else:
                logger.info("TLM is set to none, not training")

            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        if self.classification_mode == "ncm":
            logger.info("Using NCM classificaion")
            return self.ncm_classify(batch)
        elif self.classification_mode == "max":
            return self.max_classify(batch)
        else:
            raise ValueError("Invalid classification type")

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
        print("0")

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        print("1")

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            
        print("2")

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
            
        print("3")

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
        
        print("4")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        print("B", anchor_dot_contrast)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        print("C", logits)

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

        
        print("D", log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        print("E", mean_log_prob_pos)

        # loss
        loss = -1 * mean_log_prob_pos
        print("F", loss)
        loss = loss.view(anchor_count, batch_size).mean()
        print("G", loss)

        return loss

class TrainableHead(nn.Module):
    def __init__(self):
        super().__init__()

        
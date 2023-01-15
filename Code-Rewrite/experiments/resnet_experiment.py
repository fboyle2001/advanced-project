"""
The purpose of this experiment is to test train an additional head using the features
from the ViT using SCL loss to segregate the classes and then classify using NCM classification
"""

from typing import Dict, Union, List
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

class ResNetExperiment(BaseCLAlgorithm):
    """
    ResNet Experiment
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
            name="Supervised Contrastive Replay",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = 256
        self.batch_size = 32
        self.max_memory_samples = 5000
        self.lr = 0.1

        self.buffer = buffers.BalancedReplayBuffer(self.max_memory_samples)
        
        # Use the reduced model from the official repo
        self.model = scr_resnet.SupConResNet().to(self.device)  #self.model.to(self.device)
        self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr)

        self.tau = 0.07
        self.loss_criterion = nn.CrossEntropyLoss()

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
        return "secondary_experiments/resnet"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "tau (temperature)": self.tau,
            "lr": self.lr
        }

        return info

    def scl_loss(self, batch, targets) -> torch.Tensor:
        # Custom implementation of SCL, slower than the official
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

            # Greedily sample at random
            for batch_no, data in enumerate(task_dataloader, 0):
                raw_inp, raw_labels = data

                for data, target in zip(raw_inp, raw_labels):
                    self.buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())
        
        logger.info("Populated buffer")

        buffer_ds = self.buffer.to_torch_dataset(transform=self.dataset.testing_transform,)
        buffer_dl = DataLoader(buffer_ds, batch_size=32, drop_last=True)

        self.require_mean_calculation = True
        
        # Train using the samples only
        for epoch in range(1, self.epochs_per_task + 1):
            self.model.train()
            self.require_mean_calculation = True

            logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
            running_loss = 0
            short_running_loss = 0

            for batch_no, batch in enumerate(buffer_dl):
                inp, labels = batch
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                
                augmented = self.augment(inp.detach().clone())

                inp = torch.cat([inp, augmented], dim=0)
                labels = torch.cat([labels, labels], dim=0)

                clustered_features = self.model(inp)
                # print("C", clustered_features.shape)
                # f1, f2 = torch.split(clustered_features, [self.batch_size, self.batch_size], dim=0)
                # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                # print("D", features.shape)
                loss = self.loss_criterion(clustered_features, labels) # need clustered_features.unsqueeze(1) for SupConLoss

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # print(loss)

                running_loss += loss.item()
                short_running_loss += loss.item()

                if batch_no % 40 == 0 and batch_no != 0:
                    logger.info(f"{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                    short_running_loss = 0

            epoch_offset = 0

            avg_running_loss = running_loss / (len(buffer_dl) - 1)
            logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
            self.writer.add_scalar(f"Loss/Task_{epoch + 1}_Total_avg", avg_running_loss, epoch)
            self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

            running_loss = 0

            if epoch % 10 == 0 and epoch != 0:
                self.run_base_task_metrics(epoch)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        # Nearest Class Mean Classification
        self.model.eval()

        # Need to update the means
        if self.require_mean_calculation:
            logger.info("Generating fresh mean embedding vectors")
            self.mean_embeddings = dict()

            for target in self.buffer.known_classes:
                total_embed = []
                count = 0

                # For each sample in the memory, add the normalised feature embedding to the total for the class
                for hash_key in self.buffer.class_hash_pointers[target]:
                    count += 1
                    sample = self.dataset.testing_transform(Image.fromarray(self.buffer.hash_map[hash_key].astype(np.uint8))).unsqueeze(0).to(self.device) # type: ignore
                    encoded = self.model.features(sample).detach().clone().squeeze(0)
                    encoded = encoded / encoded.norm()
                    total_embed.append(encoded)
                
                # Average and normalise
                assert total_embed is not None
                mean_embed = torch.stack(total_embed).mean(0).squeeze()
                mean_embed = mean_embed / mean_embed.norm()
                self.mean_embeddings[target] = mean_embed
            
            self.require_mean_calculation = False
            logger.debug(f"Generated mean embedding vectors for: {self.mean_embeddings.keys()}")

        predictions = []
        batch = batch.to(self.device)

        # Classify the samples
        for sample in batch:
            encoded_sample = self.model.features(sample.unsqueeze(0)).squeeze(0)
            encoded_sample = encoded_sample / encoded_sample.norm()
            closest_target = None
            closest_value = None

            # Find the nearest class mean
            for target in self.mean_embeddings.keys():
                norm = (self.mean_embeddings[target] - encoded_sample).square().sum().sqrt()

                if (closest_target is None and closest_value is None) or norm < closest_value:
                    closest_target = target
                    closest_value = norm

            predictions.append(torch.tensor([closest_target]))

        return torch.cat(predictions, dim=0)

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
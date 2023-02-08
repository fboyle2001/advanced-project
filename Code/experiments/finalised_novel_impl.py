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

class NovelImplementation(BaseCLAlgorithm):
    valid_sampling_types: List[str] = ["batch_normalised", "relative_distances", "random"]
    valid_classification_types: List[str] = ["ncm", "maximisation"]

    """
    Uses a pre-trained vision transformer with a novel sample uncertainty quanitifcation approach.
    Classifies samples using NCM classification.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        max_memory_samples: int
    ):
        super().__init__(
            name="Novel Implementation",
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

        self.max_memory_size = max_memory_samples
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
        self.model = TestNet(len(self.dataset.classes)).to(self.device) # SecondaryTestNet(100).to(self.device) # 

        self.optimiser = optim.SGD(self.model.parameters(), lr=0.1)
        
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
        self.loss_criterion = SupConLoss()

        self.uncertainty_type = "batch_normalised"
        self.classification_type = "ncm"

        self.min_lr = 0.0005
        self.max_lr = 0.05

        assert self.uncertainty_type in self.valid_sampling_types, "Invalid sampling type"
        assert self.classification_type in self.valid_classification_types, "Invalid classification type"

    @staticmethod
    def get_algorithm_folder() -> str:
        return "novel"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "uncertainty_type": self.uncertainty_type,
            "max_memory_samples": self.max_memory_size,
            "classification_type": self.classification_type,
            "end_train_only": False
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

            # Either sample randomly or use the uncertainty approach
            if self.uncertainty_type == "random":
                # Greedily sample at random
                for batch_no, data in enumerate(task_dataloader, 0):
                    raw_inp, raw_labels = data

                    for data, target in zip(raw_inp, raw_labels):
                        self.buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())
            else:
                self._uncertainity_memory_update(task_dataloader)
            
            logger.info("Populated buffer")

            buffer_ds = self.buffer.to_torch_dataset(transform=self.dataset.testing_transform)
            buffer_dl = DataLoader(buffer_ds, batch_size=32)

            self.require_mean_calculation = True
            lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=self.min_lr)

            offset = self.epochs_per_task * task_no

            # Train using the samples only
            for epoch in range(1, self.epochs_per_task + 1):
                self.require_mean_calculation = True

                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0
                short_running_loss = 0

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

                for batch_no, batch in enumerate(buffer_dl):
                    inp, labels = batch
                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # tags = {x: [0 if x not in self.buffer.known_classes else len(self.buffer.class_hash_pointers[x]), 0] for x in range(10)}

                    # for t in buffer_targets:
                    #     tags[t.item()][1] += 1 # type: ignore

                    # logger.debug(tags)
                    
                    augmented = self.augment(inp.detach().clone())

                    inp = torch.cat([inp, augmented], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                    vit_features = self.f_r(self.f_e(inp))[:, 0, :]
                    clustered_features = self.model.forward(vit_features)
                    # loss = self.scl_loss(clustered_features, labels)
                    loss = self.loss_criterion(clustered_features.unsqueeze(1), labels) # need clustered_features.unsqueeze(1) for SupConLoss

                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()

                    # print(loss)

                    running_loss += loss.item()
                    short_running_loss += loss.item()

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.info(f"{task_no+1}:{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                        short_running_loss = 0
                    
                    if epoch % 8 == 0 and batch_no == 120:
                        first = self.model.linear_one(vit_features)
                        calc_first = torch.sum(torch.eq(vit_features, first)).item() / vit_features.numel()
                        relud = self.model.relu1(first)
                        calc_relud = torch.sum(torch.eq(vit_features, relud)).item() / vit_features.numel()
                        logger.warning(f"Sim: First: {calc_relud}% Relud: {calc_first}%")

                epoch_offset = offset + epoch

                avg_running_loss = running_loss / (len(buffer_dl) - 1)
                logger.info(f"{task_no+1}:{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{epoch_offset}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                running_loss = 0

                if epoch % 10 == 0 and epoch != 0:
                    self.run_base_task_metrics(epoch_offset)
        
        self.run_base_task_metrics(999)
        logger.info("Training complete")

    ## CLASSIFICATION
    
    # Maximisation of logits
    def maximisation_classify(self, batch: torch.Tensor) -> torch.Tensor:
        vit_sample_f = self.f_r(self.f_e(batch))[:, 0, :]
        output = self.model(vit_sample_f)
        _, predicted = torch.max(output.data, 1)
        return predicted
    
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
        if self.classification_type == "ncm":
            return self.ncm_classify(batch)
        elif self.classification_type == "maximisation":
            return self.maximisation_classify(batch)
        
        logger.error("Invalid classification type")
        return torch.zeros(1)

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

# Credit: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
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
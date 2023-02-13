from typing import Dict, Union, List, Optional, Tuple
from loguru import logger

from .algorithm_base import BaseCLAlgorithm
from . import buffers
from models.scr import scr_resnet

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomSolarize, RandomInvert, ColorJitter, RandomGrayscale

import numpy as np
from PIL import Image

class TrialAdaptedSCR(BaseCLAlgorithm):
    """
    Uses Supervised Contrastive Loss to segregate the features of the samples in
    the feature space. Classifies samples using nearest class mean classification.

    Reference: Mai, Zheda, et al. 2021 "Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning."
    """
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
        temperature: float, # 0.07
        lr: float # 0.1
    ):
        super().__init__(
            name="Trial-Adapted SCR",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.max_memory_samples = max_memory_samples
        self.lr = lr

        self.buffer = buffers.BalancedReplayBuffer(max_memory_samples)
        
        # Use the reduced model from the official repo
        self.model = scr_resnet.SupConResNet().to(self.device)  #self.model.to(self.device)
        self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr)

        self.memory_batch_size = memory_batch_size
        self.tau = temperature

        # Taken directly from https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/agents/scr.py
        self.augment = nn.Sequential(
            RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        ).to(self.device)
        
        self.D = 160

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

        self.memory: Dict[int, List[np.ndarray]] = {}
        self.require_mean_calculation = True
        self.mean_embeddings = torch.zeros(len(self.dataset.classes), self.D).to(self.device)
        self.uncertainty_type = "relative_distances"
        self.classification_type = "ncm"

    @staticmethod
    def get_algorithm_folder() -> str:
        return "trial_adapted_scr"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "memory_batch_size": self.memory_batch_size,
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
    
    ## UNCERTAINTY SAMPLING

    def _closest_mean_embeddings(self, batch: torch.Tensor, k: int):
        batch = batch.to(self.device)
        B = batch.shape[0]
        C = len(self.dataset.classes)
        # Get normalised features
        samples = self.model.features(batch)
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
        memory_per_class = self.max_memory_samples // seen_class_count

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
            self.model.train()

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

            for epoch in range(1, self.epochs_per_task + 1):
                self.require_mean_calculation = True

                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0
                short_running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    raw_inp, raw_labels = data

                    # If we have enough saved samples we can start using the buffer
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
                        
                        # Augment
                        augmented = self.augment(inp.detach().clone())

                        inp = torch.cat([inp, augmented], dim=0)
                        labels = torch.cat([labels, labels], dim=0)

                        # Get the features and compute the loss
                        features = self.model.forward(inp)
                        loss = self.scl_loss(features, labels) 

                        self.optimiser.zero_grad()
                        loss.backward()
                        self.optimiser.step()

                        running_loss += loss.item()
                        short_running_loss += loss.item()

                    # # Update the memory
                    # for data, target in zip(raw_inp, raw_labels):
                    #     self.buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.info(f"{task_no}:{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                        short_running_loss = 0
                
                logger.debug("Current sample stats:")

                for class_name in self.buffer.known_classes:
                    logger.debug(f"{class_name} has {len(self.buffer.class_hash_pointers[class_name])} samples")

                epoch_offset = self.epochs_per_task * task_no

                # Log data
                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                running_loss = 0

            # Evaluate the model
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    ## CLASSIFICATION
    
    # Maximisation of logits
    def maximisation_classify(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.model(batch)
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
                encoded = self.model.forward(buffer_batch).detach().clone()
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
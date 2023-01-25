from typing import Dict, Union, Callable, List, Tuple, Optional
from loguru import logger

from .algorithm_base import BaseCLAlgorithm
from . import buffers

import torch
import datasets
import torch.utils.tensorboard
import torchvision
import numpy as np
import random
from algorithms import utils

import torch.optim as optim
from datasets import CustomImageDataset
from torch.utils.data import DataLoader

from PIL import Image
import math
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomSolarize, RandomInvert, ColorJitter, RandomGrayscale

import copy

class RainbowOnlineNCM(BaseCLAlgorithm):
    """
    """
    # random could have repeated samples
    # j = index, D = total images in class, k = sample size
    sample_idx_functions: Dict[str, Callable[[int, int, int], int]] = {
        "diverse": lambda j, D, k: (j * D) // k,
        "central": lambda j, D, k: j,
        "edge": lambda j, D, k: D - j - 1,
        "random": lambda j, D, k: 0,
        "proportional": lambda j, D, k: 0,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        batch_size: int,
        max_memory_samples: int,
        epochs_per_task: int,
        gradient_clip: Union[None, int],
        max_lr: float,
        min_lr: float,
        cutmix_probability: float
    ):
        super().__init__(
            name="Rainbow Adjusted Online NCM",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.batch_size = batch_size
        self.max_memory_samples = max_memory_samples
        self.epochs_per_task = epochs_per_task
        self.gradient_clip = gradient_clip
        self.max_lr = max_lr
        self.min_lr = min_lr
        
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
        
        self.memory: Dict[int, List[np.ndarray]] = {}
        self.cutmix_probability = cutmix_probability
        self.uncertainty_type = "relative_distances"
        self.D = 4096
        self.require_mean_calculation = True
        self.mean_embeddings = torch.zeros(len(self.dataset.classes), self.D).to(self.device)

        self.augment = torch.nn.Sequential(
            RandomResizedCrop(size=(32, 32), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        ).to(self.device)

        self.raw_model = copy.deepcopy(self.model).to("cpu")

    @staticmethod
    def get_algorithm_folder() -> str:
        return "rainbow_ncm"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "epochs_per_task": f"{self.epochs_per_task} (On Exemplar Buffer Only)",
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "cutmix_probability": self.cutmix_probability,
            "uncertainty_type": self.uncertainty_type
        }

        return info

    ## UNCERTAINTY SAMPLING

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.features(batch) # type: ignore

    def _closest_mean_embeddings(self, batch: torch.Tensor, k: int):
        batch = batch.to(self.device)
        B = batch.shape[0]
        C = len(self.dataset.classes)
        # Get normalised features
        samples = self.get_features(batch)
        samples = samples.reshape(B, -1)
        samples = samples / torch.linalg.norm(samples, dim=0)
        # Extend the vectors
        tiled = samples.tile(dims=(1, C))
        tiled = tiled.reshape(B, C, self.D)
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

            logger.info("Populating replay buffer")
            
            # Either sample randomly or use the uncertainty approach
            if self.uncertainty_type == "random":
                # Greedily sample at random
                for batch_no, data in enumerate(task_dataloader, 0):
                    raw_inp, raw_labels = data

                    for data, target in zip(raw_inp, raw_labels):
                        self.buffer.add_sample(data.detach().cpu().numpy(), target.detach().cpu().item())
            else:
                self._uncertainity_memory_update(task_dataloader)
        
            logger.info("Replay buffer populated")
            logger.info(f"Buffer keys: {self.buffer.known_classes}")

            # Check the replay buffer is balanced
            for class_name in self.buffer.known_classes:
                logger.info(f"{class_name} has {len(self.buffer.class_hash_pointers[class_name])} samples")

            self.old_model = self.model
            self.model = copy.deepcopy(self.raw_model).to(self.device)
            self.optimiser = torch.optim.SGD(self.model.parameters(), lr=0.1)
            logger.info("Copied new raw model")

            # Convert the raw images to a PyTorch dataset with a dataloader
            buffer_dataset = self.buffer.to_torch_dataset(transform=self.dataset.training_transform)
            buffer_dataloader = DataLoader(buffer_dataset, batch_size=self.batch_size, shuffle=True)

            offset = self.epochs_per_task * task_no

            logger.info("Training model for inference from buffer")

            lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=self.min_lr)
            # unique_imgs = set()

            for epoch in range(1, self.epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                # logger.info(f"Unique images: {len(unique_imgs)}")
                running_loss = 0

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

                for batch_no, data in enumerate(buffer_dataloader, 0):
                    inp, labels = data

                    # for ix in inp:
                    #     unique_imgs.add(hash(pickle.dumps(ix.detach().cpu())))

                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # Apply cutmix
                    apply_cutmix = random.uniform(0, 1) < self.cutmix_probability
                    lam, labels_a, labels_b = None, None, None

                    # Cannot merge the two if statements because inp will change causing issues in autograd
                    if apply_cutmix: 
                        inp, labels_a, labels_b, lam = utils.cutmix_data(x=inp, y=labels, alpha=1.0)

                    self.optimiser.zero_grad()
                    predictions = self.model(inp)
                    
                    if apply_cutmix: 
                        assert lam is not None and labels_a is not None and labels_b is not None
                        loss = lam * self.loss_criterion(predictions, labels_a) + (1 - lam) * self.loss_criterion(predictions, labels_b)
                    else:
                        loss = self.loss_criterion(predictions, labels)
                    
                    loss.backward()

                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()

                    running_loss += loss.item()

                # Metrics
                avg_running_loss = running_loss / (len(buffer_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch)
                running_loss = 0

                if epoch > 0 and epoch % 10 == 0:
                    self.model.eval()
                    self.run_base_task_metrics(task_no=offset+epoch)
                    self.model.train()
        
        self.run_base_task_metrics(task_no=6 * self.epochs_per_task + 1)
        logger.info("Training completed")

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
                encoded = self.get_features(buffer_batch).detach().clone()
                encoded = encoded.reshape(len(buffer_batch), -1)
                encoded = encoded / torch.linalg.norm(encoded, dim=0)

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
    
    # def classify(self, batch: torch.Tensor) -> torch.Tensor:
    #     return self.ncm_classify(batch)

"""
Implementation from https://github.com/clovaai/rainbow-memory/blob/cccc8b0a08cbcb0245c361c6fd0988e4efd2fb36/utils/augment.py#L266
"""
class Cutout:
    def __init__(self, size=16) -> None:
        self.size = size

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        # assert img_height == img_width

        # Sample center where cutout mask will be applied
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[
            upper_coord[0] : lower_coord[0], upper_coord[1] : lower_coord[1], :
        ] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        _, upper_coord, lower_coord = self._create_cutout_mask(
            img_height, img_width, num_channels, self.size
        )
        pixels = pil_img.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
        return pil_img
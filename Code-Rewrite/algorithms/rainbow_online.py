from typing import Dict, Union, Callable
from loguru import logger

from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
import torchvision
import numpy as np
import math
import random
from algorithms import utils

import torch.optim as optim
from datasets import CustomImageDataset
from torch.utils.data import DataLoader

from PIL import Image

class RainbowOnline(BaseCLAlgorithm):
    # random could have repeated samples
    sample_idx_functions: Dict[str, Callable[[int, int, int], int]] = {
        "diverse": lambda j, D, k: (j * D) // k,
        "central": lambda j, D, k: j,
        "edge": lambda j, D, k: D - k - 1,
        "random": lambda j, D, k: 0
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
        cutmix_probability: float,
        sampling_strategy: str
    ):
        super().__init__(
            name="Rainbow Online",
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

        """
        Four potential sampling techniques:
        * diverse: **DEFAULT FOR RAINBOW** Sort by uncertainty and then select at evenly spaced interval from the list 
        * central: Sort by uncertainty and then select the first n from the list 
        * edge: Sort by uncertainty and then select the last n from the list 
        * random: This is effectively GDumb, just pick any samples
        """
        self.sampling_strategy = sampling_strategy

        if self.sampling_strategy not in self.sample_idx_functions.keys():
            logger.critical(f"Invalid sampling technique for Rainbow Online. Must select from {self.sample_idx_functions.keys()}")
            assert self.sampling_strategy in self.sample_idx_functions.keys(), f"Must select a valid sampling technique from: {self.sample_idx_functions.keys()}"
        
        self.calculate_sample_idx = self.sample_idx_functions[self.sampling_strategy]
        self.cutmix_probability = cutmix_probability

    @staticmethod
    def get_algorithm_folder() -> str:
        return "rainbow_online"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "epochs_per_task": f"{self.epochs_per_task} (On Exemplar Buffer Only)",
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "cutmix_probability": self.cutmix_probability,
            "sampling_technique": self.sampling_strategy
        }

        return info

    def calculate_sample_uncertainty(self, sample: np.ndarray):
        assert not self.model.training, "Need to be in eval mode for uncertainty calculation"
        """
        'Computes the uncertainty of a sample by measuring the variance of model outputs of
        perturbed samples by various transformation methods for data augmentation include:
        colour jitter, shear, and cutout ... approximate the uncertainty by Monte-Carlo (MC)
        method of the distribution p(y = c | x) when given the prior of the perturbed sample ~x
        as p(~x | x). We define the perturbation prior p(~x | x) as a uniform mixture of the 
        various perturbations'
        """

        # These transformations are the same as the original implementation
        # https://github.com/clovaai/rainbow-memory/blob/cccc8b0a08cbcb0245c361c6fd0988e4efd2fb36/methods/finetune.py#L563
        # Want certainty so p=1 for all where applicable
        possible_transforms = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.RandomSolarize(threshold=32, p=1),
            torchvision.transforms.RandomSolarize(threshold=64, p=1),
            torchvision.transforms.RandomSolarize(threshold=128, p=1),
            torchvision.transforms.RandomInvert(p=1),
            Cutout(size=8),
            Cutout(size=16),
            Cutout(size=24),
            Cutout(size=32)
        ]

        sample_img = Image.fromarray(sample.astype(np.uint8))
        transform_uncertainties = []

        # For each possible transform, compute the uncertainty
        for transform in possible_transforms:
            testable_transforms = torchvision.transforms.Compose([transform] + self.dataset.testing_transform.transforms)

            with torch.no_grad():
                transformed_sample: torch.Tensor = testable_transforms(sample_img).to(self.device) # type: ignore will be tensor
                assert type(transformed_sample) is torch.Tensor

                logits: torch.Tensor = self.model(transformed_sample.unsqueeze(0)) # Convert to a 4D tensor
                logits = logits.detach().cpu().squeeze(0) # Back to a 3D tensor
                
                # Compute 1 - logit for each logit, closer to 0 the better
                transform_uncertainties.append(torch.ones_like(logits) - logits)
        
        # For each transform, determine which class the model thoughout they belonged to
        # This is eq 4.1
        predicted_class_counter = [0 for _ in range(len(self.dataset.classes))]

        for uncertainty in transform_uncertainties:
            predicted_class = torch.argmin(uncertainty)
            predicted_class_counter[predicted_class] += 1

        calculated_uncertainty = 1 - (1 / len(possible_transforms)) * max(predicted_class_counter) # eq 4.2
        return calculated_uncertainty

    def perform_diversity_aware_memory_update(self, exemplars, exemplar_targets, task_samples, task_targets):
        """
        K denotes memory size
        N denotes the number of classes seen
        S denotes stream data at current task
        M_t [shortened to M] denotes the stored exemplars at the start of the current task
        Output is M_(t+1) the stored exemplars at the end of the current task (start of next task)

        1. M_(t+1) initialised as empty
        2. k = floor(K / N) i.e. number of samples per class for balanced exemplars
        3. For class c = 1...N
            a. Get all samples from M and S in class c, denote D
            b. Sort D by u(x) (the uncertainty) for each x in D as computed by eq. (4)
            c. For index j = 1...k
                i. Compute i = j * |D| / k
                ii. Put D[i] into M_(t+1)
        4. Return M_(t+1)
        """

        exemplar_iter = zip(exemplars, exemplar_targets) # M_t
        task_iter = zip(task_samples, task_targets) # S

        logger.info("Segmenting classes")

        # Segment the data into the classes
        class_segmented_samples = {}

        for data, target in exemplar_iter:
            if type(target) is torch.Tensor:
                target = target.detach().cpu().item()

            if target not in class_segmented_samples.keys():
                class_segmented_samples[target] = []
                logger.debug(f"Keys {class_segmented_samples.keys()}")
            
            assert type(data) is not torch.Tensor
            class_segmented_samples[target].append(data)

        for data, target in task_iter:
            if type(target) is torch.Tensor:
                target = target.detach().cpu().item()

            if target not in class_segmented_samples.keys():
                class_segmented_samples[target] = []

            assert type(data) is not torch.Tensor
            class_segmented_samples[target].append(data)

        logger.debug(f"Seen classes: {len(class_segmented_samples.keys())}")
        memory_per_class = self.max_memory_samples // len(class_segmented_samples.keys()) # this is k

        logger.info(f"Sorting class samples by uncertainty and generating next buffer: memory size per class {memory_per_class}")

        # For each class, sort by the uncertainty and produce the new memory buffer
        sorted_class_segmented_samples = {}
        next_memory_samples = []
        next_memory_targets = []

        if self.sampling_strategy == "random":
            for class_name in class_segmented_samples.keys():
                num_class_samples = len(class_segmented_samples[class_name])
                random_indices = random.sample(range(num_class_samples), k=memory_per_class)
                logger.debug(f"Random indices for class {class_name}: {random_indices}")
                count = 0

                for j in random_indices:
                    next_memory_samples.append(class_segmented_samples[class_name][j])
                    next_memory_targets.append(class_name)
                    count += 1
                
                logger.info(f"There are {count} samples from class {class_name}")
        else:
            for class_name in class_segmented_samples.keys():
                logger.debug(f"Class {class_name} under sampling strategy {self.sampling_strategy}")
                sorted_class_segmented_samples[class_name] = sorted(class_segmented_samples[class_name], key=self.calculate_sample_uncertainty)
                num_class_samples = len(sorted_class_segmented_samples[class_name])
                count = 0

                for j in range(memory_per_class):
                    sample_idx = self.calculate_sample_idx(j, num_class_samples, memory_per_class) # (num_class_samples * j) // memory_per_class
                    # logger.debug(f"j: {j}, k: {memory_per_class}, |D|: {num_class_samples}, ~i: {j * memory_per_class // num_class_samples} ~~i: {(num_class_samples * j) // memory_per_class}")
                    # logger.debug(f"idx: {sample_idx} with score {self.calculate_sample_uncertainty(sorted_class_segmented_samples[class_name][sample_idx])}")
                    next_memory_samples.append(sorted_class_segmented_samples[class_name][sample_idx])
                    next_memory_targets.append(class_name)
                    count += 1

                logger.info(f"There are {count} samples from class {class_name}")

        logger.info("Generated next exemplar buffer")

        return next_memory_samples, next_memory_targets

    def train(self) -> None:
        super().train()
        self.model.train()
        
        exemplar_data, exemplar_targets = [], []

        for task_no in range(len(self.dataset.task_datasets)):
            logger.info(f"Starting Task {task_no + 1} / {len(self.dataset.task_datasets)}")
            current_task_dataset = self.dataset.task_datasets[task_no]

            # Update the memory
            logger.info("Updating memory...")

            self.model.eval()
            logger.debug(f"T_b: {type(exemplar_data[0]) if len(exemplar_data) != 0 else None}")
            exemplar_data, exemplar_targets = self.perform_diversity_aware_memory_update(
                exemplar_data, exemplar_targets, current_task_dataset.data, current_task_dataset.targets # type: ignore
            )
            logger.debug(f"T_a: {type(exemplar_data[0])}")
            self.model.train()

            # For online version, generate the buffer at the start as we do a single pass on the data 

            logger.info("Using memory buffer exclusively")
            exemplar_dataset = CustomImageDataset(exemplar_data, exemplar_targets, transform=self.dataset.training_transform)
            exemplar_dataloader = DataLoader(exemplar_dataset, batch_size=self.batch_size, shuffle=True)

            lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=0.0005)

            # Train on this task
            for epoch in range(1, self.epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0

                # Apply learning rate warmup
                if epoch == 0:
                    for param_group in self.optimiser.param_groups:
                        param_group['lr'] = self.max_lr * 0.1

                    self.writer.add_scalar("LR/Current_LR", self.max_lr * 0.1, task_no * self.epochs_per_task + epoch)
                elif epoch == 1:
                    for param_group in self.optimiser.param_groups:
                        param_group['lr'] = self.max_lr

                    self.writer.add_scalar("LR/Current_LR", self.max_lr, task_no * self.epochs_per_task + epoch)
                else:
                    lr_warmer.step()
                    self.writer.add_scalar("LR/Current_LR", lr_warmer.get_last_lr()[-1], task_no * self.epochs_per_task + epoch)

                for batch_no, data in enumerate(exemplar_dataloader):
                    # Only draw from the exemplar buffer
                    inp, labels = data

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
                avg_running_loss = running_loss / (len(exemplar_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, task_no * self.epochs_per_task + epoch)
                running_loss = 0

                if (epoch > 0 and epoch % 10 == 0) or epoch == self.epochs_per_task:
                    self.model.eval()
                    self.run_base_task_metrics(task_no=task_no * self.epochs_per_task + epoch)
                    self.model.train()

        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)

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
from typing import Dict, Union, Tuple
from loguru import logger

from .algorithm_base import BaseCLAlgorithm

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
import matplotlib.pyplot as plt
import scipy.stats as stats

class RainbowOnlineExperimental(BaseCLAlgorithm):
    # random could have repeated samples
    sample_distribution_parameters: Dict[str, Tuple[float, float]] = {
        "endpoint_peak": (0.5, 0.5),
        "midpoint_peak": (2, 2),
        "edge_skewed_1": (5, 3)
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
        sampling_strategy: str,
        all_occurrences: bool
    ):
        super().__init__(
            name="Rainbow Online Experimental",
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
        Potential sampling techniques:
        *
        *
        """
        self.sampling_strategy = sampling_strategy
        self.all_occurrences = all_occurrences

        if self.sampling_strategy not in self.sample_distribution_parameters:
            logger.critical(f"Invalid strategy for Rainbow Online Experimental. Must select from {self.sample_distribution_parameters}")
            assert self.sampling_strategy in self.sample_distribution_parameters, f"Must select a valid strategy from: {self.sample_distribution_parameters}"
        
        self.beta_a, self.beta_b = self.sample_distribution_parameters[self.sampling_strategy]
        
        self.cutmix_probability = cutmix_probability

    @staticmethod
    def get_algorithm_folder() -> str:
        return "rainbow_online_experimental"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "batch_size": self.batch_size,
            "max_memory_samples": self.max_memory_samples,
            "epochs_per_task": f"{self.epochs_per_task} (On Exemplar Buffer Only)",
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "cutmix_probability": self.cutmix_probability,
            "sampling_technique": self.sampling_strategy,
            "all_occurrences": self.all_occurrences,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b
        }

        return info

    def calculate_sample_uncertainty(self, sample: np.ndarray, scale: bool = True):
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

        if scale:
            calculated_uncertainty = 1 - (1 / len(possible_transforms)) * max(predicted_class_counter) # eq 4.2
        else:
            calculated_uncertainty = len(possible_transforms) - max(predicted_class_counter)

        return calculated_uncertainty

    def perform_diversity_aware_memory_update(self, exemplars, exemplar_targets, task_samples, task_targets, task_no):
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

        unseen_classes = []

        for data, target in task_iter:
            if type(target) is torch.Tensor:
                target = target.detach().cpu().item()

            if target not in class_segmented_samples.keys():
                class_segmented_samples[target] = []
                unseen_classes.append(target)

            assert type(data) is not torch.Tensor
            class_segmented_samples[target].append(data)

        logger.debug(f"Seen classes: {len(class_segmented_samples.keys())}")
        memory_per_class = self.max_memory_samples // len(class_segmented_samples.keys()) # this is k

        logger.info(f"Sorting class samples by uncertainty and generating next buffer: memory size per class {memory_per_class}")

        # For each class, sort by the uncertainty and produce the new memory buffer
        sorted_class_segmented_samples = {}
        next_memory_samples = []
        next_memory_targets = []

        """
        1. Sort by uncertainty
        2. If this is the first time we have encountered the class:
            a. 
        3. Otherwise number of samples at each uncertainty level
            a. Proportionally select samples based on the size of the bins
        """

        for class_name in class_segmented_samples.keys():
            logger.debug(f"Class {class_name} under sampling strategy {self.sampling_strategy}")
            uncertainty_bins = {}

            def proportional_sort_key(sample):
                unscaled_uncertainty = self.calculate_sample_uncertainty(sample, scale=False)

                if unscaled_uncertainty not in uncertainty_bins.keys():
                    uncertainty_bins[unscaled_uncertainty] = 0
                
                uncertainty_bins[unscaled_uncertainty] += 1
                return unscaled_uncertainty

            sorted_class_segmented_samples[class_name] = sorted(class_segmented_samples[class_name], key=proportional_sort_key)
            num_class_samples = len(sorted_class_segmented_samples[class_name])
            start_idx = 0
            count = 0

            temp_samples_for_logging = []
            sampled_indexes = []

            if class_name in unseen_classes or self.all_occurrences:
                while len(sampled_indexes) < memory_per_class:
                    sample_idx = round(num_class_samples * np.random.beta(a=self.beta_a, b=self.beta_b))

                    if sample_idx < 0:
                        logger.warning(f"{sample_idx} was less than 0")
                        sample_idx = 0
                    elif sample_idx >= num_class_samples:
                        logger.warning(f"{sample_idx} was greater than max")
                        sample_idx = num_class_samples - 1
                    
                    if sample_idx in sampled_indexes:
                        logger.debug(f"Already seen idx {sample_idx}")
                        continue
                    
                    sample = sorted_class_segmented_samples[class_name][sample_idx]

                    next_memory_samples.append(sample)
                    next_memory_targets.append(class_name)

                    temp_samples_for_logging.append(torchvision.transforms.ToTensor()(Image.fromarray(sample)))
                    count += 1
                    
                    sampled_indexes.append(sample_idx)

                # Just easier for logging
                sampled_indexes = sorted(sampled_indexes)
            else:
                # Otherwise do it proportionally
                for uncertainty_bin in sorted(uncertainty_bins.keys()):
                    bin_size = uncertainty_bins[uncertainty_bin]
                    proportional_sample_size = round(memory_per_class * bin_size / num_class_samples)
                    
                    logger.debug(f"Bin: {uncertainty_bin}: {bin_size} will draw {proportional_sample_size} samples (proportion: {(100 * bin_size / num_class_samples):.2f}%)")

                    for j in range(proportional_sample_size):
                        sample_idx = start_idx + (j * bin_size // proportional_sample_size)

                        next_memory_samples.append(sorted_class_segmented_samples[class_name][sample_idx])
                        next_memory_targets.append(class_name)
                        temp_samples_for_logging.append(torchvision.transforms.ToTensor()(Image.fromarray(sorted_class_segmented_samples[class_name][sample_idx])))
                        sampled_indexes.append(sample_idx)
                        count += 1

                    start_idx += bin_size
            
            logging_img_tensors = torch.stack(temp_samples_for_logging)
            img_grid = torchvision.utils.make_grid(logging_img_tensors, nrow=math.ceil(math.sqrt(len(logging_img_tensors))))
            self.writer.add_image(f"Task {task_no} Exemplars ({self.sampling_strategy}: {len(logging_img_tensors)})/{self.dataset.classes[class_name]}", img_grid)

            logger.debug(f"Indexes: {sampled_indexes}")
            logger.info("Saved sample images to Tensorboard")
            logger.info(f"There are {count} samples from class {class_name}")

        logger.info("Generated next exemplar buffer")

        return next_memory_samples, next_memory_targets

    def train(self) -> None:
        super().train()
        exemplar_data, exemplar_targets = [], []

        # Start by plotting the beta distribution
        self.plot_beta()

        for task_no in range(len(self.dataset.task_datasets)):
            logger.info(f"Starting Task {task_no + 1} / {len(self.dataset.task_datasets)}")
            current_task_dataset = self.dataset.task_datasets[task_no]

            # Update the memory
            logger.info("Updating memory...")

            self.model.eval()
            logger.debug(f"T_b: {type(exemplar_data[0]) if len(exemplar_data) != 0 else None}")
            exemplar_data, exemplar_targets = self.perform_diversity_aware_memory_update(
                exemplar_data, exemplar_targets, current_task_dataset.data, current_task_dataset.targets, task_no # type: ignore
            )
            logger.debug(f"T_a: {type(exemplar_data[0])}")
            self.model.train()

            # For online version, generate the buffer at the start as we do a single pass on the data 

            logger.info("Using memory buffer exclusively")
            exemplar_dataset = CustomImageDataset(exemplar_data, exemplar_targets, transform=self.dataset.training_transform)
            exemplar_dataloader = DataLoader(exemplar_dataset, batch_size=self.batch_size, shuffle=True)

            lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=self.min_lr)

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

    def plot_beta(self) -> None:
        samples = 100
        xs = np.linspace(0, 1, samples)

        pdf_ys = stats.beta.pdf(xs, self.beta_a, self.beta_b)
        cdf_ys = stats.beta.cdf(xs, self.beta_a, self.beta_b)

        pdf_fig, pdf_ax = plt.subplots()

        pdf_ax.scatter(xs, pdf_ys)
        pdf_ax.set_title("Beta PDF")
        pdf_ax.set_xlabel("Density")
        pdf_ax.set_ylabel("PDF")

        self.writer.add_figure("Beta Sampling Distribution/PDF", pdf_fig)

        cdf_fig, cdf_ax = plt.subplots()

        cdf_ax.scatter(xs, cdf_ys)
        cdf_ax.set_title("Beta CDF")
        cdf_ax.set_xlabel("Cumulative Probability")
        cdf_ax.set_ylabel("CDF")

        self.writer.add_figure("Beta Sampling Distribution/CDF", cdf_fig)

        logger.info(f"Plotted beta distribution PDF and CDF on to Tensorboard with a={self.beta_a}, b={self.beta_b} over {samples} samples")

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
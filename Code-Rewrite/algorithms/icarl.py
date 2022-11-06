from typing import Dict, Union, List
from loguru import logger

from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard

class iCaRL(BaseCLAlgorithm):
    """
    Incremental Classifier and Representation Learning (iCaRL) (Rebuffi et al. 2017)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        max_epochs_per_task: int,
        batch_size: int
    ):
        super().__init__(
            name="iCaRL",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.max_epochs_per_task = max_epochs_per_task
        self.batch_size = batch_size

        self.class_exemplars = iCaRLExemplars(max_size=10)

    @staticmethod
    def get_algorithm_folder() -> str:
        return "icarl"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "max_epochs_per_task": self.max_epochs_per_task,
            "batch_size": self.batch_size
        }

        return info

    def train(self) -> None:
        super().train()

        seen_classes = set()

        """
        Directly from the paper:
        * Every time data for new classes is available iCaRL calls an update routine 
        * iCaRL makes use of a convolutional neural network
        * Interpret the network as a trainable feature extractor, followed by a single classification
          layer with as many sigmoid output nodes as classes observed so far
        * All feature vectors are L2-normalized, and the results of any operation on feature vectors, e.g. averages, are also re-normalized
        * denote the parameters of the network by Θ, split into a fixed number of parameters for the feature extraction part
          and a variable number of weight vectors. We denote the latter by w1, . . . , wt ∈ Rd
        * iCaRL uses the network only for representation learning, not for the actual classification step
        """

        # Update Representation
        # Reduce Exemplar Set
        # Construct Exemplar Set

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            for epoch in range(1, self.max_epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.max_epochs_per_task}")
                running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    inp, labels = data
                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # Split the samples into per-class sets

                    # Update model parameters
                    # Reduce exemplar sets to 

                    self.optimiser.zero_grad()
                    predictions = self.model(inp)
                    loss = self.loss_criterion(predictions, labels)
                    loss.backward()
                    self.optimiser.step()

                    running_loss += loss.item()

                epoch_offset = self.max_epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                running_loss = 0
        
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        # Algorithm 1

        # Compute a prototype vector for each class observed so far
        # This could be done once and saved as it is independent of the sample
        mean_of_exemplars = self.class_exemplars.generate_mean_of_exemplars()

        # Compute the feature vector of the image that should be classified (prototype vector)
        # Assign the class label with most similar prototype
        predicted = []

        for img, _ in batch:
            lowest_norm = None
            best_class = None

            for exemplar_class in mean_of_exemplars.keys():
                mean = mean_of_exemplars[exemplar_class]
                norm = torch.linalg.vector_norm(img - mean)

                if lowest_norm is None or norm < lowest_norm:
                    lowest_norm = norm
                    best_class = exemplar_class
            
            predicted.append(best_class)

        return torch.tensor(predicted)

class iCaRLExemplars:
    def __init__(self, max_size):
        self.max_size = max_size
        self.count = 0
        self.class_exemplars: Dict[int, List[torch.Tensor]] = {}

        self.mean_of_exemplars: Dict[int, torch.Tensor] = {}
        self.requires_mean_generation = True

    def generate_mean_of_exemplars(self) -> Dict[int, torch.Tensor]:
        # Part of Algorithm 1
        if not self.requires_mean_generation:
            return self.mean_of_exemplars
        
        new_means = {}

        for label in self.mean_of_exemplars.keys():
            new_means[label] = torch.mean(torch.tensor(self.mean_of_exemplars[label]))
        
        self.requires_mean_generation = False
        self.mean_of_exemplars = new_means

        return new_means

    def reduce_exemplar_sets(self) -> None:
        # Algorithm 5
        if self.count < self.max_size:
            return
        
        target_exemplars = self.max_size // len(self.class_exemplars.keys())
        new_exemplars = {}

        # NOTE: The lists are actually priortised so those nearer to index 0 are more important

        for clazz in self.class_exemplars.keys():
            new_exemplars[clazz] = self.class_exemplars[clazz][:target_exemplars]

        self.class_exemplars = new_exemplars

    def add_exemplar(self, feature_function, img, label):
        # Algorithm 4
        pass
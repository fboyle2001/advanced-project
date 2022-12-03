import abc
import json
from typing import Dict, Union, List, Any
from dataclasses import dataclass

from loguru import logger
import torch.utils.tensorboard

import torch
import datasets

import matplotlib.pyplot as plt
import matplotlib.figure

class BaseCLAlgorithm(abc.ABC):
    """
    Complete training and classification package for a Continual Learning algorithm.
    """
    def __init__(
        self,
        name: str,
        model_instance: torch.nn.Module,
        dataset_instance: datasets.BaseCLDataset,
        optimiser_instance: torch.optim.Optimizer,
        loss_criterion_instance: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter
    ): 
        """
        Represents a complete training and classification package for a Continual Learning algorithm.

        Args:
            name (str): Name of the algorithm
            model_instance (torch.nn.Module): Instance of the model to train
            dataset_instance (datasets.BaseCLDataset): Instance of the dataset to use for training
            optimiser_instance (torch.optim.Optimizer): Instance of the optimiser to use with parameters set
            loss_criterion_instance (torch.nn.modules.loss._Loss): Instance of the loss criterion instance to use
            writer (torch.utils.tensorboard.writer.SummaryWriter): TensorBoard writer for this instance
        """
        self.name = name

        self.model = model_instance
        self.device: torch.device = next(self.model.parameters()).device
        
        if "cuda" not in self.device.type:
            logger.warning("Not running on CUDA device")

        self.dataset = dataset_instance
        self.optimiser = optimiser_instance
        self.loss_criterion = loss_criterion_instance

        self.writer = writer
        self.directory = writer.log_dir

        self.task_metrics: Dict[int, TaskStats] = {}

    @staticmethod
    @abc.abstractmethod
    def get_algorithm_folder() -> str:
        """
        Returns the directory name for this algorithm

        Returns:
            str: Directory name for this algorithm
        """
        pass

    @abc.abstractmethod
    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        """
        Return non-standard parameters that are set for this method.
        If there are none then this should return {}

        Returns:
            Dict[str, Union[str, int, float]]: A dictionary with unique information about the algorithm
        """
        pass

    def get_information(self) -> Dict[str, Union[str, int]]:
        """
        Generate a dictionary containing key information about the algorithm. 
        Used to generate the string representation of this object.

        Returns:
            Dict[str, Union[str, int]]: A dictionary with key information about the algorithm
        """
        info = {
            "name": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "model_class": f"{self.model.__class__.__module__}.{self.model.__class__.__qualname__}" if self.model is not None else "Ignored",
            "device": str(self.device),
            "dataset_class": f"{self.dataset.__class__.__module__}.{self.dataset.__class__.__qualname__}",
            "optimiser_class": f"{self.optimiser.__class__.__module__}.{self.optimiser.__class__.__qualname__}" if self.optimiser is not None else "Ignored",
            "loss_criterion_class": f"{self.loss_criterion.__class__.__module__}.{self.loss_criterion.__class__.__qualname__}" if self.loss_criterion is not None else "Ignored",
        }

        info = {**info, **self.get_unique_information()}

        return info
    
    def __str__(self) -> str:
        return json.dumps(self.get_information(), indent=2)

    def _setup_training(self) -> None:
        """
        Internal method called to prepare the model for training
        """
        if self.model is not None:
            self.model.to(self.device)
            self.model.train()
            logger.debug(f"Model moved to {self.device} and set to train mode")

    @abc.abstractmethod
    def train(self) -> None:
        """
        Trains the model. This method should be overridden.
        """
        logger.info(self)
        self._setup_training()
        logger.info("Starting training")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Classify samples using max of the outputs of the model

        Args:
            batch (torch.Tensor): The batch of images to classify

        Returns:
            torch.Tensor: The predicted classes of the batch of images
        """
        self.model.eval()
        output = self.model(batch)
        _, predicted = torch.max(output.data, 1)
        return predicted

    def run_base_task_metrics(self, task_no: int, eval_batch_size=32):
        """
        Calculates metrics for comparisons and stores them per task

        Args:
            task_no (int): _description_
        """

        base_name = f"Task {task_no}"
        base_label = f"task_{task_no}"
        
        logger.info(f"Running metrics: {base_name}")
        
        test_loader = self.dataset.create_evaluation_dataloader(batch_size=eval_batch_size)

        """
        Firstly, classify the samples into classes
        For a class x and any class y,
        true_positive: image classified as class x when it is class x
        false_positive: image classified as class x when it is class y
        false_negative: image classified as class y when it is class x
        """
        class_evaluation: Dict[Union[int, str], ClassStats] = {}

        for class_name in self.dataset.classes:
            class_evaluation[class_name] = ClassStats()

        total = 0
        total_correct = 0

        with torch.no_grad():
            for data in test_loader:
                images, ground_truth = data

                images = images.to(self.device)
                ground_truth = ground_truth.to(self.device)

                predicted = self.classify(images)

                total += ground_truth.size(0)
                
                for i, truth_tensor in enumerate(ground_truth):
                    truth = self.dataset.classes[truth_tensor.item()]
                    prediction = self.dataset.classes[predicted[i].item()] # type: ignore

                    class_evaluation[truth].real_total += 1

                    if truth == prediction:
                        total_correct += 1
                        class_evaluation[truth].true_positives += 1
                    else:
                        class_evaluation[truth].false_negative += 1
                        class_evaluation[prediction].false_positive += 1

        overall_accuracy = total_correct / total

        task_stats = TaskStats(
            overall_accuracy=overall_accuracy,
            task_classes={n: self.dataset.resolve_class_indexes(split) for n, split in enumerate(self.dataset.task_splits)}, # type: ignore
            per_class_stats=class_evaluation
        )

        self.task_metrics[task_no] = task_stats

        with open(f"{self.directory}/{base_label}_results.json", "w+") as fp:
            json.dump(task_stats.to_dict(), fp, indent=2, default=lambda o: o.__dict__)

        logger.debug(f"Raw classification accuracy results saved to {self.directory}/{base_label}_accuracy_results.json")
        logger.info(f"Correctly classified {total_correct} / {total} samples ({(100 * overall_accuracy):.2f}% correct)")
        self.writer.add_scalar("Acc/Overall", overall_accuracy, task_no)

        accuracy_bar_figure = generate_accuracy_bar_chart(f"{base_name} Classification Accuracy", class_evaluation)
        self.writer.add_figure(f"Acc_Plots/{base_label}", accuracy_bar_figure)

@dataclass
class ClassStats:
    real_total: int = 0
    true_positives: int = 0
    false_negative: int = 0
    false_positive: int = 0

@dataclass
class TaskStats:
    overall_accuracy: float
    task_classes: Dict[Union[int, str], List[Union[int, str]]]
    per_class_stats: Dict[Union[int, str], ClassStats]

    def calculate_per_class_accuracy(self) -> Dict[Union[int, str], float]:
        return {
            target: self.per_class_stats[target].true_positives / self.per_class_stats[target].real_total
            for target in self.per_class_stats.keys()
        }

    def calculate_per_task_accuracy(self) -> Dict[Union[int, str], float]:
        task_accuracies: Dict[Union[int, str], float] = {}
        per_class_accuracy = self.calculate_per_class_accuracy()

        for task in self.task_classes.keys():
            targets = self.task_classes[task]
            not_seen = False

            for target in targets:
                if target not in self.per_class_stats.keys():
                    task_accuracies[task] = 0
                    not_seen = True
            
            if not_seen:
                continue

            task_accuracy_sum = sum([per_class_accuracy[target] for target in targets])
            task_accuracy = task_accuracy_sum / len(targets)
            task_accuracies[task] = task_accuracy 
        
        return task_accuracies

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "per_class_accuracy": self.calculate_per_class_accuracy(),
            "per_task_accuracy": self.calculate_per_task_accuracy()
        }


def generate_accuracy_bar_chart(
    plot_title: str,
    data: Dict[Union[str, int], ClassStats]
) -> matplotlib.figure.Figure:
    """
    Generates a bar chart from the accuracy data generated by :func:`evaluate_accuracy`

    Args:
        plot_title (str): Title to display at the top of the plot
        data (Dict[Union[str, int], Dict[str, int]]): Data generated from :func:`evaluate_accuracy`

    Returns:
        matplotlib.figure.Figure: The figure containing the bar chart
    """
    total_correct = 0
    total_count = 0

    xs = []
    ys = []

    for clazz in data.keys():
        clazz_data = data[clazz]

        total_correct += clazz_data.true_positives
        total_count += clazz_data.real_total

        accuracy = (clazz_data.true_positives / clazz_data.real_total) * 100
        xs.append(clazz)
        ys.append(accuracy)

    xs.append("total")
    ys.append((total_correct / total_count) * 100)

    fig, ax = plt.subplots()
    ax.bar(xs, ys)
    ax.set_title(plot_title)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    for i in range(len(xs)):
        ax.text(xs[i], ys[i] // 2, f"{ys[i]:.1f}", ha="center", color="white")

    return fig
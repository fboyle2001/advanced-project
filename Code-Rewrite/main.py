import os
import time
from typing import Tuple

from loguru import logger
from torch.utils.tensorboard.writer import SummaryWriter
import atexit

import datasets
import algorithms
import utils

import torch
from torchvision.models import resnet18

PARENT_DIRECTORY = "./store/models"

ALGORITHM_DEFAULTS = {
    algorithms.OfflineTraining: {
        "epochs_per_task": 1,
        "batch_size": 64,
        "gradient_clip": 10,
        "apply_learning_rate_annealing": True,
        "max_lr": 0.05,
        "min_lr": 0.0005,
    },
    algorithms.Finetuning: {
        "batch_size": 64
    },
    algorithms.GDumb: {
        "batch_size": 16,
        "max_memory_samples": 500,
        "post_population_max_epochs": 256,
        "gradient_clip": 10,
        "max_lr": 0.05,
        "min_lr": 0.0005,
        "cutmix_probability": 0.5
    },
    algorithms.ElasticWeightConsolidation: {
        "max_epochs_per_task": 5,
        "batch_size": 64,
        "task_importance": 1000
    },
    algorithms.Rainbow: {
        "batch_size": 16,
        "max_memory_samples": 1000,
        "epochs_per_task": 1,
        "gradient_clip": 10,
        "max_lr": 0.05,
        "min_lr": 0.0005,
        "cutmix_probability": 0.5
    },
    algorithms.RainbowOnline: {
        "batch_size": 16,
        "max_memory_samples": 1000,
        "epochs_per_task": 50,
        "gradient_clip": 10,
        "max_lr": 0.05,
        "min_lr": 0.0005,
        "cutmix_probability": 0.5,
        "sampling_strategy": ["diverse", "central", "edge", "random", "proportional"][0] # change the index to 0, 1, 2, 3, 4 
    },
    algorithms.RainbowOnlineExperimental: {
        "batch_size": 16,
        "max_memory_samples": 1000,
        "epochs_per_task": 50,
        "gradient_clip": 10,
        "max_lr": 0.05,
        "min_lr": 0.0005,
        "cutmix_probability": 0.5,
        "sampling_strategy": ["endpoint_peak", "midpoint_peak", "edge_skewed_1"][0], # change the index to 0, 1, 2
        "all_occurrences": False
    },
    algorithms.HindsightAnchor: {
        "batch_size": 16,
        "max_memory_per_class": 5,
        "epochs_per_task": 1,
        "gradient_clip": 10,
        "apply_learning_rate_annealing": False,
        "max_lr": 1e-3,
        "min_lr": 1e-3,
        "cutmix_probability": 0,
    }
}

DATASET_DEFAULTS = {
    datasets.CIFAR10: {
        "disjoint": True,
        "classes_per_task": 2
    },
    datasets.MNIST: {
        "disjoint": False,
        "classes_per_task": 0
    }
}

def setup_files(algorithm_folder, dataset_class, experiment_name) -> Tuple[str, SummaryWriter]:
    directory = f"{PARENT_DIRECTORY}/{algorithm_folder}/{time.time()}_{dataset_class.__qualname__}"

    if experiment_name is not None:
        directory = f"{directory}_{experiment_name}"

    os.makedirs(directory, exist_ok=False)

    logger.add(f"{directory}/log.log", backtrace=True, diagnose=True)
    writer = SummaryWriter(log_dir=directory)

    logger.info(f"Experiment Name: {experiment_name if experiment_name is not None else 'Not Set'}")

    return directory, writer

class MLP(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=400, classes=10):
        super().__init__()

        self.input = torch.nn.Linear(input_size, hidden_size)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x
        # return torch.nn.functional.softmax(x)

def execute(algorithm_class, dataset_class, directory, writer):
    dataset = dataset_class(**DATASET_DEFAULTS[dataset_class])

    algorithm = algorithm_class(
        model=model,
        dataset=dataset,
        optimiser=torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-6), #torch.optim.Adam(model.parameters())
        loss_criterion=torch.nn.CrossEntropyLoss(),
        writer=writer,
        **ALGORITHM_DEFAULTS[algorithm_class]
    )

    algorithm.train()

    model_save_loc = f"{directory}/model.pth"

    logger.info(f"Saving model to {model_save_loc}")
    torch.save(algorithm.model.state_dict(), model_save_loc)

if __name__ == "__main__":
    utils.seed_everything(0)

    algorithm_class = algorithms.HindsightAnchor
    dataset_class = datasets.CIFAR10
    experiment_name = "CHECK"

    device = torch.device("cuda:0")

    # model = resnet18(weights=None)
    # model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

    # Has higher performance, need to analyse why in the future
    model = utils.get_gdumb_resnet_impl()

    model.to(device)

    directory, writer = setup_files(algorithm_class.get_algorithm_folder(), dataset_class, experiment_name)

    def close_tensorboard_writer():
        writer.flush()
        writer.close()
        logger.info("Shut down TensorBoard")

    logger.info(f"To access TensorBoard, run the command: tensorboard --logdir={writer.log_dir}")
    atexit.register(close_tensorboard_writer)

    with logger.catch(exception=BaseException, reraise=True):
        execute(algorithm_class, dataset_class, directory, writer)
    
    
    


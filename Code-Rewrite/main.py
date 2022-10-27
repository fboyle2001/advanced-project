import os
import time

from loguru import logger

import datasets
import algorithms
import metrics

import torch
from torchvision.models import resnet18

PARENT_DIRECTORY = "./store/models"
ALGORITHM_DEFAULTS = {
    algorithms.OfflineTraining: {
        "max_epochs_per_task": 5,
        "batch_size": 64
    },
    algorithms.Finetuning: {
        "batch_size": 64
    }
}

def setup_files(algorithm_folder) -> str:
    directory = f"{PARENT_DIRECTORY}/{algorithm_folder}/{time.time()}"
    os.makedirs(directory, exist_ok=False)

    logger.add(f"{directory}/log.log", backtrace=True, diagnose=True)

    return directory

def execute(algorithm_class, directory):
    # dataset = datasets.CIFAR10(disjoint=True, classes_per_task=5)
    dataset = datasets.CIFAR10(disjoint=False)

    algorithm = algorithm_class(
        model=model,
        dataset=dataset,
        optimiser=torch.optim.Adam(model.parameters()),
        loss_criterion=torch.nn.CrossEntropyLoss(),
        **ALGORITHM_DEFAULTS[algorithm_class]
    )

    algorithm.train()

    model_save_loc = f"{directory}/model.pth"

    logger.info(f"Saving model to {model_save_loc}")
    torch.save(algorithm.model.state_dict(), model_save_loc)

    metrics.run_metrics(algorithm, dataset, directory)

if __name__ == "__main__":
    algorithm_class = algorithms.Finetuning
    device = torch.device("cuda:0")
    model = resnet18(weights=None)
    model.to(device)

    directory = setup_files(algorithm_class.get_algorithm_folder())

    with logger.catch(exception=BaseException, reraise=True):
        execute(algorithm_class, directory)

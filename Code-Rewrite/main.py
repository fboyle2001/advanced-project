import os
import time
from typing import Tuple

from models import cifar
from dotmap import DotMap

from loguru import logger
from torch.utils.tensorboard.writer import SummaryWriter
import atexit

import datasets
import algorithms
# import algorithms.metrics.metrics as metrics

import torch
from torchvision.models import resnet18

PARENT_DIRECTORY = "./store/models"

ALGORITHM_DEFAULTS = {
    algorithms.OfflineTraining: {
        "max_epochs_per_task": 100,
        "batch_size": 64
    },
    algorithms.Finetuning: {
        "batch_size": 64
    },
    algorithms.GDumb: {
        "batch_size": 16,
        "max_memory_samples": 1000,
        "post_population_max_epochs": 256,
        "gradient_clip": 10,
        "max_lr": 0.05,
        "min_lr": 0.0005
    },
    algorithms.ElasticWeightConsolidation: {
        "max_epochs_per_task": 5,
        "batch_size": 64,
        "task_importance": 1000
    }
}

DATASET_DEFAULTS = {
    datasets.CIFAR10: {
        "disjoint": False,
        "classes_per_task": 0
    },
    datasets.MNIST: {
        "disjoint": False,
        "classes_per_task": 0
    }
}

def setup_files(algorithm_folder, dataset_class) -> Tuple[str, SummaryWriter]:
    directory = f"{PARENT_DIRECTORY}/{algorithm_folder}/{time.time()}_{dataset_class.__qualname__}"
    os.makedirs(directory, exist_ok=False)

    logger.add(f"{directory}/log.log", backtrace=True, diagnose=True)
    writer = SummaryWriter(log_dir=directory)

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

def execute(algorithm_class, dataset_class, directory, writer, opt):
    dataset = dataset_class(**DATASET_DEFAULTS[dataset_class])

    algorithm = algorithm_class(
        model=model,
        dataset=dataset,
        optimiser=torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-6), #torch.optim.Adam(model.parameters())
        loss_criterion=torch.nn.CrossEntropyLoss().cuda(),
        writer=writer,
        **ALGORITHM_DEFAULTS[algorithm_class],
        opt=opt
    )

    algorithm.train()

    model_save_loc = f"{directory}/model.pth"

    logger.info(f"Saving model to {model_save_loc}")
    torch.save(algorithm.model.state_dict(), model_save_loc)

if __name__ == "__main__":
    algorithm_class = algorithms.GDumb
    dataset_class = datasets.CIFAR10

    opts = {
        "depth": 18,
        "in_channels": 3,
        "normtype": "BatchNorm",
        "pooltype": "MaxPool2d",
        "preact": False,
        "activetype": "ReLU",
        "bn": True,
        "affine_bn": True,
        "bn_eps": 1e-6,
        "compression": 0.5,
        "num_classes": 10,
        "inp_size": 32,
        "dataset": "CIFAR10",
        "data_dir": "./store/data",
        "workers": 0,
        "batch_size": 16,
        "num_tasks": 5,
        "num_classes_per_task": 2,
        "num_pretrain_classes": 0,
        "memory_size": 1000
    }

    opts = DotMap(opts)

    device = torch.device("cuda:0")
    # model = resnet18(weights=None)
    model = cifar.ResNet(opts)
    model.to(device)

    directory, writer = setup_files(algorithm_class.get_algorithm_folder(), dataset_class)

    def close_tensorboard_writer():
        writer.flush()
        writer.close()
        logger.info("Shut down TensorBoard")

    logger.info(f"To access TensorBoard, run the command: tensorboard --logdir={writer.log_dir}")
    atexit.register(close_tensorboard_writer)

    with logger.catch(exception=BaseException, reraise=True):
        execute(algorithm_class, dataset_class, directory, writer, opts)
    
    
    


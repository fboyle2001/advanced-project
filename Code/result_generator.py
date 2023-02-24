from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from loguru import logger
from torch.utils.tensorboard.writer import SummaryWriter

import torch

import algorithms
import datasets
import utils
import experiments

import argparse
import time
import os
import ssl
import atexit
import psutil
import json

ssl._create_default_https_context = ssl._create_unverified_context

PARENT_DIRECTORY = "./output_cifar10_0.5k"
@dataclass
class Configuration:
    algorithm: str
    dataset: str
    classes_per_task: int
    seed: int
    samples: Optional[int]

@dataclass
class AlgorithmSetup:
    algorithm: Any
    options: Dict[str, Any]
    reduced_model: bool
    img_size: int

algorithm_setups = {
    "offline": AlgorithmSetup(
        algorithm=algorithms.OfflineTraining,
        options={
            "epochs_per_task": 250,
            "batch_size": 64,
            "gradient_clip": 10,
            "apply_learning_rate_annealing": True,
            "max_lr": 0.05,
            "min_lr": 0.0005,
        },
        reduced_model=False,
        img_size=32
    ),
    "finetuning": AlgorithmSetup(
        algorithm=algorithms.Finetuning,
        options={
            "batch_size": 64
        },
        reduced_model=False,
        img_size=32
    ),
    "gdumb": AlgorithmSetup(
        algorithm=algorithms.GDumb,
        options={
            "batch_size": 32,
            "max_memory_samples": 500,
            "post_population_max_epochs": 256,
            "gradient_clip": 10,
            "max_lr": 0.05,
            "min_lr": 0.0005,
            "cutmix_probability": 0.5
        },
        reduced_model=False,
        img_size=32
    ),
    "ewc": AlgorithmSetup(
        algorithm=algorithms.ElasticWeightConsolidation,
        options={
            "max_epochs_per_task": 50,
            "batch_size": 64,
            "task_importance": 175000 # works, 20k didnt maybe drop a little bit? #250k worked reasonably well
        },
        reduced_model=False,
        img_size=32
    ),
    "rainbow": AlgorithmSetup(
        algorithm=algorithms.RainbowOnline,
        options={
            "batch_size": 32,
            "max_memory_samples": 500,
            "epochs_per_task": 50,
            "gradient_clip": 10,
            "max_lr": 0.05,
            "min_lr": 0.0005,
            "cutmix_probability": 0.5,
            "sampling_strategy": "diverse" # ["diverse", "central", "edge", "random", "proportional"]
        },
        reduced_model=False,
        img_size=32
    ),
    "l2p": AlgorithmSetup(
        algorithm=algorithms.LearningToPrompt,
        options={
            "epochs_per_task": 1,
            "batch_size": 16,
            "K_lr": 1e-3,
            "P_lr": 1e-3,
            "g_phi_lr": 1e-3,
            "N": 2,
            "L_p": 5,
            "M": 10,
            "balancing_lambda": 0.5,
            "prompt_frequency_strategy": "minmax" # ["disabled", "minmax", "scaled_frequency"]
        },
        reduced_model=False,
        img_size=224
    ),
    "scr": AlgorithmSetup(
        algorithm=algorithms.SupervisedContrastiveReplay,
        options={
            "epochs_per_task": 1,
            "batch_size": 10,
            "max_memory_samples": 500,
            "memory_batch_size": 100,
            "temperature": 0.07,
            "lr": 0.1
        },
        reduced_model=True,
        img_size=32
    ),
    "der": AlgorithmSetup(
        algorithm=algorithms.DarkExperiencePlusPlus,
        options={
            "epochs_per_task": 1,
            "batch_size": 16,
            "max_memory_samples": 500,
            "alpha": 0.5,
            "beta": 0 # Beta = 0 is equivalent to DER
        },
        reduced_model=False,
        img_size=32
    ),
    "derpp": AlgorithmSetup(
        algorithm=algorithms.DarkExperiencePlusPlus,
        options={
            "epochs_per_task": 1,
            "batch_size": 16,
            "max_memory_samples": 500,
            "alpha": 0.5,
            "beta": 0.5
        },
        reduced_model=False,
        img_size=32
    ),
    "novel_bn": AlgorithmSetup(
        algorithm=experiments.NovelImplementation,
        options={
            "max_memory_samples": 500,
            "uncertainty_type": "batch_normalised"
        },
        reduced_model=False,
        img_size=224
    ),
    "novel_rd": AlgorithmSetup(
        algorithm=experiments.NovelImplementation,
        options={
            "max_memory_samples": 500,
            "uncertainty_type": "relative_distances"
        },
        reduced_model=False,
        img_size=224
    ),
    "non_mlp": AlgorithmSetup(
        algorithm=experiments.NonMLPImplementation,
        options={},
        reduced_model=False,
        img_size=224
    ),
    "novel_et": AlgorithmSetup(
        algorithm=experiments.NovelImplementationETOnly,
        options={},
        reduced_model=False,
        img_size=224
    ),
    "rainbow_ncm": AlgorithmSetup(
        algorithm=algorithms.RainbowOnlineNCM,
        options={
            "batch_size": 32,
            "max_memory_samples": 5000,
            "epochs_per_task": 256,
            "gradient_clip": 10,
            "max_lr": 0.05,
            "min_lr": 0.0005,
            "cutmix_probability": 0.5
        },
        reduced_model=False,
        img_size=32
    )
}

dataset_map = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100
}

# Setup the directory and writer to store everything
def setup_files(algorithm_folder, dataset_class, experiment_name) -> Tuple[str, SummaryWriter]:
    directory = f"{PARENT_DIRECTORY}/{algorithm_folder}/{time.time()}_{dataset_class.__qualname__}"

    if experiment_name is not None:
        directory = f"{directory}_{experiment_name}"

    os.makedirs(directory, exist_ok=False)

    logger.add(f"{directory}/log.log", backtrace=True, diagnose=True)
    writer = SummaryWriter(log_dir=directory)

    logger.info(f"Experiment Name: {experiment_name if experiment_name is not None else 'Not Set'}")

    return directory, writer

def parse_arguments() -> Configuration:
    parser = argparse.ArgumentParser()

    # e.g. python result_generator.py --algorithm scr --dataset cifar100 --tasks 5 --seed 2001
    parser.add_argument("--algorithm", action="store", choices=algorithm_setups.keys(), required=True)
    parser.add_argument("--dataset", action="store", choices=dataset_map.keys(), default="cifar100")
    parser.add_argument("--cpt", "--classes_per_task", action="store", type=int, default=20)
    parser.add_argument("--seed", action="store", type=int, default=2001)
    parser.add_argument("--samples", action="store", type=int)

    arguments = parser.parse_args()

    return Configuration(
        algorithm=arguments.algorithm,
        dataset=arguments.dataset,
        classes_per_task=arguments.cpt,
        seed=arguments.seed,
        samples=arguments.samples
    )

def select_model(dataset_name, reduced):
    if dataset_name == "cifar10":
        return utils.get_gdumb_resnet_18_impl(reduced=reduced)
    
    if dataset_name == "cifar100":
        return utils.get_gdumb_resnet_32_impl(reduced=reduced)

    raise ValueError("Invalid dataset name")

def get_device():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        exit(-1)

    return torch.device("cuda:0")

def setup_shutdown_hook(writer):
    def close_tensorboard_writer():
        writer.flush()
        writer.close()
        logger.info("Shut down TensorBoard")

    logger.info(f"To access TensorBoard, run the command: tensorboard --logdir={writer.log_dir}")
    atexit.register(close_tensorboard_writer)

def force_cuda_load(device):
    logger.info("Forcing CUDA initialisation...")

    random_tensor = torch.randn(size=(10, 10))
    random_tensor.to(device)
    random_tensor.square()
    torch.cuda.synchronize(device)

def run_experiment(config: Configuration, algorithm_setup: AlgorithmSetup, device, model, directory, writer):
    classes_per_task = config.classes_per_task
    disjoint = classes_per_task != 0
    dataset = dataset_map[config.dataset](disjoint=disjoint, classes_per_task=classes_per_task, size=algorithm_setup.img_size)

    algorithm = algorithm_setup.algorithm(
        model=model,
        dataset=dataset,
        optimiser=torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-6),
        loss_criterion=torch.nn.CrossEntropyLoss(),
        writer=writer,
        **algorithm_setup.options
    )

    torch.cuda.reset_max_memory_allocated(device)
    process = psutil.Process()

    logger.debug(f"PID: {process.pid}")

    logger.info("Starting...")

    start_time = time.time()
    algorithm.train()
    end_time = time.time()

    logger.info("Training and evaluation completed!")

    max_process_memory_used = process.memory_info().peak_wset
    max_gpu_memory_used = torch.cuda.max_memory_allocated(device)

    stats = {
        "running_duration": end_time - start_time,
        "max_gpu_memory_used": max_gpu_memory_used,
        "max_process_memory_used": max_process_memory_used
    }

    with open(f"{directory}/stats.json", "w+") as fp:
        json.dump(stats, fp, indent=2, default=lambda o: o.__dict__)

    logger.info("Wrote usage stats to file")

def main():
    # Parse command line arguments
    config = parse_arguments()

    # Get the algorithm
    algorithm_setup = algorithm_setups[config.algorithm]

    # Set the samples
    if config.samples is not None:
        if config.algorithm in ["gdumb", "rainbow", "scr", "der", "derpp", "novel_bn", "novel_rd"]:
            algorithm_setup.options["max_memory_samples"] = config.samples
        elif config.algorithm in ["offline", "l2p", "ewc", "finetuning"]:
            print(f"{config.algorithm} is not a sample-based approach, ignoring --samples flag")
        else:
            assert 1 == 0, f"Invalid algorithm {config.algorithm} for sample setting"

    # Setup the directory and get the writer
    directory, writer = setup_files(algorithm_setup.algorithm.get_algorithm_folder(), dataset_map[config.dataset], "OFFICIAL_RUN")
    logger.info(f"Command line arguments: {config}")

    # Set the random seed everywhere 
    utils.seed_everything(config.seed)
    logger.info(f"Seeded random to {config.seed}")

    # Get the GPU
    device = get_device()
    logger.info("Got CUDA device")

    # Force CUDA initialisation to prevent incorrect timings
    force_cuda_load(device)
    logger.info("CUDA initialised")

    # Setup the writer shutdown hook
    setup_shutdown_hook(writer)
    logger.info("Setup shutdown hooks")

    # Select the model
    model = select_model(config.dataset, algorithm_setup.reduced_model)
    model.to(device)
    logger.info("Model is setup")

    logger.info("Complete pre-setup, starting experiment")

    # Run the experiment
    with logger.catch(exception=BaseException, reraise=True):
        run_experiment(config, algorithm_setup, device, model, directory, writer)

    logger.info("Complete")

if __name__ == "__main__":
    main()
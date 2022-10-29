from typing import Dict, Tuple, Union

from loguru import logger
from torch.utils.tensorboard.writer import SummaryWriter

import json

import torch
import datasets
import algorithms

import matplotlib.pyplot as plt

def accuracy_figure(data: Dict[Union[str, int], Dict[str, int]], name: str):
    total_correct = 0
    total_count = 0

    x = []
    y = []

    for clazz in data.keys():
        clazz_data = data[clazz]

        total_correct += clazz_data["true_positives"]
        total_count += clazz_data["real_total"]

        accuracy = clazz_data["true_positives"]  / clazz_data["real_total"]
        x.append(clazz)
        y.append(accuracy)

    x.append("total")
    y.append(total_correct / total_count)
    y = [yi * 100 for yi in y]

    #fig = plt.figure()
    fig, ax = plt.subplots()
    # r = plt.bar(x, y)
    # fig.add_subplot(r, )
    # ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x, y)
    ax.set_title(name)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    # ax.set_xticks(x)
    # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # ax.set_title("Test")
    # ax.yaxis.set_label("Y Leg")
    # ax.xaxis.set_label("X Leg")

    return fig

def evaluate_accuracy(
    algorithm: algorithms.BaseCLAlgorithm,
    dataset: datasets.BaseCLDataset,
    batch_size: int = 32
) -> Tuple[int, int, Dict[Union[str, int], Dict[str, int]]]:
    test_loader = dataset.create_evaluation_dataloader(batch_size=batch_size)

    """
    For a class x and any class y,
    true_positive: image classified as class x when it is class x
    false_positive: image classified as class x when it is class y
    false_negative: image classified as class y when it is class x
    """
    class_evaluation = {}

    total = 0
    total_correct = 0

    with torch.no_grad():
        for data in test_loader:
            images, ground_truth = data

            images = images.to(algorithm.device)
            ground_truth = ground_truth.to(algorithm.device)

            predicted = algorithm.classify(images)

            total += ground_truth.size(0)
            
            for i, truth_tensor in enumerate(ground_truth):
                truth = dataset.classes[truth_tensor.item()]
                prediction = dataset.classes[predicted[i].item()] # type: ignore

                if truth not in class_evaluation.keys():
                    class_evaluation[truth] = {
                        "true_positives": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "real_total": 0
                    }
                
                if prediction not in class_evaluation.keys():
                    class_evaluation[prediction] = {
                        "true_positives": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "real_total": 0
                    }

                class_evaluation[truth]["real_total"] += 1

                if truth == prediction:
                    total_correct += 1
                    class_evaluation[truth]["true_positives"] += 1
                else:
                    class_evaluation[truth]["false_negative"] += 1
                    class_evaluation[prediction]["false_positive"] += 1
    
    return total, total_correct, class_evaluation

def run_metrics(
    algorithm: algorithms.BaseCLAlgorithm,
    dataset: datasets.BaseCLDataset,
    directory: str,
    writer: SummaryWriter
):
    logger.info("Evaluating classification accuracy for each class")
    total, total_correct, class_eval = evaluate_accuracy(algorithm, dataset)

    with open(f"{directory}/accuracy_results.json", "w+") as fp:
        json.dump(class_eval, fp, indent=2)

    logger.debug(f"Raw results saved to {directory}/accuracy_results.json")
    logger.info(f"Classified {total_correct} / {total} samples correctly ({100*total_correct/total:.2f})%")


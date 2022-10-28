from typing import Dict, Tuple, Union

from loguru import logger
import json

import torch
import datasets
import algorithms

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
    directory: str
):
    logger.info("Evaluating classification accuracy for each class")
    total, total_correct, class_eval = evaluate_accuracy(algorithm, dataset)

    with open(f"{directory}/accuracy_results.json", "w+") as fp:
        json.dump(class_eval, fp, indent=2)

    logger.debug(f"Raw results saved to {directory}/accuracy_results.json")
    logger.info(f"Classified {total_correct} / {total} samples correctly ({100*total_correct/total:.2f})%")


from torchvision.models import resnet18, resnet34

import torch_utils
import algorithms
import datasets
import metrics

import json

## DEFAULT ALGORITHM SETTINGS

algorithm_kwargs_lookup = {
    algorithms.Finetuning: {
        "init": {},
        "train": {}
    },
    algorithms.OfflineTraining: {
        "init": {},
        "train": {
            "max_epochs": 50
        }
    },
    algorithms.GDumb: {
        "init": {
            "max_memory_samples": 2000
        },
        "train": {
            "batch_size": 16,
            "max_epochs": 5
        }
    }
}

## PARAMETERS

batch_size = 64
algorithm_class = algorithms.Finetuning
dataset_class = datasets.CIFAR10
model = resnet18(weights=None)
per_split = 2

algorithm_kwargs_overrides = None

## LEAVE BELOW UNCHANGED

device = torch_utils.get_gpu()

algorithm_kwargs = algorithm_kwargs_overrides if algorithm_kwargs_overrides is not None else algorithm_kwargs_lookup[algorithm_class]
dataset_kwargs = {} if per_split is None else { "per_split": per_split, "randomised_split": True }

trainer = algorithm_class(device, **algorithm_kwargs["init"])
dataset = dataset_class(batch_size, **dataset_kwargs)

trainer.train(model, dataset, **algorithm_kwargs["train"])
trainer.dump_model(model)

total, total_correct, class_eval = metrics.evaluate_accuracy(model, device, dataset)

with open(f"{trainer.save_directory}/classification_results.json", "w+") as fp:
    json.dump(class_eval, fp, indent=2)

trainer.logger.info(f"Classified {total_correct} / {total} samples correctly ({100*total_correct/total:.2f})%")
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
            "max_epochs": 200
        }
    }
}

## PARAMETERS

batch_size = 64
algorithm_class = algorithms.GDumb
dataset_class = datasets.CIFAR10
model = resnet18(weights=None)

algorithm_kwargs_overrides = None

## LEAVE BELOW UNCHANGED

device = torch_utils.get_gpu()

algorithm_kwargs = algorithm_kwargs_overrides if algorithm_kwargs_overrides is not None else algorithm_kwargs_lookup[algorithm_class]

trainer = algorithm_class(device, **algorithm_kwargs["init"])
dataset = dataset_class(batch_size)

trainer.train(model, dataset, **algorithm_kwargs["train"])
trainer.dump_model(model)

total, total_correct, class_eval = metrics.evaluate_accuracy(model, device, dataset)

with open(f"{trainer.save_directory}/classification_results.json", "w+") as fp:
    json.dump(class_eval, fp, indent=2)

trainer.logger.info(f"Classified {total_correct} / {total} samples correctly ({100*total_correct/total:.2f})%")
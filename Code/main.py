from torchvision.models import resnet18, resnet34

import torch_utils
import algorithms
import datasets
import metrics

## DEFAULT ALGORITHM SETTINGS

algorithm_kwargs_lookup = {
    algorithms.Finetuning: {
        "init": {},
        "train": {
            "epochs_per_task": 1
        }
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
    },
    algorithms.ElasticWeightConsolidation: {
        "init": {
            "task_importance": 1000
        },
        "train": {
            "epochs_per_task": 5
        }
    }
}

## PARAMETERS

batch_size = 64
algorithm_class = algorithms.ElasticWeightConsolidation
dataset_class = datasets.CIFAR10
model = resnet18(weights=None)
per_split = 5

algorithm_kwargs_overrides = None

## LEAVE BELOW UNCHANGED

device = torch_utils.get_gpu()

algorithm_kwargs = algorithm_kwargs_overrides if algorithm_kwargs_overrides is not None else algorithm_kwargs_lookup[algorithm_class]
dataset_kwargs = {} if per_split is None else { "per_split": per_split, "randomised_split": True }

trainer = algorithm_class(device, **algorithm_kwargs["init"])
dataset = dataset_class(batch_size, **dataset_kwargs)

trainer.train(model, dataset, **algorithm_kwargs["train"])
trainer.dump_model(model)

metrics.run_metrics(model, device, dataset, trainer.save_directory, trainer.logger)
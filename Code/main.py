from torchvision.models import resnet18
import torch_utils

import algorithms
import datasets

import torchvision
from torchvision import transforms

device = torch_utils.get_gpu()
trainer = algorithms.OfflineTrainingAlgorithm(device)
model = resnet18(weights=None)

### START: TO CHANGE

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar = datasets.BaseDataset(
    torchvision.datasets.CIFAR10,
    { "train": True, "download": True },
    { "train": False, "download": True },
    transform
)

### END: TO CHANGE

trainer.train(model, cifar.training_loader, 5)
trainer.dump_model(model)
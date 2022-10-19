from torchvision.models import resnet18

import torch_utils
import algorithms
import datasets

batch_size = 64

device = torch_utils.get_gpu()
trainer = algorithms.OfflineTrainingAlgorithm(device)
model = resnet18(weights=None)
dataset = datasets.CIFAR10(batch_size)

trainer.train(model, dataset, 5)
trainer.dump_model(model)
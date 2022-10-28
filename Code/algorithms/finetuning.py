import torch.optim as optim
import torch.nn as nn

from .algorithm_base import BaseTrainingAlgorithm

"""
Finetuning

Disjoint Task Formulation: Yes
Online CL: No
Class Incremental: Yes
"""
class Finetuning(BaseTrainingAlgorithm):
    def __init__(self, device, verbose=True, log_to_file=True, log_to_console=True):
        super().__init__(
            name="Finetuning",
            alg_directory="finetune",
            optimiser_class=optim.Adam, 
            initial_optimiser_parameters={ "lr": 1e-3 },
            criterion_class=nn.CrossEntropyLoss,
            device=device,
            verbose=verbose,
            log_to_file=log_to_file,
            log_to_console=log_to_console
        )

    def train(self, model, dataset, epochs_per_task: int = 1):
        super().train(model, dataset)
        self.logger.info(f"Performing finetuning with {epochs_per_task} epochs pers task for {len(dataset.task_splits)} tasks")

        running_loss = 0

        for task_id, (task_split, task_training_loader) in enumerate(dataset.iterate_task_dataloaders()):
            self.logger.info(f"Finetuning on Task #{task_id + 1} with class split {task_split} (classes: {dataset.resolve_class_indexes(task_split)})")

            for epoch in range(epochs_per_task): 
                self.logger.info(f"Epoch {epoch + 1} / {epochs_per_task} for Task #{task_id + 1}")

                for i, data in enumerate(task_training_loader, 0):
                    inp, labels = data
                    inp = inp.to(self.device)
                    labels = labels.to(self.device)
                    self.optimiser.zero_grad()
                    predictions = model(inp)
                    loss = self.criterion(predictions, labels)
                    loss.backward()
                    self.optimiser.step()

                    running_loss += loss.item()

                    if i == len(task_training_loader) - 1:
                        self.logger.info(f"Loss: {running_loss / 2000:.3f}")
                        running_loss = 0
        
        self.logger.info("Training completed")
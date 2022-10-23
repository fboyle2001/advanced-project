import torch.optim as optim
import torch.nn as nn

from .algorithm_base import BaseTrainingAlgorithm

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

    def train(self, model, dataset):
        super().train(model, dataset)
        train_loader = dataset.training_loader

        self.logger.info(f"Performing single finetuning pass")
        running_loss = 0

        for i, data in enumerate(train_loader, 0):
            inp, labels = data
            inp = inp.to(self.device)
            labels = labels.to(self.device)
            self.optimiser.zero_grad()
            predictions = model(inp)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimiser.step()

            running_loss += loss.item()

            if i == len(train_loader) - 1:
                self.logger.info(f"Loss: {running_loss / 2000:.3f}")
                running_loss = 0
        
        self.logger.info("Training completed")
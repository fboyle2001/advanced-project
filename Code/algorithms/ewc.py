from .algorithm_base import BaseTrainingAlgorithm

import torch.optim as optim
import torch.nn as nn

"""
Elastic Weight Consolidation (Kirkpatrick et al. 2017)

For each parameter in the model, compute the corresponding Fisher information
and use this in the loss function to constrain the model by applying a quadratic
penalty based on the difference between the current parameter value and the new
parameter value
"""
class ElasticWeightConsolidation(BaseTrainingAlgorithm):
    def __init__(self, device, verbose=True, log_to_file=True, log_to_console=True):
        super().__init__(
            name="",
            alg_directory="",
            optimiser_class=optim.Adam, 
            initial_optimiser_parameters={ "lr": 1e-3 },
            criterion_class=nn.CrossEntropyLoss,
            device=device,
            verbose=verbose,
            log_to_file=log_to_file,
            log_to_console=log_to_console
        )

    def train(self, model, dataset, max_epochs):
        super().train(model, dataset)
        train_loader = dataset.training_loader

        for epoch in range(max_epochs):
            self.logger.info(f"Starting epoch {epoch+1} / {max_epochs}")
        
        self.logger.info("Training completed")
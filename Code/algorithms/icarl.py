from .algorithm_base import BaseTrainingAlgorithm

import torch.optim as optim
import torch.nn as nn

"""
Incremental Classifier and Representation Learning (Rebuffi et al. 2017)

Consists of three components:
1) Classification by nearest-mean-of-exemplars
2) Herding-based exemplar selection
3) Representation learning
Learns classifier and feature representation simultaneously

Disjoint Task Formulation: Yes / No
Online CL: Yes / No
Class Incremental: Yes / No
"""
class ExampleTrainingAlgorithm(BaseTrainingAlgorithm):
    def __init__(self, device, verbose=True, log_to_file=True, log_to_console=True):
        super().__init__(
            name="iCaRL",
            alg_directory="icarl",
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
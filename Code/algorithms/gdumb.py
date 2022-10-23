from .algorithm_base import BaseTrainingAlgorithm
from .utils import HashReplayBuffer

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

"""

"""
class GDumb(BaseTrainingAlgorithm):
    def __init__(self, device, max_memory_samples, verbose=True, log_to_file=True, log_to_console=True):
        super().__init__(
            name="GDumb",
            alg_directory="gdumb",
            optimiser_class=optim.Adam, 
            initial_optimiser_parameters={ "lr": 1e-3 },
            criterion_class=nn.CrossEntropyLoss,
            device=device,
            verbose=verbose,
            log_to_file=log_to_file,
            log_to_console=log_to_console
        )

        self._max_memory_samples = max_memory_samples
        self.replay_buffer = HashReplayBuffer(max_memory_samples, "random_from_largest_class")

    def train(self, model, dataset, batch_size, max_epochs):
        super().train(model, dataset)

        self.logger.info("Populating the replay buffer")

        # Populate the replay buffer
        for img, label in dataset.training_data:
            self.replay_buffer.add_to_buffer(img, label)

        self.logger.info(f"Replay buffer populated with {self.replay_buffer.count} samples")
        self.logger.info("Starting model training")

        # Now train the model
        rb_dataset = self.replay_buffer.to_torch_dataset()
        rb_dataloader = DataLoader(rb_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(max_epochs):
            self.logger.info(f"Starting epoch {epoch + 1} / {max_epochs}")
            running_loss = 0

            for i, data in enumerate(rb_dataloader, 0):
                inp, labels = data
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                self.optimiser.zero_grad()
                predictions = model(inp)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

                if i == len(rb_dataloader) - 1:
                    self.logger.info(f"{epoch + 1} loss: {running_loss / 2000:.3f}")
                    running_loss = 0
        
        self.logger.info("Training completed")
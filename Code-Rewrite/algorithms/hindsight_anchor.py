from typing import Dict, Union, Optional, Tuple
from loguru import logger

from . import utils, buffers
from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision

import random
import copy
import numpy as np
import sys
from PIL import Image
import time

import other_utils

class HindsightAnchor(BaseCLAlgorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        epochs_per_task: int,
        batch_size: int,
        gradient_clip: Optional[float],
        apply_learning_rate_annealing: bool,
        max_lr: Optional[float],
        min_lr: Optional[float],
        cutmix_probability: float,
        max_memory_per_class: int
    ):
        super().__init__(
            name="Hindsight Anchor",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        assert self.epochs_per_task == 1
        self.batch_size = 16
        self.gradient_clip = gradient_clip

        self.apply_learning_rate_annealing = apply_learning_rate_annealing
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cutmix_probability = cutmix_probability

        self.max_memory_per_class = 5
        self.fifo_buffer = buffers.FIFORingReplayBuffer(max_memory_per_class=self.max_memory_per_class)
        self.finetuning_epochs = 50
        self.feature_embedding_moving_beta = 0.5 # Beta
        self.regularisation_strength = 1.0 # This is lambda
        self.gamma = 0.1 # No idea for the correct value
        self.anchor_epochs = 100

        self.mean_feature_embeddings = None

    @staticmethod
    def get_algorithm_folder() -> str:
        return "hindsight_anchor_rew"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "apply_learning_rate_annealing": self.apply_learning_rate_annealing,
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": str(self.max_lr),
            "min_lr": str(self.min_lr),
            "cutmix_probability": self.cutmix_probability,
            "max_memory_per_class": self.max_memory_per_class
        }

        return info

    def _update_fifo_buffer(self, inp: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Pass NON-TRANSFORMED inputs to the buffer
        """
        for idx in range(len(inp)):
            raw_sample = inp[idx].detach().cpu().numpy()
            raw_target = targets[idx].detach().cpu().numpy().item()

            self.fifo_buffer.add_sample(raw_sample, raw_target)

    def _preprocess_data(
        self, 
        B_untransformed: torch.Tensor,
        B_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B_M_untransformed, B_M_targets = self.fifo_buffer.draw_sample(self.batch_size, transform=None)
        A_untransformed = torch.cat((B_untransformed, B_M_untransformed), dim=0)
        A_targets = torch.cat((B_targets, B_M_targets), dim=0).long()

        # logger.debug(f"B_untransformed: {B_untransformed.shape}, B_targets: {B_targets.shape}")
        # logger.debug(f"B_M_untransformed: {B_M_untransformed.shape}, B_M_targets: {B_M_targets.shape})")
        # logger.debug(f"A_untransformed: {A_untransformed.shape}, A_targets: {A_targets.shape})")
        
        B = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in B_untransformed]) # type: ignore
        B_M = None
        
        if B_M_untransformed.shape[0] != 0:
            B_M = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in B_M_untransformed]) # type: ignore
        
        A = torch.stack([self.dataset.training_transform(Image.fromarray(x.numpy().astype(np.uint8))) for x in A_untransformed]) # type: ignore
        
        # logger.debug(f"B_untransformed: {B.shape}")

        # if B_M is not None:
        #     logger.debug(f"B_M_untransformed: {B_M.shape}")

        # logger.debug(f"A_untransformed: {A.shape}")
        
        B_untransformed = B_untransformed.to(self.device)
        B_targets = B_targets.to(self.device)
        B_M_untransformed = B_M_untransformed.to(self.device)
        B_M_targets = B_M_targets.to(self.device)
        A_untransformed = A_untransformed.to(self.device)
        A_targets = A_targets.to(self.device)

        B = B.to(self.device)

        if B_M is not None:
            B_M = B_M.to(self.device)

        A = A.to(self.device)

        return B_untransformed, B_targets, B_M_untransformed, B_M_targets, A_untransformed, A_targets, B, B_M, A

    def _finetune_model(self):
        finetuned_model = copy.deepcopy(self.model).to(self.device)
        finetuned_model_opt = optim.SGD(finetuned_model.parameters(), lr=0.05)
        M_dataloader = torch.utils.data.DataLoader(
            dataset=self.fifo_buffer.to_torch_dataset(transform=self.dataset.training_transform),
            batch_size=self.batch_size,
            shuffle=True
        )

        for finetuning_epoch in range(self.finetuning_epochs):
            finetuning_running_loss = 0

            for finetuning_batch_no, (B_M, B_M_targets) in enumerate(M_dataloader):
                B_M = B_M.to(self.device)
                B_M_targets = B_M_targets.to(self.device)

                finetuned_model_opt.zero_grad()
                finetuned_predictions = finetuned_model(B_M)
                finetuned_loss = self.loss_criterion(finetuned_predictions, B_M_targets)
                finetuned_loss.backward()

                # Clip gradients
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(finetuned_model.parameters(), self.gradient_clip) # type: ignore
                
                finetuned_model_opt.step()
                finetuning_running_loss += finetuned_loss.item()

            if finetuning_epoch % 10 == 0 and finetuning_epoch != 0:
                logger.debug(f"Finetuning Epoch {finetuning_epoch}, RL: {finetuning_running_loss / 40}")
                finetuning_running_loss = 0
        
        return finetuned_model

    def _update_mean_embeddings(self, B, B_targets): 
        # Update the mean embedding
        frozen_feature_extractor = copy.deepcopy(self.model)
        frozen_feature_extractor.final = torch.nn.Identity()

        # Experiment with transformed vs untransformed
        for transformed_sample, target in zip(B, B_targets):
            target = target.detach().cpu().item()
            sample_features = frozen_feature_extractor(transformed_sample.unsqueeze(0).to(self.device)).squeeze(0).to(self.device)
            updated_mean_feature_embedding = (self.feature_embedding_moving_beta * self.mean_feature_embeddings[target] + (1 - self.feature_embedding_moving_beta) * sample_features) # type: ignore
            self.mean_feature_embeddings[target] = updated_mean_feature_embedding.detach() # type: ignore - must detach

    def _future_model_update(self, A, A_targets):
        # Copy the copy for a temporary update first
        future_model = copy.deepcopy(self.model).to(self.device)
        future_model_opt = optim.SGD(future_model.parameters(), lr=0.05)

        future_model_opt.zero_grad()
        future_predictions = future_model(A)
        future_loss = self.loss_criterion(future_predictions, A_targets)
        future_loss.backward()
        future_model_opt.step()

        return future_model

    def _confirm_new_anchors(
        self,
        detached_new_anchor_data: torch.Tensor, 
        anchor_data: Optional[torch.Tensor],
        new_anchor_classes: torch.Tensor,
        anchor_classes: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if anchor_data is None:
            assert anchor_classes is None
            return detached_new_anchor_data.to(self.device), new_anchor_classes.long().to(self.device)
        else:
            assert anchor_classes is not None
            return torch.cat([anchor_data, detached_new_anchor_data], dim=0).to(self.device), torch.cat([anchor_classes, new_anchor_classes], dim=0).long().to(self.device)

    def train(self) -> None:
        super().train()

        ### START OF PARAMETERS 
        # batch_size = 16 (see self.batch_size)
        # episodic_memory_size = 5 * 5 * 10 # mem_size * num_tasks * total_classes
        # feature_embedding_moving_beta = 0.9 # (see self.feature_embedding_moving_beta)
        # lambda = 1.0
        # gamma = ? 
        ### END OF PARAMETERS

        # Get the raw, untransformed data and their targets
        untransformed_task_datasets = {
            task_no: { 
                "data": self.dataset.task_datasets[task_no].data, # type: ignore
                "targets": self.dataset.task_datasets[task_no].targets # type: ignore
            } for task_no in range(len(self.dataset.task_datasets))
        }

        anchor_data = None
        anchor_classes = None

        current_task_anchors = None

        for task_no in range(self.dataset.task_count):
            self.model.train()
            current_task_anchors = torch.zeros([10, 3, 32, 32])

            if anchor_data is not None:
                assert anchor_classes is not None
                logger.debug(f"Anchor shape (d/t): {anchor_data.shape}, {anchor_classes.shape}")

            # Store the mean feature embedding for each class
            self.mean_feature_embeddings = torch.zeros([10, 512]).to(self.device)

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(self.dataset.task_splits[task_no])}")

            untransformed_task_dataset = datasets.CustomImageDataset(
                data=untransformed_task_datasets[task_no]["data"],
                targets=untransformed_task_datasets[task_no]["targets"],
                transform=None
            )

            untransformed_task_dataloader = torch.utils.data.DataLoader(
                dataset=untransformed_task_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            running_loss = 0
            running_anchor_loss = 0

            for batch_no, (B_untransformed, B_targets) in enumerate(untransformed_task_dataloader):
                # Preprocess the data into B, B_M, and A. Put all on the correct device.
                B_untransformed, B_targets, B_M_untransformed, B_M_targets, A_untransformed, A_targets, B, B_M, A = self._preprocess_data(B_untransformed, B_targets)

                # # Disable EPS_MEM
                # A = B
                # A_targets = B_targets

                if task_no == 0: 
                    # There are no anchors to train on for the first task!
                    self.optimiser.zero_grad()
                    predictions = self.model(A)
                    loss = self.loss_criterion(predictions, A_targets)
                    loss.backward()

                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()
                    running_loss += loss.item()

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.debug(f"Batch {batch_no}, RL: {running_loss / 40}")
                        running_loss = 0
                else:
                    # Now use the anchors as well
                    future_model = self._future_model_update(A, A_targets)
                    overall_loss = 0

                    self.optimiser.zero_grad()
                    traditional_predictions = self.model(A)
                    traditional_loss = self.loss_criterion(traditional_predictions, A_targets)
                    overall_loss += traditional_loss

                    future_anchor_predictions = future_model(anchor_data)
                    current_anchor_predictions = self.model(anchor_data)

                    anchor_loss = (current_anchor_predictions - future_anchor_predictions).square().mean(dim=1).sum()
                    running_anchor_loss += self.regularisation_strength * anchor_loss.item()
                    overall_loss += self.regularisation_strength * anchor_loss

                    overall_loss.backward()
                    
                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()
                    running_loss += overall_loss.item()

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.debug(f"Batch {batch_no}, RL: {running_loss / 40}, AL: {running_anchor_loss / 40}")
                        running_loss = 0
                        running_anchor_loss = 0

                # Update the mean embeddings
                self._update_mean_embeddings(B, B_targets)

                # Update the episodic memory FIFO buffer
                self._update_fifo_buffer(B_untransformed, B_targets)

                # Populate the unfilled anchors
                for sample, target in zip(B, B_targets): # B or B_untransformed?
                    current_task_anchors[target] = sample #.permute(2, 0, 1)

                    if batch_no == 0:
                        logger.debug(f"S: {sample.shape}, {sample.square().mean()}")

            # Just get the anchors for this task
            new_anchors = []
            new_anchor_classes = []

            for new_anchor_class, new_anchor in enumerate(current_task_anchors):
                # Skip those that are all zero i.e. classes that never appeared
                if torch.all(torch.eq(new_anchor, torch.zeros_like(new_anchor))):
                    continue
                    
                new_anchors.append(new_anchor)
                new_anchor_classes.append(torch.LongTensor([new_anchor_class]))

            new_anchors = torch.stack(new_anchors).to(self.device)
            new_anchor_classes = torch.stack(new_anchor_classes).to(self.device)

            ## ANCHOR TRAINING
            # Once the model has been updated, we finetune on the episodic memory
            finetuned_model = self._finetune_model()

            # self.model.eval()
            # self.run_base_task_metrics(task_no=100-task_no)
            # self.model.train()

            if anchor_data is not None:
                pre_opt_anchors = copy.deepcopy(anchor_data)
                torchvision.utils.save_image(pre_opt_anchors.detach().cpu(), f"pre_opt_{task_no}.png")
            else:
                pre_opt_anchors = None
            # anchor_opt = optim.SGD([anchor_data], lr=0.05)

            frozen_feature_extractor = copy.deepcopy(self.model)
            frozen_feature_extractor.final = torch.nn.Identity()
            frozen_model = copy.deepcopy(self.model).to(self.device)
            frozen_model.eval()
            finetuned_model.eval()

            optimisable_anchors = [anchor.detach().clone().to(self.device).requires_grad_() for anchor in new_anchors]
            logger.critical(f"OPT ANCHORS: {len(optimisable_anchors)}")
            anchor_optimisers = [optim.SGD([anchor], lr=0.05) for anchor in optimisable_anchors]
            running_avg_anch_loss = 0

            for anchor_epoch in range(self.anchor_epochs):
                epoch_loss = 0
                # extracted_features = frozen_feature_extractor(torch.stack(optimisable_anchors, dim=0))

                for anchor_idx, anchor in enumerate(optimisable_anchors):
                    anchor_opt = anchor_optimisers[anchor_idx]
                    anchor_class = new_anchor_classes[anchor_idx]#.squeeze()
                    individual_loss = 0

                    finetuned_prediction = finetuned_model(anchor.unsqueeze(0))
                    finetuned_loss = self.loss_criterion(finetuned_prediction, anchor_class)
                    individual_loss += finetuned_loss

                    actual_prediction = frozen_model(anchor.unsqueeze(0))
                    actual_loss = self.loss_criterion(actual_prediction, anchor_class)
                    individual_loss -= actual_loss

                    extracted_feature = frozen_feature_extractor(anchor.unsqueeze(0)).squeeze(0)
                    feature_loss = extracted_feature - self.mean_feature_embeddings[anchor_class]
                    feature_loss = feature_loss.square().mean()
                    individual_loss -= feature_loss

                    (-individual_loss).backward()
                    anchor_opt.step()
                    epoch_loss += individual_loss.item()
                
                running_avg_anch_loss += epoch_loss / len(optimisable_anchors)

                if anchor_epoch % 10 == 0:
                    logger.debug(f"Epoch {anchor_epoch}: {running_avg_anch_loss / 10}")
                    running_avg_anch_loss = 0
                elif anchor_epoch == 0:
                    running_avg_anch_loss = 0

            detached_anchors = [optimised_anchor.detach().clone().to(self.device) for optimised_anchor in optimisable_anchors]
            detached_new_anchor_data = torch.stack(detached_anchors).to(self.device)
            
            # Save the newly selected anchors
            anchor_data, anchor_classes = self._confirm_new_anchors(detached_new_anchor_data, anchor_data, new_anchor_classes, anchor_classes)

            for anchor in anchor_data:
                logger.debug(f"A: {anchor.shape}, {anchor.square().mean()}")

            # anchor_data.detach_()
            torchvision.utils.save_image(anchor_data.detach().cpu(), f"post_opt_{task_no}.png")

            self.model.eval()
            self.run_base_task_metrics(task_no=task_no)
            self.model.train()

        logger.info("Completed tasks, now training a model just on the anchors?")

        m = other_utils.get_gdumb_resnet_impl().to(self.device)
        m.train()
        o = optim.SGD(m.parameters(), lr=0.05)

        logger.debug(f"ANCHOR SHAPE: {anchor_data.shape}")
        logger.debug(f"ANCHOR CLASS SHAPE: {anchor_classes.squeeze().shape}")

        for epoch in range(5):
            o.zero_grad()
            pr = m(anchor_data)
            logger.info(f"pr={pr.detach().cpu()}")
            logger.info(f"ac: {anchor_classes.detach().cpu().squeeze()}")
            ls = self.loss_criterion(pr, anchor_classes.squeeze())
            ls.backward()
            o.step()

        self.model = m
        self.model.eval()
        self.run_base_task_metrics(task_no=10)

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
from typing import Dict, Union, Optional
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
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip

        self.apply_learning_rate_annealing = apply_learning_rate_annealing
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cutmix_probability = cutmix_probability

        self.max_memory_per_class = max_memory_per_class
        self.fifo_buffer = buffers.FIFORingReplayBuffer(max_memory_per_class=self.max_memory_per_class)

    @staticmethod
    def get_algorithm_folder() -> str:
        return "hindsight_anchor"

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
        for idx in range(len(inp)):
            raw_sample = inp[idx].detach().cpu().numpy()
            raw_target = targets[idx].detach().cpu().numpy().item()

            self.fifo_buffer.add_sample(raw_sample, raw_target)

    def train(self) -> None:
        super().train()

        anchors_per_class = 1
        raw_data_batch_size = 16
        max_exemplar_batch_size = 16

        # Targets = index
        anchors = torch.zeros([10, 3, 32, 32])
        next_anchors = torch.zeros([10, 3, 32, 32])
        averaging_beta = 0.9

        seen_classes = set()

        bct = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            seen_classes |= set(self.dataset.task_splits[task_no])

            # Reset the average image vectors
            ### Discard phi_t step
            average_mean_embedding = torch.zeros([512]).to(self.device) # Remove hardcoded values

            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")
            logger.info(f"Classes seen so far: {seen_classes}")

            raw_task_dataset = datasets.CustomImageDataset(self.dataset.task_datasets[task_no].data, self.dataset.task_datasets[task_no].targets, transform=None) # type: ignore
            raw_task_dataloader = torch.utils.data.DataLoader(raw_task_dataset, batch_size=raw_data_batch_size, shuffle=True)

            logger.debug("Raw task dataset and dataloader prepared")

            rl = 0

            for batch_no, data in enumerate(raw_task_dataloader):
                raw_inp, raw_inp_targets = data

                ### START: Sample B_M step from the algorithm
                # Pull some random exemplars here --> can merge like we did in a previous method
                exemplar_dataset = self.fifo_buffer.to_torch_dataset(transform=None)
                exemplar_batch_size = min(max_exemplar_batch_size, len(exemplar_dataset))
                exemplar_idxs = random.sample(range(len(exemplar_dataset)), k=exemplar_batch_size)
                exemplar_inp = torch.Tensor(np.array([exemplar_dataset.data[idx] for idx in exemplar_idxs]))
                exemplar_targets = torch.LongTensor(np.array([exemplar_dataset.targets[idx] for idx in exemplar_idxs]))

                if batch_no == 0:
                    logger.debug(f"Sampled exemplars, shapes: inp: {exemplar_inp.shape}, targets: {exemplar_targets.shape}")

                # Merge to form a whole batch
                inp_arr = [x.detach().numpy() for x in raw_inp] + [x.detach().numpy() for x in exemplar_inp] #  torch.concat((raw_inp, exemplar_inp), dim=0).to(self.device)
                targets_arr = [y for y in raw_inp_targets] + [y for y in exemplar_targets] # torch.concat((raw_inp_targets, exemplar_targets), dim=0).to(self.device)
                batch_ds = datasets.CustomImageDataset(inp_arr, targets_arr, transform=bct)

                inp = batch_ds.get_transformed_data().to(self.device)
                targets = batch_ds.targets.to(self.device)

                # logger.info(f"Inp: {inp.shape}")

                if batch_no == 0:
                    logger.debug(f"Concat shapes: inp: {inp.shape}, targets: {targets.shape}")

                ### END: SAMPLE B_M step from the algorithm

                if task_no == 0:
                    if batch_no == 0:
                        logger.debug("Normal training for a single epoch")

                    # Train the model without anchors (i.e. normal training)
                    self.optimiser.zero_grad()
                    predictions = self.model(inp)

                    loss = self.loss_criterion(predictions, targets)
                    loss.backward()

                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()
                    rl += loss.item()

                    if batch_no % 40 == 0:
                        logger.info(f"{batch_no}: {rl / 40}")
                        rl = 0
                else:
                    if batch_no == 0:
                        logger.debug("Anchored training for a single epoch")
                    # Train the model with anchors
                    # Firstly we have to do a temporary update to the model parameters
                    temp_learning_rate = self.optimiser.param_groups[0]['lr']
                    temp_model = copy.deepcopy(self.model).to(self.device)
                    temp_opt = torch.optim.SGD(temp_model.parameters(), lr=temp_learning_rate) # Assumes SGD
                    temp_opt.zero_grad()

                    temp_inp, temp_targets = copy.deepcopy(inp).to(self.device), copy.deepcopy(targets).to(self.device)
                    temp_predictions = temp_model(temp_inp)

                    temp_loss = self.loss_criterion(temp_predictions, temp_targets)
                    temp_loss.backward()
                    temp_opt.step()

                    # Now do the real update
                    self.optimiser.zero_grad()
                    predictions = self.model(inp)
                    loss = self.loss_criterion(predictions, targets)

                    total_anchor_loss = None
                    if batch_no == 0:
                        logger.debug(f"Whole anchor shape: {len(anchors)}")

                    # Disable batch norm layers
                    self.model.eval()
                    temp_model.eval()

                    for class_name in range(len(anchors)):
                        if class_name not in seen_classes:
                            if batch_no == 0:
                                logger.debug(f"Not seen {self.dataset.classes[class_name]}")
                            continue

                        # Don't optimise on this tasks anchors as they haven't been updated yet
                        if class_name in self.dataset.task_splits[task_no]:
                            continue

                        class_anchor = anchors[class_name].unsqueeze(0).to(self.device)
                        
                        if batch_no == 0:
                            logger.debug(f"Class Anchor shape: {class_anchor.shape}")

                        model_out = self.model(class_anchor)
                        temp_out = temp_model(class_anchor)

                        if batch_no == 0:
                            logger.debug(f"Out shapes: model: {model_out.shape}, temp: {temp_out.shape}")

                        anchor_loss = (model_out - temp_out).mean().square()

                        if total_anchor_loss is None:
                            total_anchor_loss = anchor_loss
                        else:
                            total_anchor_loss += anchor_loss
                    
                    self.model.train()
                    temp_model.train()

                    assert total_anchor_loss is not None, "No anchor loss"
                    anchor_weighting = 0.1
                    loss += anchor_weighting * total_anchor_loss

                    loss.backward()

                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()

                    rl += loss.item()

                    if batch_no % 40 == 0:
                        logger.info(f"{batch_no}: {rl / 40}")
                        rl = 0
            
                # Update average image vectors, only uses the raw data
                ### Update phi_t step
                frozen_feature_extractor = copy.deepcopy(self.model)
                frozen_feature_extractor.final = torch.nn.Identity()

                # for sample, target in zip(inp, targets): #raw inp!
                #     arr_idx = target.detach().cpu().item()
                #     feature_sample = frozen_feature_extractor(sample.unsqueeze(0).to(self.device)).squeeze(0).to(self.device)

                #     updated = averaging_beta * average_class_vectors[arr_idx] + (1 - averaging_beta) * ()

                #     updated = average_class_vectors[arr_idx] - (1 - averaging_alpha) * ( - feature_sample)
                #     average_class_vectors[arr_idx] = updated.detach() # detach needed to prevent unlimited memory growth

                #     if task_no == 0:
                #         # Randomly update the anchors if first task
                #         anchors[target] = sample.to(self.device)

                task_only_transformed_inp = inp[:len(raw_inp)]

                # Prepopulate the next anchors
                for sample, target in zip(task_only_transformed_inp, raw_inp_targets):
                    next_anchors[target.detach().cpu().item()] = sample.detach()#.to(self.device)

                ## Update phi_t
                feature_samples = frozen_feature_extractor(task_only_transformed_inp)

                for fs in feature_samples:
                    average_mean_embedding = (averaging_beta * average_mean_embedding + (1 - averaging_beta) * fs).detach()
                
                if batch_no == 0:
                    logger.info("Updated average class vectors")

                # Update FIFO buffer with the training data
                ### Update M by adding B to the FIFO Buffer
                self._update_fifo_buffer(raw_inp, raw_inp_targets)

                if batch_no % 40 == 0:
                    logger.debug(f"FIFO Keys: {self.fifo_buffer.known_classes}")
                    self.fifo_buffer.log_debug()
            
            self.model.eval()
            self.run_base_task_metrics(task_no=2*task_no)
            self.model.train()

            if task_no != self.dataset.task_count:
                logger.info("Starting anchor hindsight update")

                # Update the anchors with the hindsight update
                # Finetune the model
                mem_learning_rate = self.optimiser.param_groups[0]['lr']
                mem_model = copy.deepcopy(self.model).to(self.device)
                mem_opt = torch.optim.SGD(mem_model.parameters(), lr=mem_learning_rate) # Assumes SGD

                mem_epochs = 50
                mem_dataloader = torch.utils.data.DataLoader(self.fifo_buffer.to_torch_dataset(transform=bct), batch_size=16, shuffle=True)

                for mem_epoch in range(mem_epochs):
                    for mem_batch_no, mem_data in enumerate(mem_dataloader):
                        mem_inp, mem_targets = mem_data
                        mep_inp, mem_targets = mem_inp.to(self.device), mem_targets.to(self.device)

                        mem_opt.zero_grad()
                        mem_predictions = mem_model(mep_inp)

                        mem_loss = self.loss_criterion(mem_predictions, mem_targets)
                        mem_loss.backward()

                        if self.gradient_clip is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                        mem_opt.step()

                logger.info("Finetuned on episodic memory")
                logger.info("Starting anchor update")

                # Now update the anchors
                anchor_epochs = 100
                frozen_feature_extractor = copy.deepcopy(self.model)
                frozen_feature_extractor.final = torch.nn.Identity()

                mem_model.eval()
                self.model.eval()

                anchors = next_anchors
                next_anchors = torch.zeros([10, 3, 32, 32])

                anchor_update_alpha = 1e-3

                __copied_anchors = copy.deepcopy(anchors)

                updatable_anchors = nn.ParameterList()
                updatable_anchors_classes = []

                for class_name in range(len(anchors)):
                    if class_name not in seen_classes:
                        continue 

                    updatable_anchors.append(nn.Parameter(anchors[class_name].unsqueeze(0))) # type: ignore
                    updatable_anchors_classes.append(class_name)
                
                logger.debug(f"There are {len(updatable_anchors)} updatable anchors")
                
                updatable_anchors.to(self.device)
                anchor_opt = optim.SGD(updatable_anchors, lr=anchor_update_alpha)

                for anchor_epoch in range(anchor_epochs):
                    anchor_opt.zero_grad()
                    anc_loss = 0
                    
                    for updatable_anchor_idx in range(len(updatable_anchors)):
                        if anchor_epoch == 0:
                            logger.debug(f"First update for {updatable_anchor_idx}")

                        class_anchor = updatable_anchors[updatable_anchor_idx]
                        class_target = torch.LongTensor([updatable_anchors_classes[updatable_anchor_idx]]).to(self.device)
                        
                        mem_anchor_pred = mem_model(class_anchor)
                        anchor_loss = self.loss_criterion(mem_anchor_pred, class_target)
                        real_anchor_pred = self.model(class_anchor)#.detach()
                        anchor_loss -= self.loss_criterion(real_anchor_pred, class_target)
                        anchor_eta = 0.1
                        anchor_feature_diff = (frozen_feature_extractor(class_anchor) - average_mean_embedding).mean().square()
                        anchor_loss -= anchor_eta * anchor_feature_diff
                        anc_loss += anchor_loss

                    anc_loss *= -1 / len(updatable_anchors)
                    anc_loss.backward()

                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    anchor_opt.step()

                    logger.debug(f"AL {anchor_epoch}: {anc_loss.item()}")
                    # Also look at L175, do they have more anchors the older a task is?
                
                anchors = torch.zeros([10, 3, 32, 32])

                for anch, targ in zip(updatable_anchors, updatable_anchors_classes):
                    anchors[targ] = anch.detach()
                    
                mem_model.train()
                self.model.train()

                logger.info("Finished anchor updates")

                logger.critical(f"Does anch eq: {torch.all(__copied_anchors.eq(anchors))}")

            

            # Test the model
            self.model.eval()
            self.run_base_task_metrics(task_no=2*task_no + 1)
            self.model.train()
                    
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
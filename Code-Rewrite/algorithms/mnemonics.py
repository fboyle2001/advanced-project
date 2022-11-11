from typing import Dict, Union, Optional
from loguru import logger

from algorithms import utils, BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard

import torch.optim as optim
from datasets import CustomImageDataset
from torch.utils.data import DataLoader

import random

class Mnemonics(BaseCLAlgorithm):
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
        cutmix_probability: float
    ):
        super().__init__(
            name="Mnemonics",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip

        self.apply_learning_rate_annealing = apply_learning_rate_annealing
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cutmix_probability = cutmix_probability

        self.mnemonics_per_class = 20

    @staticmethod
    def get_algorithm_folder() -> str:
        return "mnemonics"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "apply_learning_rate_annealing": self.apply_learning_rate_annealing,
            "gradient_clip": self.gradient_clip if self.gradient_clip is not None else "disabled",
            "max_lr": str(self.max_lr),
            "min_lr": str(self.min_lr),
            "cutmix_probability": self.cutmix_probability
        }

        return info

    # This is the exemplar phase?
    def perform_mnemonics_memory_update(self, exemplars, exemplar_targets, task_samples, task_targets):
        # Take a subset of the task_samples to initialise as the exemplars for this task

        # Segment the data into the classes
        task_iter = zip(task_samples, task_targets) # D_i
        class_segmented_samples = {}

        for data, target in task_iter:
            if type(target) is torch.Tensor:
                target = target.detach().cpu().item()

            if target not in class_segmented_samples.keys():
                class_segmented_samples[target] = []

            assert type(data) is not torch.Tensor
            class_segmented_samples[target].append(data)

        # task_mnemonics = {}
        mnemonics_data = []
        mnemonics_targets = []

        # Draw a random subset from each class
        for class_name in class_segmented_samples.keys():
            sample_indices = random.sample(range(len(class_segmented_samples[class_name])), k=self.mnemonics_per_class)
            # task_mnemonics[class_name] = []

            for j in sample_indices:
                # We want them as Tensors
                # task_mnemonics[class_name].append(torch.tensor(class_segmented_samples[class_name][j]))

                mnemonics_data.append(class_segmented_samples[class_name][j])
                mnemonics_targets.append(class_name)
        
        # Now have the starting point for E_i
        # Create theta'_i without the final FC layer
        temporary_feature_model = torch.nn.Sequential(*list(self.model.children())[:-1])
        mnemonics_optimizer = optim.SGD(self.mnemonics, lr=self.args.mnemonics_outer_lr, momentum=0.9, weight_decay=5e-4) # type:ignore TODO

        mnemonics_training_epochs = 50 # self.args.mnemonics_total_epoch
        mnemonics_dataset = CustomImageDataset(mnemonics_data, mnemonics_targets, transform=self.dataset.training_transform)
        mnemonics_dataloader = DataLoader(mnemonics_dataset, batch_size=self.batch_size, shuffle=True)
        alpha_two = 0.01
        
        for epoch in range(mnemonics_training_epochs):
            for batch_no, data in enumerate(mnemonics_dataloader, 0):
                    inp, labels = data

                    # for ix in inp:
                    #     unique_imgs.add(hash(pickle.dumps(ix.detach().cpu())))

                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # # Apply cutmix
                    # apply_cutmix = random.uniform(0, 1) < self.cutmix_probability
                    # lam, labels_a, labels_b = None, None, None

                    # # Cannot merge the two if statements because inp will change causing issues in autograd
                    # if apply_cutmix: 
                    #     inp, labels_a, labels_b, lam = utils.cutmix_data(x=inp, y=labels, alpha=1.0)

                    mnemonics_optimizer.zero_grad()
                    predictions = self.model(inp)
                    
                    # if apply_cutmix: 
                    #     assert lam is not None and labels_a is not None and labels_b is not None
                    #     loss = lam * self.loss_criterion(predictions, labels_a) + (1 - lam) * self.loss_criterion(predictions, labels_b)
                    # else:
                    
                    loss = self.loss_criterion(predictions, labels)
                    loss.backward()

                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    mnemonics_optimizer.step()
                    # running_loss += loss.item()


    def train(self) -> None:
        """ 
        Authors combine theirs with LUCIR, going to go with GDumb as it makes more sense?
        Mnemonics just draws the exemplars and then we can use these to train the model instead?

        Mnemonics training alternates the learning of classification models and mnemonics exemplars across all phases
        Mnemonics exemplars are not just data samples but can be optimised and adjusted online
        Formulate as a global Bilevel Optimisation Problem (BOP) composed of model-level and exemplar-level problems

        Classification model is incrementally trained in each phase on the union of new class data and old class mnemonics
        **Omitting = data no longer available (in this paper)**
        Based on this model, the new class mnemonics (i.e. parameters of the exemplars) are trained before omitting new class data
        Optimality of model derives a constrain to optimising the exemplars
        Formulate this relationship with a global BOP in which each phase uses the optimal model to optimise exemplars and vice versa

        In the i-th phase, aim to learn a model Θ_i to approximate ideal Θ*_i which minimises L_c on both D_i and D_(0:i-1)
        (where L_c is classification loss, D_i is the data for the current task, D_(0:i-1) is the data for the previous tasks that is omitted)
        [see Eq 3 for the ideal sol]

        Since we only have access to E_(0:i-1) and D_i in the i-th phase, we approximate E_(0:i-1) towards the optimal replacement of D_(0:i-1)
        (where E_(0:i-1) are the mnemonics exemplars stored in memory that are accessible)
        [see Eq 4a (the model-level problem) and 4b (the exemplar-level problem)]

        First solve the model-level problem with the mnemonics exemplars with E_(0:i-1) as part of the input and previous Θ_(i-1) as the model init
        [see Eq 5, uses hyperparameter lambda]

        Then, Θ_i is used to train the parameters of the mnemonics exemplars i.e. to solve the exemplar-level problem
        Note that |E_(0:i-1)| << |D_i| typically
        Mnemonics exemplars are differentiable
        Train a temporary model Θ'_i on E_i to maximise the prediction on D_i
        Use D_i to compute a validation loss to penalize this temporary training with respect to the parameters of E_i
        Formulated as a local BOP where local means within a single phase
        [see Eq 7a (meta-level optimisation) and Eq 7b (base-level optimisation)]

        Training:
        Initialise parameters of E_i by a random sample subset S of D_i
        Initialise a temporary model Θ'_i using Θ_i and train Θ'_i on E_i for 'a few iterations' by gradient descent
        [see Eq 8, uses hyperparameter a_2]
        Both Θ'_i and E_i are differentiable so compute the loss of Θ'_i on D_i and backprop to optimise E_i
        [see Eq 9, uses hyperparameter b_1]

        The mnemonics exemplars of previous classes, E_(0:i) were trained when the class occurred
        Desirable to adjust them to the changing data distribution online
        Old class data D_(0:i-1) are not accessible (can't apply Eq 9)
        Instead, split E_(0:i-1) into two subsets E(A)_(0:i-1) and E(B)_(0:i-1) s.t. E = E(A) u E(B) [indices omitted]
        We use one, e.g. E(B)_(0:i-1), as the validation set in placed of D_(0:i-1) to optimise the other, e.g. E(A)_(0:i-1)
        This alternates to optimise both subsets
        [see Eq 10a and Eq 10b, uses hyperparameters b_2, also uses Eq 8 with E(A), E(B) subbed in]
        Propose to finetune Θ_i on E_i u E_(0:i) rather than D_i u E_(0:i) due to class imbalance as well
        """
        super().train()
        exemplar_data, exemplar_targets = [], []

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Starting Task {task_no + 1} / {len(self.dataset.task_datasets)}")
            current_task_dataset = self.dataset.task_datasets[task_no]

            # Update the memory
            logger.info("Updating memory...")

            self.model.eval()
            logger.debug(f"T_b: {type(exemplar_data[0]) if len(exemplar_data) != 0 else None}")
            exemplar_data, exemplar_targets = self.perform_mnemonics_memory_update(
                exemplar_data, exemplar_targets, current_task_dataset.data, current_task_dataset.targets # type: ignore
            )
            logger.debug(f"T_a: {type(exemplar_data[0])}")
            self.model.train()

            # For online version, generate the buffer at the start as we do a single pass on the data

            logger.info("Using memory buffer exclusively")
            exemplar_dataset = CustomImageDataset(exemplar_data, exemplar_targets, transform=self.dataset.training_transform)
            exemplar_dataloader = DataLoader(exemplar_dataset, batch_size=self.batch_size, shuffle=True)

            lr_warmer = None

            if self.apply_learning_rate_annealing:
                assert self.max_lr is not None and self.min_lr is not None, "Must set min and max LRs for annealing"
                lr_warmer = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=1, T_mult=2, eta_min=self.min_lr)
                logger.info("Annealing Scheduler setup")

            for epoch in range(1, self.epochs_per_task + 1):
                # Apply learning rate warmup if turned on
                if lr_warmer is not None and self.min_lr is not None and self.max_lr is not None:
                    if epoch == 0:
                        for param_group in self.optimiser.param_groups:
                            param_group['lr'] = self.max_lr * 0.1

                        self.writer.add_scalar("LR/Current_LR", self.max_lr * 0.1, task_no * self.epochs_per_task + epoch)
                    elif epoch == 1:
                        for param_group in self.optimiser.param_groups:
                            param_group['lr'] = self.max_lr

                        self.writer.add_scalar("LR/Current_LR", self.max_lr, task_no * self.epochs_per_task + epoch)
                    else:
                        lr_warmer.step()
                        self.writer.add_scalar("LR/Current_LR", lr_warmer.get_last_lr()[-1], task_no * self.epochs_per_task + epoch)

                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0

                for batch_no, data in enumerate(exemplar_dataloader, 0):
                    inp, labels = data

                    # for ix in inp:
                    #     unique_imgs.add(hash(pickle.dumps(ix.detach().cpu())))

                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # Apply cutmix
                    apply_cutmix = random.uniform(0, 1) < self.cutmix_probability
                    lam, labels_a, labels_b = None, None, None

                    # Cannot merge the two if statements because inp will change causing issues in autograd
                    if apply_cutmix: 
                        inp, labels_a, labels_b, lam = utils.cutmix_data(x=inp, y=labels, alpha=1.0)

                    self.optimiser.zero_grad()
                    predictions = self.model(inp)
                    
                    if apply_cutmix: 
                        assert lam is not None and labels_a is not None and labels_b is not None
                        loss = lam * self.loss_criterion(predictions, labels_a) + (1 - lam) * self.loss_criterion(predictions, labels_b)
                    else:
                        loss = self.loss_criterion(predictions, labels)
                    
                    loss.backward()

                    # Clip gradients
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # type: ignore

                    self.optimiser.step()

                    running_loss += loss.item()

                epoch_offset = self.epochs_per_task * task_no

                avg_running_loss = running_loss / (len(exemplar_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)

                running_loss = 0
        
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        return super().classify(batch)
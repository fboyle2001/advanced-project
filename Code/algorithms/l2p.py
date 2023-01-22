from typing import Dict, Union, List
from loguru import logger

from .algorithm_base import BaseCLAlgorithm
import datasets

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch.optim as optim

import models.vit.vit_models as vit_models

class LearningToPrompt(BaseCLAlgorithm):
    """
    Learning to Prompt introduces the use of a pretrained vision transformer and 
    the idea of trainable prompts from Natural Language Processing to guide the model.

    Reference: Wang, Zifeng, et al. 2022 "Learning to prompt for continual learning."
    """

    valid_prompt_freq_strategies: List[str] = ["disabled", "minmax", "scaled_frequency"]

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: datasets.BaseCLDataset,
        optimiser: torch.optim.Optimizer,
        loss_criterion: torch.nn.modules.loss._Loss,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        epochs_per_task: int,
        batch_size: int,
        K_lr: float,
        P_lr: float,
        g_phi_lr: float,
        N: int,
        L_p: int,
        M: int,
        balancing_lambda: float,
        prompt_frequency_strategy: str
    ):
        super().__init__(
            name="L2P",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.optimiser = None
        self.model = None

        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size

        # Hyperparameters
        self.K_lr = K_lr
        self.P_lr = P_lr
        self.g_phi_lr = g_phi_lr
        self.balancing_lambda = balancing_lambda
        self.N = N # prompt selection length
        self.L_p = L_p # prompt token length
        self.M = M # total number of prompts

        # Create and load the pretrained model
        self.pretrained_vit = vit_models.create_model().to(self.device)
        self.D = self.pretrained_vit.feat_dim # number of features
        self.D_k = self.D
        self.g_phi = AvgPoolClassifier(self.D, len(self.dataset.classes)).to(self.device)
        self.prompt_frequency = torch.zeros(self.M).to(self.device)
        assert prompt_frequency_strategy in self.valid_prompt_freq_strategies, "Invalid prompt frequency strategy"
        self.prompt_frequency_strategy = prompt_frequency_strategy

        # Initialise the keys
        self.prompt_keys = [
            torch.nn.parameter.Parameter(((-0.0625 - 0.0625) * torch.rand(self.D_k) + 0.0625).to(self.device), requires_grad=True)
            for _ in range(self.M)
        ]

        # Each key has its own optimiser
        self.K_opts = [
            optim.Adam([K], lr=self.K_lr, betas=(0.9, 0.999))
            for K in self.prompt_keys
        ]

        # Initialise the prompts
        self.prompts = [
            torch.nn.parameter.Parameter(((-0.0625 - 0.0625) * torch.rand(self.L_p, self.D) + 0.0625).to(self.device), requires_grad=True)
            for _ in range(self.M)
        ]

        # Each prompt has its own optimiser
        self.P_opts = [
            optim.Adam([P], lr=self.P_lr, betas=(0.9, 0.999))
            for P in self.prompts
        ]

        self.g_phi_opt = optim.Adam(self.g_phi.parameters(), lr=self.g_phi_lr, betas=(0.9, 0.999))

    @staticmethod
    def get_algorithm_folder() -> str:
        return "l2p"

    def get_unique_information(self) -> Dict[str, Union[str, int, float]]:
        info: Dict[str, Union[str, int, float]] = {
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "K_lr": self.K_lr,
            "P_lr": self.P_lr,
            "g_phi_lr": self.g_phi_lr,
            "D": self.D,
            "D_k": self.D_k,
            "N": self.N,
            "L_p": self.L_p,
            "M": self.M,
            "balancing_lambda": self.balancing_lambda,
            "prompt_frequency_strategy": self.prompt_frequency_strategy
        }

        return info

    def gamma(self, sample, compare):
        # Computes the similarity between a sample and a reference value (i.e. a key)
        query_features, _ = self.pretrained_vit.enc.transformer(sample)
        query_features = query_features[:, 0]
        # print("q", query_features.shape)
        sim_calc = torch.nn.CosineSimilarity()
        return 1 - sim_calc(query_features, compare)

    def calculate_top_N_keys(self, sample, weights):
        # qf should be 1x768
        # k should be Mx768
        # Selects the top N keys to draw the corresponding prompts
        compare = torch.cat([x.unsqueeze(0) for x in self.prompt_keys], dim=0)
        # Apply the frequency weighting to stop some prompts dominating
        distance = self.gamma(sample, compare) * weights
        indices = [i for i in range(self.M)]
        argmin = sorted(indices, key=lambda i: distance[i])[:self.N]
        top_N_keys = [self.prompt_keys[i] for i in argmin] # type: ignore
        return argmin, top_N_keys

    def f_e(self, img_batch):
        # f_e is the embedding layer
        embeddings = self.pretrained_vit.enc.transformer.embeddings(img_batch)
        return embeddings

    def f_r(self, embedding, prompts): 
        # f_r is the self-attention layers
        # input should have the format [0, 1:prompts, prompts+1:]
        # prompts = prompts.reshape(-1, self.D)
        prompts = prompts.reshape((-1, self.D))
        prompt_prepended_embeddings = torch.cat([embedding[:1, :], prompts, embedding[1:, :]], dim=0)
        prompt_prepended_embeddings = prompt_prepended_embeddings.unsqueeze(0)
        encoded, attn_weights = self.pretrained_vit.enc.transformer.encoder(prompt_prepended_embeddings)
        return encoded

    def train(self) -> None:
        super().train()
        torch.set_printoptions(sci_mode=False)

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            # Count the frequency that each prompt is selected
            previous_prompt_freq = torch.ones_like(self.prompt_frequency).to(self.device)

            # Optionally weight the prompts to prevent overfitting to certain prompts
            if task_no > 0 and self.prompt_frequency_strategy != "disabled":
                if self.prompt_frequency_strategy == "minmax":
                    ppf_min = torch.min(self.prompt_frequency)
                    ppf_max = torch.max(self.prompt_frequency)

                    previous_prompt_freq = (self.prompt_frequency - ppf_min) / (ppf_max - ppf_min)
                elif self.prompt_frequency_strategy == "scaled_frequency":
                    previous_prompt_freq = self.prompt_frequency / self.prompt_frequency.sum()
                else:
                    assert False, ""

            for epoch in range(1, self.epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0
                short_running_loss = 0
                ce_l = 0
                ky_l = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    inp, labels = data

                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    # Get the feature embeddings of the input
                    xs_e = self.f_e(inp)

                    selected_key_indices = set()

                    batch_loss = 0

                    # Process the samples individually
                    for i, img in enumerate(inp):
                        img = img.unsqueeze(0)
                        # Draw the prompts
                        top_N_indices, top_N_keys = self.calculate_top_N_keys(img, previous_prompt_freq)
                        selected_key_indices |= set(top_N_indices)

                        # x_p = torch.cat([xs_e[i, :1, :], self.prompts[top_N_keys], xs_e[i, 1:, :]], dim=1)
                        selected_prompts = torch.cat([self.prompts[i].unsqueeze(0) for i in top_N_indices], dim=0)

                        # Update the frequency
                        for prompt_index in top_N_indices:
                            self.prompt_frequency[prompt_index] += 1

                        # Prepend prompts and get the feature representations
                        x_p = self.f_r(xs_e[i], selected_prompts)
                        x_p = x_p[:, 0:(self.L_p * self.N + 1)]

                        # Get the output prediction
                        prediction = self.g_phi(x_p)
                        ce_loss = self.loss_criterion(prediction, labels[i].unsqueeze(0))
                        key_loss = self.balancing_lambda * self.gamma(img, torch.cat([x.unsqueeze(0) for x in top_N_keys], dim=0)).sum()
                        loss_x = ce_loss + key_loss
                        batch_loss += loss_x
                        ce_l += ce_loss.item()
                        ky_l += key_loss.item()

                    running_loss += batch_loss.item() # type: ignore
                    short_running_loss += batch_loss.item() # type: ignore

                    # Update phi
                    self.g_phi_opt.zero_grad()

                    for seen_key in selected_key_indices:
                        # Update K
                        self.K_opts[seen_key].zero_grad()
                        # Update P 
                        self.P_opts[seen_key].zero_grad()

                    batch_loss.backward() # type: ignore
                    
                    for seen_key in selected_key_indices:
                        # Update K
                        self.K_opts[seen_key].step()
                        # Update P 
                        self.P_opts[seen_key].step()

                    # Update phi
                    self.g_phi_opt.step()

                    # Logging
                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.info(f"{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                        logger.info(f"KL: {ky_l / 40}, CE: {ce_l / 40}")
                        logger.info(f"Prev Freq: {previous_prompt_freq}")
                        logger.info(f"Freq: {self.prompt_frequency}")

                        for i, key in enumerate(self.prompt_keys):
                            detached = key.detach().clone().cpu()
                            k_mean = detached.mean().item()
                            k_min = detached.min().item()
                            k_max = detached.max().item()
                            k_sum = detached.sum().item()

                            logger.debug(f"Key {i} (mean, min, max, sum): {k_mean}, {k_min}, {k_max}, {k_sum}")

                        for i, prompt in enumerate(self.prompts):
                            detached = prompt.detach().clone().cpu()
                            p_mean = detached.mean().item()
                            p_min = detached.min().item()
                            p_max = detached.max().item()
                            p_sum = detached.sum().item()

                            logger.debug(f"Prompt {i} (mean, min, max, sum): {p_mean}, {p_min}, {p_max}, {p_sum}")

                        short_running_loss = 0
                        ce_l = 0
                        ky_l = 0

                epoch_offset = self.epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)
                running_loss = 0

            # Evaluate the model
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        # Can't use simple maximisation classification 
        # Need to draw prompts
        predictions = []
        xs_e = self.f_e(batch)

        weights = torch.ones(self.M).to(self.device)

        # Count the prompt frequency applied to the unseen data
        prompt_occurs = torch.zeros(10)

        for i, img in enumerate(batch):
            # Draw the prompts and classify
            img = img.unsqueeze(0)
            top_N_indices, top_N_keys = self.calculate_top_N_keys(img, weights)

            for index in top_N_indices:
                prompt_occurs[index] += 1

            # x_p = torch.cat([xs_e[i, :1, :], self.prompts[top_N_keys], xs_e[i, 1:, :]], dim=1)
            selected_prompts = torch.cat([self.prompts[i].unsqueeze(0) for i in top_N_indices], dim=0)
            x_p = self.f_r(xs_e[i], selected_prompts)
            x_p = x_p[:, 0:(self.L_p * self.N + 1)]
            _, prediction = torch.max(self.g_phi(x_p), 1)
            predictions.append(prediction)
        
        logger.debug(f"Selected prompts: {prompt_occurs}")
        return torch.cat(predictions, dim=0)

class AvgPoolClassifier(nn.Module):
    """
    Final classification head, 1D Avg Pool followed by Linear layer
    """
    def __init__(self, feature_size, classes):
        super().__init__()
        self.classifier = nn.Linear(in_features=feature_size, out_features=classes)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x.squeeze(dim=1)

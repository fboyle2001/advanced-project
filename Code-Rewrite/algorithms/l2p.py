from typing import Dict, Union, Optional
from loguru import logger

from . import utils
from .algorithm_base import BaseCLAlgorithm

import torch
import datasets
import torch.utils.tensorboard
import torch.optim as optim

import vit.vit_models

import random

class LearningToPrompt(BaseCLAlgorithm):
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
            name="L2P",
            model_instance=model,
            dataset_instance=dataset,
            optimiser_instance=optimiser,
            loss_criterion_instance=loss_criterion,
            writer=writer
        )

        self.epochs_per_task = 1
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip

        self.apply_learning_rate_annealing = apply_learning_rate_annealing
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cutmix_probability = cutmix_probability
        self.pretrained_vit = vit.vit_models.create_model().to(self.device)
        self.D = self.pretrained_vit.feat_dim # number of features
        self.D_k = self.D
        self.N = 2 # prompt selection length
        self.L_p = 5 # prompt token length
        self.M = 10 # total number of prompts
        self.g_phi = AvgPoolClassifier(self.D, 10).to(self.device)
        self.balancing_lambda = 0.5
        self.prompt_frequency = torch.zeros(self.M).to(self.device)
        self.prompt_frequency_strategy = ["minmax", "scaled_frequency"][0]

        self.prompt_keys = [
            torch.nn.parameter.Parameter(((-0.0625 - 0.0625) * torch.rand(self.D_k) + 0.0625).to(self.device), requires_grad=True)
            for _ in range(self.M)
        ]

        self.K_opts = [
            optim.Adam([K], lr=1e-3, betas=(0.9, 0.999))
            for K in self.prompt_keys
        ]

        self.prompts = [
            torch.nn.parameter.Parameter(((-0.0625 - 0.0625) * torch.rand(self.L_p, self.D) + 0.0625).to(self.device), requires_grad=True) # maybe should be upping this to L_p, D instead? -> yes!
            for _ in range(self.M)
        ]

        self.P_opts = [
            optim.Adam([P], lr=1e-3, betas=(0.9, 0.999))
            for P in self.prompts
        ]

    @staticmethod
    def get_algorithm_folder() -> str:
        return "l2p"

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

    def gamma(self, sample, compare):
        query_features, _ = self.pretrained_vit.enc.transformer(sample)
        query_features = query_features[:, 0]
        # print("q", query_features.shape)
        sim_calc = torch.nn.CosineSimilarity()
        return 1 - sim_calc(query_features, compare)

    def calculate_top_N_keys(self, sample, weights):
        # qf should be 1x768
        # k should be Mx768
        compare = torch.cat([x.unsqueeze(0) for x in self.prompt_keys], dim=0)
        # print("C", compare.shape)
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
        g_phi_opt = optim.Adam(self.g_phi.parameters(), lr=1e-3, betas=(0.9, 0.999))

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            previous_prompt_freq = torch.ones_like(self.prompt_frequency).to(self.device)

            if task_no > 0:
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

                    xs_e = self.f_e(inp)

                    selected_key_indices = set()

                    batch_loss = 0

                    for i, img in enumerate(inp):
                        img = img.unsqueeze(0)
                        top_N_indices, top_N_keys = self.calculate_top_N_keys(img, previous_prompt_freq)
                        selected_key_indices |= set(top_N_indices)

                        # x_p = torch.cat([xs_e[i, :1, :], self.prompts[top_N_keys], xs_e[i, 1:, :]], dim=1)
                        selected_prompts = torch.cat([self.prompts[i].unsqueeze(0) for i in top_N_indices], dim=0)

                        for prompt_index in top_N_indices:
                            self.prompt_frequency[prompt_index] += 1

                        x_p = self.f_r(xs_e[i], selected_prompts)
                        x_p = x_p[:, 0:(self.L_p * self.N + 1)]
                        prediction = self.g_phi(x_p)
                        ce_loss = self.loss_criterion(prediction, labels[i].unsqueeze(0))
                        key_loss = self.balancing_lambda * self.gamma(img, torch.cat([x.unsqueeze(0) for x in top_N_keys], dim=0)).sum()
                        loss_x = ce_loss + key_loss
                        batch_loss += loss_x
                        ce_l += ce_loss.item()
                        ky_l += key_loss.item()


                    running_loss += batch_loss.item() # type: ignore
                    short_running_loss += batch_loss.item() # type: ignore
                    
                    copied_keys = {i: self.prompt_keys[i].detach().clone() for i in selected_key_indices}
                    copied_prompts = {i: self.prompts[i].detach().clone() for i in selected_key_indices}

                    # Update phi
                    g_phi_opt.zero_grad()

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
                    g_phi_opt.step()

                    # for seen_key in selected_key_indices:
                    #     logger.debug(f"K {seen_key}: {not torch.all(torch.eq(copied_keys[seen_key], self.prompt_keys[seen_key]))}")
                    #     logger.debug(f"P {seen_key}: {not torch.all(torch.eq(copied_prompts[seen_key], self.prompts[seen_key]))}")

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.info(f"{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                        logger.info(f"KL: {ky_l / 40}, CE: {ce_l / 40}")
                        logger.info(f"Prev Freq: {previous_prompt_freq}")
                        logger.info(f"Freq: {self.prompt_frequency}")
                        short_running_loss = 0
                        ce_l = 0
                        ky_l = 0
                    
                    # logger.debug(f"Keys: {sorted(selected_key_indices)}")

                epoch_offset = self.epochs_per_task * task_no

                avg_running_loss = running_loss / (len(task_dataloader) - 1)
                logger.info(f"{epoch}, loss: {avg_running_loss:.3f}")
                self.writer.add_scalar(f"Loss/Task_{task_no + 1}_Total_avg", avg_running_loss, epoch)
                self.writer.add_scalar("Loss/Overall_Total_avg", avg_running_loss, epoch_offset + epoch)
                running_loss = 0
        
            self.run_base_task_metrics(task_no)
        
        logger.info("Training complete")

    def classify(self, batch: torch.Tensor) -> torch.Tensor:
        predictions = []
        xs_e = self.f_e(batch)

        weights = torch.ones(self.M).to(self.device)
        prompt_occurs = torch.zeros(10)

        for i, img in enumerate(batch):
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

import torch.nn as nn

class AvgPoolClassifier(nn.Module):
    def __init__(self, feature_size, classes):
        super().__init__()
        self.classifier = nn.Linear(in_features=feature_size, out_features=classes)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x.squeeze(dim=1)

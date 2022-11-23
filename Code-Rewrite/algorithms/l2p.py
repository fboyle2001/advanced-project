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
        self.N = 5 # prompt subset length
        self.M = 10 # total number of prompts
        self.g_phi = AvgPoolClassifier(self.N, self.D, 10).to(self.device)
        self.balancing_lambda = 0.5

        self.prompt_keys = [
            torch.nn.parameter.Parameter(torch.randn(self.D).to(self.device), requires_grad=True)
            for _ in range(self.M)
        ]

        self.K_opts = [
            optim.Adam([K], lr=0.03, betas=(0.9, 0.999))
            for K in self.prompt_keys
        ]

        self.prompts = [
            torch.nn.parameter.Parameter(torch.randn(self.D).to(self.device), requires_grad=True)
            for _ in range(self.M)
        ]

        self.P_opts = [
            optim.Adam([P], lr=0.03, betas=(0.9, 0.999))
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

    def calculate_top_N_keys(self, sample):
        # qf should be 1x768
        # k should be Mx768
        compare = torch.cat([x.unsqueeze(0) for x in self.prompt_keys], dim=0)
        # print("C", compare.shape)
        distance = self.gamma(sample, compare)
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
        prompt_prepended_embeddings = torch.cat([embedding[:1, :], prompts, embedding[1:, :]], dim=0)
        prompt_prepended_embeddings = prompt_prepended_embeddings.unsqueeze(0)
        encoded, attn_weights = self.pretrained_vit.enc.transformer.encoder(prompt_prepended_embeddings)
        return encoded

    def train(self) -> None:
        super().train()
        g_phi_opt = optim.Adam(self.g_phi.parameters(), lr=0.03, betas=(0.9, 0.999))

        for task_no, (task_indices, task_dataloader) in enumerate(self.dataset.iterate_task_dataloaders(batch_size=self.batch_size)):
            logger.info(f"Task {task_no + 1} / {self.dataset.task_count}")
            logger.info(f"Classes in task: {self.dataset.resolve_class_indexes(task_indices)}")

            for epoch in range(1, self.epochs_per_task + 1):
                logger.info(f"Starting epoch {epoch} / {self.epochs_per_task}")
                running_loss = 0
                short_running_loss = 0

                for batch_no, data in enumerate(task_dataloader, 0):
                    inp, labels = data

                    inp = inp.to(self.device)
                    labels = labels.to(self.device)

                    xs_e = self.f_e(inp)

                    selected_key_indices = set()

                    batch_loss = 0

                    for i, img in enumerate(inp):
                        img = img.unsqueeze(0)
                        top_N_indices, top_N_keys = self.calculate_top_N_keys(img)
                        selected_key_indices |= set(top_N_indices)

                        # x_p = torch.cat([xs_e[i, :1, :], self.prompts[top_N_keys], xs_e[i, 1:, :]], dim=1)
                        selected_prompts = torch.cat([self.prompts[i].unsqueeze(0) for i in top_N_indices], dim=0)
                        x_p = self.f_r(xs_e[i], selected_prompts)
                        x_p = x_p[:, 0:(self.N + 1)]
                        prediction = self.g_phi(x_p)
                        ce_loss = self.loss_criterion(prediction, labels[i].unsqueeze(0))
                        loss_x = ce_loss + self.balancing_lambda * self.gamma(img, torch.cat([x.unsqueeze(0) for x in top_N_keys], dim=0)).sum()
                        batch_loss += loss_x

                    running_loss += batch_loss.item() # type: ignore
                    short_running_loss += batch_loss.item() # type: ignore
                    
                    # Update phi
                    g_phi_opt.zero_grad()

                    for seen_key in selected_key_indices:
                        # Update K
                        self.K_opts[seen_key].step()
                        # Update P 
                        self.P_opts[seen_key].step()

                    batch_loss.backward() # type: ignore
                    
                    for seen_key in selected_key_indices:
                        # Update K
                        self.K_opts[seen_key].step()
                        # Update P 
                        self.P_opts[seen_key].step()

                    # Update phi
                    g_phi_opt.step()

                    if batch_no % 40 == 0 and batch_no != 0:
                        logger.info(f"{epoch}:{batch_no}, loss: {short_running_loss / 40:.3f}")
                        short_running_loss = 0

                epoch_offset = self.epochs_per_task * task_no
                self.run_base_task_metrics(epoch)

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

        for i, img in enumerate(batch):
            img = img.unsqueeze(0)
            top_N_indices, top_N_keys = self.calculate_top_N_keys(img)

            # x_p = torch.cat([xs_e[i, :1, :], self.prompts[top_N_keys], xs_e[i, 1:, :]], dim=1)
            selected_prompts = torch.cat([self.prompts[i].unsqueeze(0) for i in top_N_indices], dim=0)
            x_p = self.f_r(xs_e[i], selected_prompts)
            x_p = x_p[:, 0:(self.N + 1)]
            _, prediction = torch.max(self.g_phi(x_p), 1)
            predictions.append(prediction)
        
        return torch.cat(predictions, dim=0)

import torch.nn as nn

class AvgPoolClassifier(nn.Module):
    def __init__(self, prompt_size, feature_size, classes):
        super().__init__()
        out_f = feature_size // (prompt_size + 1)
        self.pool = nn.AvgPool2d(kernel_size=prompt_size + 1)
        self.classifier = nn.Linear(in_features=out_f, out_features=classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.classifier(x)
        return x.squeeze(dim=1)

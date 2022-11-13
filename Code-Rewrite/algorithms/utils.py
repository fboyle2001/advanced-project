# Cutmix, implemented by GDumb
# https://github.com/drimpossible/GDumb/blob/ca38afcec332fa523ceff0cc8d3846e2bcf78697/src/utils.py
# Taken from official implementation
from loguru import logger
from typing import Union
import copy

import numpy as np
import torch

import torchvision.models
import models.cifar.resnet

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.intc(W * cut_rat)
    cut_h = np.intc(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5, device="cuda:0"):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    index = index.to(device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def strip_to_feature_extractor(model: Union[models.cifar.resnet.ResNet, torchvision.models.ResNet], copy_model: bool):
    if type(model) is models.cifar.resnet.ResNet:
        logger.debug(f"Model type is models.cifar.resnet.ResNet")

        if copy:
            model = copy.deepcopy(model)
        
        model.final = torch.nn.Identity() # type: ignore
        return model
    elif type(model) is torchvision.models.ResNet:
        # https://stackoverflow.com/a/52548419
        logger.debug(f"Model type is torchvision.models.ResNet")

        if copy:
            model = copy.deepcopy(model)

        return torch.nn.Sequential(*(list(model.children())[:-1]))
    else:
        logger.critical(f"Invalid model type {type(model)}")
        assert 1==0
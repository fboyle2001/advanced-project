import random
import numpy as np
import torch
import os

from models.cifar.resnet import ResNet
from dotmap import DotMap

# From GDumb
def seed_everything(seed):
    
    '''
    Fixes the class-to-task assignments and most other sources of randomness, except CUDA training aspects.
    '''
    # Avoid all sorts of randomness for better replication
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # type: ignore

def get_gdumb_resnet_impl(reduced=False):
    opt = {
        "depth": 18,
        "num_classes": 10,
        "bn": True,
        "preact": False,
        "normtype": "BatchNorm",
        "affine_bn": True, 
        "bn_eps": 1e-6,
        "activetype": "ReLU",
        "in_channels": 3
    }
    
    model = ResNet(DotMap(opt), reduced=reduced)
    return model
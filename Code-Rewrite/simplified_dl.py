import torch, torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random, copy
import argparse
import numpy as np

from dotmap import DotMap
from loguru import logger

import time
import json

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class VisionDataset(object):
    """
    Code to load the dataloaders for the storage memory (implicitly performs greedy sampling) for training GDumb. Should be easily readable and extendable to any new dataset.
    Should generate class_mask, cltrain_loader, cltest_loader; with support for pretraining dataloaders given as pretrain_loader and pretest_loader.
    """
    def __init__(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        self.total_num_classes = 10
        self.inp_size = 32
        self.in_channels = 3

        self.workers = 0
        self.batch_size = 16
        self.dataset = "CIFAR10"
        self.data_dir = "./store/data"
        self.num_tasks = 5
        self.num_classes_per_task = 2
        self.memory_size = 1000
        self.num_pretrain_classes = 0
        
        self.opt = DotMap({
            "workers": self.workers,
            "batch_size": self.batch_size,
            "dataset": self.dataset,
            "data_dir": self.data_dir,
            "num_tasks": self.num_tasks,
            "num_classes_per_task": self.num_classes_per_task,
            "memory_size": self.memory_size,
            "num_pretrain_classes": self.num_pretrain_classes,
            "total_num_classes": self.total_num_classes,
            "inp_size": self.inp_size,
            "in_channels": self.in_channels
        })

        # Sets parameters of the dataset. For adding new datasets, please add the dataset details in `get_statistics` function.
        # _, _, opt.total_num_classes, opt.inp_size, opt.in_channels = get_statistics(self.dataset)
        
        self.class_order = None

        # Generates the standard data augmentation transforms
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(self.inp_size, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean, std)
        ])

        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean, std)
        ])

        # Creates the supervised baseline dataloader (upper bound for continual learning methods)
        self.supervised_trainloader = self.get_loader(indices=None, transforms=self.train_transforms, train=True)
        self.supervised_testloader = self.get_loader(indices=None, transforms=self.test_transforms, train=False)

    def get_loader(self, indices, transforms, train, shuffle=True, target_transforms=None):
        sampler = None

        if indices is not None: 
            if shuffle and train:
                sampler = SubsetRandomSampler(indices)  
            else:
                sampler = SubsetSequentialSampler(indices)       
        
        return DataLoader(torchvision.datasets.CIFAR10(
            root=self.opt.data_dir, train=train, download=True, transform=transforms, target_transform=target_transforms
        ), sampler=sampler, num_workers=0, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def gen_cl_mapping(self):
        ## START OF CLASS BINNING
        # This part here is simply dividing the items into their class bins
        train_class_labels_dict = classwise_split(targets=self.supervised_trainloader.dataset.targets) # type: ignore
        dumpable_trcld = {int(key): [int(x) for x in train_class_labels_dict[key]] for key in train_class_labels_dict.keys()}

        with open("train_class_labels_dict.json", "w+") as f:
            json.dump(dumpable_trcld, f, indent=2)

        test_class_labels_dict = classwise_split(targets=self.supervised_testloader.dataset.targets) # type: ignore
        dumpable_tecld = {int(key): [int(x) for x in test_class_labels_dict[key]] for key in test_class_labels_dict.keys()}

        with open("test_class_labels_dict.json", "w+") as f:
            json.dump(dumpable_tecld, f, indent=2)

        ## END OF CLASS BINNING

        # Sets classes to be 0 to n-1 if class order is not specified, else sets it to class order. To produce different effects tweak here.
        cl_class_list = list(range(self.opt.total_num_classes))
        random.shuffle(cl_class_list) # Generates different class-to-task assignment
        
        self.class_mask = torch.from_numpy(np.kron(np.eye(self.opt.num_tasks,dtype=int),np.ones((self.opt.num_classes_per_task,self.opt.num_classes_per_task)))).cuda() #Generates equal num_classes for all tasks. 
        continual_target_transform = ReorderTargets(cl_class_list)  # Remaps the class order to a 0-n order, required for crossentropy loss using class list
        trainidx, testidx = [], []
        mem_per_cls = self.opt.memory_size//(self.opt.num_classes_per_task*self.opt.num_tasks)
        for cl in cl_class_list: # Selects classes from the continual learning list and loads memory and test indices, which are then passed to a subset sampler
            num_memory_samples = min(len(train_class_labels_dict[cl][:]), mem_per_cls)
            trainidx += train_class_labels_dict[cl][:num_memory_samples] # This is class-balanced greedy sampling (Selects the first n samples).
            testidx += test_class_labels_dict[cl][:]
        assert(len(trainidx) <= self.opt.memory_size), "ERROR: Cannot exceed max. memory samples!"
        self.cltrain_loader = self.get_loader(indices=trainidx, transforms=self.train_transforms, train=True, target_transforms=continual_target_transform)
        self.cltest_loader = self.get_loader(indices=testidx, transforms=self.test_transforms, train=False, target_transforms=continual_target_transform)


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)



class ReorderTargets(object):
    """
    Converts the class-orders to 0 -- (n-1) irrespective of order passed.
    """
    def __init__(self, class_order):
        self.class_order = np.array(class_order) 

    def __call__(self, target):
        return np.where(self.class_order==target)[0][0]

        
def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        indices (sequence): a sequence of indices
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict
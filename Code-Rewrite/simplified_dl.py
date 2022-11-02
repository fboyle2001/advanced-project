from copyreg import pickle
import torch, torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random, copy
import argparse
import numpy as np

import pickle

from dotmap import DotMap
from PIL import Image
from loguru import logger

import time
import json
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.data.sampler import RandomSampler


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
        # self.supervised_trainloader = self.get_loader(indices=None, transforms=self.train_transforms, train=True)
        self.train_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.train_transforms)

        # for idx, s in self.train_dataset:
        #     print(idx, s)

        # self.supervised_testloader = self.get_loader(indices=None, transforms=self.test_transforms, train=False)
        self.test_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.test_transforms)

    def get_cl_loader(self, indices, transforms, train, target_transforms=None):
        sampler = None
        dataset = None

        if train:
            sampler = SubsetRandomSampler(indices)
            dataset = self.train_dataset
        else:
            sampler = SubsetSequentialSampler(indices) 
            dataset = self.test_dataset

        # torchvision.datasets.CIFAR10(
        #     root=self.data_dir, train=train, download=True, transform=transforms, target_transform=target_transforms
        # )
        
        return DataLoader(dataset=dataset, sampler=sampler, num_workers=0, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def gen_cl_mapping(self):
        ## START OF CLASS BINNING
        # This part here is simply dividing the items into their class bins
        train_class_labels_dict = classwise_split(targets=self.train_dataset.targets) # type: ignore
        dumpable_trcld = {int(key): [int(x) for x in train_class_labels_dict[key]] for key in train_class_labels_dict.keys()}

        with open("train_class_labels_dict.json", "w+") as f:
            json.dump(dumpable_trcld, f, indent=2)

        test_class_labels_dict = classwise_split(targets=self.test_dataset.targets) # type: ignore
        dumpable_tecld = {int(key): [int(x) for x in test_class_labels_dict[key]] for key in test_class_labels_dict.keys()}

        with open("test_class_labels_dict.json", "w+") as f:
            json.dump(dumpable_tecld, f, indent=2)

        ## END OF CLASS BINNING

        # Sets classes to be 0 to n-1 if class order is not specified, else sets it to class order. To produce different effects tweak here.
        cl_class_list = list(range(self.total_num_classes))
        random.shuffle(cl_class_list) # Generates different class-to-task assignment

        # This makes a small difference but no where near able to explain the difference I'm seeing
        # continual_target_transform = ReorderTargets(cl_class_list)  # Remaps the class order to a 0-n order, required for crossentropy loss using class list
        continual_target_transform = None
 
        # This is the memory
        trainidx, testidx = [], []
        trainimgs = []
        trainclz = []
        mem_per_cls = self.memory_size // (self.total_num_classes)

        assert self.train_dataset is not None
        assert self.train_dataset.transform is not None

        for cl in cl_class_list: # Selects classes from the continual learning list and loads memory and test indices, which are then passed to a subset sample
            num_memory_samples = min(len(train_class_labels_dict[cl][:]), mem_per_cls)
            # Limited no of training samples
            q = train_class_labels_dict[cl][:num_memory_samples]
            trainidx += q # This is class-balanced greedy sampling (Selects the first n samples).
            trainimgs += [self.train_dataset.data[idx] for idx in q]
            trainclz += [cl] * num_memory_samples
            logger.info(f"Taking {len(q)} from {cl}")

            # Take all of the test samples
            testidx += test_class_labels_dict[cl][:]

        logger.info(f"{len(trainimgs)}, {len(trainclz)}, {type(trainimgs[0])}, {type(trainclz[0])}")

        # trainidx stores the index of all of the samples that form the replay buffer
        # We can check this by pulling their labels

        # for idx in trainidx:
        #     clzz = self.train_dataset.targets[idx]
        #     other_clzz = self.train_dataset[idx][1]
        #     logger.info(f"{clzz} == {other_clzz}")
        #     assert clzz == other_clzz

        # It is certainly true
        # So this is where it gets weird
        # If we pull all of the imgs associated with idxs and put them in DataLoader rather
        # than sampling by their indices we get far worse accuracy which is insane

        # Data types in CIFAR-10
        logger.info(f"{len(self.train_dataset.data)}, {len(self.train_dataset.targets)}")

        

        #     print(len(sample), type(sample[0]), type(sample[1]))

        assert(len(trainidx) <= self.memory_size), "ERROR: Cannot exceed max. memory samples!"

        ds = CustomImageDataset(trainimgs, trainclz, transform=self.train_dataset.transform)
        self.cltrain_loader = DataLoader(dataset=ds, batch_size=16, num_workers=0, shuffle=True, pin_memory=True)

        # This here actually does work
        # subset = Subset(self.train_dataset, trainidx)

        # self.cltrain_loader = DataLoader(subset, batch_size=16, num_workers=0, shuffle=True, pin_memory=True)
        # self.cltrain_loader = self.get_cl_loader(indices=trainidx, transforms=self.train_transforms, train=True)# , target_transforms=continual_target_transform)
        self.cltest_loader = self.get_cl_loader(indices=testidx, transforms=self.test_transforms, train=False)# , target_transforms=continual_target_transform)


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

# https://stackoverflow.com/a/59661024
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

    def get_sample(self, sample_size):
        return random.sample(self.data, k=sample_size)
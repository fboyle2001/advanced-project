from typing import List, Union
from .dataset_base import BaseCLDataset

import torchvision

training_cifar_transform = lambda x: [
    torchvision.transforms.RandomCrop(x, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
]

# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2627261#gistcomment-2627261
base_cifar_transform = lambda x: [
    torchvision.transforms.Resize((x, x)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) 
]

cifar_classes: List[Union[str, int]] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", 
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", 
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", 
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", 
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", 
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", 
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", 
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

class CIFAR100(BaseCLDataset):
    def __init__(
        self,
        disjoint: bool,
        size: int,
        classes_per_task: Union[int, None] = None
    ):
        super().__init__(
            dataset_class=torchvision.datasets.CIFAR100,
            training_dataset_parameters={ "train": True, "download": True },
            testing_dataset_parameters={ "train": False, "download": True },
            training_transform=torchvision.transforms.Compose(training_cifar_transform(size) + base_cifar_transform(size)),
            testing_transform=torchvision.transforms.Compose(base_cifar_transform(size)),
            classes=cifar_classes,
            disjoint=disjoint,
            classes_per_task=classes_per_task
        )
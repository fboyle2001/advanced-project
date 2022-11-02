import torchvision

class CiSplit:
    def __init__(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        self.inp_size = 32
        self.data_dir = "./store/data"

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

        self.train_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.train_transforms)
        self.test_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.test_transforms)

    def _perform_cl_splitting(self):
        # First, split into bins
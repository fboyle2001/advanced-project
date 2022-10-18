import torch

def get_gpu(fatal_on_error=True):
    cuda_available = torch.cuda.is_available()
    if not cuda_available and fatal_on_error:
        print(cuda_available, fatal_on_error)
        raise RuntimeError("No GPU available")
    
    return torch.device("cuda" if cuda_available else "cpu")

def create_iterator(dataloader): 
    # helper function to make getting another batch of data easier
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    return iter(cycle(dataloader))
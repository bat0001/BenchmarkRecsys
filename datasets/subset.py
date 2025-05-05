from torch.utils.data import Subset

def create_subset(dataset, indices):
    return Subset(dataset, indices)
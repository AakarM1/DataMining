from pathlib import Path
from torch.utils.data import Subset, random_split, DataLoader
import numpy as np
import random
from data.dataset import LiTSDataset

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
def reduce_channels(x):
    return x.mean(dim=0, keepdim=True)  # Reduce channels to 1

def create_data_loaders(dataset, batch_size=2):
    # Add a transform to ensure 5D data
    dataset.transform = Lambda(reduce_channels)  # Replace lambda with named function

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )

    return train_loader, val_loader


def split_clients(dataset, num_clients=4, strategy="iid"):
    """
    Split dataset into client-specific subsets based on strategy.

    Args:
        dataset (Dataset): The full dataset.
        num_clients (int): Number of clients.
        strategy (str): Strategy ('iid', 'non-iid', 'random').

    Returns:
        client_datasets (list of Subset): List of client-specific datasets.
    """
    indices = list(range(len(dataset)))

    if strategy == "iid":
        random.shuffle(indices)
        client_splits = np.array_split(indices, num_clients)

    elif strategy == "random":
        random.shuffle(indices)
        client_splits = np.array_split(indices, num_clients)

    elif strategy == "non-iid":
        # Example: Divide by intensity ranges (customize as needed)
        sorted_indices = sorted(indices, key=lambda idx: dataset[idx][0].mean())
        client_splits = np.array_split(sorted_indices, num_clients)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [Subset(dataset, split) for split in client_splits]

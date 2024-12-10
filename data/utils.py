from pathlib import Path
from torch.utils.data import Subset, random_split, DataLoader
import numpy as np
import random
from data.dataset import LiTSDataset

def create_dataloaders(image_dir, mask_dir, batch_size, val_split=0.2, input_shape=(96, 192, 192), augment=True):
    """
    Create train and validation DataLoaders.

    Args:
        image_dir (str): Path to the images directory.
        mask_dir (str): Path to the masks directory.
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of data to use for validation.
        input_shape (tuple): Target shape for resizing.
        augment (bool): Whether to apply data augmentation.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    dataset = LiTSDataset(
        image_paths=sorted(Path(image_dir).glob("*.nii")),
        mask_paths=sorted(Path(mask_dir).glob("*.nii")),
        target_shape=input_shape,
        augment=augment
    )
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

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

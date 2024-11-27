import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data.preprocess import Preprocessor

class MedicalDataset(Dataset):
    """
    A custom dataset class for medical imaging data.
    """
    def __init__(self, data_dir, input_shape=(1, 64, 64)):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.preprocessor = Preprocessor(input_shape)
        self.data, self.targets = self._load_data()

    def _load_data(self):
        # Load data from the directory (dummy implementation)
        # Replace with actual dataset loading logic
        data = torch.rand(100, *self.input_shape)  # Example: 100 synthetic samples
        targets = torch.rand(100, *self.input_shape)
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.preprocessor.preprocess(self.data[idx])
        target = self.targets[idx]
        return data, target


def get_dataloader(dataset_name="medical", data_dir="./data", batch_size=8, input_shape=(1, 64, 64)):
    """
    Return a DataLoader for the specified dataset.
    """
    if dataset_name.lower() == "medical":
        dataset = MedicalDataset(data_dir, input_shape)
    elif dataset_name.lower() == "dummy":
        dataset = MedicalDataset(data_dir)  # Reuse MedicalDataset for dummy data
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class LiTSDataset(Dataset):
    """
    Dataset class for the LiTS dataset (Liver Tumor Segmentation).
    """
    def __init__(self, data_dir, input_shape=(1, 64, 64)):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.preprocessor = Preprocessor(input_shape)
        self.data, self.targets = self._load_data()

    def _load_data(self):
        # Load all images and masks (replace with real paths)
        image_dir = os.path.join(self.data_dir, "images")
        mask_dir = os.path.join(self.data_dir, "masks")

        images = []
        masks = []

        for image_file in sorted(os.listdir(image_dir)):
            mask_file = os.path.join(mask_dir, os.path.basename(image_file))

            # Load image and mask as numpy arrays
            image = np.load(os.path.join(image_dir, image_file))
            mask = np.load(mask_file)

            images.append(image)
            masks.append(mask)

        return torch.tensor(images), torch.tensor(masks)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.preprocessor.preprocess(self.data[idx])
        target = self.targets[idx]
        return data, target


def get_dataloader(dataset_name="medical", data_dir="./data", batch_size=8, input_shape=(1, 64, 64)):
    """
    Return a DataLoader for the specified dataset.
    """
    if dataset_name.lower() == "lits":
        dataset = LiTSDataset(data_dir, input_shape)
    elif dataset_name.lower() == "dummy":
        dataset = MedicalDataset(data_dir, input_shape)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

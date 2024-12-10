import os
import urllib.request
import tarfile
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data.preprocess import Preprocessor  # Assumed to exist
import kagglehub
import shutil
from pathlib import Path
import nibabel as nib
from PIL import Image
import random
from torch.utils.data import Subset, random_split


@DeprecationWarning
def combine_volume_folders(data_dir="./data/lits"):
    """
    Combine all `volume_pt*` folders into a single directory `volumes_combined`
    and delete the original folders after combining.

    Args:
        data_dir (str): Path to the main LiTS dataset directory.

    Returns:
        Path: Path to the combined volumes directory.

    Raises:
        FileNotFoundError: If any of the `volume_pt*` directories are missing.
    """
    volume_dirs = [Path(data_dir) / f"volume_pt{i}" for i in range(1, 6)]
    combined_dir = Path(data_dir) / "volumes_combined"

    # Ensure all volume directories exist
    if not all(volume_dir.exists() for volume_dir in volume_dirs):
        raise FileNotFoundError(f"One or more `volume_pt*` directories are missing in {data_dir}.")

    combined_dir.mkdir(exist_ok=True)

    # Copy all `.nii` files into the combined directory
    for volume_dir in volume_dirs:
        for file in volume_dir.glob("*.nii"):
            shutil.copy(file, combined_dir / file.name)

        # Delete the individual folder after copying its contents
        shutil.rmtree(volume_dir)
        print(f"[INFO] Deleted folder: {volume_dir}")

    print(f"[INFO] Combined volumes saved to {combined_dir}.")
    return combined_dir


def prepare_client_folders(data_dir="./data/lits", client_base_dir="./data/clients/client_{i}", num_clients=4):
    """
    Prepare client-specific folders by matching images and masks by name and splitting them among clients.

    Args:
        data_dir (str): Path to the main LiTS dataset directory.
        client_base_dir (str): Template path for client-specific directories.
        num_clients (int): Number of clients.

    Raises:
        FileNotFoundError: If the required directories are missing.
        ValueError: If no matching files are found.
    """
    image_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"

    # Verify directories exist
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Required directories do not exist: {image_dir}, {mask_dir}")

    # Collect and match files by name
    image_files = sorted(image_dir.glob("*.png"))
    mask_files = sorted(mask_dir.glob("*.png"))

    # Match images and masks by file name
    mask_dict = {f.name: f for f in mask_files}
    matched_images = []
    matched_masks = []

    for image_file in image_files:
        if image_file.name in mask_dict:
            matched_images.append(image_file)
            matched_masks.append(mask_dict[image_file.name])

    if not matched_images or not matched_masks:
        raise ValueError("No matching image and mask files found.")

    if len(matched_images) != len(matched_masks):
        raise ValueError(
            f"Mismatch after filtering: {len(matched_images)} images and {len(matched_masks)} masks."
        )

    print(f"[DEBUG] Matched {len(matched_images)} images and masks.")

    # Split data across clients
    num_files_per_client = len(matched_images) // num_clients
    for client_id in range(num_clients):
        client_dir = Path(client_base_dir.format(i=client_id))
        client_image_dir = client_dir / "images"
        client_mask_dir = client_dir / "masks"

        client_image_dir.mkdir(parents=True, exist_ok=True)
        client_mask_dir.mkdir(parents=True, exist_ok=True)

        start_idx = client_id * num_files_per_client
        end_idx = start_idx + num_files_per_client

        # Copy files for the client
        for image_file, mask_file in zip(matched_images[start_idx:end_idx], matched_masks[start_idx:end_idx]):
            shutil.copy(image_file, client_image_dir / image_file.name)
            shutil.copy(mask_file, client_mask_dir / mask_file.name)

    print("[INFO] Client-specific dataset folders prepared successfully.")



def download_and_prepare_lits(data_dir="./data/lits", kaggle_dataset_list=None, download=True):
    """
    Download and prepare the LiTS dataset by combining `volume_pt*` folders into a single `images` folder.

    Args:
        data_dir (str): Path to the local directory where the dataset should be stored.
        kaggle_dataset_list (list): List of Kaggle dataset identifiers to download.
        download (bool): Whether to download the datasets.

    Raises:
        Exception: If downloading or combining the dataset fails.
    """
    if kaggle_dataset_list is None:
        kaggle_dataset_list = ["andrewmvd/liver-tumor-segmentation", "andrewmvd/liver-tumor-segmentation-part-2"]

    # Check if the dataset directory already exists
    images_dir = Path(data_dir) / "images"
    if not download and images_dir.exists():
        print(f"[INFO] LiTS dataset already exists at {images_dir}.")
        return

    print(f"[INFO] Preparing LiTS dataset at {data_dir}...")

    try:
        for kaggle_dataset in kaggle_dataset_list:
            # Download the dataset using KaggleHub
            print(f"[INFO] Downloading {kaggle_dataset}...")
            path = kagglehub.dataset_download(kaggle_dataset)
            print(f"[INFO] Dataset downloaded to cache at {path}.")

            # Combine all volume_pt* folders into `images`
            for volume_dir in Path(path).glob("volume_pt*"):
                if volume_dir.is_dir():
                    print(f"[INFO] Moving files from {volume_dir} to {images_dir}...")
                    images_dir.mkdir(parents=True, exist_ok=True)
                    for file in volume_dir.glob("*.nii"):
                        shutil.copy(file, images_dir / file.name)
                    shutil.rmtree(volume_dir)
                    print(f"[INFO] Deleted folder: {volume_dir}")

        print(f"[INFO] Dataset successfully prepared in {images_dir}.")
    except Exception as e:
        print(f"[ERROR] Failed to download or prepare the LiTS dataset: {str(e)}")
        raise


def get_dataloader(dataset_name="lits", data_dir="./data", batch_size=8, input_shape=(1, 64, 64)):
    """
    Return a DataLoader for the specified dataset.

    Args:
        dataset_name (str): The name of the dataset ("lits" or others).
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        input_shape (tuple): Shape to resize the images to.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    if dataset_name.lower() == "lits":
        dataset = LiTSDataset(data_dir, input_shape)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_dummy_data(client_id, dataset_name, num_samples=10, input_shape=(1, 64, 64)):
    """
    Generate dummy images and masks for testing.

    Args:
        client_id (int): ID of the client for which to generate data.
        dataset_name (str): Name of the dataset ("dummy").
        num_samples (int): Number of samples to generate.
        input_shape (tuple): Shape of the images/masks.
    """
    base_dir = f"./client_{client_id}_{dataset_name}"
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_samples):
        image = np.random.rand(*input_shape).astype(np.float32)
        mask = np.random.randint(0, 2, size=input_shape).astype(np.float32)

        np.save(os.path.join(image_dir, f"image_{i}.npy"), image)
        np.save(os.path.join(mask_dir, f"mask_{i}.npy"), mask)

    print(f"Dummy data generated for {base_dir}")

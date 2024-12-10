import numpy as np
from scipy import ndimage
import nibabel as nib
import torch
from torch.utils.data import Dataset


class DataAugmentation3D:
    def __init__(self, target_shape=(96, 192, 192), p_flip=0.5, rotate_angle=15):
        """
        3D Data Augmentation with Fixed Target Shape
        
        Args:
        - target_shape: Consistent output shape for all augmented volumes
        - p_flip: Probability of flipping
        - rotate_angle: Maximum rotation angle in degrees
        """
        self.target_shape = target_shape
        self.p_flip = p_flip
        self.rotate_angle = rotate_angle

    def __call__(self, image, mask):
        """
        Apply random augmentations to image and mask
        
        Args:
        - image: 3D numpy array
        - mask: 3D numpy array (binary)
        
        Returns:
        - Augmented image and mask of consistent shape
        """
        # Horizontal Flip
        if np.random.random() < self.p_flip:
            image, mask = self.flip(image, mask, axis=1)
        
        # Vertical Flip
        if np.random.random() < self.p_flip:
            image, mask = self.flip(image, mask, axis=0)
        
        # Depth Flip
        if np.random.random() < self.p_flip:
            image, mask = self.flip(image, mask, axis=2)
        
        # Random Rotation
        angles = np.random.uniform(-self.rotate_angle, self.rotate_angle, size=3)
        image = self.rotate(image, angles)
        mask = self.rotate(mask, angles, is_mask=True)
        
        # Resize to target shape
        image_resized = self.resize(image, self.target_shape, order=1)
        mask_resized = self.resize(mask, self.target_shape, order=0)

        return image_resized, mask_resized

    @staticmethod
    def flip(image, mask, axis):
        return np.flip(image, axis=axis), np.flip(mask, axis=axis)

    @staticmethod
    def rotate(volume, angles, is_mask=False):
        """
        Apply 3D rotation to the volume along x, y, and z axes.

        Args:
        - volume: 3D numpy array
        - angles: List of rotation angles for x, y, and z
        - is_mask: Boolean to apply nearest neighbor interpolation for masks
        
        Returns:
        - Rotated volume
        """
        order = 0 if is_mask else 1
        volume = ndimage.rotate(volume, angles[0], axes=(1, 2), reshape=False, order=order, mode='nearest')
        volume = ndimage.rotate(volume, angles[1], axes=(0, 2), reshape=False, order=order, mode='nearest')
        volume = ndimage.rotate(volume, angles[2], axes=(0, 1), reshape=False, order=order, mode='nearest')
        return volume

    @staticmethod
    def resize(volume, target_shape, order=1):
        """
        Resize a 3D volume to the target shape.
        
        Args:
        - volume: 3D numpy array
        - target_shape: Desired shape for the volume
        - order: Interpolation order (1 for linear, 0 for nearest neighbor)
        
        Returns:
        - Resized volume
        """
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        return ndimage.zoom(volume, factors, order=order)


class LiTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_shape=(96, 192, 192), augment=True):
        """
        Dataset for LiTS with optional augmentation.
        
        Args:
        - image_paths: List of paths to image files
        - mask_paths: List of paths to mask files
        - target_shape: Desired shape for the 3D volumes
        - augment: Boolean to apply augmentation
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_shape = target_shape
        self.augmentation = DataAugmentation3D(target_shape=target_shape) if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load NIfTI images
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Preprocess
        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)

        # Apply augmentations if enabled
        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension

        return image, mask

    def preprocess_image(self, volume):
        """
        Preprocess image volume (clip, normalize, resize).
        """
        volume = np.clip(volume, -100, 200)
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
        return DataAugmentation3D.resize(volume, self.target_shape, order=1)

    def preprocess_mask(self, mask):
        """
        Preprocess binary mask volume (resize, threshold).
        """
        resized_mask = DataAugmentation3D.resize(mask, self.target_shape, order=0)
        return (resized_mask > 0.5).astype(np.float32)

import numpy as np
from scipy import ndimage
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Lambda, Compose


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
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        
        # Vertical Flip
        if np.random.random() < self.p_flip:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        
        # Depth Flip
        if np.random.random() < self.p_flip:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=2)
        
        # Rotation
        angle_x = np.random.uniform(-self.rotate_angle, self.rotate_angle)
        angle_y = np.random.uniform(-self.rotate_angle, self.rotate_angle)
        angle_z = np.random.uniform(-self.rotate_angle, self.rotate_angle)
        
        image = ndimage.rotate(image, angle_x, reshape=False, mode='nearest')
        image = ndimage.rotate(image, angle_y, axes=(0,2), reshape=False, mode='nearest')
        image = ndimage.rotate(image, angle_z, axes=(0,1), reshape=False, mode='nearest')
        
        mask = ndimage.rotate(mask, angle_x, reshape=False, mode='nearest')
        mask = ndimage.rotate(mask, angle_y, axes=(0,2), reshape=False, mode='nearest')
        mask = ndimage.rotate(mask, angle_z, axes=(0,1), reshape=False, mode='nearest')
        
        # Resize to target shape to ensure consistency
        image_resized = ndimage.zoom(image, 
                               (self.target_shape[0]/image.shape[0], 
                                self.target_shape[1]/image.shape[1], 
                                self.target_shape[2]/image.shape[2]), 
                               order=1)
        
        mask_resized = ndimage.zoom(mask, 
                               (self.target_shape[0]/mask.shape[0], 
                                self.target_shape[1]/mask.shape[1], 
                                self.target_shape[2]/mask.shape[2]), 
                               order=0)  # Use nearest neighbor for masks
        
        # Ensure binary mask
        mask_resized = (mask_resized > 0.5).astype(np.float32)
        
        return image_resized, mask_resized

# class LiTSDataset(Dataset):
#     def __init__(self, image_paths, mask_paths, target_shape=(96, 192, 192), augment=True):
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.target_shape = target_shape
#         self.augmentation = DataAugmentation3D(target_shape=target_shape) if augment else None

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # Load NIfTI images
#         image = nib.load(self.image_paths[idx]).get_fdata()
#         mask = nib.load(self.mask_paths[idx]).get_fdata()

#         # Preprocess
#         image = self.preprocess_image(image)
#         mask = self.preprocess_mask(mask)

#         # Apply augmentations if enabled
#         if self.augmentation is not None:
#             image, mask = self.augmentation(image, mask)

#         # Convert to tensors
#         image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
#         mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension

#         return image, mask

#     def preprocess_image(self, volume):
#         # Clip and normalize
#         volume = np.clip(volume, -100, 200)
#         volume = (volume - volume.min()) / (volume.max() - volume.min())
        
#         # Resize using scipy
#         resized_volume = ndimage.zoom(volume, 
#                                (self.target_shape[0]/volume.shape[0], 
#                                 self.target_shape[1]/volume.shape[1], 
#                                 self.target_shape[2]/volume.shape[2]), 
#                                order=1)
#         return resized_volume

#     def preprocess_mask(self, mask):
#         # Binary mask preprocessing
#         resized_mask = ndimage.zoom(mask, 
#                             (self.target_shape[0]/mask.shape[0], 
#                              self.target_shape[1]/mask.shape[1], 
#                              self.target_shape[2]/mask.shape[2]), 
#                             order=0)  # Nearest neighbor for binary masks
#         return (resized_mask > 0.5).astype(np.float32)
    

class LiTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_shape=(96, 192, 192), augment=True):
        """
        Dataset for LiTS with optional augmentation for 3D volumes.

        Args:
        - image_paths: List of paths to image files.
        - mask_paths: List of paths to mask files.
        - target_shape: Desired shape for the 3D volumes (depth, height, width).
        - augment: Boolean to apply data augmentation.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_shape = target_shape
        self.augment = augment
        self.transform = Compose([
                        Lambda(self.reduce_channels),
                    ])
    def reduce_channels(x):
        return x.mean(dim=0, keepdim=True)  # Reduce channels to 1

    def __len__(self):
        return len(self.image_paths)

    # Dataset __getitem__ method
    def __getitem__(self, idx):
        # Load NIfTI volumes
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Preprocess image and mask
        image = self.preprocess_image(image)  # (depth, height, width)
        mask = self.preprocess_mask(mask)    # (depth, height, width)

        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension: (1, depth, height, width)
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension: (1, depth, height, width)

         # Apply transformation
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)  # Optional if mask needs to match

        print(f"Shape of the image is {image.shape} and shape of the mask is {mask.shape}")
        return image, mask
    

    def preprocess_image(self, volume):
        """
        Clip, normalize, and resize the image volume.
        """
        volume = np.clip(volume, -100, 200)  # Clip intensity values
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)  # Normalize
        factors = [
            self.target_shape[i] / volume.shape[i] for i in range(3)
        ]  # Compute resizing factors
        return ndimage.zoom(volume, factors, order=1)

    def preprocess_mask(self, mask):
        """
        Resize the mask volume and ensure binary values.
        """
        factors = [
            self.target_shape[i] / mask.shape[i] for i in range(3)
        ]  # Compute resizing factors
        mask = ndimage.zoom(mask, factors, order=0)  # Nearest-neighbor interpolation for masks
        return (mask > 0.5).astype(np.float32)
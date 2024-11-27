import torch
import numpy as np
from torchvision import transforms

class Preprocessor:
    """
    Handles dataset preprocessing tasks such as normalization and resizing.
    """
    def __init__(self, input_shape=(1, 64, 64)):
        self.input_shape = input_shape
        self.transforms = transforms.Compose([
            transforms.Resize(input_shape[1:]),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess(self, data):
        """
        Apply preprocessing steps to the input data.
        """
        if isinstance(data, torch.Tensor):
            data = data.float()  # Ensure float tensor for neural networks
            return self.transforms(data)
        else:
            raise ValueError("Data must be a torch.Tensor.")

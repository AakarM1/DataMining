import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

    def set_weights(self, weights):
        """
        Update the model's weights using the provided state_dict.
        """
        self.load_state_dict(weights)

    def get_weights(self):
        """
        Retrieve the model's weights as a state_dict.
        """
        return self.state_dict()
    def compute_cmcr_loss(self, x_ct, x_mri):
        """
        Compute Cross-Modal Consistency Regularization loss.
        """
        features_ct = self.forward(x_ct)
        features_mri = self.forward(x_mri)

        cmcr_loss = torch.mean((features_ct - features_mri) ** 2)
        return cmcr_loss

    def save_model(self, path="model.pth"):
        """
        Save the model weights.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path="model.pth"):
        """
        Load the model weights.
        """
        self.load_state_dict(torch.load(path))

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(EnhancedUNet3D, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.pool = nn.MaxPool3d(2, ceil_mode=True)  # Adjust pooling

        self.middle = self.conv_block(64, 128)

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=1)
        self.decoder1 = self.conv_block(128, 64)

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, output_padding=1)
        self.decoder2 = self.conv_block(64, 32)

        self.output_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Check if input is 4D and add a dummy depth dimension
        if len(x.shape) == 4:  # Shape: (batch_size, channels, height, width)
            x = x.unsqueeze(2)  # Shape: (batch_size, channels, depth=1, height, width)

        # Encoder path
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))

        # Middle
        x3 = self.middle(self.pool(x2))

        # Decoder path with skip connections
        x4 = self.up1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.decoder1(x4)

        x5 = self.up2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.decoder2(x5)

        return self.output_conv(x5)

    def set_weights(self, weights):
        """
        Update the model's weights using the provided state_dict.
        """
        self.load_state_dict(weights)

    def get_weights(self):
        """
        Retrieve the model's weights as a state_dict.
        """
        return self.state_dict()

    def save_model(self, path="model.pth"):
        """
        Save the model weights.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path="model.pth"):
        """
        Load the model weights.
        """
        self.load_state_dict(torch.load(path))

    
    # TODO
    # def compute_cmcr_loss(self, x_ct, x_mri):
    #     """
    #     Compute Cross-Modal Consistency Regularization loss.
    #     """
    #     features_ct = self.forward(x_ct)
    #     features_mri = self.forward(x_mri)

    #     cmcr_loss = torch.mean((features_ct - features_mri) ** 2)
    #     return cmcr_loss

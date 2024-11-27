import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

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
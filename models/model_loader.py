from models.unet import UNet, EnhancedUNet3D
import torch

def get_model(model_name):
    """
    Return the model class based on the provided name.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name.lower() == "unet":
        return UNet()
    elif model_name.lower() == "eunet":
        return EnhancedUNet3D().to(device)
    # Add other models here as needed
    else:
        raise ValueError(f"Unknown model: {model_name}")

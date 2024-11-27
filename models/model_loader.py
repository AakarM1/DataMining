from models.unet import UNet

def get_model(model_name):
    """
    Return the model class based on the provided name.
    """
    if model_name.lower() == "unet":
        return UNet()
    # Add other models here as needed
    else:
        raise ValueError(f"Unknown model: {model_name}")

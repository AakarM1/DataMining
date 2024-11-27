import torch

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Similarity Coefficient.
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    return (2.0 * intersection + smooth) / (union + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    """
    Compute the Intersection over Union.
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)

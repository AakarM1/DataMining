import torch
import json
import csv

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

def precision_recall_f1(y_true, y_pred, smooth=1e-6):
    """
    Compute precision, recall, and F1-score.
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    tp = (y_true * y_pred).sum().item()
    fp = ((1 - y_true) * y_pred).sum().item()
    fn = (y_true * (1 - y_pred)).sum().item()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return precision, recall, f1


def save_metrics_to_json(metrics, path="metrics.json"):
    """
    Save metrics dictionary to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def save_metrics_to_csv(metrics, path="metrics.csv"):
    """
    Save metrics dictionary to a CSV file.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Dice", "IoU", "Precision", "Recall", "F1-Score"])
        for epoch, metric in metrics.items():
            writer.writerow([
                epoch,
                metric["Dice"],
                metric["IoU"],
                metric["Precision"],
                metric["Recall"],
                metric["F1-Score"]
            ])

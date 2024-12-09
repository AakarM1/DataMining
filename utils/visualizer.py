import matplotlib.pyplot as plt
import torch

def visualize_predictions(data, target, prediction, epoch, save_path="./visualizations"):
    """
    Visualize input data, target mask, and predicted mask.

    Args:
        data (Tensor): Input data.
        target (Tensor): Ground truth.
        prediction (Tensor): Predicted output.
        epoch (int): Current epoch number.
        save_path (str): Path to save the visualization.
    """
    # Ensure the directory exists
    import os
    os.makedirs(save_path, exist_ok=True)

    # Convert tensors to numpy arrays
    data = data.cpu().numpy()
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(data[0, 0], cmap="gray")
    axes[0].set_title("Input Data")
    axes[1].imshow(target[0, 0], cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(prediction[0, 0], cmap="gray")
    axes[2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(f"{save_path}/epoch_{epoch}.png")
    plt.close(fig)

def plot_metrics(metrics, save_path="./metrics_plot.png"):
    """
    Plot metrics (Dice, IoU, Precision, Recall, F1-score) over epochs.
    """
    import matplotlib.pyplot as plt

    epochs = list(metrics.keys())
    dice = [metrics[epoch]["Dice"] for epoch in epochs]
    iou = [metrics[epoch]["IoU"] for epoch in epochs]
    precision = [metrics[epoch]["Precision"] for epoch in epochs]
    recall = [metrics[epoch]["Recall"] for epoch in epochs]
    f1 = [metrics[epoch]["F1-Score"] for epoch in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dice, label="Dice")
    plt.plot(epochs, iou, label="IoU")
    plt.plot(epochs, precision, label="Precision")
    plt.plot(epochs, recall, label="Recall")
    plt.plot(epochs, f1, label="F1-Score")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Metrics Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

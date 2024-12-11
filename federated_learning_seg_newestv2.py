import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import torch.nn.functional as F
import csv
from multiprocessing import Pool, cpu_count

def iou_score(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)  # Apply sigmoid for probabilities
    inputs = (inputs > 0.5).float()  # Binarize predictions
    intersection = (inputs * targets).sum(dim=(2, 3, 4))
    union = (inputs + targets).sum(dim=(2, 3, 4)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()

def precision_recall_f1(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)  # Apply sigmoid for probabilities
    inputs = (inputs > 0.5).float()  # Binarize predictions
    
    tp = (inputs * targets).sum(dim=(2, 3, 4))  # True positives
    fp = ((inputs == 1) & (targets == 0)).sum(dim=(2, 3, 4))  # False positives
    fn = ((inputs == 0) & (targets == 1)).sum(dim=(2, 3, 4))  # False negatives

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return precision.mean().item(), recall.mean().item(), f1.mean().item()


# Dice Coefficient
def dice_coefficient(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    return (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

# Dice + BCE Loss Function
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return bce_loss + dice_loss

# Enhanced 3D U-Net Model
class EnhancedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(EnhancedUNet3D, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.pool = nn.MaxPool3d(2)
        self.middle = self.conv_block(64, 128)
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
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
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.middle(self.pool(x2))
        x4 = self.up1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.decoder1(x4)
        x5 = self.up2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.decoder2(x5)
        return self.output_conv(x5)

# Adaptive Client Weighting
def compute_client_weights(client_data_stats):
    weights = []
    for stats in client_data_stats:
        variance = stats['variance']
        weight = stats['size'] / (1 + variance)
        weights.append(weight)
    total_weight = sum(weights)
    return [w / total_weight for w in weights]

# Gradient-Adjusted Update Aggregation
def aggregate_updates(global_model, client_models, client_weights):
    averaged_state = {}
    for key in global_model.state_dict():
        weighted_sum = sum(client_weights[i] * client_models[i].state_dict()[key] for i in range(len(client_models)))
        averaged_state[key] = weighted_sum
    global_model.load_state_dict(averaged_state)
    return global_model

# Save Metrics to CSV
def save_metrics_to_csv(round_num, epoch, client, train_loss, train_dice, val_loss, val_dice, file_name="metrics.csv"):
    header = ["Round", "Epoch", "Client", "Train Loss", "Train Dice", "Val Loss", "Val Dice"]
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([round_num, epoch, client, train_loss, train_dice, val_loss, val_dice])

# Validation Loop
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    val_iou = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            val_dice += dice_coefficient(outputs, masks).item()
            val_iou += iou_score(outputs, masks)
            precision, recall, f1 = precision_recall_f1(outputs, masks)
            val_precision += precision
            val_recall += recall
            val_f1 += f1

    num_batches = len(val_loader)
    val_loss /= num_batches
    val_dice /= num_batches
    val_iou /= num_batches
    val_precision /= num_batches
    val_recall /= num_batches
    val_f1 /= num_batches

    return val_loss, val_dice, val_iou, val_precision, val_recall, val_f1


def create_data_loaders(dataset, batch_size=2):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Use pin_memory and num_workers for better performance
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )

    return train_loader, val_loader
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

class AugmentedLiverDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_shape=(96, 192, 192), augment=True):
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
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension

        return image, mask

    def preprocess_image(self, volume):
        # Clip and normalize
        volume = np.clip(volume, -100, 200)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Resize using scipy
        resized_volume = ndimage.zoom(volume, 
                               (self.target_shape[0]/volume.shape[0], 
                                self.target_shape[1]/volume.shape[1], 
                                self.target_shape[2]/volume.shape[2]), 
                               order=1)
        return resized_volume

    def preprocess_mask(self, mask):
        # Binary mask preprocessing
        resized_mask = ndimage.zoom(mask, 
                            (self.target_shape[0]/mask.shape[0], 
                             self.target_shape[1]/mask.shape[1], 
                             self.target_shape[2]/mask.shape[2]), 
                            order=0)  # Nearest neighbor for binary masks
        return (resized_mask > 0.5).astype(np.float32)
    

# Save Metrics to CSV (updated for all 5 metrics)
def save_metrics_to_csv_v2(round_num, epoch, client, train_loss, train_dice, train_iou, train_precision, train_recall, train_f1,
                           val_loss, val_dice, val_iou, val_precision, val_recall, val_f1, file_name="metricsv2.csv"):
    header = ["Round", "Epoch", "Client", 
              "Train Loss", "Train Dice", "Train IoU", "Train Precision", "Train Recall", "Train F1", 
              "Val Loss", "Val Dice", "Val IoU", "Val Precision", "Val Recall", "Val F1"]
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([round_num, epoch, client, 
                         train_loss, train_dice, train_iou, train_precision, train_recall, train_f1, 
                         val_loss, val_dice, val_iou, val_precision, val_recall, val_f1])

# Updated Federated Learning Training Loop
def federated_learning(global_model, client_datasets, criterion, optimizer_fn, rounds=10, epochs=5, device='cuda'):
    best_val_loss = float('inf')

    for round_num in range(rounds):
        print(f"=== Federated Learning Round {round_num + 1}/{rounds} ===")

        client_models = []
        client_metrics = {
            "train_loss": [], "train_dice": [], "train_iou": [], "train_precision": [], "train_recall": [], "train_f1": [],
            "val_loss": [], "val_dice": [], "val_iou": [], "val_precision": [], "val_recall": [], "val_f1": []
        }

        for client_id, (train_loader, val_loader) in enumerate(client_datasets):
            print(f"\nClient {client_id + 1}/{len(client_datasets)}:")
            client_model = EnhancedUNet3D().to(device)
            client_model.load_state_dict(global_model.state_dict())
            optimizer = optimizer_fn(client_model.parameters())

            for epoch in range(epochs):
                # Training Loop
                client_model.train()
                epoch_train_loss, epoch_train_dice, epoch_train_iou = 0, 0, 0
                epoch_train_precision, epoch_train_recall, epoch_train_f1 = 0, 0, 0

                for images, masks in train_loader:
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    epoch_train_dice += dice_coefficient(outputs, masks).item()
                    epoch_train_iou += iou_score(outputs, masks)
                    precision, recall, f1 = precision_recall_f1(outputs, masks)
                    epoch_train_precision += precision
                    epoch_train_recall += recall
                    epoch_train_f1 += f1

                num_train_batches = len(train_loader)
                avg_train_loss = epoch_train_loss / num_train_batches
                avg_train_dice = epoch_train_dice / num_train_batches
                avg_train_iou = epoch_train_iou / num_train_batches
                avg_train_precision = epoch_train_precision / num_train_batches
                avg_train_recall = epoch_train_recall / num_train_batches
                avg_train_f1 = epoch_train_f1 / num_train_batches

                # Validation Loop
                val_loss, val_dice, val_iou, val_precision, val_recall, val_f1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                if val_loader:
                    val_metrics = validate(client_model, val_loader, criterion, device)
                    val_loss, val_dice, val_iou, val_precision, val_recall, val_f1 = val_metrics

                # Save metrics to CSV
                save_metrics_to_csv_v2(round_num + 1, epoch + 1, client_id + 1, 
                                       avg_train_loss, avg_train_dice, avg_train_iou, avg_train_precision, avg_train_recall, avg_train_f1,
                                       val_loss, val_dice, val_iou, val_precision, val_recall, val_f1)

                # Print metrics for the current epoch
                print(f"  Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train IoU: {avg_train_iou:.4f}, "
                      f"Train Precision: {avg_train_precision:.4f}, Train Recall: {avg_train_recall:.4f}, Train F1: {avg_train_f1:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, "
                      f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

            # Collect metrics for this client
            client_metrics["train_loss"].append(avg_train_loss)
            client_metrics["train_dice"].append(avg_train_dice)
            client_metrics["train_iou"].append(avg_train_iou)
            client_metrics["train_precision"].append(avg_train_precision)
            client_metrics["train_recall"].append(avg_train_recall)
            client_metrics["train_f1"].append(avg_train_f1)
            client_metrics["val_loss"].append(val_loss)
            client_metrics["val_dice"].append(val_dice)
            client_metrics["val_iou"].append(val_iou)
            client_metrics["val_precision"].append(val_precision)
            client_metrics["val_recall"].append(val_recall)
            client_metrics["val_f1"].append(val_f1)

            client_models.append(client_model)

        # Aggregate Updates
        global_model = aggregate_updates(global_model, client_models, [1.0 / len(client_datasets)] * len(client_datasets))

        # Print round summary
        print(f"\n=== Summary for Round {round_num + 1} ===")
        print(f"Avg Val Loss: {np.mean(client_metrics['val_loss']):.4f}, Avg Val Dice: {np.mean(client_metrics['val_dice']):.4f}, "
              f"Avg Val IoU: {np.mean(client_metrics['val_iou']):.4f}, Avg Val Precision: {np.mean(client_metrics['val_precision']):.4f}, "
              f"Avg Val Recall: {np.mean(client_metrics['val_recall']):.4f}, Avg Val F1: {np.mean(client_metrics['val_f1']):.4f}")
    return global_model

def main():

    # Update these paths to match your actual dataset location
    base_path = "C:\\Users\\Student\\Desktop\\0210\\DM\\Project\\atlas-train-dataset-1.0.1\\preprocessed-new"
    image_paths = [os.path.join(base_path, "images", f"im{i}.nii.gz") for i in range(191)]
    mask_paths = [os.path.join(base_path, "labels", f"lb{i}.nii.gz") for i in range(191)]

    target_shape = (96, 192, 192)  
    # Create the dataset
    dataset = AugmentedLiverDataset(image_paths, mask_paths, augment=True)

    # Split dataset among clients
    num_clients = 6
    client_datasets = []
    client_size = len(dataset) // num_clients

    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size if i < num_clients - 1 else len(dataset)
        
        client_image_paths = image_paths[start:end]
        client_mask_paths = mask_paths[start:end]

        client_dataset = AugmentedLiverDataset(client_image_paths, client_mask_paths, target_shape=target_shape)
        train_loader, val_loader = create_data_loaders(client_dataset)

        client_datasets.append((train_loader, val_loader))

    # Model, Criterion, and Optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global_model = EnhancedUNet3D().to(device)
    criterion = DiceBCELoss()
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)

    # Run Federated Learning
    federated_learning(global_model, 
                       client_datasets=client_datasets, 
                       criterion=criterion, 
                       optimizer_fn=optimizer_fn, 
                       rounds=2, 
                       epochs=100,
                       device=device)

if __name__ == "__main__":
    main()
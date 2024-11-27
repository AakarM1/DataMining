import sys
import os
import torch
from utils.logger import setup_logger
from fedseg.federated import FederatedServer
from fedseg.client import FederatedClient
from models.model_loader import get_model
from data.loaders import get_dataloader
from utils.metrics import dice_coefficient, iou

from data.loaders import get_dataloader

def main(args):
    logger = setup_logger()

    try:
        logger.info(f"Initializing {args.model} model for Federated Server...")
        shared_model = get_model(args.model)
        server = FederatedServer(shared_model, learning_rate=args.learning_rate)

        logger.info(f"Creating {args.num_clients} clients...")
        clients = []
        for i in range(args.num_clients):
            model = get_model(args.model)
            dataset = get_dataloader(
                dataset_name="medical",
                data_dir="./data",
                batch_size=args.batch_size,
                input_shape=(1, 64, 64)
            )
            clients.append(FederatedClient(model, dataset, heterogeneity_factor=1.0))

        logger.info("Starting Federated Training...")
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")

            client_updates = []
            client_weights = []
            for i, client in enumerate(clients):
                try:
                    logger.info(f"Client {i}: Starting local training...")
                    update, weight = client.compute_update(server.global_weights)
                    client_updates.append(update)
                    client_weights.append(weight)
                    logger.info(f"Client {i}: Training completed successfully.")
                except Exception as e:
                    logger.error(f"Client {i} failed to compute update: {str(e)}")

            try:
                server.aggregate_updates(client_updates, client_weights)
                logger.info("Global weights updated successfully.")
            except Exception as e:
                logger.error(f"Server aggregation failed: {str(e)}")

            logger.info("Evaluating global model...")
            try:
                val_data = next(iter(clients[0].dataset))  # Grab a batch from client 0
                data, target = val_data
                with torch.no_grad():
                    output = shared_model(data)
                    dice = dice_coefficient(target, (output > 0.5).float())
                    iou_score = iou(target, (output > 0.5).float())
                    logger.info(f"Evaluation - Dice: {dice:.4f}, IoU: {iou_score:.4f}")
            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")

            logger.info(f"Epoch {epoch + 1}/{args.epochs} completed.")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")

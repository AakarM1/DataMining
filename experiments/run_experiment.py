
import sys
import os
import torch
from utils.logger import setup_logger
from fedseg.federated import FederatedServer
from fedseg.client import FederatedClient
from models.model_loader import get_model
from data.loaders import get_dataloader
from utils.metrics import dice_coefficient, iou, precision_recall_f1, save_metrics_to_json, save_metrics_to_csv
from utils.visualizer import visualize_predictions, plot_metrics

from data.loaders import get_dataloader
import yaml

import multiprocessing
from tqdm import tqdm
from fedseg.client import client_train

def load_config(path="experiments/config.yaml"):
    """
    Load configuration file.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)



def main(args):
    logger = setup_logger()

    try:
        # Update main function in run_experiment.py
        # config = load_config()

        # # Override default args with config values
        # args.model = config['model']
        # args.epochs = config['training']['epochs']
        # args.batch_size = config['training']['batch_size']
        # args.learning_rate = config['training']['learning_rate']
        # args.num_clients = config['training']['num_clients']

        logger.info(f"Initializing {args.model} model for Federated Server...")
        shared_model = get_model(args.model)
        server = FederatedServer(shared_model, learning_rate=args.learning_rate)

        logger.info(f"Creating {args.num_clients} clients...")
        clients = []
        for i in range(args.num_clients):
            model = get_model(args.model)
            dataset = get_dataloader(
                dataset_name=args.dataset,
                data_dir=f"./client_{i}_{args.dataset}",
                batch_size=args.batch_size,
                input_shape=(1, 64, 64)
            )
            clients.append(FederatedClient(model, dataset, heterogeneity_factor=1.0))

        logger.info("Starting Federated Training...")
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")

            # Distributed client training
            with multiprocessing.Pool(processes=args.num_clients) as pool:
                results = list(tqdm(pool.starmap(client_train, [(client, server.global_weights) for client in clients]),
                        total=len(clients),
                        desc="Client Training Progress"))
                
            # Filter valid results
            client_updates = [res[0] for res in results if res[0] is not None]
            client_weights = [res[1] for res in results if res[1] is not None]

            if not client_updates or not client_weights:
                logger.error("No valid updates received from clients.")
                continue

            # Aggregate updates
            try:
                server.aggregate_updates(client_updates, client_weights)
                logger.info("Global weights updated successfully.")
            except Exception as e:
                logger.error(f"Server aggregation failed: {str(e)}")

            # Evaluate global model
            logger.info("Evaluating global model...")
            metrics_log = {}
            try:
                val_data = next(iter(clients[0].dataset))
                data, target = val_data
                with torch.no_grad():
                    output = shared_model(data)
                    dice = dice_coefficient(target, (output > 0.5).float())
                    iou_score = iou(target, (output > 0.5).float())
                    precision, recall, f1 = precision_recall_f1(target, (output > 0.5).float())

                    metrics_log[epoch + 1] = {
                        "Dice": dice.item(),
                        "IoU": iou_score.item(),
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1
                    }

                    logger.info(f"Evaluation - Dice: {dice:.4f}, IoU: {iou_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")

            logger.info(f"Epoch {epoch + 1}/{args.epochs} completed.")
        logger.info("Saving metrics...")
        save_metrics_to_json(metrics_log, path="metrics.json")
        save_metrics_to_csv(metrics_log, path="metrics.csv")
        plot_metrics(metrics_log, save_path="metrics_plot.png")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
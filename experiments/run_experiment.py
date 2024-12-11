from models.unet import UNet, EnhancedUNet3D
import sys
import os
import torch
from utils.logger import setup_logger
from fedseg.federated import FederatedServer
from fedseg.client import FederatedClient
from models.model_loader import get_model
from data.loaders import get_dataloader, generate_dummy_data, download_and_prepare_lits, prepare_client_folders
from data.dataset import LiTSDataset
from data.utils import create_data_loaders, split_clients
from utils.metrics import dice_coefficient, iou, precision_recall_f1, save_metrics_to_json, save_metrics_to_csv
from utils.visualizer import visualize_predictions, plot_metrics
import yaml
from torchvision.transforms import Lambda, Compose
import multiprocessing
from tqdm import tqdm
from fedseg.client import client_train

from torch.utils.data import DataLoader

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
        
        # if args.dataset == "dummy":
        #     logger.info("Generating dummy data for all clients...")
        #     for client_id in range(args.num_clients):
        #         generate_dummy_data(
        #             client_id=client_id,
        #             dataset_name=args.dataset,
        #             num_samples=10,  # Adjust the number of samples as needed
        #             input_shape=(1, 64, 64)
        #         )
        #     logger.info("Dummy data generation complete.")

        if args.dataset == "lits":
            logger.info("Checking for LiTS dataset...")
            download_and_prepare_lits(data_dir="./data/lits", kaggle_dataset_list = ["andrewmvd/liver-tumor-segmentation","andrewmvd/liver-tumor-segmentation-part-2"])
            logger.info("LiTS data is ready.")

            '''# Update these paths to match your actual dataset location
            base_path = "C:\\Users\\Student\\Desktop\\0210\\DM\\Aakar\\DataMining\\data\\lits"
            image_paths = [os.path.join(base_path, "images", f"im{i}.nii.gz") for i in range(131)]
            mask_paths = [os.path.join(base_path, "labels", f"lb{i}.nii.gz") for i in range(131)]

            target_shape = (96, 192, 192)  
            # Create the dataset
            dataset = LiTSDataset(image_paths, mask_paths, augment=True)         

            # Split dataset among clients
            num_clients = args.num_clients
            client_datasets = []
            client_size = len(dataset) // num_clients

            for i in range(num_clients):
                start = i * client_size
                end = (i + 1) * client_size if i < num_clients - 1 else len(dataset)
                
                client_image_paths = image_paths[start:end]
                client_mask_paths = mask_paths[start:end]

                client_dataset = LiTSDataset(client_image_paths, client_mask_paths, target_shape=target_shape)
                train_loader, val_loader = create_data_loaders(client_dataset)

                client_datasets.append((train_loader, val_loader))
                
            print(f"[INFO] Created {args.num_clients} client DataLoaders using {args.split_strategy} strategy.")
            print(f"[INFO] Training and validation DataLoaders are ready.")'''
            
            # Prepare client-specific folders
            # logger.info("Preparing client-specific dataset folders...")
            # prepare_client_folders(
            #     data_dir="./data/lits",
            #     client_base_dir="./client_{i}_lits",
            #     num_clients=args.num_clients
            # )

        logger.info(f"Initializing {args.model} model for Federated Server with {args.dataset}...")

        shared_model = get_model(args.model)
        
        # Model, Criterion, and Optimizer
        
        # global_model = get_model(args.model) #EnhancedUNet3D().to(device)
        # criterion = DiceBCELoss()
        # optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
        server = FederatedServer(shared_model, learning_rate=args.learning_rate)

        # Prepare client folders
        # prepare_client_folders(data_dir="./data/lits", client_base_dir="./data/clients/client_{i}", num_clients=args.num_clients)

        # Initialize Client
        logger.info(f"Creating {args.num_clients} clients...")
        clients = []
        for i in range(args.num_clients):
            # model = get_model(args.model)
            
            '''if args.dataset == "lits":
                dataset = LiTSDataset(data_dir=f"./data/clients/client_{i}", input_shape=(1, 64, 64))

            clients.append(FederatedClient(model, dataset, heterogeneity_factor=1.0))'''
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if args.model.lower() == "unet":
                model =  UNet().to(device)
            elif args.model.lower() == "eunet":
                model =  EnhancedUNet3D().to(device)

            # Update these paths to match your actual dataset location
            base_path = "C:\\Users\\Student\\Desktop\\0210\\DM\\Aakar\\DataMining\\data\\lits"
            image_paths = [os.path.join(base_path, "images", f"volume-{i}.nii") for i in range(131)]
            mask_paths = [os.path.join(base_path, "segmentations", f"segmentation-{i}.nii") for i in range(131)]

            target_shape = (96, 192, 192)  
            # Create the dataset
            if args.dataset == "lits":
                # Define a transformation pipeline
                print("error here")
                
                dataset = LiTSDataset(image_paths, mask_paths, target_shape=target_shape, augment=True)         

            # Split dataset among clients
            num_clients = args.num_clients
            client_datasets = []
            client_size = len(dataset) // num_clients

            for i in range(num_clients):
                start = i * client_size
                end = (i + 1) * client_size if i < num_clients - 1 else len(dataset)
                
                client_image_paths = image_paths[start:end]
                client_mask_paths = mask_paths[start:end]

                
                client_dataset = LiTSDataset(client_image_paths, client_mask_paths, target_shape=target_shape)
                train_loader, val_loader = create_data_loaders(client_dataset)

                client_datasets.append((train_loader, val_loader))
                clients.append(FederatedClient(model, client_dataset, heterogeneity_factor=1.0))
                
            print(f"[INFO] Created {args.num_clients} client DataLoaders using {args.split_strategy} strategy.")
            print(f"[INFO] Training and validation DataLoaders are ready.")


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
        logger.info("Federated Training Complete.")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
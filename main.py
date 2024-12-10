import argparse
from experiments.run_experiment import main as run_experiment
from models.model_loader import get_model  # Ensure this import is present
from utils.logger import setup_logger

def parse_args():
    """
    Parse command-line arguments for the main script.
    """
    parser = argparse.ArgumentParser(description="FedSeg Framework")
    
    # General options
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test'], help='Task to perform')
    
    # Experiment options
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the masks directory")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for local training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--num_clients', type=int, default=5, help='Number of clients in the federated setup')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of parallel processes for client training')
    parser.add_argument('--dataset', type=str, default='dummy', choices=['dummy', 'lits'], help='Dataset to use')
    parser.add_argument("--split_strategy", type=str, default="iid", choices=["iid", "non-iid", "random"],
                    help="Splitting strategy for clients")

    # Model options
    parser.add_argument('--model', type=str, default='unet', choices=['unet'], help='Model to use')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a pre-trained model to load')  # Add this line

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    logger = setup_logger()

    # Initialize shared model
    logger.info(f"Initializing {args.model} model...")
    shared_model = get_model(args.model)

    # Load pre-trained model if specified
    if args.load_model:
        try:
            logger.info(f"Loading pre-trained model from {args.load_model}...")
            shared_model.load_model(path=args.load_model)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {str(e)}")
            return

    # Pass control to run_experiment
    if args.task == 'train':
        logger.info(f"Starting training on {args.dataset} dataset...")
        run_experiment(args)
    elif args.task == 'test':
        logger.info("Testing is not implemented yet.")
    else:
        logger.error(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()

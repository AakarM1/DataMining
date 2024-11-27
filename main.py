import argparse
from experiments.run_experiment import main as run_experiment

def parse_args():
    """
    Parse command-line arguments for the main script.
    """
    parser = argparse.ArgumentParser(description="FedSeg Framework")
    
    # General options
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test'], help='Task to perform')
    
    # Experiment options
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for local training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--num_clients', type=int, default=5, help='Number of clients in the federated setup')

    # Model options
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'other_model'], help='Model to use')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.task == 'train':
        print("Starting training...")
        run_experiment(args)
    elif args.task == 'test':
        print("Testing is not implemented yet.")
    else:
        print(f"Unknown task: {args.task}")

if __name__ == "__main__":
    main()

import logging

def setup_logger(log_file="experiment.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def log_epoch_progress(logger, epoch, total_epochs):
    logger.info(f"Epoch {epoch + 1}/{total_epochs} completed.")

def log_client_progress(logger, client_id, message):
    logger.info(f"Client {client_id}: {message}")

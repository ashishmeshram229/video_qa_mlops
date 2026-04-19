import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from src.config.core import config

def get_logger(logger_name: str) -> logging.Logger:
    """
    Creates and returns a production-grade logger.
    Outputs to console (stdout) and a rotating file system.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(logger_name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 1. Console Handler (Crucial for Docker/Airflow container logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 2. Rotating File Handler (Prevents log files from consuming all disk space)
    log_file_path = os.path.join(config.LOG_DIR, "system_runtime.log")
    file_handler = RotatingFileHandler(
        filename=log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
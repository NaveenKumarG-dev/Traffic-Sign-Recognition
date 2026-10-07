"""
Logging configuration for Traffic Sign Recognition project.
Provides a centralized logger with both console and file output.
"""

import os
import logging
from src.config import PROJECT_ROOT, load_config


def setup_logger(name="traffic_sign_recognition", log_file=None):
    """Set up and return a configured logger.
    
    Args:
        name: Logger name
        log_file: Path to log file. If None, uses config default.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    config = load_config()
    
    # Determine log file path
    if log_file is None:
        log_file = os.path.join(
            PROJECT_ROOT, config.get("logging", {}).get("log_file", "artifacts/logs/training.log")
        )
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added yet
    if not logger.handlers:
        log_level = getattr(
            logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO
        )
        logger.setLevel(log_level)
        
        # Log format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name="traffic_sign_recognition"):
    """Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger

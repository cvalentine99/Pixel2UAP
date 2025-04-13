"""
Logging configuration for the application.
"""

import os
import logging
from datetime import datetime

from voxel_projector_v2 import CONFIG_PATH


def setup_logging(level="INFO"):
    """
    Configure application-wide logging.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logs directory
    logs_dir = os.path.join(CONFIG_PATH, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(logs_dir, f"voxel_projector_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log initial information
    logging.info(f"Logging initialized at level {level}")
    logging.info(f"Log file: {log_file}")
    
    return log_file
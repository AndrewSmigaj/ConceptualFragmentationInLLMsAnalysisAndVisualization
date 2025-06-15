"""
Centralized logging configuration for Unified CTA
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__ from calling module)
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - one file per day
    timestamp = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(
        log_path / f'unified_cta_{timestamp}.log'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create module-level logger
logger = setup_logging(__name__)
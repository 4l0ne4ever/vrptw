"""
Logging system for VRP-GA System.
Provides centralized logging with file and console handlers.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None, 
                 level: int = logging.INFO, log_dir: str = "logs") -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path. If None, uses default naming.
        level: Logging level (default: INFO)
        log_dir: Directory for log files (default: "logs")
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__, 'logs/vrp_ga.log')
        >>> logger.info("Starting GA optimization...")
        >>> logger.debug(f"Generation {gen}: best fitness = {fitness}")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        # Default log file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'vrp_ga_{timestamp}.log')
    else:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, os.path.basename(log_file))
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one with default settings.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create with default settings if not exists
        return setup_logger(name)
    
    return logger


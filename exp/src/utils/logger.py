"""Logging setup utilities."""

import logging
from datetime import datetime


def setup_logger(name: str = "tiered_pricing", level: int = logging.INFO) -> logging.Logger:
    """Configure and return logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

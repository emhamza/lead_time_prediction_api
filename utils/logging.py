import logging
import sys
from loguru import logger

def setup_logging():
    """Configure logging for the API"""
    logging.basicConfig(level=logging.INFO)
    logger.configure(
        handlers=[
            {"sink": sys.stdout, "format": "{time} {level} {message}"},
            {"sink": "logs/api.log", "rotation": "500 MB", "retention": "10 days"}
        ]
    )
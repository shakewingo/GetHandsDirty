import logging
from pathlib import Path

def get_logger(name: str, level=logging.INFO, stream=False):
    """
    Create a logger with file handler and optionally stream handler
    
    Args:
        name: Name of the logger/module
        level: Logging level
        stream: Whether to add StreamHandler (default: False)
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f"{Path(name).stem}.log")
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    # Optionally add stream handler
    if stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
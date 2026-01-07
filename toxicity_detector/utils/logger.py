import logging
from pathlib import Path


# Logger configuration
def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger instance with the given name and level.

    Parameters
    ----------
    name : str
        Name of the logger.
    log_file : Path, optional
        Path to the log file. If provided, a file handler will be added to the logger.
    level : int, optional
        Logging level. Default is logging.INFO.

    Returns
    -------s
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
        if log_file is not None:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            
            logger.addHandler(fh)
            
    return logger

import logging
import logging.config
import os
from pathlib import Path

from utils import read_json


def setup_logging(
    save_dir,
    log_config=os.path.join(os.path.dirname(__file__), "logger_config.json"),
    default_level=logging.INFO,
):
    """
    logging conf setup
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = os.path.join(save_dir, handler["filename"])
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def get_logger(name, verbosity=2):

    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    assert (
        verbosity in log_levels
    ), f"Verbosity option {verbosity} is invalid. Valid options are {log_levels}"
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger

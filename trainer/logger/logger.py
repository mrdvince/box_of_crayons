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
                handler["filename"] = str(save_dir / handler["filename"])
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

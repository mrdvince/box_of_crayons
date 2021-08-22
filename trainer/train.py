from datetime import datetime
from pathlib import Path
import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from config import Config
from trainer import Trainer
from logger import get_logger
from utils.util import prepare_device
import os

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = get_logger("train")

    # setup data_loader instances
    data_loader = getattr(module_data, config.data_loader["type"])(
        **config.data_loader["args"]
    )  # config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = getattr(module_arch, config.arch["type"])(**config.arch["args"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config.n_gpu)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config.loss)
    metrics = [getattr(module_metric, met) for met in config.metrics]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = getattr(torch.optim, config.optimizer["type"])(
        model.parameters(), **config.optimizer["args"]
    )
    lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler["type"])(
        optimizer, **config.lr_scheduler["args"]
    )

    experiment_name = config.name
    run_id = datetime.now().strftime("%d%m_%H%M%S")
    _save_dir = os.path.join(os.path.dirname(__file__), config.trainer["save_dir"])

    Path(os.path.join(_save_dir, "models", experiment_name, run_id)).mkdir(
        parents=True, exist_ok=True
    )

    Path(os.path.join(_save_dir, "log", experiment_name, run_id)).mkdir(
        parents=True, exist_ok=True
    )
    config.save_dir = os.path.join(_save_dir, "models", experiment_name, run_id)
    config.log_dir = os.path.join(_save_dir, "log", experiment_name, run_id)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    config = Config()
    main(config)

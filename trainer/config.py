from model import model
from data_loader import data_loaders

import torch


class Config:
    name = "box_of_crayons[boc]"
    n_gpu = 1
    loss = "nll_loss"

    @property
    def arch(self):
        return {"type": "model", "args": {}}

    @property
    def metrics(self):
        return ["accuracy", "top_k_acc"]

    @property
    def lr_scheduler(self):
        return {"type": "StepLR", "args": {"step_size": 20, "gamma": 0.1}}

    @property
    def trainer(self):
        return {
            "epochs": 15,
            "save_dir": "saved/",
            "save_period": 1,
            "verbosity": 2,
            "monitor": "min val_loss",
            "early_stop": 10,
            "tensorboard": True,
        }

    @property
    def optimizer(self):
        return {
            "type": "Adam",
            "args": {"lr": 0.001, "weight_decay": 0, "amsgrad": True},
        }

    @property
    def data_loader(self):
        return {
            "type": "Loader",
            "args": {
                "data_dir": "/mnt/c/Users/vinc3/Pictures",
                "batch_size": 16,
                "shuffle": True,
                "validation_split": 0.1,
                "num_workers": 2,
            },
        }


# config = Config()
# config.ld = "awesome"
# # dl = getattr(data_loaders, config.data_loader['type'])(**config.data_loader["args"])
# # print(dl.dataset.class_to_idx)
# import numpy as np
# ml = getattr(model, config.arch["type"])(**config.arch["args"])
# trainable_params = ml.classifier.parameters()
# opt = getattr(torch.optim, config.optimizer["type"])(
#     ml.parameters(), **config.optimizer["args"]
# )
# print(sum(p.numel() for p in ml.parameters() if p.requires_grad))
# # print(ml)#sum([np.prod(p.size()) for p in trainable_params]))
# # # lr = getattr(torch.optim.lr_scheduler, config.lr_scheduler["type"])(opt, **config.lr_scheduler["args"])
# # from pathlib import Path
# # import os
# # print(Path(os.path.join(os.path.dirname(__file__),config.trainer['save_dir'])))
# save_dir
# log_dir
# resume

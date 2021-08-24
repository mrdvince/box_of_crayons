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
                "data_dir": "/data/nebo/PlantVillage",
                "batch_size": 64,
                "shuffle": True,
                "validation_split": 0.2,
                "num_workers": 4 * self.n_gpu,
                "pin_memory": True,
            },
        }

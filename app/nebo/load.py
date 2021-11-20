"""Loads a yolo model with given parameters
"Borrowed from" https://github.com/ultralytics/yolov5/blob/master/hubconf.py
"""
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../", "trainer"))

from models.experimental import attempt_load
from models.yolo import Model
from utils.downloads import attempt_download
from utils.general import set_logging


def _create(
    name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True
):
    """Load and return a yolo model"""
    file = Path(__file__).resolve()
    set_logging(verbose=verbose)

    save_dir = Path("") if str(name).endswith(".pt") else file.parent
    path = (save_dir / name).with_suffix(".pt")  # checkpoint path
    try:
        device = "0" if torch.cuda.is_available() else "cpu"

        if pretrained and channels == 3 and classes == 80:
            model = attempt_load(path, map_location=device)  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / "models").rglob(f"{name}.yaml"))[
                0
            ]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                msd = model.state_dict()  # model state_dict
                csd = (
                    ckpt["model"].float().state_dict()
                )  # checkpoint state_dict as FP32
                csd = {
                    k: v for k, v in csd.items() if msd[k].shape == v.shape
                }  # filter
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt["model"].names) == classes:
                    model.names = ckpt["model"].names  # set class names attribute
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)

    except Exception as e:
        s = "Couldn't load model, double check filename path."
        raise Exception(s) from e


def load_model(path="path/to/model.pt", autoshape=True, verbose=True):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=verbose)

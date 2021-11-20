import glob
import os
import re
from collections import namedtuple
from io import BytesIO
from pathlib import Path

import wandb
from app.nebo.load import load_model
from PIL import Image

project = os.environ["PROJECT"]
entity = os.environ["ENTITY"]
model = os.environ["WANDB_MODEL"]
model_name = os.environ["MODEL_NAME"]

weights = f'{os.environ["ARTIFACT_DIR"]}/{model}/{model_name}'


def get_model():
    wandb.login(key=os.environ["WANDB_KEY"])
    run = wandb.init(project=project, entity=entity)
    artifact = run.use_artifact(f"{entity}/{project}/{model}", type="model")
    artifact_dir = artifact.download(f'{os.environ["ARTIFACT_DIR"]}/{model}')


if not os.path.exists(weights):
    get_model()


def read_imagefile(data) -> Image.Image:
    img_stream = BytesIO(data)
    img = Image.open(img_stream)
    return img


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/ --> runs/{sep}2, runs/{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    if not path.exists() and mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def read_imagefile(data) -> Image.Image:
    img_stream = BytesIO(data)
    img = Image.open(img_stream)
    return img


def predict(thres: namedtuple, bytes: BytesIO, id: str):
    """
    Return a prediction given a file name
    """
    img = read_imagefile(bytes.file.read())
    filename = bytes.filename
    filename_path = f"{os.environ['IMAGE_DIR']}/{filename}"
    img.save(f"{filename_path}")
    model = load_model(path=weights)
    model.conf = thres.conf_thres
    model.iou = thres.iou_thres
    results = model(filename_path, size=640)
    save_dir = increment_path(f"{os.environ['RUNS']}/{id}", sep=".", mkdir=True)
    results.save(save_dir=save_dir)
    results.crop(save=True, save_dir=save_dir)

    return (
        (save_dir, filename),
        list(Path(save_dir).glob("crops/*/*")),
        results.pandas().xyxy[0][["confidence", "name"]].to_json(orient="records"),
    )

import os
from pathlib import Path
from typing import Any

import wandb
from app import crud, models
from app.api import deps
from app.checkpoint.pred import predict
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, responses
from models.experimental import attempt_load
from sqlalchemy.orm import Session

from utils.torch_utils import select_device

router = APIRouter()


model_name = "best.pt"
artifact_dir = "/app/artifacts"
weights = os.path.join(
    os.path.dirname(__file__),
    "../../",
    artifact_dir,
    f"{os.environ['WANDB_MODEL']+'/'+model_name}",
)
if not os.path.exists(weights):
    wandb.login(key=os.environ["WANDB_KEY"])
    run = wandb.init(project="box_of_crayons", entity="droid")
    artifact = run.use_artifact(
        f"droid/box_of_crayons/{os.environ['WANDB_MODEL']}", type="model"
    )
    artifact_dir = artifact.download("/app/artifacts")


def preload_model(weights, device):
    w = weights[0] if isinstance(weights, list) else weights
    _, suffix = False, Path(w).suffix.lower()
    pt, _, _, _, _ = (
        suffix == x for x in [".pt", ".onnx", ".tflite", ".pb", ""]
    )  # backend
    stride, _ = 64, [f"class{i}" for i in range(1000)]  # assign defaults

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    return pt, stride, model


# Load model
pt, stride, model = preload_model(weights, select_device())


@router.post("/")
def get_predictions(
    conf_thresh: float,
    iou_thresh: float = 0.4,
    db: Session = Depends(deps.get_db),
    file: UploadFile = File(...),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Upload image and get prediction
    """
    if crud.user.is_active(current_user):
        res_str, inf_ls, cinames = predict(
            pt,
            stride,
            model,
            file,
            iou_thres=iou_thresh,
            conf_thres=conf_thresh,
            weights=weights,
        )
        pathogens = set([str(cname).split("/")[3] for cname in cinames])
        res_str = {
            "no_pathogens": len(pathogens),
            "pathogen_names": list(pathogens),
            "image_crops": cinames,
        }
        return res_str, responses.FileResponse(inf_ls[0], filename=inf_ls[1])
    else:
        raise HTTPException(
            status_code=400,
            detail="The user with this username isn't active.",
        )

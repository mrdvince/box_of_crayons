import os
from collections import namedtuple
from pathlib import Path
from typing import Any

from app import crud, models
from app.api import deps
from app.nebo.pred import predict
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, responses

router = APIRouter()


@router.post("/")
def get_predictions(
    id: str,
    conf_thresh: float,
    iou_thresh: float = 0.4,
    file: UploadFile = File(...),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Upload image and get prediction
    """
    if crud.user.is_active(current_user):
        thres = namedtuple("thres", ["conf_thres", "iou_thres"])
        inf_list, crops, results = predict(
            thres(conf_thres=conf_thresh, iou_thres=iou_thresh), file, id
        )
        # pylint: disable=R1718
        pathogens = set([str(cname).split("/")[3] for cname in crops])
        return {
            "no_pathogens": len(pathogens),
            "pathogen_names": list(pathogens),
            "image_crops": crops,
            "results": results,
            "file_response": responses.FileResponse(
                inf_list[0], filename=inf_list[1].split(".")[0] + ".jpg"
            ),
        }
    raise HTTPException(
        status_code=400,
        detail="The user with this username isn't active.",
    )

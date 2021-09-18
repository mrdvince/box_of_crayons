from typing import Any
from app.checkpoint.pred import predict

from fastapi import APIRouter, Depends, UploadFile, File, Response, HTTPException
from sqlalchemy.orm import Session
from app import models, crud

from app.api import deps


router = APIRouter()


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
        img, res_str = predict(file, iou_thres=iou_thresh, conf_thres=conf_thresh)

        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
        }
        headers["response_string"] = str(res_str)

        response = Response(
            content=img.getvalue(), media_type="image/jpg", headers=headers
        )
        return response
    else:
        raise HTTPException(
            status_code=400,
            detail="The user with this username isn't active.",
        )

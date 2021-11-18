from typing import Any

from app import crud, models
from app.api import deps
from app.nebo.gpt_davinci import davinci
from fastapi import APIRouter, Depends, HTTPException


router = APIRouter()


@router.post("/")
def gpt_davinci(
    question: str,
    chat_log: str = None,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Upload image and get prediction
    """
    if crud.user.is_active(current_user):
        return davinci(question=question, chat_log=chat_log)
    else:
        raise HTTPException(
            status_code=400,
            detail="The user with this username isn't active.",
        )

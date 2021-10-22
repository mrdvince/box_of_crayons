from app.api.api_v1.endpoints import login, predict, users
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(predict.router, prefix="/pred", tags=["pred image"])
api_router.include_router(login.router, tags=["users"])
api_router.include_router(users.router, prefix="/users", tags=["users"])

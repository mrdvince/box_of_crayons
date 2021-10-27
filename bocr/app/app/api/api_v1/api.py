from app.api.api_v1.endpoints import login, predict, users, gpt_davinci
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(gpt_davinci.router, prefix="/gpt", tags=["nebo"])
api_router.include_router(predict.router, prefix="/pred", tags=["nebo"])
api_router.include_router(login.router, tags=["users"])
api_router.include_router(users.router, prefix="/users", tags=["users"])

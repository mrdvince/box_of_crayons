from typing import Any

from fastapi import FastAPI  # File, Response, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from app.api.api_v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# setup cors
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/", tags=["redirect"])
def redirect_to_docs() -> Any:
    return RedirectResponse(url="redoc")


app.include_router(api_router, prefix=settings.API_V1_STR)

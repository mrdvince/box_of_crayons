from typing import Any

from fastapi import FastAPI  # File, Response, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.api_v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
Given a plant leaf photo, 
return a bounding box of the type of disease it has
(limited to the 8 classes trained on). 

Also get some answers to your question using OpenAI's GPT3 (Am on free credits so will be switching to the ada model from davinci)

Model used is a YoloV5
> Note: the dataset used was manually labelled and still needs to be improved for a better and more accurate model

### Disease types
- Apple Scab
- Apple Cedar Rust
- Apple Frogeye Spot
- Maize Gray Leaf Spot
- Maize Leaf Blight
- Potato Blight
- Tomato Bacteria Spot
- Tomato Blight
    """,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)
app.mount("/runs", StaticFiles(directory="runs", html=True), name="runs")

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

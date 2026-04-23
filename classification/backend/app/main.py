"""FastAPI application for classification pipeline."""

import logging
from pathlib import Path

# Load .env from repo root so Azure OpenAI (appendage subtype) credentials are available
try:
    import dotenv
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent
    dotenv.load_dotenv(_repo_root / ".env")
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import (
    sessions,
    type_classification,
    decoration_classification,
    color_classification,
    texture_classification,
    pottery_classification,
    images,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Classification Pipeline API",
    description="Local API for pottery classification (type, decoration, color, texture)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175", "http://localhost:5174", "http://127.0.0.1:5175", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(type_classification.router, prefix="/sessions", tags=["type-classification"])
app.include_router(decoration_classification.router, prefix="/sessions", tags=["decoration-classification"])
app.include_router(color_classification.router, prefix="/sessions", tags=["color-classification"])
app.include_router(texture_classification.router, prefix="/sessions", tags=["texture-classification"])
app.include_router(pottery_classification.router, prefix="/sessions", tags=["pottery-classification"])
app.include_router(images.router, prefix="/sessions", tags=["images"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "Classification Pipeline API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "classification-pipeline-api", "version": "1.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})

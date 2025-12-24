"""FastAPI main application for preprocess pipeline."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.routes import sessions, stage1_cards, stage2_mask, stage3_scale, images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Preprocess Pipeline API",
    description="Local API for 3-stage preprocessing pipeline (card detection, segmentation, scale calculation)",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(stage1_cards.router, prefix="/sessions", tags=["stage1-cards"])
app.include_router(stage2_mask.router, prefix="/sessions", tags=["stage2-mask"])
app.include_router(stage3_scale.router, prefix="/sessions", tags=["stage3-scale"])
app.include_router(images.router, prefix="/sessions", tags=["images"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Preprocess Pipeline API is running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": "preprocess-pipeline-api",
        "version": "1.0.0"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


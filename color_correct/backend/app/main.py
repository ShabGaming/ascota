"""FastAPI main application for color correction."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.routes import sessions, cluster, preview, export as export_route, reference_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Color Correction API",
    description="Local API for batch color correction with clustering",
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
app.include_router(cluster.router, prefix="/sessions", tags=["clustering"])
app.include_router(preview.router, prefix="/sessions", tags=["preview"])
app.include_router(export_route.router, prefix="/sessions", tags=["export"])
app.include_router(reference_images.router, prefix="/sessions", tags=["reference-images"])
# Add preset references endpoints
from app.routes import reference_images
app.add_api_route("/preset-references", reference_images.list_preset_references, methods=["GET"], tags=["reference-images"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Color Correction API is running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": "color-correction-api",
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


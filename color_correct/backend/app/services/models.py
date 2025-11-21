"""Pydantic models for color correction API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class JobStatus(str, Enum):
    """Status of a background job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageSourceMode(str, Enum):
    """Image source selection mode."""
    PROXY_450 = "450px"
    PROXY_1500 = "1500px"
    PROXY_3000 = "3000px"
    RAW_MODE = "raw_mode"


class SessionOptions(BaseModel):
    """Options for a color correction session."""
    image_source: ImageSourceMode = Field(ImageSourceMode.PROXY_3000, description="Primary image source with fallback")
    overwrite: bool = Field(False, description="Overwrite existing files if True")
    custom_k: Optional[int] = Field(None, description="Number of clusters (auto-detect if None)")
    sensitivity: float = Field(1.0, description="Clustering sensitivity")
    preview_resolution: int = Field(1500, description="Preview resolution width in pixels (450, 1500, or 3000)")


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    contexts: List[str] = Field(..., description="List of context directory paths")
    options: SessionOptions


class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str


class ImageItem(BaseModel):
    """Represents a single image with all its variants."""
    id: str = Field(..., description="Unique identifier (hash of primary path)")
    context_id: str = Field(..., description="Context directory this image belongs to")
    find_number: str = Field(..., description="Find number within context")
    raw_path: Optional[str] = Field(None, description="Path to RAW file (.CR3/.CR2)")
    proxy_3000: Optional[str] = Field(None, description="Path to -3000 proxy")
    proxy_1500: Optional[str] = Field(None, description="Path to -1500 proxy")
    proxy_450: Optional[str] = Field(None, description="Path to base/450 proxy")
    primary_path: str = Field(..., description="Primary path used for clustering")


class CorrectionParams(BaseModel):
    """Parameters for color correction."""
    temperature: float = Field(0.0, description="Temperature adjustment (-100 to 100)")
    tint: float = Field(0.0, description="Tint adjustment (-100 to 100)")
    exposure: float = Field(0.0, description="Exposure adjustment (-2 to 2 EV)")
    contrast: float = Field(1.0, description="Contrast multiplier (0.5 to 2.0)")
    saturation: float = Field(1.0, description="Saturation multiplier (0 to 2.0)")
    red_gain: float = Field(1.0, description="Red channel gain (0.5 to 2.0)")
    green_gain: float = Field(1.0, description="Green channel gain (0.5 to 2.0)")
    blue_gain: float = Field(1.0, description="Blue channel gain (0.5 to 2.0)")


class Cluster(BaseModel):
    """A cluster of similar images."""
    id: str
    image_ids: List[str]
    correction_params: Optional[CorrectionParams] = None


class ClusterResponse(BaseModel):
    """Response containing clustering results."""
    clusters: List[Cluster]
    images: Dict[str, ImageItem]


class MoveImageRequest(BaseModel):
    """Request to move an image to a different cluster."""
    image_id: str
    target_cluster_id: str


class SetCorrectionRequest(BaseModel):
    """Request to set correction parameters for a cluster."""
    params: CorrectionParams


class JobStatusResponse(BaseModel):
    """Response for job status queries."""
    status: JobStatus
    progress: float = Field(0.0, description="Progress from 0.0 to 1.0")
    message: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class ExportSummary(BaseModel):
    """Summary of export operation."""
    total_images: int
    total_files_written: int
    overwritten_count: int
    new_files_count: int
    failed_count: int
    errors: List[str] = []


class ExportResponse(BaseModel):
    """Response for export operation."""
    job_id: str
    summary: Optional[ExportSummary] = None


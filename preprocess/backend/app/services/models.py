"""Pydantic models for preprocess pipeline API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ImageItem(BaseModel):
    """Represents a single image with all its variants."""
    id: str = Field(..., description="Unique identifier (hash of primary path)")
    context_id: str = Field(..., description="Context directory this image belongs to")
    find_number: str = Field(..., description="Find number within context")
    raw_path: Optional[str] = Field(None, description="Path to RAW file (.CR3/.CR2)")
    proxy_3000: Optional[str] = Field(None, description="Path to -3000 proxy")
    proxy_1500: Optional[str] = Field(None, description="Path to -1500 proxy")
    proxy_450: Optional[str] = Field(None, description="Path to base/450 proxy")
    primary_path: str = Field(..., description="Primary path used")


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    contexts: List[str] = Field(..., description="List of context directory paths")


class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str


class CardDetection(BaseModel):
    """A detected color card."""
    card_id: str = Field(..., description="Unique identifier for this card")
    card_type: str = Field(..., description="Type: 24_color_card, 8_hybrid_card, or checker_card")
    coordinates: List[List[float]] = Field(..., description="4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")


class ImageCardResult(BaseModel):
    """Card detection results for a single image."""
    image_id: str
    image_size: List[int] = Field(..., description="[width, height] in pixels")
    cards: List[CardDetection] = Field(default_factory=list)
    error: Optional[str] = None


class Stage1Results(BaseModel):
    """Results from Stage 1 card detection."""
    results: Dict[str, ImageCardResult] = Field(default_factory=dict)


class UpdateCardsRequest(BaseModel):
    """Request to update card coordinates for an image."""
    cards: List[CardDetection]


class MaskResult(BaseModel):
    """Mask generation result for an image."""
    image_id: str
    mask_path: Optional[str] = Field(None, description="Path to mask file relative to .ascota folder")
    error: Optional[str] = None


class Stage2Results(BaseModel):
    """Results from Stage 2 mask generation."""
    results: Dict[str, MaskResult] = Field(default_factory=dict)


class UpdateMaskRequest(BaseModel):
    """Request to update mask (base64 encoded PNG)."""
    mask_data: str = Field(..., description="Base64 encoded PNG image data")


class ScaleResult(BaseModel):
    """Scale calculation result for an image."""
    image_id: str
    pixels_per_cm: Optional[float] = None
    surface_area_cm2: Optional[float] = None
    method: Optional[str] = Field(None, description="checker_card or 8_hybrid_card")
    card_used: Optional[str] = None
    centers: Optional[List[List[float]]] = Field(None, description="Circle centers for 8_hybrid_card in original image coordinates [[x1,y1], [x2,y2], [x3,y3]]")
    error: Optional[str] = None


class Stage3Results(BaseModel):
    """Results from Stage 3 scale calculation."""
    results: Dict[str, ScaleResult] = Field(default_factory=dict)


class UpdateCentersRequest(BaseModel):
    """Request to update 8-hybrid circle centers."""
    centers: List[List[float]] = Field(..., description="3 circle centers [[x1,y1], [x2,y2], [x3,y3]]")


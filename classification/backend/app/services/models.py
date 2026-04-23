"""Pydantic models for classification API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request to create a classification session (single context)."""
    context_path: str = Field(..., description="Path to context directory")
    classification_type: str = Field("type", description="One of: type, decoration, color, texture, pottery")


class LoadSessionRequest(BaseModel):
    """Request to load/restore a session from disk."""
    session_id: str = Field(..., description="Session ID to restore")


class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str
    items_count: int


class TypeRunRequest(BaseModel):
    """Request to run type classification."""
    enable_appendage_subtype: bool = Field(True, description="Use Azure OpenAI for appendage subtypes")
    resolution: int = Field(1500, description="Target resolution: 1000, 1500, or 3000")


class TypeRunResponse(BaseModel):
    """Response from type run."""
    results: List[Dict[str, Any]] = Field(default_factory=list)


class UpdateTypeResultRequest(BaseModel):
    """Request to update a single type result (manual override)."""
    label: str = Field(..., description="New classification label")


class PotteryRunRequest(BaseModel):
    """Request to run pottery vs non-pottery classification."""
    resolution: int = Field(1500, description="Target resolution: 1000, 1500, or 3000")


class PotteryRunResponse(BaseModel):
    """Response from pottery binary run."""
    results: List[Dict[str, Any]] = Field(default_factory=list)


class UpdatePotteryResultRequest(BaseModel):
    """Request to update a single pottery binary result (manual override)."""
    label: str = Field(..., description="pottery or non_pottery")


class DecorationRunRequest(BaseModel):
    """Request to run decoration classification (no extra options)."""
    resolution: int = Field(1500, description="Target resolution: 1000, 1500, or 3000")


class DecorationRunResponse(BaseModel):
    """Response from decoration run."""
    results: List[Dict[str, Any]] = Field(default_factory=list)


class UpdateDecorationResultRequest(BaseModel):
    """Request to update a single decoration result (manual override)."""
    label: str = Field(..., description="New classification label (Impressed, Incised, or custom)")


class ExportResponse(BaseModel):
    """Response from export."""
    saved_finds: int


# Color clustering
class ColorRunRequest(BaseModel):
    """Optional options for color run (e.g. preview resolution)."""
    resolution: Optional[int] = Field(1500, description="Preview resolution: 1000, 1500, or 3000")


class ColorRunResponse(BaseModel):
    """Response from color run."""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    clusters: List[Dict[str, Any]] = Field(default_factory=list)  # [{ cluster_id, item_ids }]
    noise_item_ids: List[str] = Field(default_factory=list)
    cluster_names: Dict[int, str] = Field(default_factory=dict)


class ColorResultsResponse(BaseModel):
    """Response for color results GET."""
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cluster_names: Dict[int, str] = Field(default_factory=dict)
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    noise_item_ids: List[str] = Field(default_factory=list)


# Texture clustering (same shape as color; stored under classification_type "texture")
class TextureRunRequest(BaseModel):
    """Optional options for texture run (e.g. preview resolution)."""
    resolution: Optional[int] = Field(1500, description="Preview resolution: 1000, 1500, or 3000")


class TextureRunResponse(BaseModel):
    """Response from texture run."""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    noise_item_ids: List[str] = Field(default_factory=list)
    cluster_names: Dict[int, str] = Field(default_factory=dict)


class TextureResultsResponse(BaseModel):
    """Response for texture results GET."""
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cluster_names: Dict[int, str] = Field(default_factory=dict)
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    noise_item_ids: List[str] = Field(default_factory=list)


class ReclusterRequest(BaseModel):
    """Request to recluster with new HDBSCAN params."""
    min_cluster_size: Optional[int] = Field(None, description="HDBSCAN min_cluster_size")
    min_samples: Optional[int] = Field(None, description="HDBSCAN min_samples")
    cluster_selection_epsilon: Optional[float] = Field(None, description="HDBSCAN cluster_selection_epsilon")
    cluster_selection_method: Optional[str] = Field(None, description="eom or leaf")


class UpdateColorResultRequest(BaseModel):
    """Request to move item to another cluster."""
    cluster_id: int = Field(..., description="Target cluster id (-1 for noise)")


class SetClusterNameRequest(BaseModel):
    """Request to set cluster display name."""
    name: str = Field(..., description="Display name for the cluster")

"""MobileSAM session API for interactive point-prompt segmentation.

Loads the model once; call set_image once per frame, then predict() with
multi-point prompts without reloading weights or re-encoding the image.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

Point = Tuple[int, int, int]  # x, y, label (1 foreground, 0 background)

_mobile_sam_model = None
_mobile_sam_predictor = None
_model_loaded = False
_load_lock = threading.Lock()


def get_mobile_sam_checkpoint_path() -> Path:
    """Resolve path to mobile_sam.pt (same search order as preprocess service)."""
    core_dir = Path(__file__).resolve().parent
    repo_root = core_dir.parent.parent

    preprocess_weights = repo_root / "preprocess" / "backend" / "weights" / "mobile_sam.pt"
    if preprocess_weights.exists():
        return preprocess_weights

    project_weights = repo_root / "weights" / "mobile_sam.pt"
    if project_weights.exists():
        return project_weights

    try:
        import mobile_sam

        mobile_sam_path = Path(mobile_sam.__file__).resolve().parent.parent / "weights" / "mobile_sam.pt"
        if mobile_sam_path.exists():
            return mobile_sam_path
    except ImportError:
        pass

    return preprocess_weights


def _load_predictor():
    """Lazy-init global MobileSAM model and SamPredictor."""
    global _mobile_sam_model, _mobile_sam_predictor, _model_loaded

    with _load_lock:
        if _model_loaded:
            return _mobile_sam_predictor

        from mobile_sam import SamPredictor, sam_model_registry

        checkpoint_path = get_mobile_sam_checkpoint_path()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found at {checkpoint_path}. "
                "Download mobile_sam.pt into repo weights/ or preprocess/backend/weights/."
            )

        model_type = "vit_t"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _mobile_sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        _mobile_sam_model.to(device=device)
        _mobile_sam_model.eval()
        _mobile_sam_predictor = SamPredictor(_mobile_sam_model)
        _model_loaded = True

        return _mobile_sam_predictor


class MobileSamSession:
    """One predictor image at a time; thread-safe for single-worker use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image_shape: Optional[Tuple[int, int]] = None

    def set_image(self, rgb: np.ndarray) -> None:
        """Encode image for SAM (call once when the displayed image changes)."""
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 RGB, got shape {arr.shape}")

        predictor = _load_predictor()
        with self._lock:
            predictor.set_image(arr)
            self._image_shape = (arr.shape[0], arr.shape[1])

    @property
    def image_shape(self) -> Optional[Tuple[int, int]]:
        return self._image_shape

    def predict(self, points: List[Point]) -> np.ndarray:
        """Run SAM with multiple point prompts. Labels: 1=foreground, 0=background."""
        if not points:
            raise ValueError("points must be non-empty")
        if self._image_shape is None:
            raise RuntimeError("Call set_image() before predict()")

        h, w = self._image_shape
        coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        labels = np.array([p[2] for p in points], dtype=np.int64)

        for i, (x, y, _) in enumerate(points):
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"Point {i} ({x}, {y}) out of bounds for image size {w}x{h}")

        predictor = _load_predictor()
        with self._lock:
            with torch.inference_mode():
                masks, _scores, _logits = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )

        mask = masks[0]
        return mask.astype(np.uint8)

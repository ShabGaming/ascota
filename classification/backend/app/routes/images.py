"""Image serving for classification session items."""

import io
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from PIL import Image
from fastapi.responses import FileResponse, Response

from app.services.classification_store import get_classification_store
from app.services.image_utils import (
    build_rgba_from_find_and_mask,
    crop_to_foreground_bounds,
    crop_image_to_mask_bounds,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Thumbnail size for focus-on-pottery (transparent) view
FOCUS_IMAGE_SIZE = 500


@router.get("/{session_id}/image/{item_id}")
async def serve_item_image(
    session_id: str,
    item_id: str,
    transparent: int = Query(0, description="If 1, serve masked (transparent background) image"),
    focus: int = Query(0, description="If 1, crop to foreground (use with transparent)"),
):
    """Serve the find image. transparent=1: RGBA masked. focus=1: crop to foreground bbox (implies masked)."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    item = session.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if focus == 1 and transparent == 0:
        try:
            img = crop_image_to_mask_bounds(
                image_3000_path=item["image_3000_path"],
                find_path=item["find_path"],
                mask_path_rel=item["mask_path"],
                padding=8,
            )
            w, h = img.size
            if max(w, h) > FOCUS_IMAGE_SIZE:
                if w >= h:
                    new_w = FOCUS_IMAGE_SIZE
                    new_h = int(round(h * FOCUS_IMAGE_SIZE / w))
                else:
                    new_h = FOCUS_IMAGE_SIZE
                    new_w = int(round(w * FOCUS_IMAGE_SIZE / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return Response(content=buf.getvalue(), media_type="image/jpeg")
        except Exception as e:
            logger.warning(f"Focus (no transparent) image failed for {item_id}: {e}")
            raise HTTPException(status_code=500, detail="Could not build image")

    if transparent == 1 or focus == 1:
        try:
            rgba = build_rgba_from_find_and_mask(
                find_path=item["find_path"],
                image_filename=item["image_filename"],
                image_3000_path=item["image_3000_path"],
                mask_path_rel=item["mask_path"],
                target_size=None,
            )
            if focus == 1:
                rgba = crop_to_foreground_bounds(rgba, padding=8)
            w, h = rgba.size
            if max(w, h) > FOCUS_IMAGE_SIZE:
                if w >= h:
                    new_w = FOCUS_IMAGE_SIZE
                    new_h = int(round(h * FOCUS_IMAGE_SIZE / w))
                else:
                    new_h = FOCUS_IMAGE_SIZE
                    new_w = int(round(w * FOCUS_IMAGE_SIZE / h))
                rgba = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgba.save(buf, format="PNG")
            buf.seek(0)
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as e:
            logger.warning(f"Masked/focus image failed for {item_id}: {e}")
            raise HTTPException(status_code=500, detail="Could not build image")

    path = Path(item["image_3000_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    media_type = "image/jpeg"
    if path.suffix.lower() in [".png", ".PNG"]:
        media_type = "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)

"""Scanner service for discovering classification items (finds with preprocess + masks)."""

from pathlib import Path
from typing import List, Dict, Any
import logging

from app.services.metadata import load_preprocess_json

logger = logging.getLogger(__name__)


def _make_item_id(find_number: str, image_filename: str) -> str:
    """Stable id for (find_number, image_filename)."""
    return f"{find_number}_{image_filename}"


def _find_image_3000_path(find_path: Path, image_filename: str) -> Path | None:
    """Return path to -3000 image in find/photos if it exists."""
    photos = find_path / "photos"
    if not photos.exists():
        return None
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
        p = photos / f"{image_filename}-3000{ext}"
        if p.exists():
            return p
    return None


def scan_context_for_classification(context_path: str) -> List[Dict[str, Any]]:
    """
    Scan a context directory for items that have preprocess + stage2 masks.
    Returns list of classification items with find_path, find_number, image_filename,
    image_3000_path, mask_path (relative to .ascota), and item_id.
    """
    context_dir = Path(context_path)
    finds_base = context_dir / "finds" / "individual"
    if not finds_base.exists():
        logger.warning(f"No finds directory at {finds_base}")
        return []

    items: List[Dict[str, Any]] = []
    for find_dir in finds_base.iterdir():
        if not find_dir.is_dir():
            continue
        find_number = find_dir.name
        find_path = str(find_dir)
        preprocess = load_preprocess_json(find_path)
        stage2 = preprocess.get("stage2") or {}
        masks_data = stage2.get("masks") or {}

        for image_filename, mask_info in masks_data.items():
            if not isinstance(mask_info, dict):
                continue
            mask_path_rel = mask_info.get("mask_path")
            if not mask_path_rel:
                continue
            # mask_path is e.g. "masks/1_mask.png"
            mask_full = find_dir / ".ascota" / mask_path_rel
            if not mask_full.exists():
                continue
            image_3000 = _find_image_3000_path(find_dir, image_filename)
            if not image_3000 or not image_3000.exists():
                continue
            item_id = _make_item_id(find_number, image_filename)
            items.append({
                "item_id": item_id,
                "find_path": find_path,
                "find_number": find_number,
                "image_filename": image_filename,
                "image_3000_path": str(image_3000),
                "mask_path": mask_path_rel,
            })
    logger.info(f"Scanned {context_path}: found {len(items)} classification items")
    return items

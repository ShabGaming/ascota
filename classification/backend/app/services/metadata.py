"""Service for managing .ascota folders and classification.json storage."""

import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Windows hidden attribute constant
FILE_ATTRIBUTE_HIDDEN = 0x02


def set_hidden_attribute_windows(path: Path) -> None:
    """Set hidden attribute on Windows."""
    try:
        import ctypes
        import platform

        if platform.system() == "Windows":
            abs_path = str(path.absolute())
            ctypes.windll.kernel32.SetFileAttributesW(abs_path, FILE_ATTRIBUTE_HIDDEN)
            logger.debug(f"Set hidden attribute on {path}")
    except Exception as e:
        logger.warning(f"Failed to set hidden attribute on {path}: {e}")


def ensure_ascota_folder(find_path: str) -> Path:
    """Ensure .ascota folder exists in find directory and set hidden attribute on Windows."""
    find_dir = Path(find_path)
    ascota_dir = find_dir / ".ascota"
    ascota_dir.mkdir(parents=True, exist_ok=True)
    set_hidden_attribute_windows(ascota_dir)
    return ascota_dir


def ensure_context_ascota_folder(context_path: str) -> Path:
    """Ensure .ascota folder exists in context directory."""
    context_dir = Path(context_path)
    ascota_dir = context_dir / ".ascota"
    ascota_dir.mkdir(parents=True, exist_ok=True)
    set_hidden_attribute_windows(ascota_dir)
    return ascota_dir


def get_classification_ascota_dir() -> Path:
    """Return path to classification/.ascota (hidden temp folder)."""
    # From backend/app/services -> backend -> classification
    backend_dir = Path(__file__).resolve().parent.parent
    classification_dir = backend_dir.parent
    ascota_dir = classification_dir / ".ascota"
    ascota_dir.mkdir(parents=True, exist_ok=True)
    set_hidden_attribute_windows(ascota_dir)
    return ascota_dir


def load_preprocess_json(find_path: str) -> Dict[str, Any]:
    """Load preprocess.json from find directory."""
    find_dir = Path(find_path)
    json_path = find_dir / ".ascota" / "preprocess.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load preprocess.json from {json_path}: {e}", exc_info=True)
        return {}


def load_classification_json(find_path: str) -> Dict[str, Any]:
    """Load classification.json from find's .ascota folder."""
    ascota_dir = ensure_ascota_folder(find_path)
    json_path = ascota_dir / "classification.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load classification.json from {json_path}: {e}", exc_info=True)
        return {}


def save_classification_json(find_path: str, data: Dict[str, Any]) -> bool:
    """Save classification.json to find's .ascota folder (merge-friendly)."""
    try:
        ascota_dir = ensure_ascota_folder(find_path)
        json_path = ascota_dir / "classification.json"
        temp_file = json_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        temp_file.replace(json_path)
        logger.info(f"Saved classification.json to {json_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save classification.json to {find_path}: {e}", exc_info=True)
        return False

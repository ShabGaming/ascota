"""Session management and context status routes."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.services.metadata import ensure_context_ascota_folder
from app.services.scanner import scan_context_for_classification
from app.services.classification_store import get_classification_store
from app.services.pottery_gate import assess_pottery_gate, hydrate_pottery_results_from_disk
from app.services.models import CreateSessionRequest, CreateSessionResponse, ExportResponse, LoadSessionRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _is_context_preprocessed(context_path: str) -> bool:
    """Check if context has preprocess_status true in context_status.json."""
    try:
        ascota_dir = ensure_context_ascota_folder(context_path)
        status_file = ascota_dir / "context_status.json"
        if not status_file.exists():
            return False
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "status_history" not in data or not data["status_history"]:
            return False
        for entry in reversed(data["status_history"]):
            if "preprocess_status" in entry:
                return bool(entry.get("preprocess_status", False))
        return False
    except Exception as e:
        logger.warning(f"Failed to read context_status.json for {context_path}: {e}")
        return False


@router.get("/check-context-status")
async def check_context_status(
    context_path: str = Query(..., description="Path to context directory"),
):
    """Check if a context directory has been preprocessed."""
    return {"is_preprocessed": _is_context_preprocessed(context_path)}


@router.get("/check-pottery-gate")
async def check_pottery_gate_route(
    context_path: str = Query(..., description="Path to context directory"),
):
    """Report whether every scanned image has on-disk pottery labels (for enabling category sessions)."""
    trimmed = context_path.strip()
    path = Path(trimmed)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="Context path does not exist or is not a directory")
    if not _is_context_preprocessed(trimmed):
        return {
            "complete": False,
            "total": 0,
            "missing_count": 0,
            "pottery_on_disk_count": 0,
            "is_preprocessed": False,
        }
    items = scan_context_for_classification(trimmed)
    if not items:
        return {
            "complete": False,
            "total": 0,
            "missing_count": 0,
            "pottery_on_disk_count": 0,
            "is_preprocessed": True,
        }
    complete, missing, _pottery_items, pottery_count = assess_pottery_gate(items)
    return {
        "complete": complete,
        "total": len(items),
        "missing_count": len(missing),
        "pottery_on_disk_count": pottery_count,
        "is_preprocessed": True,
    }


@router.get("/list")
async def list_sessions():
    """List all sessions (in-memory and from .ascota run files). Each has context_path, classification_type, items_count, results_count."""
    store = get_classification_store()
    return {"sessions": store.list_sessions()}


@router.post("/load")
async def load_session(request: LoadSessionRequest):
    """Restore a session from disk by session_id. Returns session summary so frontend can navigate."""
    session_id = request.session_id
    store = get_classification_store()
    session = store.restore_session_from_disk(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or could not be restored")
    return {
        "session_id": session.session_id,
        "context_path": session.context_path,
        "classification_type": session.classification_type,
        "items_count": len(session.items),
        "has_results": len(session.results) > 0,
        "options": session.options,
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Remove session from store and delete its run file."""
    store = get_classification_store()
    store.delete_session(session_id)
    return {"message": "Session deleted"}


@router.post("", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new classification session (single context). Requires context to be preprocessed."""
    context_path = request.context_path.strip()
    path = Path(context_path)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="Context path does not exist or is not a directory")
    if not _is_context_preprocessed(context_path):
        raise HTTPException(
            status_code=400,
            detail="Context has not been preprocessed. Run the preprocess pipeline first.",
        )
    items = scan_context_for_classification(context_path)
    if not items:
        raise HTTPException(
            status_code=400,
            detail="No classification items found (finds with preprocess and masks).",
        )
    classification_type = (request.classification_type or "type").strip().lower()
    if classification_type not in ("type", "decoration", "color", "texture", "pottery"):
        classification_type = "type"

    store = get_classification_store()

    if classification_type == "pottery":
        session_id = store.create_session(context_path, classification_type, items)
        session = store.get_session(session_id)
        if session:
            hydrated = hydrate_pottery_results_from_disk(items)
            session.results.update(hydrated)
            store.persist_run_state(session_id)
        return CreateSessionResponse(session_id=session_id, items_count=len(items))

    complete, missing, pottery_items, _ = assess_pottery_gate(items)
    if not complete:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Pottery labels missing on disk for {len(missing)} image(s). "
                "Run and export a Pottery vs non-pottery session first."
            ),
        )
    if not pottery_items:
        raise HTTPException(
            status_code=400,
            detail="No pottery images in this context (all labeled non-pottery on disk).",
        )

    session_id = store.create_session(context_path, classification_type, pottery_items)
    return CreateSessionResponse(session_id=session_id, items_count=len(pottery_items))


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get session summary."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "context_path": session.context_path,
        "classification_type": session.classification_type,
        "items_count": len(session.items),
        "has_results": len(session.results) > 0,
        "options": session.options,
    }


@router.post("/{session_id}/export", response_model=ExportResponse)
async def export_session(session_id: str):
    """Export results to each find's .ascota/classification.json (key = classification_type), then clear temp and remove session."""
    from app.services.export import export_classification_results

    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    saved = export_classification_results(session)
    store.delete_session(session_id)
    return ExportResponse(saved_finds=saved)

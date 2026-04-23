"""Decoration/pattern classification routes: run, results, update, undo."""

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.classification_store import get_classification_store
from app.services.find_consensus import propagate_max_confidence_within_find
from app.services.models import (
    DecorationRunRequest,
    DecorationRunResponse,
    UpdateDecorationResultRequest,
)
from app.services.image_utils import build_rgba_from_find_and_mask

logger = logging.getLogger(__name__)
router = APIRouter()

_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@router.post("/{session_id}/decoration/run", response_model=DecorationRunResponse)
async def run_decoration_classification(session_id: str, request: DecorationRunRequest):
    """Build RGBA images, run batch decoration classification. Only classifies items not already in session.results."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.classification_type != "decoration":
        raise HTTPException(status_code=400, detail="Session is not a decoration session")

    resolution = request.resolution
    if resolution not in (1000, 1500, 3000):
        raise HTTPException(status_code=400, detail="resolution must be 1000, 1500, or 3000")

    unclassified = [it for it in session.items if it["item_id"] not in session.results]
    if not unclassified:
        propagate_max_confidence_within_find(session.items, session.results)
        store.persist_run_state(session_id)
        out = [
            {"item_id": iid, "label": session.results[iid]["label"], "confidence": session.results[iid]["confidence"]}
            for iid in (it["item_id"] for it in session.items)
        ]
        return DecorationRunResponse(results=out)

    from ascota_classification.decoration import batch_classify_pottery_decoration

    images = []
    order = []
    for it in unclassified:
        try:
            rgba = build_rgba_from_find_and_mask(
                find_path=it["find_path"],
                image_filename=it["image_filename"],
                image_3000_path=it["image_3000_path"],
                mask_path_rel=it["mask_path"],
                target_size=resolution,
            )
            images.append(rgba)
            order.append(it["item_id"])
        except Exception as e:
            logger.warning(f"Failed to build RGBA for {it['item_id']}: {e}")
            continue

    if not images:
        raise HTTPException(status_code=500, detail="No images could be built from finds and masks")

    results_list = batch_classify_pottery_decoration(
        images,
        return_confidence=True,
        debug=False,
    )

    session.options = {"resolution": resolution}
    for item_id, res in zip(order, results_list):
        session.results[item_id] = {
            "label": res.get("label", "Impressed"),
            "confidence": float(res.get("confidence", 0.0)),
        }

    propagate_max_confidence_within_find(session.items, session.results)

    store.persist_run_state(session_id)

    out = [
        {"item_id": iid, "label": session.results[iid]["label"], "confidence": session.results[iid]["confidence"]}
        for iid in (it["item_id"] for it in session.items)
    ]
    return DecorationRunResponse(results=out)


@router.get("/{session_id}/decoration/results")
async def get_decoration_results(session_id: str):
    """Return current decoration results (after edits/undo)."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"results": session.results}


@router.patch("/{session_id}/decoration/results/{item_id}")
async def update_decoration_result(session_id: str, item_id: str, request: UpdateDecorationResultRequest):
    """Set result for item to label and confidence 1.0; push undo state. Label can be Impressed, Incised, or custom."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get_item(item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    session.push_undo()
    session.results[item_id] = {"label": request.label.strip() or "Impressed", "confidence": 1.0}
    store.persist_run_state(session_id)
    return {"item_id": item_id, "label": session.results[item_id]["label"], "confidence": 1.0}


@router.post("/{session_id}/decoration/undo")
async def undo_decoration_edit(session_id: str):
    """Pop last edit and restore previous results."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.pop_undo():
        return {"restored": False, "message": "Nothing to undo"}
    store.persist_run_state(session_id)
    return {"restored": True, "results": session.results}

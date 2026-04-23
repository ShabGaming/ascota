"""Type classification routes: run, results, update, undo."""

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.classification_store import get_classification_store
from app.services.find_consensus import propagate_max_confidence_within_find
from app.services.models import TypeRunRequest, TypeRunResponse, UpdateTypeResultRequest
from app.services.image_utils import build_rgba_from_find_and_mask

logger = logging.getLogger(__name__)
router = APIRouter()

# Add repo src to path for ascota_classification (routes -> app -> backend -> classification -> repo)
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@router.post("/{session_id}/type/run", response_model=TypeRunResponse)
async def run_type_classification(session_id: str, request: TypeRunRequest):
    """Build RGBA images, run batch type classification, store results. Only classifies items not already in session.results."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    resolution = request.resolution
    if resolution not in (1000, 1500, 3000):
        raise HTTPException(status_code=400, detail="resolution must be 1000, 1500, or 3000")

    # Only run on items that don't have a result yet (incremental)
    unclassified = [it for it in session.items if it["item_id"] not in session.results]
    if not unclassified:
        propagate_max_confidence_within_find(session.items, session.results)
        store.persist_run_state(session_id)
        # Already all classified; just return current results
        out = [
            {"item_id": iid, "label": session.results[iid]["label"], "confidence": session.results[iid]["confidence"]}
            for iid in (it["item_id"] for it in session.items)
        ]
        return TypeRunResponse(results=out)

    from ascota_classification.type import batch_classify_pottery_type

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

    results_list = batch_classify_pottery_type(
        images,
        use_azure_openai=request.enable_appendage_subtype,
        debug=False,
    )

    session.options = {
        "enable_appendage_subtype": request.enable_appendage_subtype,
        "resolution": resolution,
    }
    for item_id, res in zip(order, results_list):
        session.results[item_id] = {
            "label": res.get("label", "body"),
            "confidence": res.get("stage1", {}).get("confidence", 0.0),
        }
        if "stage2" in res:
            session.results[item_id]["confidence"] = res["stage2"].get("confidence", 0.0)
        if "stage3" in res:
            session.results[item_id]["confidence"] = res["stage3"].get("confidence", 0.0)

    propagate_max_confidence_within_find(session.items, session.results)

    store.persist_run_state(session_id)

    out = [
        {"item_id": iid, "label": session.results[iid]["label"], "confidence": session.results[iid]["confidence"]}
        for iid in (it["item_id"] for it in session.items)
    ]
    return TypeRunResponse(results=out)


@router.get("/{session_id}/type/results")
async def get_type_results(session_id: str):
    """Return current type results (after edits/undo)."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"results": session.results}


@router.patch("/{session_id}/type/results/{item_id}")
async def update_type_result(session_id: str, item_id: str, request: UpdateTypeResultRequest):
    """Set result for item to label and confidence 1.0; push undo state."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get_item(item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    session.push_undo()
    session.results[item_id] = {"label": request.label, "confidence": 1.0}
    store.persist_run_state(session_id)
    return {"item_id": item_id, "label": request.label, "confidence": 1.0}


@router.post("/{session_id}/type/undo")
async def undo_type_edit(session_id: str):
    """Pop last edit and restore previous results."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.pop_undo():
        return {"restored": False, "message": "Nothing to undo"}
    store.persist_run_state(session_id)
    return {"restored": True, "results": session.results}

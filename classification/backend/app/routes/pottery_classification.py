"""Pottery vs non-pottery routes: run, results, update, undo."""

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.classification_store import get_classification_store
from app.services.find_consensus import propagate_max_confidence_within_find
from app.services.models import PotteryRunRequest, PotteryRunResponse, UpdatePotteryResultRequest
from app.services.image_utils import build_rgba_from_find_and_mask

logger = logging.getLogger(__name__)
router = APIRouter()

_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_VALID_LABELS = frozenset({"pottery", "non_pottery"})


@router.post("/{session_id}/pottery/run", response_model=PotteryRunResponse)
async def run_pottery_classification(session_id: str, request: PotteryRunRequest):
    """Run batch pottery vs non-pottery; only items not already in session.results."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.classification_type != "pottery":
        raise HTTPException(status_code=400, detail="Session is not a pottery session")

    resolution = request.resolution
    if resolution not in (1000, 1500, 3000):
        raise HTTPException(status_code=400, detail="resolution must be 1000, 1500, or 3000")

    unclassified = [it for it in session.items if it["item_id"] not in session.results]
    if not unclassified:
        propagate_max_confidence_within_find(session.items, session.results)
        store.persist_run_state(session_id)
        out = []
        for it in session.items:
            iid = it["item_id"]
            r = session.results[iid]
            row = {"item_id": iid, "label": r["label"], "confidence": r["confidence"]}
            if "p_pottery" in r:
                row["p_pottery"] = r["p_pottery"]
            out.append(row)
        return PotteryRunResponse(results=out)

    from ascota_classification.type import batch_classify_pottery_vs_non_pottery

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

    results_list = batch_classify_pottery_vs_non_pottery(images, return_confidence=True, debug=False)

    session.options = {"resolution": resolution}
    for item_id, res in zip(order, results_list):
        session.results[item_id] = {
            "label": res.get("label", "non_pottery"),
            "confidence": float(res.get("confidence", 0.0)),
            "p_pottery": float(res.get("p_pottery", 0.0)),
        }

    propagate_max_confidence_within_find(session.items, session.results)

    store.persist_run_state(session_id)

    out = []
    for it in session.items:
        iid = it["item_id"]
        r = session.results[iid]
        row = {"item_id": iid, "label": r["label"], "confidence": r["confidence"]}
        if "p_pottery" in r:
            row["p_pottery"] = r["p_pottery"]
        out.append(row)
    return PotteryRunResponse(results=out)


@router.get("/{session_id}/pottery/results")
async def get_pottery_results(session_id: str):
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"results": session.results}


@router.patch("/{session_id}/pottery/results/{item_id}")
async def update_pottery_result(session_id: str, item_id: str, request: UpdatePotteryResultRequest):
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.classification_type != "pottery":
        raise HTTPException(status_code=400, detail="Session is not a pottery session")
    if session.get_item(item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    label = (request.label or "").strip().lower()
    if label not in _VALID_LABELS:
        raise HTTPException(status_code=400, detail="label must be pottery or non_pottery")

    session.push_undo()
    prev = session.results.get(item_id, {})
    session.results[item_id] = {"label": label, "confidence": 1.0}
    if "p_pottery" in prev:
        session.results[item_id]["p_pottery"] = prev["p_pottery"]
    store.persist_run_state(session_id)
    return {"item_id": item_id, "label": label, "confidence": 1.0}


@router.post("/{session_id}/pottery/undo")
async def undo_pottery_edit(session_id: str):
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.pop_undo():
        return {"restored": False, "message": "Nothing to undo"}
    store.persist_run_state(session_id)
    return {"restored": True, "results": session.results}

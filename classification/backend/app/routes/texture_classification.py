"""Texture clustering routes: run (LBP+GLCM + scaler + PCA + HDBSCAN), results, recluster, update, rename, undo."""

import logging
import sys
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from app.services.classification_store import get_classification_store
from app.services.models import (
    TextureRunRequest,
    TextureRunResponse,
    TextureResultsResponse,
    ReclusterRequest,
    UpdateColorResultRequest,
    SetClusterNameRequest,
)
from app.services.image_utils import build_rgba_from_find_and_mask

logger = logging.getLogger(__name__)
router = APIRouter()

_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ascota_classification.texture import (
    extract_texture_pca_features,
    cluster_images_hdbscan,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_CLUSTER_SELECTION_EPSILON,
    DEFAULT_CLUSTER_SELECTION_METHOD,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_RESIZE_MAX,
)


def _build_texture_results_response(session) -> dict:
    """Build clusters, noise_item_ids, results, cluster_names from session."""
    results = session.results
    cluster_names = getattr(session, "texture_cluster_names", {}) or {}
    by_cluster: dict = {}
    noise_item_ids = []
    for it in session.items:
        item_id = it["item_id"]
        r = results.get(item_id)
        if r is None:
            continue
        cid = r.get("cluster_id", -1)
        if cid == -1:
            noise_item_ids.append(item_id)
        else:
            by_cluster.setdefault(cid, []).append(item_id)
    clusters = [{"cluster_id": cid, "item_ids": item_ids} for cid, item_ids in sorted(by_cluster.items())]
    return {
        "results": results,
        "cluster_names": cluster_names,
        "clusters": clusters,
        "noise_item_ids": noise_item_ids,
    }


@router.post("/{session_id}/texture/run", response_model=TextureRunResponse)
async def run_texture_clustering(session_id: str, request: TextureRunRequest = TextureRunRequest()):
    """Build RGBA for all items, extract texture+PCA features, cluster with HDBSCAN. Store features for recluster."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.classification_type != "texture":
        raise HTTPException(status_code=400, detail="Session is not a texture session")

    resolution = request.resolution if request.resolution in (1000, 1500, 3000) else 1500
    session.options["resolution"] = resolution

    images = []
    order = []
    for it in session.items:
        try:
            rgba = build_rgba_from_find_and_mask(
                find_path=it["find_path"],
                image_filename=it["image_filename"],
                image_3000_path=it["image_3000_path"],
                mask_path_rel=it["mask_path"],
                target_size=512,
            )
            images.append(rgba)
            order.append(it["item_id"])
        except Exception as e:
            logger.warning(f"Failed to build RGBA for {it['item_id']}: {e}")
            continue

    if not images:
        raise HTTPException(status_code=500, detail="No images could be built from finds and masks")

    features, _, _ = extract_texture_pca_features(
        images,
        pca_components=DEFAULT_PCA_COMPONENTS,
        alpha_threshold=DEFAULT_ALPHA_THRESHOLD,
        resize_max=DEFAULT_RESIZE_MAX,
    )
    labels, _ = cluster_images_hdbscan(
        features,
        min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
        min_samples=DEFAULT_MIN_SAMPLES,
        cluster_selection_epsilon=DEFAULT_CLUSTER_SELECTION_EPSILON,
        cluster_selection_method=DEFAULT_CLUSTER_SELECTION_METHOD,
    )

    session.texture_features = features.tolist()
    for i, item_id in enumerate(order):
        session.results[item_id] = {"cluster_id": int(labels[i])}

    unique_labels = sorted(set(int(l) for l in labels))
    session.texture_cluster_names = {}
    for cid in unique_labels:
        if cid == -1:
            session.texture_cluster_names[-1] = "Noise"
        else:
            session.texture_cluster_names[cid] = f"Cluster {cid}"
    store.persist_run_state(session_id)

    out = _build_texture_results_response(session)
    out["results"] = [{"item_id": iid, "cluster_id": session.results[iid]["cluster_id"]} for iid in order]
    return TextureRunResponse(**out)


@router.get("/{session_id}/texture/results", response_model=TextureResultsResponse)
async def get_texture_results(session_id: str):
    """Return current texture results (clusters, noise, cluster names)."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return TextureResultsResponse(**_build_texture_results_response(session))


@router.post("/{session_id}/texture/recluster", response_model=TextureResultsResponse)
async def recluster_texture(session_id: str, request: ReclusterRequest):
    """Re-run HDBSCAN on stored PCA features with given params; reset cluster names."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.classification_type != "texture":
        raise HTTPException(status_code=400, detail="Session is not a texture session")
    if not session.texture_features:
        raise HTTPException(status_code=400, detail="Run texture clustering first")

    min_cluster_size = request.min_cluster_size if request.min_cluster_size is not None else DEFAULT_MIN_CLUSTER_SIZE
    min_samples = request.min_samples if request.min_samples is not None else DEFAULT_MIN_SAMPLES
    cluster_selection_epsilon = (
        request.cluster_selection_epsilon
        if request.cluster_selection_epsilon is not None
        else DEFAULT_CLUSTER_SELECTION_EPSILON
    )
    cluster_selection_method = (
        request.cluster_selection_method
        if request.cluster_selection_method is not None
        else DEFAULT_CLUSTER_SELECTION_METHOD
    )

    features = np.array(session.texture_features, dtype=np.float64)
    labels, _ = cluster_images_hdbscan(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
    )

    item_ids = [it["item_id"] for it in session.items]
    for i, item_id in enumerate(item_ids):
        if i < len(labels):
            session.results[item_id] = {"cluster_id": int(labels[i])}
    unique_labels = sorted(set(int(l) for l in labels))
    session.texture_cluster_names = {}
    for cid in unique_labels:
        if cid == -1:
            session.texture_cluster_names[-1] = "Noise"
        else:
            session.texture_cluster_names[cid] = f"Cluster {cid}"
    store.persist_run_state(session_id)

    return TextureResultsResponse(**_build_texture_results_response(session))


@router.patch("/{session_id}/texture/results/{item_id}")
async def update_texture_result(session_id: str, item_id: str, request: UpdateColorResultRequest):
    """Move item to another cluster; push undo."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get_item(item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    session.push_undo()
    cid = request.cluster_id
    session.results[item_id] = {"cluster_id": cid}
    if cid != -1:
        names = getattr(session, "texture_cluster_names", None) or {}
        if cid not in names:
            if not hasattr(session, "texture_cluster_names"):
                session.texture_cluster_names = {}
            session.texture_cluster_names[cid] = f"Cluster {cid}"
    store.persist_run_state(session_id)
    return {"item_id": item_id, "cluster_id": cid}


@router.patch("/{session_id}/texture/cluster/{cluster_id}/name")
async def set_texture_cluster_name(session_id: str, cluster_id: str, request: SetClusterNameRequest):
    """Set display name for a cluster; push undo. cluster_id can be -1 for noise."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    cid = int(cluster_id)
    session.push_undo()
    if not hasattr(session, "texture_cluster_names"):
        session.texture_cluster_names = {}
    session.texture_cluster_names[cid] = request.name.strip() or (cid == -1 and "Noise" or f"Cluster {cid}")
    store.persist_run_state(session_id)
    return {"cluster_id": cid, "name": session.texture_cluster_names[cid]}


@router.post("/{session_id}/texture/undo")
async def undo_texture_edit(session_id: str):
    """Pop last edit and restore previous results."""
    store = get_classification_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.pop_undo():
        return {"restored": False, "message": "Nothing to undo"}
    store.persist_run_state(session_id)
    return {"restored": True, "results": session.results}

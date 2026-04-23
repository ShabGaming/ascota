"""Export classification results to each find's .ascota/classification.json."""

import logging
from typing import Dict, Any

from app.services.metadata import load_classification_json, save_classification_json

logger = logging.getLogger(__name__)


def export_classification_results(session: Any) -> int:
    """
    Group session results by find_number, merge into each find's classification.json
    under key session.classification_type (e.g. "type", "decoration"), then save.
    Returns number of finds written.
    """
    key = session.classification_type or "type"
    find_path_by_number: Dict[str, str] = {}
    for it in session.items:
        fn = it.get("find_number")
        fp = it.get("find_path")
        if fn and fp:
            find_path_by_number[fn] = fp

    by_find: Dict[str, Dict[str, Dict[str, Any]]] = {}
    is_color = key == "color"
    is_texture = key == "texture"
    is_pottery = key == "pottery"
    for item_id, result in session.results.items():
        item = session.get_item(item_id)
        if not item:
            continue
        find_number = item.get("find_number")
        image_filename = item.get("image_filename")
        if not find_number or not image_filename:
            continue
        if find_number not in by_find:
            by_find[find_number] = {}
        if is_color:
            cluster_id = result.get("cluster_id", -1)
            color_name = getattr(session, "color_cluster_names", {}).get(cluster_id, "Noise")
            by_find[find_number][image_filename] = {
                "cluster_id": cluster_id,
                "color_name": color_name,
            }
        elif is_texture:
            cluster_id = result.get("cluster_id", -1)
            texture_name = getattr(session, "texture_cluster_names", {}).get(cluster_id, "Noise")
            by_find[find_number][image_filename] = {
                "cluster_id": cluster_id,
                "texture_name": texture_name,
            }
        elif is_pottery:
            row: Dict[str, Any] = {
                "label": result.get("label", ""),
                "confidence": float(result.get("confidence", 0.0)),
            }
            if "p_pottery" in result and result["p_pottery"] is not None:
                row["p_pottery"] = float(result["p_pottery"])
            by_find[find_number][image_filename] = row
        else:
            by_find[find_number][image_filename] = {
                "label": result.get("label", ""),
                "confidence": result.get("confidence", 0.0),
            }

    saved = 0
    for find_number, type_results in by_find.items():
        find_path = find_path_by_number.get(find_number)
        if not find_path:
            continue
        data = load_classification_json(find_path)
        data[key] = type_results
        if save_classification_json(find_path, data):
            saved += 1
    logger.info(f"Exported {key} classification to {saved} finds")
    return saved

"""Pottery labels on disk: gate for type/decoration/color/texture sessions."""

from typing import Any, Dict, List, Tuple

from app.services.metadata import load_classification_json


def _pottery_record_for_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
    data = load_classification_json(item["find_path"])
    pottery_map = data.get("pottery") or {}
    rec = pottery_map.get(item["image_filename"])
    if not rec or not isinstance(rec, dict):
        return None
    label = rec.get("label")
    if not label:
        return None
    return rec


def assess_pottery_gate(all_items: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[Dict[str, Any]], int]:
    """
    Returns:
        complete: every item has a pottery record on disk
        missing_item_ids: items without pottery entry
        filtered_pottery_items: items whose on-disk label is pottery
        pottery_on_disk_count: number of items labeled pottery on disk
    """
    missing: List[str] = []
    pottery_items: List[Dict[str, Any]] = []
    pottery_count = 0
    for it in all_items:
        rec = _pottery_record_for_item(it)
        if rec is None:
            missing.append(it["item_id"])
        elif rec.get("label") == "pottery":
            pottery_items.append(it)
            pottery_count += 1
    complete = len(missing) == 0
    return complete, missing, pottery_items, pottery_count


def hydrate_pottery_results_from_disk(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build item_id -> results dict from each find's classification.json pottery section."""
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        rec = _pottery_record_for_item(it)
        if rec is None:
            continue
        row: Dict[str, Any] = {
            "label": rec["label"],
            "confidence": float(rec.get("confidence", 0.0)),
        }
        if "p_pottery" in rec and rec["p_pottery"] is not None:
            row["p_pottery"] = float(rec["p_pottery"])
        out[it["item_id"]] = row
    return out

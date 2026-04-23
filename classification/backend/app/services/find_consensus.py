"""Propagate classification results within each find (folder) from highest-confidence sibling."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List


def propagate_max_confidence_within_find(
    items: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
) -> None:
    """
    For each find_number, copy the result dict of the item with maximum confidence
    to every item in that find (including siblings without a prior result).

    Mutates ``results`` in place.
    """
    by_find: Dict[str, List[str]] = defaultdict(list)
    order: List[str] = []
    seen: set[str] = set()
    for it in items:
        fn = it.get("find_number")
        iid = it.get("item_id")
        if not fn or not iid:
            continue
        if iid not in seen:
            seen.add(iid)
            order.append(iid)
        by_find[fn].append(iid)

    for _fn, item_ids in by_find.items():
        item_ids = list(dict.fromkeys(item_ids))
        present = [iid for iid in item_ids if iid in results]
        if not present:
            continue

        def order_index(iid: str) -> int:
            try:
                return order.index(iid)
            except ValueError:
                return len(order)

        # Max confidence; tie-break: first item in stable session order wins.
        winner_id = max(
            present,
            key=lambda iid: (
                float(results[iid].get("confidence", 0.0)),
                -order_index(iid),
            ),
        )
        template = copy.deepcopy(results[winner_id])
        for iid in item_ids:
            results[iid] = copy.deepcopy(template)

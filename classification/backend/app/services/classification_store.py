"""In-memory session store for classification; persists to classification/.ascota."""

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from app.services.metadata import get_classification_ascota_dir

logger = logging.getLogger(__name__)


class ClassificationSession:
    """Single classification session: one context, one type, items + results + undo history."""

    def __init__(
        self,
        session_id: str,
        context_path: str,
        classification_type: str,
        items: List[Dict[str, Any]],
    ):
        self.session_id = session_id
        self.context_path = context_path
        self.classification_type = classification_type
        self.items = items  # list of { item_id, find_path, find_number, image_filename, image_3000_path, mask_path }
        self.options: Dict[str, Any] = {}  # appendage_subtype, resolution
        self.results: Dict[str, Dict[str, Any]] = {}  # item_id -> { label, confidence } or { cluster_id } for color/texture
        self._history: List[Dict[str, Dict[str, Any]]] = []  # stack of previous results for undo
        self.color_features: Optional[List[List[float]]] = None  # one row per item (color sessions only)
        self.color_cluster_names: Dict[int, str] = {}  # cluster_id -> display name (color sessions only)
        self.texture_features: Optional[List[List[float]]] = None  # PCA rows after texture run (texture sessions only)
        self.texture_cluster_names: Dict[int, str] = {}  # cluster_id -> display name (texture sessions only)

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        for it in self.items:
            if it.get("item_id") == item_id:
                return it
        return None

    def push_undo(self) -> None:
        """Save current results to history (before an edit)."""
        self._history.append(dict(self.results))

    def pop_undo(self) -> bool:
        """Restore previous results from history. Returns True if restored."""
        if not self._history:
            return False
        self.results = self._history.pop()
        return True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "session_id": self.session_id,
            "context_path": self.context_path,
            "classification_type": self.classification_type,
            "items": self.items,
            "options": self.options,
            "results": self.results,
            "history_count": len(self._history),
        }
        if self.color_features is not None:
            out["color_features"] = self.color_features
        if self.color_cluster_names:
            out["color_cluster_names"] = {str(k): v for k, v in self.color_cluster_names.items()}
        if self.texture_features is not None:
            out["texture_features"] = self.texture_features
        if self.texture_cluster_names:
            out["texture_cluster_names"] = {str(k): v for k, v in self.texture_cluster_names.items()}
        return out


class ClassificationStore:
    """Store for all classification sessions; persists run state to classification/.ascota."""

    def __init__(self):
        self._sessions: Dict[str, ClassificationSession] = {}
        self._lock = threading.RLock()

    def create_session(self, context_path: str, classification_type: str, items: List[Dict[str, Any]]) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            session = ClassificationSession(session_id, context_path, classification_type, items)
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id} with {len(items)} items")
        return session_id

    def get_session(self, session_id: str) -> Optional[ClassificationSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            self._delete_run_file(session_id)
            logger.info(f"Deleted session {session_id}")
            return True

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return all sessions: in-memory plus those restored from run files. Deduped by session_id (in-memory wins)."""
        with self._lock:
            seen: Dict[str, Dict[str, Any]] = {}
            for sid, session in self._sessions.items():
                seen[sid] = {
                    "session_id": sid,
                    "context_path": session.context_path,
                    "classification_type": session.classification_type,
                    "items_count": len(session.items),
                    "results_count": len(session.results),
                    "in_memory": True,
                }
            ascota = get_classification_ascota_dir()
            for path in ascota.glob("run_*.json"):
                try:
                    raw = path.stem.replace("run_", "")
                    if raw in seen:
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    seen[raw] = {
                        "session_id": data.get("session_id", raw),
                        "context_path": data.get("context_path", ""),
                        "classification_type": data.get("classification_type", "type"),
                        "items_count": len(data.get("items", [])),
                        "results_count": len(data.get("results", {})),
                        "in_memory": False,
                    }
                except Exception as e:
                    logger.warning(f"Failed to read run file {path}: {e}")
            return list(seen.values())

    def restore_session_from_disk(self, session_id: str) -> Optional[ClassificationSession]:
        """Load session from run file into store. Returns session if restored, None if not found."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            path = self._run_file_path(session_id)
            if not path.exists():
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = ClassificationSession(
                    session_id=data["session_id"],
                    context_path=data["context_path"],
                    classification_type=data["classification_type"],
                    items=data.get("items", []),
                )
                session.options = data.get("options", {})
                session.results = data.get("results", {})
                session.color_features = data.get("color_features")
                raw_names = data.get("color_cluster_names", {})
                session.color_cluster_names = {int(k): v for k, v in raw_names.items()}
                session.texture_features = data.get("texture_features")
                raw_texture_names = data.get("texture_cluster_names", {})
                session.texture_cluster_names = {int(k): v for k, v in raw_texture_names.items()}
                self._sessions[session_id] = session
                logger.info(f"Restored session {session_id} from disk")
                return session
            except Exception as e:
                logger.warning(f"Failed to restore session {session_id}: {e}")
                return None

    def _run_file_path(self, session_id: str) -> Path:
        return get_classification_ascota_dir() / f"run_{session_id}.json"

    def _delete_run_file(self, session_id: str) -> None:
        path = self._run_file_path(session_id)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete run file {path}: {e}")

    def persist_run_state(self, session_id: str) -> None:
        """Write session state to classification/.ascota/run_{session_id}.json."""
        session = self.get_session(session_id)
        if not session:
            return
        path = self._run_file_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to persist run state for {session_id}: {e}")


_store: Optional[ClassificationStore] = None


def get_classification_store() -> ClassificationStore:
    global _store
    if _store is None:
        _store = ClassificationStore()
    return _store

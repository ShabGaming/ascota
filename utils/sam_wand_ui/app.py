"""
MobileSAM wand UI: folder navigation, point prompts, mask or RGBA export.
Run from repo root: python -m utils.sam_wand_ui
"""

from __future__ import annotations

import sys
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Ensure ascota_core is importable when running as python -m utils.sam_wand_ui
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ascota_core.mobilesam_wand import MobileSamSession, Point  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
OUTPUT_MASKS = "masks"
OUTPUT_TRANSPARENT = "transparent"
DISPLAY_MAX_SIDE = 1600


class ExportMode(Enum):
    MASK = auto()
    TRANSPARENT = auto()


def list_image_files(folder: Path) -> List[Path]:
    """Non-recursive list of image files in folder (not in output subfolders)."""
    out: List[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rel = p.relative_to(folder)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] in (OUTPUT_MASKS, OUTPUT_TRANSPARENT):
            continue
        out.append(p)
    return out


def filter_unprocessed_images(paths: List[Path], out_dir: Path) -> List[Path]:
    """Keep only images that do not already have an exported PNG in out_dir."""
    filtered: List[Path] = []
    for p in paths:
        if not (out_dir / f"{p.stem}.png").is_file():
            filtered.append(p)
    return filtered


def numpy_rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """HxWx3 uint8 RGB -> QImage."""
    h, w, _ = rgb.shape
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def resize_for_display(rgb: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> Tuple[np.ndarray, float]:
    """Return scaled RGB and scale factor relative to original (display / original)."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return rgb, 1.0
    scale = max_side / float(side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    pil = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS)
    return np.array(pil), scale


def mask_to_overlay_rgba(mask: np.ndarray, color: Tuple[int, int, int, int]) -> np.ndarray:
    """HxW uint8 0/1 -> HxW RGBA overlay."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    m = mask.astype(bool)
    out[m] = color
    return out


class SamWorker(QObject):
    """Runs MobileSamSession on a dedicated thread."""

    image_ready = pyqtSignal(int)
    result_ready = pyqtSignal(object, int)
    error_signal = pyqtSignal(str, int)

    def __init__(self) -> None:
        super().__init__()
        self._session = MobileSamSession()

    @pyqtSlot(int, object)
    def on_load_image(self, load_id: int, rgb: object) -> None:
        try:
            arr = np.asarray(rgb, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError("RGB array must be HxWx3")
            self._session.set_image(np.ascontiguousarray(arr))
            self.image_ready.emit(load_id)
        except Exception as e:
            self.error_signal.emit(str(e), load_id)

    @pyqtSlot(object, int)
    def on_predict(self, points: object, req_id: int) -> None:
        try:
            pts: List[Point] = [(int(p[0]), int(p[1]), int(p[2])) for p in points]
            mask = self._session.predict(pts)
            self.result_ready.emit(mask, req_id)
        except Exception as e:
            self.error_signal.emit(str(e), req_id)


class PreviewCanvas(QLabel):
    """Shows scaled image with optional mask overlay; maps clicks to full-res coords."""

    point_added = pyqtSignal(int, int, int)  # x, y, label (1 fg, 0 bg)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(False)
        self._full_w = 0
        self._full_h = 0
        self._disp_w = 0
        self._disp_h = 0
        self._base_qimage: Optional[QImage] = None
        self._overlay_qimage: Optional[QImage] = None

    def set_image(self, rgb: np.ndarray) -> None:
        disp, _scale = resize_for_display(rgb)
        self._full_w = rgb.shape[1]
        self._full_h = rgb.shape[0]
        self._disp_w = disp.shape[1]
        self._disp_h = disp.shape[0]
        self._base_qimage = numpy_rgb_to_qimage(disp)
        self._overlay_qimage = None
        self._compose()

    def set_mask_overlay(self, mask_full: np.ndarray) -> None:
        if self._base_qimage is None:
            return
        if mask_full.shape[:2] != (self._full_h, self._full_w):
            return
        # Downscale mask with nearest neighbor
        pil_m = Image.fromarray((mask_full * 255).astype(np.uint8), mode="L")
        pil_m = pil_m.resize((self._disp_w, self._disp_h), Image.Resampling.NEAREST)
        m_small = (np.array(pil_m) > 127).astype(np.uint8)
        ov = mask_to_overlay_rgba(m_small, (80, 255, 120, 100))
        h, w = ov.shape[:2]
        if not ov.flags["C_CONTIGUOUS"]:
            ov = np.ascontiguousarray(ov)
        self._overlay_qimage = QImage(ov.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self._compose()

    def clear_overlay(self) -> None:
        self._overlay_qimage = None
        self._compose()

    def _compose(self) -> None:
        if self._base_qimage is None:
            self.clear()
            return
        w, h = self._disp_w, self._disp_h
        canvas = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(0xFF404040)
        p = QPainter(canvas)
        p.drawImage(0, 0, self._base_qimage)
        if self._overlay_qimage is not None:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.drawImage(0, 0, self._overlay_qimage)
        p.end()
        self.setPixmap(QPixmap.fromImage(canvas))

    def _widget_to_image_coords(self, wx: int, wy: int) -> Optional[Tuple[int, int]]:
        pix = self.pixmap()
        if pix is None or self._disp_w <= 0:
            return None
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        off_x = (lw - pw) // 2
        off_y = (lh - ph) // 2
        lx = wx - off_x
        ly = wy - off_y
        if lx < 0 or ly < 0 or lx >= pw or ly >= ph:
            return None
        ix = int(lx * self._full_w / pw)
        iy = int(ly * self._full_h / ph)
        ix = max(0, min(self._full_w - 1, ix))
        iy = max(0, min(self._full_h - 1, iy))
        return ix, iy

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = self._widget_to_image_coords(int(event.position().x()), int(event.position().y()))
        if pos is None:
            return
        ix, iy = pos
        if event.button() == Qt.MouseButton.LeftButton:
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:
                self.point_added.emit(ix, iy, 0)
            else:
                self.point_added.emit(ix, iy, 1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.point_added.emit(ix, iy, 0)
        super().mousePressEvent(event)


class ModeDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export mode")
        self._mode = ExportMode.MASK
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose how to save results:"))
        self._btn_mask = QPushButton("Mask mode — binary PNG masks in subfolder \"masks/\"")
        self._btn_trans = QPushButton("Transparent image mode — RGBA PNGs in subfolder \"transparent/\"")
        layout.addWidget(self._btn_mask)
        layout.addWidget(self._btn_trans)
        self._btn_mask.clicked.connect(self._pick_mask)
        self._btn_trans.clicked.connect(self._pick_trans)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _pick_mask(self) -> None:
        self._mode = ExportMode.MASK
        self.accept()

    def _pick_trans(self) -> None:
        self._mode = ExportMode.TRANSPARENT
        self.accept()


def load_mask_from_export_png(
    path: Path, mode: ExportMode, rgb_shape: Tuple[int, int, int]
) -> Optional[np.ndarray]:
    """Load binary mask (HxW uint8 0/1) from a saved export PNG; resize to rgb HxW if needed."""
    h, w = rgb_shape[0], rgb_shape[1]
    try:
        if mode == ExportMode.MASK:
            im = Image.open(path).convert("L")
            arr = np.array(im)
            mask = (arr > 127).astype(np.uint8)
        else:
            im = Image.open(path).convert("RGBA")
            arr = np.array(im)
            if arr.ndim != 3 or arr.shape[2] < 4:
                return None
            alpha = arr[:, :, 3]
            mask = (alpha > 127).astype(np.uint8)
    except OSError:
        return None
    if mask.shape[:2] != (h, w):
        pil_m = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        pil_m = pil_m.resize((w, h), Image.Resampling.NEAREST)
        mask = (np.array(pil_m) > 127).astype(np.uint8)
    return mask


class MainWindow(QWidget):
    sig_load_image = pyqtSignal(int, object)
    sig_predict = pyqtSignal(object, int)

    def __init__(self, folder: Path, mode: ExportMode) -> None:
        super().__init__()
        self._folder = Path(folder).resolve()
        self._mode = mode
        all_paths = list_image_files(self._folder)
        self._index = 0
        self._points: List[Tuple[int, int, int]] = []
        self._current_rgb: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._next_rgb_cache: Optional[np.ndarray] = None
        self._next_rgb_cache_idx: Optional[int] = None
        self._load_seq = 0
        self._predict_seq = 0
        self._image_load_id = 0

        out_name = OUTPUT_MASKS if mode == ExportMode.MASK else OUTPUT_TRANSPARENT
        self._out_dir = self._folder / out_name
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._paths = filter_unprocessed_images(all_paths, self._out_dir)

        self.setWindowTitle(f"SAM wand — {folder.name} — {out_name}/")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        root = QVBoxLayout(self)
        self._status = QLabel()
        self._export_path_label = QLabel()
        self._export_path_label.setText(f"Export folder: {self._out_dir}")
        self._export_path_label.setWordWrap(True)
        self._export_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._help = QLabel(
            "Left-click: foreground · Right-click or Ctrl+click: background · Clear: reset points\n"
            "Left arrow: previous · Right arrow: save PNG to export folder, then next · Down: next without saving · Space: clear current points/mask\n"
            f"Files are written under .../{out_name}/  (open that subfolder to see PNGs)."
        )
        self._help.setWordWrap(True)
        root.addWidget(self._status)
        root.addWidget(self._export_path_label)
        root.addWidget(self._help)

        self._canvas = PreviewCanvas()
        self._canvas.point_added.connect(self._on_point_added)
        root.addWidget(self._canvas, stretch=1)

        row = QHBoxLayout()
        self._btn_clear = QPushButton("Clear points")
        self._btn_clear.clicked.connect(self._clear_points)
        row.addWidget(self._btn_clear)
        row.addStretch()
        root.addLayout(row)

        self._thread = QThread()
        self._worker = SamWorker()
        self._worker.moveToThread(self._thread)
        self._thread.start()

        self.sig_load_image.connect(
            self._worker.on_load_image, Qt.ConnectionType.QueuedConnection
        )
        self.sig_predict.connect(self._worker.on_predict, Qt.ConnectionType.QueuedConnection)

        self._worker.image_ready.connect(self._on_image_ready)
        self._worker.result_ready.connect(self._on_predict_result)
        self._worker.error_signal.connect(self._on_worker_error)

        if not self._paths:
            self._status.setText("No images found in folder.")
        else:
            self._load_current_image()

    def _update_status(self) -> None:
        n = len(self._paths)
        if n == 0:
            self._status.setText("No images.")
            return
        path = self._paths[self._index]
        self._status.setText(f"{self._index + 1} / {n} — {path.name}")

    def _load_current_image(self) -> None:
        if not self._paths:
            return
        path = self._paths[self._index]
        if self._next_rgb_cache_idx == self._index and self._next_rgb_cache is not None:
            self._current_rgb = self._next_rgb_cache
        else:
            try:
                img = Image.open(path).convert("RGB")
                self._current_rgb = np.array(img)
            except OSError as e:
                QMessageBox.warning(self, "Open failed", f"{path}: {e}")
                return

        self._points.clear()
        self._current_mask = None
        self._canvas.set_image(self._current_rgb)
        self._canvas.clear_overlay()

        self._load_seq += 1
        self._image_load_id = self._load_seq
        self.sig_load_image.emit(self._image_load_id, self._current_rgb.copy())
        self._update_status()
        self._restore_saved_mask_if_any()
        self._prefetch_next_rgb()

    def _prefetch_next_rgb(self) -> None:
        """Pre-decode next image so Right-arrow navigation feels instant."""
        if not self._paths:
            self._next_rgb_cache = None
            self._next_rgb_cache_idx = None
            return
        next_idx = (self._index + 1) % len(self._paths)
        if self._next_rgb_cache_idx == next_idx and self._next_rgb_cache is not None:
            return
        next_path = self._paths[next_idx]
        try:
            img = Image.open(next_path).convert("RGB")
            self._next_rgb_cache = np.array(img)
            self._next_rgb_cache_idx = next_idx
        except OSError:
            self._next_rgb_cache = None
            self._next_rgb_cache_idx = None

    def _export_path_for_current(self) -> Path:
        stem = self._paths[self._index].stem
        return self._out_dir / f"{stem}.png"

    def _restore_saved_mask_if_any(self) -> None:
        """Reload mask from disk when revisiting an image so navigation does not wipe work."""
        out_path = self._export_path_for_current()
        if not out_path.is_file():
            return
        mask = load_mask_from_export_png(out_path, self._mode, self._current_rgb.shape)
        if mask is None:
            return
        self._current_mask = mask
        self._canvas.set_mask_overlay(mask)

    def _on_image_ready(self, load_id: int) -> None:
        if load_id != self._image_load_id:
            return

    def _on_point_added(self, x: int, y: int, label: int) -> None:
        self._points.append((x, y, label))
        self._queue_predict()

    def _clear_points(self) -> None:
        self._points.clear()
        self._current_mask = None
        self._canvas.clear_overlay()
        self._predict_seq += 1

    def _queue_predict(self) -> None:
        if not self._points or self._current_rgb is None:
            return
        if not any(p[2] == 1 for p in self._points):
            return
        self._predict_seq += 1
        rid = self._predict_seq
        self.sig_predict.emit(list(self._points), rid)

    def _on_predict_result(self, mask: object, req_id: int) -> None:
        if req_id != self._predict_seq:
            return
        m = np.asarray(mask, dtype=np.uint8)
        self._current_mask = m
        self._canvas.set_mask_overlay(m)

    def _on_worker_error(self, msg: str, req_id: int) -> None:
        if req_id == self._image_load_id or req_id == -1:
            QMessageBox.critical(self, "SAM error", msg)
        elif req_id == self._predict_seq:
            QMessageBox.warning(self, "Predict error", msg)

    def _save_current(self) -> bool:
        if self._current_rgb is None or not self._paths:
            return False
        if self._current_mask is None:
            QMessageBox.information(
                self,
                "Save",
                "No mask to save — click on the object (foreground) to build a mask, "
                "or open an image that already has a file in the export folder.",
            )
            return False

        src = self._paths[self._index]
        stem = src.stem
        out_path = self._out_dir / f"{stem}.png"

        mask = self._current_mask
        if mask.shape[:2] != self._current_rgb.shape[:2]:
            QMessageBox.warning(self, "Save", "Mask size mismatch; try Clear and predict again.")
            return False

        try:
            if self._mode == ExportMode.MASK:
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                    out_path, compress_level=1
                )
            else:
                rgb = self._current_rgb
                h, w = rgb.shape[:2]
                alpha = (mask.astype(np.float32) * 255.0).astype(np.uint8)
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[:, :, :3] = rgb
                rgba[:, :, 3] = alpha
                Image.fromarray(rgba, mode="RGBA").save(out_path, compress_level=1)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return False
        self._status.setText(
            f"{self._index + 1} / {len(self._paths)} — {src.name}  ·  Saved: {out_path.name}"
        )
        return True

    def _go_previous(self) -> None:
        if not self._paths:
            return
        self._index = (self._index - 1) % len(self._paths)
        self._load_current_image()

    def _go_next(self) -> None:
        if not self._paths:
            return
        self._index = (self._index + 1) % len(self._paths)
        self._load_current_image()

    def _go_skip(self) -> None:
        self._go_next()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._go_previous()
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            if self._save_current():
                self._go_next()
            event.accept()
            return
        if key == Qt.Key.Key_Down:
            self._go_skip()
            event.accept()
            return
        if key == Qt.Key.Key_Space:
            self._clear_points()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._thread.quit()
        self._thread.wait(5000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    folder = QFileDialog.getExistingDirectory(None, "Select image folder", str(_REPO_ROOT))
    if not folder:
        sys.exit(0)
    dlg = ModeDialog()
    if dlg.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    win = MainWindow(Path(folder).resolve(), dlg._mode)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

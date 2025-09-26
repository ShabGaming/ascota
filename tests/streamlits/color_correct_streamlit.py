import io
import tempfile
import os
import numpy as np
import cv2
import streamlit as st
from typing import Optional, Tuple
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.ascota_core.imaging import process_image_pipeline
from src.ascota_core.color import match_color_cards_from_pipeline_outputs

# helper to convert PIL mask to small bw thumbnail bytes for st.image
def _mask_to_thumbnail_bytes(mask_pil: Image.Image, size=(160,120)) -> bytes:
    """Return PNG bytes for a resized black-and-white preview of mask_pil."""
    # Ensure mask is 'L'
    m = mask_pil.convert('L')
    # Resize to thumbnail while preserving aspect
    m_thumb = m.copy()
    m_thumb.thumbnail(size)
    # Convert to simple BW (0/255)
    m_bw = m_thumb.point(lambda p: 255 if p > 0 else 0).convert('L')
    buf = io.BytesIO()
    m_bw.save(buf, format='PNG')
    return buf.getvalue()

# ============================================================
# Color math helpers (16-bit internal, linear pipeline)
# ============================================================

def ensure_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """x in [0,1], sRGB encoded -> linear."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    low = x <= 0.04045
    high = ~low
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / 12.92
    out[high] = ((x[high] + a) / (1 + a)) ** 2.4
    return out


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """x in [0,1], linear -> sRGB encoded."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    low = x <= 0.0031308
    high = ~low
    out = np.empty_like(x, dtype=np.float32)
    out[low] = 12.92 * x[low]
    out[high] = (1 + a) * (x[high] ** (1 / 2.4)) - a
    return np.clip(out, 0.0, 1.0)


def bgr_linear_to_lab_u8(bgr_lin: np.ndarray) -> np.ndarray:
    """Convert linear BGR [0..1] -> CIE Lab **uint8** using OpenCV's 8-bit path.
    This avoids OpenCV's float-path (expects 0..1) pitfalls that can yield black outputs.
    """
    bgr_srgb = linear_to_srgb(np.clip(bgr_lin, 0.0, 1.0))
    bgr8 = (bgr_srgb * 255.0 + 0.5).astype(np.uint8)
    lab8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
    return lab8


def lab_u8_to_bgr_linear(lab8: np.ndarray) -> np.ndarray:
    """Convert CIE Lab **uint8** -> linear BGR [0..1] via OpenCV's 8-bit path."""
    bgr8 = cv2.cvtColor(lab8.astype(np.uint8), cv2.COLOR_LAB2BGR)
    bgr_srgb = np.clip(bgr8.astype(np.float32) / 255.0, 0.0, 1.0)
    return srgb_to_linear(bgr_srgb)


def to_uint8_preview(bgr_lin: np.ndarray) -> Image.Image:
    bgr_srgb = linear_to_srgb(np.clip(bgr_lin, 0.0, 1.0))
    rgb8 = (cv2.cvtColor((bgr_srgb * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB))
    return Image.fromarray(rgb8)


def to_png16_bytes(bgr_lin: np.ndarray) -> bytes:
    """Encode linear BGR [0..1] as 16-bit PNG bytes (sRGB encoded for writing)."""
    bgr_srgb = linear_to_srgb(np.clip(bgr_lin, 0.0, 1.0))
    bgr16 = (bgr_srgb * 65535.0 + 0.5).astype(np.uint16)
    ok, buf = cv2.imencode('.png', bgr16)
    if not ok:
        raise RuntimeError("Failed to encode PNG16 with OpenCV")
    return buf.tobytes()


# ============================================================
# I/O helpers
# ============================================================

def read_image_to_bgr_linear(file) -> Optional[np.ndarray]:
    """Return image as BGR linear float32 [0..1], or None if cannot be read. """
    name = getattr(file, 'name', None)
    data = file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        rgb8 = np.array(img, dtype=np.uint8)
        rgb_srgb = rgb8.astype(np.float32) / 255.0
        rgb_lin = srgb_to_linear(rgb_srgb)
        bgr_lin = cv2.cvtColor(rgb_lin, cv2.COLOR_RGB2BGR)
        return bgr_lin
    except Exception as e:
        st.error(f"Failed to read image '{name}': {e}")
        return None


def read_mask(mask_file, expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    try:
        arr = np.array(Image.open(mask_file).convert('L'))
    except Exception as e:
        st.error(f"Failed to read mask '{getattr(mask_file,'name',None)}': {e}")
        return None
    if (arr.shape[1], arr.shape[0]) != expected_shape:
        arr = cv2.resize(arr, expected_shape, interpolation=cv2.INTER_NEAREST)
    return arr >= 200


# ============================================================
# Manual pre-correction (applied in sRGB space, but kept 16-bit internally)
# ============================================================

def apply_manual_adjustments_linear(bgr_lin: np.ndarray, brightness: int, contrast: float, gamma: float,
                                    r_gain: float, g_gain: float, b_gain: float) -> np.ndarray:
    """Return BGR linear after manual edits. Edits are applied in sRGB space for user-friendly behavior."""
    bgr_srgb = linear_to_srgb(bgr_lin)
    # Apply per-channel gains in sRGB
    gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32).reshape(1, 1, 3)
    x = np.clip(bgr_srgb * contrast + (brightness / 255.0), 0.0, 1.0)
    x = np.clip(x * gains, 0.0, 1.0)
    # Gamma tweak in sRGB
    x = np.power(np.clip(x, 1e-6, 1.0), 1.0 / max(gamma, 1e-6))
    # Back to linear
    return srgb_to_linear(np.clip(x, 0.0, 1.0))


# ============================================================
# Matching algorithms (16-bit/float32 internal)
# ============================================================

def lab_mean_std_transfer_linear(src_bgr_lin: np.ndarray, tgt_bgr_lin: np.ndarray,
                                 src_mask: np.ndarray, tgt_mask: np.ndarray) -> np.ndarray:
    src_lab = bgr_linear_to_lab_u8(src_bgr_lin)
    tgt_lab = bgr_linear_to_lab_u8(tgt_bgr_lin)
    out = src_lab.astype(np.float32)
    for c in range(3):
        s_vals = src_lab[..., c][src_mask].astype(np.float32)
        t_vals = tgt_lab[..., c][tgt_mask].astype(np.float32)
        if len(s_vals) == 0 or len(t_vals) == 0:
            continue
        s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals) + 1e-6)
        t_mean, t_std = float(np.mean(t_vals)), float(np.std(t_vals))
        out[..., c] = (out[..., c] - s_mean) * (t_std / s_std) + t_mean
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return lab_u8_to_bgr_linear(out_u8)


def _build_monotone_quantile_mapping(src_vals: np.ndarray, tgt_vals: np.ndarray, n_knots: int = 257):
    """Return (xk, yk) monotone knot vectors to be used with np.interp. Operates on float arrays."""
    q = np.linspace(0.0, 1.0, n_knots, dtype=np.float32)
    s_q = np.quantile(src_vals.astype(np.float32), q)
    t_q = np.quantile(tgt_vals.astype(np.float32), q)
    # Enforce strict monotonicity for src knots
    s_q = np.maximum.accumulate(s_q)
    # If still flat segments, add tiny eps to make np.interp stable
    eps = 1e-4
    for i in range(1, len(s_q)):
        if s_q[i] <= s_q[i-1]:
            s_q[i] = s_q[i-1] + eps
    return s_q.astype(np.float32), t_q.astype(np.float32)


def monotone_lut_ab_only(src_bgr_lin: np.ndarray, tgt_bgr_lin: np.ndarray,
                         src_mask: np.ndarray, tgt_mask: np.ndarray,
                         n_knots: int = 257) -> np.ndarray:
    """Map a/b channels in Lab with a monotone quantile LUT. L is preserved from source.
    Uses 8-bit Lab path to avoid float-range pitfalls.
    """
    src_lab = bgr_linear_to_lab_u8(src_bgr_lin)
    tgt_lab = bgr_linear_to_lab_u8(tgt_bgr_lin)
    out = src_lab.astype(np.float32)
    for c in [1, 2]:  # a and b channels
        s_vals = src_lab[..., c][src_mask].astype(np.float32)
        t_vals = tgt_lab[..., c][tgt_mask].astype(np.float32)
        if len(s_vals) == 0 or len(t_vals) == 0:
            continue
        xk, yk = _build_monotone_quantile_mapping(s_vals, t_vals, n_knots=n_knots)
        ch = out[..., c]
        out[..., c] = np.interp(ch, xk, yk).astype(np.float32)
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return lab_u8_to_bgr_linear(out_u8)


def masked_histogram_match_rgb_linear(src_bgr_lin: np.ndarray, tgt_bgr_lin: np.ndarray,
                                      src_mask: np.ndarray, tgt_mask: np.ndarray,
                                      n_knots: int = 1025) -> np.ndarray:
    """High-precision per-channel quantile matching in sRGB RGB space; applied to full image."""
    src_srgb = linear_to_srgb(src_bgr_lin)
    tgt_srgb = linear_to_srgb(tgt_bgr_lin)
    # Work in RGB order for clarity, then convert back
    src_rgb = cv2.cvtColor(src_srgb, cv2.COLOR_BGR2RGB)
    tgt_rgb = cv2.cvtColor(tgt_srgb, cv2.COLOR_BGR2RGB)
    out_rgb = src_rgb.copy()
    for c in range(3):
        s_vals = src_rgb[..., c][src_mask].astype(np.float32)
        t_vals = tgt_rgb[..., c][tgt_mask].astype(np.float32)
        if len(s_vals) == 0 or len(t_vals) == 0:
            continue
        # Scale to 0..255 for knot finding
        s_vals255 = s_vals * 255.0
        t_vals255 = t_vals * 255.0
        xk, yk = _build_monotone_quantile_mapping(s_vals255, t_vals255, n_knots=n_knots)
        ch = (out_rgb[..., c] * 255.0).astype(np.float32)
        mapped = np.interp(ch, xk, yk) / 255.0
        out_rgb[..., c] = np.clip(mapped, 0.0, 1.0)
    out_bgr_srgb = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return srgb_to_linear(out_bgr_srgb)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Color Card Style Matcher", layout="wide")
st.title("🎨 Color Card Style Matcher (16‑bit internal)")
st.caption("Upload A/B images. The app will detect color cards automatically — no separate mask upload required. Manually tune A, then match A ↔ B using your chosen algorithm.")

with st.sidebar:
    st.header("1) Upload Images (A = source, B = target)")
    types = ["jpg", "jpeg", "png", "tif", "tiff", "cr2", "cr3"]
    a_file = st.file_uploader("Image A (uncorrected)", type=types)
    b_file = st.file_uploader("Image B (target style)", type=types)

    st.header("2) Manual Pre-Correction for A (applied before card detection)")
    brightness = st.slider("Brightness (A)", -100, 100, 0, step=1)
    contrast = st.slider("Contrast (A)", 0.50, 1.50, 1.00, step=0.01)
    gamma = st.slider("Gamma (A)", 0.50, 2.00, 1.00, step=0.01)
    r_gain = st.slider("Red gain (A)", 0.50, 1.50, 1.00, step=0.01)
    g_gain = st.slider("Green gain (A)", 0.50, 1.50, 1.00, step=0.01)
    b_gain = st.slider("Blue gain (A)", 0.50, 1.50, 1.00, step=0.01)

    st.header("3) Matching Algorithm")
    algo = st.selectbox(
        "Choose matching algorithm",
        [
            "Lab mean/std transfer",
            "Monotone LUT (a/b only)",
            "Histogram matching (masked)",
        ],
    )

    direction = st.selectbox(
        "Which image do you want to correct?",
        [
            "Correct A to match B (A→B)",
            "Correct B to match A (manual) (B→A_manual)",
        ],
    )

    run = st.button("Run Matching")

# Main processing (now only requires the two images)
if a_file is not None and b_file is not None:
    # Read uploaded files into memory
    a_bytes = a_file.read()
    b_bytes = b_file.read()

    # Convert uploaded bytes to linear BGR for preview / manual edits
    try:
        # reuse existing helper to read into linear BGR (works from file buffer)
        a_bgr_lin = read_image_to_bgr_linear(io.BytesIO(a_bytes))
        b_bgr_lin = read_image_to_bgr_linear(io.BytesIO(b_bytes))
    except Exception as e:
        st.error(f"Failed to decode uploaded images: {e}")
        st.stop()

    if a_bgr_lin is None or b_bgr_lin is None:
        st.stop()

    # Apply manual pre-correction to A (keeps linear internal)
    a_pre_lin = apply_manual_adjustments_linear(a_bgr_lin, brightness, contrast, gamma, r_gain, g_gain, b_gain)

    # Show previews
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("A (Original)")
        st.image(to_uint8_preview(a_bgr_lin), use_container_width=True)
    with col2:
        st.subheader("A (Manual Pre-Corrected)")
        st.image(to_uint8_preview(a_pre_lin), use_container_width=True)
    with col3:
        st.subheader("B (Target Style)")
        st.image(to_uint8_preview(b_bgr_lin), use_container_width=True)

    if run:
        # Save pre-corrected A and original B to temporary files so process_image_pipeline can load them
        tmp_files = []
        try:
            tmp_a = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            to_uint8_preview(a_pre_lin).save(tmp_a, format="PNG")
            tmp_a.close()
            tmp_files.append(tmp_a.name)

            tmp_b = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            # write the original B upload (use the original bytes to preserve fidelity)
            with open(tmp_b.name, "wb") as f:
                f.write(b_bytes)
            tmp_files.append(tmp_b.name)

            # Run detection pipeline on both images (A uses the pre-corrected image)
            try:
                pipe_a = process_image_pipeline(tmp_a.name, debug=False)
            except Exception as e:
                st.error(f"Card detection failed for Image A: {e}")
                raise

            try:
                pipe_b = process_image_pipeline(tmp_b.name, debug=False)
            except Exception as e:
                st.error(f"Card detection failed for Image B: {e}")
                raise

            # Map streamlit algo selection to internal method keys
            method_map = {
                "Lab mean/std transfer": "lab_mean_std_transfer",
                "Monotone LUT (a/b only)": "monotone_lut",
                "Histogram matching (masked)": "histogram_matching",
            }
            method_key = method_map.get(algo, "lab_mean_std_transfer")

            # If direction selects correcting B->A_manual, swap pipeline roles accordingly when calling matcher
            if direction.startswith("Correct A"):
                src_pipe, tgt_pipe = pipe_a, pipe_b
            else:
                src_pipe, tgt_pipe = pipe_b, pipe_a

            # Call the convenience wrapper which accepts pipeline outputs
            try:
                match_out = match_color_cards_from_pipeline_outputs(
                    original_pipeline_output=src_pipe,
                    target_pipeline_output=tgt_pipe,
                    method=method_key,
                    n_knots=513 if method_key == "monotone_lut" else (1025 if method_key == "histogram_matching" else None),
                    debug=False,
                )
            except Exception as e:
                st.error(f"Color matching failed: {e}")
                raise

            # Show result and provide downloads
            st.markdown("---")
            res_title = "Result: A matched to B (applied to full image)" if direction.startswith("Correct A") else "Result: B matched to A (manual)"
            st.subheader(res_title)

            # Display matched preview and small mask thumbnails beneath
            # matched_preview is a PIL Image
            st.image(match_out['matched_preview'], use_container_width=True)

            # Try to extract the masks used from the pipeline outputs; prefer the masks corresponding to 'used_card_type'
            used_type = match_out.get('used_card_type')
            src_masks = src_pipe.get('masks', []) if isinstance(src_pipe, dict) else []
            tgt_masks = tgt_pipe.get('masks', []) if isinstance(tgt_pipe, dict) else []
            src_types = src_pipe.get('card_types', []) if isinstance(src_pipe, dict) else []
            tgt_types = tgt_pipe.get('card_types', []) if isinstance(tgt_pipe, dict) else []

            # Find index of mask used in both
            def _index_of_type(types_list, desired):
                if not types_list:
                    return None
                for i, t in enumerate(types_list):
                    if t == desired and i < len(types_list):
                        return i
                return None

            src_idx = _index_of_type(src_types, used_type) if used_type else None
            tgt_idx = _index_of_type(tgt_types, used_type) if used_type else None

            # Display masks if available
            if src_idx is not None or tgt_idx is not None:
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Source mask (used)**")
                    if src_idx is not None and src_idx < len(src_masks):
                        try:
                            mb = _mask_to_thumbnail_bytes(src_masks[src_idx])
                            st.image(mb, use_container_width=False)
                        except Exception:
                            st.write("(failed to render source mask)")
                    else:
                        st.write("(no mask)")
                with cols[1]:
                    st.markdown("**Target mask (used)**")
                    if tgt_idx is not None and tgt_idx < len(tgt_masks):
                        try:
                            mb = _mask_to_thumbnail_bytes(tgt_masks[tgt_idx])
                            st.image(mb, use_container_width=False)
                        except Exception:
                            st.write("(failed to render target mask)")
                    else:
                        st.write("(no mask)")

            out_name_png16 = "matched_result_16bit.png"
            png16_bytes = match_out['matched_png16']
            st.download_button("⬇️ Download Corrected (16‑bit PNG)", data=png16_bytes, file_name=out_name_png16, mime="image/png")

        finally:
            # Cleanup temp files
            for p in tmp_files:
                try:
                    os.unlink(p)
                except Exception:
                    pass
else:
    st.info("⬅️ Upload Image A and Image B in the sidebar to begin.")

st.markdown("""
---
**Notes**
- Internal processing runs at high precision (linear float) and encodes outputs to **16‑bit PNG** to minimize banding.
- The app now detects color cards automatically (no separate mask upload needed). Manual pre-correction for A is applied before detection so masks reflect edits.
- Algorithms:
  - **Lab mean/std transfer**: robust baseline using masked stats in Lab.
  - **Monotone LUT (a/b only)**: smooth chroma mapping in Lab; preserves luminance/texture.
  - **Histogram matching (masked)**: high‑precision quantile mapping in sRGB; can be stronger/stylized.
""")

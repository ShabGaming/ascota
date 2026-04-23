"""
Streamlit app for pottery vs non-pottery binary classification.

Uses DINOv2-large features and the sklearn classifier trained in
notebooks/pottery_non_pottery.ipynb (joblib files under ascota_classification/models/).

Run with: streamlit run type_pottery_non_pottery_streamlit.py
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import time

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from ascota_classification.type import (
    classify_pottery_vs_non_pottery,
    batch_classify_pottery_vs_non_pottery,
    DEFAULT_MODEL_PATH_POTTERY_BINARY,
)


def load_image(uploaded_file):
    """Load and convert image to RGBA format (same behavior as type classifier app)."""
    image = Image.open(uploaded_file)
    if image.mode != "RGBA":
        if image.mode in ("LA", "L"):
            image = image.convert("RGBA")
        elif image.mode == "RGB":
            st.warning("Image has no transparency. Treating near-white as transparent.")
            img_array = np.array(image)
            white_threshold = 240
            is_white = np.all(img_array > white_threshold, axis=2)
            alpha = np.where(is_white, 0, 255).astype(np.uint8)
            image = Image.fromarray(np.dstack([img_array, alpha]))
        else:
            image = image.convert("RGBA")
    return image


def label_icon(label: str) -> str:
    return "🏺" if label == "pottery" else "🧱"


def label_color(label: str) -> str:
    return "#6d4c41" if label == "pottery" else "#546e7a"


def visualize_image_properties(image):
    """Visualize alpha channel and size."""
    st.subheader("Image analysis")
    img_array = np.array(image)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Alpha channel**")
        alpha = img_array[:, :, 3]
        st.image(alpha, caption="Transparency mask", width="stretch", clamp=True)
        opaque = np.sum(alpha > 0)
        total = alpha.size
        st.metric("Opaque coverage", f"{opaque / total * 100:.1f}%")
    with col2:
        st.markdown("**Input**")
        st.image(image, caption="Input image", width="stretch")
        st.metric("Size", f"{image.width} × {image.height}")


def display_binary_result(result: dict, elapsed_time: float):
    """Show single-image classification result."""
    label = result.get("label", "unknown")
    icon = label_icon(label)
    color = label_color(label)
    st.markdown(
        f"<h1 style='color: {color};'>{icon} {label.replace('_', ' ').upper()}</h1>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing time", f"{elapsed_time:.3f}s")
    if result.get("confidence") is not None:
        with col2:
            st.metric("Confidence (max class)", f"{result['confidence']:.4f}")
        with col3:
            st.metric("P(pottery)", f"{result.get('p_pottery', 0):.4f}")
    with st.expander("Full result"):
        st.json(result)


def resolve_model_path(custom_path: str | None) -> Path | None:
    """Return Path for custom non-empty path, else None for default."""
    if not custom_path or not str(custom_path).strip():
        return None
    p = Path(custom_path.strip())
    return p if p.is_absolute() else Path(__file__).parent.parent.parent / p


def single_mode(model_path: Path | None, debug: bool, show_analysis: bool):
    st.header("Single image")
    uploaded = st.file_uploader(
        "Upload an image (PNG with transparency recommended)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is None:
        st.info("Upload an image to classify as **pottery** or **non-pottery**.")
        return

    try:
        image = load_image(uploaded)
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return

    if show_analysis:
        with st.expander("Image properties", expanded=False):
            visualize_image_properties(image)

    st.divider()
    try:
        with st.spinner("Classifying…"):
            t0 = time.time()
            result = classify_pottery_vs_non_pottery(
                image,
                model_path=model_path,
                return_confidence=True,
                debug=debug,
            )
            elapsed = time.time() - t0
        st.subheader("Result")
        display_binary_result(result, elapsed)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Train the model in `notebooks/pottery_non_pottery.ipynb` and copy the `.pkl` files into `src/ascota_classification/models/`.")
    except Exception as e:
        st.error(f"Classification failed: {e}")
        if debug:
            st.exception(e)


def batch_mode(model_path: Path | None, debug: bool):
    st.header("Batch processing")
    files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("Upload one or more images, then click **Classify all**.")
        return

    st.caption(f"{len(files)} file(s) selected")
    if st.button("Classify all", type="primary"):
        images = []
        names = []
        for f in files:
            try:
                images.append(load_image(f))
                names.append(f.name)
            except Exception as e:
                st.error(f"Failed to load {f.name}: {e}")
        if not images:
            return
        try:
            with st.spinner(f"Classifying {len(images)} images…"):
                t0 = time.time()
                results = batch_classify_pottery_vs_non_pottery(
                    images,
                    model_path=model_path,
                    return_confidence=True,
                    debug=debug,
                )
                elapsed = time.time() - t0
            st.success(f"Done in {elapsed:.2f}s ({elapsed / len(results):.3f}s per image)")

            rows = []
            for name, r in zip(names, results):
                rows.append(
                    {
                        "Image": name,
                        "Label": r["label"],
                        "Confidence": f"{r.get('confidence', 0):.4f}",
                        "P(pottery)": f"{r.get('p_pottery', 0):.4f}",
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)

            counts = {}
            for r in results:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Pottery", counts.get("pottery", 0))
            with c2:
                st.metric("Non-pottery", counts.get("non_pottery", 0))

            if len(results) > 1:
                try:
                    import plotly.express as px

                    pie_df = pd.DataFrame(
                        {"Label": list(counts.keys()), "Count": list(counts.values())}
                    )
                    fig = px.pie(
                        pie_df,
                        values="Count",
                        names="Label",
                        title="Label distribution",
                        color="Label",
                        color_discrete_map={
                            "pottery": label_color("pottery"),
                            "non_pottery": label_color("non_pottery"),
                        },
                    )
                    st.plotly_chart(fig, width="stretch")
                except ImportError:
                    pass

            csv = df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="pottery_binary_results.csv", mime="text/csv")

            with st.expander("Per-image thumbnails"):
                for name, img, r in zip(names, images, results):
                    ic = label_icon(r["label"])
                    st.markdown(f"**{ic} {name}** → `{r['label']}` (P_pottery={r.get('p_pottery', 0):.3f})")
                    st.image(img, width=200)
                    st.divider()

        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Place trained `pottery_binary_*_optimized.pkl` under `src/ascota_classification/models/`.")
        except Exception as e:
            st.error(f"Batch classification failed: {e}")
            if debug:
                st.exception(e)


def main():
    st.set_page_config(
        page_title="Pottery vs non-pottery",
        page_icon="🏺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Pottery vs non-pottery")
    st.markdown(
        "Binary classifier: **DINOv2-large** + sklearn (see `notebooks/pottery_non_pottery.ipynb`)."
    )

    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["Single image", "Batch processing"])
    default_full = Path(__file__).parent.parent.parent / "src" / "ascota_classification" / DEFAULT_MODEL_PATH_POTTERY_BINARY
    st.sidebar.caption(f"Default model path:\n`{default_full}`")
    custom = st.sidebar.text_input(
        "Optional: custom classifier .pkl path",
        placeholder="Leave empty for default",
        help="Absolute path, or repo-relative (e.g. src/ascota_classification/models/pottery_binary_logistic_regression_optimized.pkl)",
    )
    model_path = resolve_model_path(custom)
    debug = st.sidebar.checkbox("Debug output", value=False)
    show_analysis = st.sidebar.checkbox("Show image analysis (single mode)", value=True)

    st.sidebar.divider()
    st.sidebar.markdown("**About**")
    st.sidebar.info(
        "Labels: `pottery` (positive class 1) vs `non_pottery` (0). "
        "Same preprocessing as the training notebook (RGBA→RGB, 224 crop)."
    )

    st.divider()
    if mode == "Single image":
        single_mode(model_path, debug, show_analysis)
    else:
        batch_mode(model_path, debug)


if __name__ == "__main__":
    main()

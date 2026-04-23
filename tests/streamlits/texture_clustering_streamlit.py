"""
Streamlit app for testing LBP + GLCM + PCA + HDBSCAN texture clustering.

Clusters similar pottery by grayscale texture (opaque region only), not color.
Uses luminance, Local Binary Patterns, Gray-Level Co-occurrence features,
standardization, PCA, and HDBSCAN. Defaults favor fine-grained clusters (leaf).

Run with: streamlit run texture_clustering_streamlit.py
(from tests/streamlits/)
"""

import streamlit as st
import sys
import time
from pathlib import Path
from typing import List

from PIL import Image

# Add project root to path so imports work from any cwd (e.g. streamlit run from repo root)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.ascota_classification.texture import (
    cluster_similar_images_by_texture,
    extract_texture_pca_features,
)


def load_image(uploaded_file) -> Image.Image:
    """Load image; ensure RGBA or RGB."""
    image = Image.open(uploaded_file)
    if image.mode not in ("RGBA", "RGB"):
        image = image.convert("RGBA")
    return image


def display_cluster_grid(
    images: List[Image.Image],
    indices: List[int],
    names: List[str],
    num_columns: int = 4,
) -> None:
    """Display images in a grid with captions."""
    if not indices:
        st.write("No images in this cluster.")
        return
    cols = st.columns(num_columns)
    for i, idx in enumerate(indices):
        col_idx = i % num_columns
        with cols[col_idx]:
            try:
                img = images[idx]
                img = img.copy()
                img.thumbnail((200, 200))
                caption = names[idx] if idx < len(names) else f"Image {idx}"
                st.image(img, caption=caption, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image {idx}: {e}")


def main() -> None:
    st.set_page_config(
        page_title="Texture Clustering",
        page_icon="🧱",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧱 Texture (LBP + GLCM) + PCA + HDBSCAN")
    st.markdown("""
    Upload multiple images (transparent PNGs preferred). The pipeline uses **surface texture only**
    (grayscale luminance over the opaque region—no white compositing):

    1. **LBP** histograms (uniform, two radii) and **GLCM** properties on quantized gray.
    2. **Standardize** raw features, then **PCA** reduction.
    3. **HDBSCAN** clustering (noise points appear under "Noise").

    Defaults use **`min_samples = 1`** so HDBSCAN does not mark most points as noise (2 is often
    too strict). Use **`leaf`** for finer splits or **`eom`** if you still see too much noise.
    Tune parameters in the sidebar, then click **Run clustering**.
    """)

    st.sidebar.header("HDBSCAN Parameters")
    min_cluster_size = st.sidebar.number_input(
        "min_cluster_size",
        min_value=2,
        max_value=50,
        value=2,
        help="Minimum cluster size; smaller groups become noise.",
    )
    min_samples = st.sidebar.number_input(
        "min_samples",
        min_value=1,
        max_value=50,
        value=1,
        help="1 = fewest noise points (recommended). 2+ requires denser neighborhoods and often labels most images as noise.",
    )
    cluster_selection_epsilon = st.sidebar.number_input(
        "cluster_selection_epsilon",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Distance threshold for merging clusters (0 = no merge by epsilon).",
    )
    cluster_selection_method = st.sidebar.selectbox(
        "cluster_selection_method",
        options=["eom", "leaf"],
        index=1,
        help="eom = fewer noise points, broader clusters. leaf = finer clusters (try with min_samples=1).",
    )

    st.sidebar.header("Feature Extraction")
    pca_components = st.sidebar.number_input(
        "PCA: fraction of feature dims (0–1)",
        min_value=0.1,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="Same convention as color clustering: fraction of raw feature dimensions (after scaling).",
    )
    alpha_threshold = st.sidebar.slider(
        "Alpha threshold (opaque pixels)",
        min_value=1,
        max_value=255,
        value=128,
        help="Pixels with alpha >= this define the opaque mask and bounding box.",
    )
    resize_max = st.sidebar.number_input(
        "Resize max (longer side)",
        min_value=64,
        max_value=1024,
        value=512,
        step=64,
        help="Resize before texture features for speed.",
    )
    show_pca_plot = st.sidebar.checkbox("Show PCA 2D plot (first 2 components)", value=True)

    st.sidebar.divider()
    images_per_row = st.sidebar.slider("Images per row in clusters", min_value=2, max_value=6, value=4)

    uploaded_files = st.file_uploader(
        "Choose images to cluster (transparent PNGs preferred)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Images are clustered by grayscale texture (LBP + GLCM), not color.",
    )

    if not uploaded_files:
        st.info("Upload one or more images to run clustering.")
        return

    with st.spinner("Loading images..."):
        images: List[Image.Image] = []
        names: List[str] = []
        for f in uploaded_files:
            try:
                images.append(load_image(f))
                names.append(f.name)
            except Exception as e:
                st.error(f"Error loading {f.name}: {e}")
    if not images:
        st.error("No images loaded.")
        return

    st.success(f"Loaded {len(images)} images.")

    if st.button("Run clustering", type="primary"):
        try:
            with st.spinner("Extracting texture + PCA features and clustering..."):
                start = time.perf_counter()
                result = cluster_similar_images_by_texture(
                    images,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    cluster_selection_method=cluster_selection_method,
                    pca_components=pca_components,
                    alpha_threshold=alpha_threshold,
                    resize_max=resize_max,
                    return_pca_and_labels=True,
                )
                elapsed = time.perf_counter() - start

            clusters, noise_indices, pca, labels, scaler = result
            n_clusters = len(clusters)
            n_noise = len(noise_indices)

            st.metric("Total time", f"{elapsed:.3f}s")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clusters", n_clusters)
            with col2:
                st.metric("Noise points", n_noise)
            with col3:
                st.metric("Total images", len(images))

            if show_pca_plot and pca is not None and scaler is not None and len(images) > 0:
                try:
                    features, _, _ = extract_texture_pca_features(
                        images,
                        pca_components=pca_components,
                        alpha_threshold=alpha_threshold,
                        resize_max=resize_max,
                    )
                    if features.shape[1] >= 2:
                        import plotly.express as px
                        import pandas as pd

                        df = pd.DataFrame({
                            "PC1": features[:, 0],
                            "PC2": features[:, 1],
                            "label": [f"C{int(l)}" if l >= 0 else "Noise" for l in labels],
                            "name": names,
                        })
                        fig = px.scatter(df, x="PC1", y="PC2", color="label", hover_data=["name"])
                        st.subheader("PCA (first 2 components)")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not draw PCA plot: {e}")

            st.divider()
            st.subheader("Clusters")

            for i, indices in enumerate(clusters):
                cluster_name = f"Cluster {i + 1} ({len(indices)} images)"
                with st.expander(cluster_name, expanded=(i == 0)):
                    display_cluster_grid(images, indices, names, num_columns=images_per_row)
                    with st.expander("View indices", expanded=False):
                        st.write(indices)

            if noise_indices:
                with st.expander(f"Noise ({len(noise_indices)} images)", expanded=False):
                    display_cluster_grid(images, noise_indices, names, num_columns=images_per_row)
                    with st.expander("View indices", expanded=False):
                        st.write(noise_indices)

        except Exception as e:
            st.error(f"Clustering failed: {e}")
            raise


if __name__ == "__main__":
    main()

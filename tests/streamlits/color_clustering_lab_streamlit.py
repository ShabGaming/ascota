"""
Streamlit app for testing Lab + PCA + HDBSCAN color clustering of transparent images.

Clusters similar images by CIE Lab color summary (opaque pixels only), PCA reduction,
and HDBSCAN. Use for grouping images by dominant color/lighting.

Run with: streamlit run color_clustering_lab_streamlit.py
(from tests/streamlits/)
"""

import streamlit as st
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# Add project root to path so imports work from any cwd (e.g. streamlit run from repo root)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.ascota_classification.color import (
    cluster_similar_images,
    extract_lab_pca_features,
    cluster_images_hdbscan,
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
        page_title="Lab Color Clustering",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎨 Lab + PCA + HDBSCAN Color Clustering")
    st.markdown("""
    Upload multiple images (transparent PNGs preferred). The pipeline:
    1. Converts each image to **CIE Lab** and summarizes color over opaque pixels.
    2. Reduces dimension with **PCA**.
    3. Clusters with **HDBSCAN** (noise points get their own "Noise" group).

    Tune HDBSCAN and PCA in the sidebar, then click **Run clustering**.
    """)

    # Sidebar: HDBSCAN and feature params
    st.sidebar.header("HDBSCAN Parameters")
    min_cluster_size = st.sidebar.number_input(
        "min_cluster_size",
        min_value=2,
        max_value=50,
        value=5,
        help="Minimum size of a cluster; smaller groups become noise.",
    )
    min_samples = st.sidebar.number_input(
        "min_samples",
        min_value=1,
        max_value=50,
        value=5,
        help="Core point neighborhood; often same as min_cluster_size.",
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
        index=0,
        help="eom = excess of mass; leaf = leaf extraction.",
    )

    st.sidebar.header("Feature Extraction")
    pca_components = st.sidebar.number_input(
        "PCA variance to retain (0–1) or components (int)",
        min_value=0.1,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="Fraction of variance (e.g. 0.95) or use 1.0 for no reduction (6 dims).",
    )
    alpha_threshold = st.sidebar.slider(
        "Alpha threshold (opaque pixels)",
        min_value=1,
        max_value=255,
        value=128,
        help="Pixels with alpha >= this are used for Lab summary.",
    )
    resize_max = st.sidebar.number_input(
        "Resize max (longer side)",
        min_value=64,
        max_value=1024,
        value=512,
        step=64,
        help="Resize images before Lab to speed up; None = no resize.",
    )
    show_pca_plot = st.sidebar.checkbox("Show PCA 2D plot (first 2 components)", value=True)

    st.sidebar.divider()
    images_per_row = st.sidebar.slider("Images per row in clusters", min_value=2, max_value=6, value=4)

    # Main: file upload
    uploaded_files = st.file_uploader(
        "Choose images to cluster (transparent PNGs preferred)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload multiple images; they will be clustered by Lab color summary.",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more images to run clustering.")
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
            with st.spinner("Extracting Lab+PCA features and clustering..."):
                start = time.perf_counter()
                result = cluster_similar_images(
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

            clusters, noise_indices, pca, labels = result
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

            # Optional: PCA 2D plot
            if show_pca_plot and pca is not None and len(images) > 0:
                try:
                    features, _ = extract_lab_pca_features(
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

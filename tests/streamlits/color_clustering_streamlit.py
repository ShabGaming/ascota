"""
Streamlit application for clustering images by lighting conditions using corner analysis.
"""

import streamlit as st
import os
import sys
from typing import List, Optional
from PIL import Image
import numpy as np

# Add the src directory to the path so we can import from ascota_core
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src')
sys.path.insert(0, src_path)

try:
    from ascota_core.color import group_similar_images_by_lighting
except ImportError as e:
    st.error(f"Failed to import clustering function: {e}")
    st.stop()


def display_images_in_columns(image_paths: List[str], num_columns: int = 3):
    """Display images in a grid layout using Streamlit columns."""
    if not image_paths:
        st.write("No images in this cluster.")
        return
    
    # Create columns for image display
    cols = st.columns(num_columns)
    
    for i, img_path in enumerate(image_paths):
        col_idx = i % num_columns
        
        with cols[col_idx]:
            try:
                # Load and display image
                img = Image.open(img_path)
                
                # Create a reasonable thumbnail size
                img.thumbnail((300, 300))
                
                # Display image with filename as caption
                filename = os.path.basename(img_path)
                st.image(img, caption=filename, width='stretch')
                
            except Exception as e:
                st.error(f"Failed to load image {img_path}: {e}")


def main():
    st.set_page_config(
        page_title="Image Clustering by Lighting",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Image Clustering by Lighting Conditions")
    st.markdown("""
    This tool analyzes images in a directory and groups them based on similar lighting conditions.
    It examines the corners of each image, selects the brightest corners (avoiding color cards),
    and clusters images with similar lighting characteristics.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Directory input
    directory_path = st.sidebar.text_input(
        "Directory Path",
        placeholder="Enter the path to your image directory",
        help="Provide the full path to the directory containing images to cluster"
    )
    
    # Optional k parameter
    use_custom_k = st.sidebar.checkbox(
        "Override number of clusters (k)",
        help="Check this to manually specify the number of clusters instead of auto-detection"
    )
    
    custom_k = None
    if use_custom_k:
        custom_k = st.sidebar.number_input(
            "Number of clusters (k)",
            min_value=1,
            max_value=20,
            value=3,
            help="Number of clusters to create. If not specified, will be determined automatically using BIC."
        )
    
    # File extensions filter
    st.sidebar.subheader("File Extensions")
    default_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    selected_extensions = st.sidebar.multiselect(
        "Select image file extensions to process",
        options=default_extensions,
        default=['.jpg', '.jpeg', '.png'],
        help="Choose which image file types to include in the clustering analysis"
    )
    
    # Display options
    st.sidebar.subheader("Display Options")
    images_per_row = st.sidebar.slider(
        "Images per row",
        min_value=1,
        max_value=6,
        value=3,
        help="Number of images to display per row in each cluster"
    )

    # Sensitivity slider for clustering
    sensitivity = st.sidebar.slider(
        "Clustering sensitivity",
        min_value=0.2,
        max_value=1.5,
        value=1.0,
        step=0.1,
        help="Controls how sensitive the clustering is to small lighting/color differences. Higher = more clusters."
    )
    
    # Main content area
    if not directory_path:
        st.info("👆 Please enter a directory path in the sidebar to get started.")
        return
    
    if not os.path.exists(directory_path):
        st.error(f"❌ Directory does not exist: {directory_path}")
        return
    
    if not selected_extensions:
        st.error("❌ Please select at least one file extension to process.")
        return
    
    # Run clustering button
    if st.button("🔄 Run Clustering Analysis", type="primary"):
        
        with st.spinner("Analyzing images and clustering by lighting conditions..."):
            try:
                # Run the clustering algorithm
                clustered_images = group_similar_images_by_lighting(
                    directory=directory_path,
                    k=custom_k,
                    extensions=selected_extensions,
                    sensitivity=sensitivity,
                    debug=True  # Set to True for debugging output
                )
                
                # Store results in session state
                st.session_state.clustered_images = clustered_images
                st.session_state.directory_path = directory_path
                st.session_state.custom_k = custom_k
                st.session_state.sensitivity = sensitivity
                
            except Exception as e:
                st.error(f"❌ Error during clustering: {e}")
                return
        
        st.success(f"✅ Clustering completed! Found {len(clustered_images)} clusters.")
    
    # Display results if available
    if hasattr(st.session_state, 'clustered_images') and st.session_state.clustered_images:
        clustered_images = st.session_state.clustered_images
        
        st.header("📊 Clustering Results")
        
        # Summary statistics
        total_images = sum(len(cluster) for cluster in clustered_images)
        st.metric("Total Images Processed", total_images)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", len(clustered_images))
        with col2:
            largest_cluster_size = max(len(cluster) for cluster in clustered_images) if clustered_images else 0
            st.metric("Largest Cluster Size", largest_cluster_size)
        with col3:
            avg_cluster_size = total_images / len(clustered_images) if clustered_images else 0
            st.metric("Average Cluster Size", f"{avg_cluster_size:.1f}")
        
        st.markdown("---")
        
        # Display clusters with expandable sections
        for i, cluster_images in enumerate(clustered_images):
            cluster_name = f"Cluster {i+1} ({len(cluster_images)} images)"
            
            with st.expander(cluster_name, expanded=(i == 0)):  # First cluster expanded by default
                st.write(f"**Images in {cluster_name}:**")
                
                # Show file paths in a collapsible section
                with st.expander("📁 View file paths", expanded=False):
                    for img_path in cluster_images:
                        st.text(img_path)
                
                # Display images in grid
                display_images_in_columns(cluster_images, num_columns=images_per_row)
        
        # Export options
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        if st.button("📄 Generate Text Report"):
            report_lines = [
                f"Image Clustering Report",
                f"Directory: {st.session_state.directory_path}",
                f"Total Images: {total_images}",
                f"Number of Clusters: {len(clustered_images)}",
                f"Custom K: {st.session_state.custom_k if st.session_state.custom_k else 'Auto-detected'}",
                f"Sensitivity: {st.session_state.sensitivity}",
                "",
                "Cluster Details:",
                "=" * 50
            ]
            
            for i, cluster_images in enumerate(clustered_images):
                report_lines.append(f"\nCluster {i+1} ({len(cluster_images)} images):")
                report_lines.append("-" * 30)
                for img_path in cluster_images:
                    report_lines.append(f"  {img_path}")
            
            report_text = "\n".join(report_lines)
            
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"clustering_report_{len(clustered_images)}clusters.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()

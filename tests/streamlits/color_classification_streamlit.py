"""
Streamlit app for testing and showcasing pottery color classification.

This app demonstrates three different classification methods:
1. LAB Threshold
2. K-means LAB
3. CLIP ViT

Run with: streamlit run color_classification.py
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

from ascota_classification.color import classify_pottery_color


def load_image(uploaded_file):
    """Load and convert image to RGBA format."""
    image = Image.open(uploaded_file)
    if image.mode != 'RGBA':
        # Try to preserve transparency if it exists
        if image.mode in ('LA', 'L'):
            image = image.convert('RGBA')
        elif image.mode == 'RGB':
            st.warning("Image has no transparency. Adding a white background as transparent.")
            # Create alpha channel from white pixels
            img_array = np.array(image)
            # Assuming white background (can be adjusted)
            white_threshold = 240
            is_white = np.all(img_array > white_threshold, axis=2)
            alpha = np.where(is_white, 0, 255).astype(np.uint8)
            image = Image.fromarray(np.dstack([img_array, alpha]))
        else:
            image = image.convert('RGBA')
    return image


def display_result_card(method_name, result, elapsed_time):
    """Display a styled result card for a classification method."""
    with st.container():
        st.markdown(f"### {method_name}")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display label with color coding
            label = result.get('label', 'Unknown')
            if 'Red' in label:
                st.markdown(f"<h2 style='color: #d32f2f;'>🔴 {label}</h2>", unsafe_allow_html=True)
            elif 'Black' in label:
                st.markdown(f"<h2 style='color: #424242;'>⚫ {label}</h2>", unsafe_allow_html=True)
            elif 'Mixed' in label:
                st.markdown(f"<h2 style='color: #ff9800;'>🟠 {label}</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: #757575;'>❓ {label}</h2>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Processing Time", f"{elapsed_time:.3f}s")
        
        # Display scores if available (CLIP method)
        if 'scores' in result:
            st.subheader("Confidence Scores")
            scores_df = pd.DataFrame([
                {"Label": label, "Score": f"{score:.4f}", "Percentage": f"{score*100:.2f}%"}
                for label, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(scores_df, width="stretch", hide_index=True)
            
            # Create a bar chart
            import plotly.express as px
            fig = px.bar(
                scores_df, 
                x='Label', 
                y='Score',
                title='Classification Confidence',
                color='Score',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display method details
        with st.expander("Method Details"):
            for key, value in result.items():
                if key not in ['label', 'scores']:
                    st.write(f"**{key}:** {value}")
        
        st.divider()


def visualize_image_properties(image):
    """Visualize image properties like alpha channel and LAB distribution."""
    st.subheader("Image Analysis")
    
    img_array = np.array(image)
    
    # Create columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Alpha Channel**")
        alpha = img_array[:, :, 3]
        st.image(alpha, caption="Transparency Mask", width="stretch", clamp=True)
        
        # Stats
        pottery_pixels = np.sum(alpha > 0)
        total_pixels = alpha.size
        st.metric("Pottery Coverage", f"{pottery_pixels/total_pixels*100:.1f}%")
    
    with col2:
        st.markdown("**Original Image**")
        st.image(image, caption="Input Image", width="stretch")
        
        # Image stats
        st.metric("Image Size", f"{image.width} × {image.height}")


def main():
    st.set_page_config(
        page_title="Pottery Color Classifier",
        page_icon="🏺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏺 Pottery Color Classification")
    st.markdown("""
    This app classifies pottery color using three different methods:
    - **LAB Threshold**: CIELAB color space thresholds
    - **K-means LAB**: Clustering in CIELAB space
    - **CLIP ViT**: Deep learning-based classification
    
    Upload a pottery image with a transparent background to get started!
    """)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("LAB Threshold Settings")
    st.sidebar.markdown("*No parameters to adjust*")
    
    st.sidebar.subheader("K-means LAB Settings")
    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=5,
        value=2,
        help="Number of clusters for K-means algorithm"
    )
    
    st.sidebar.subheader("CLIP ViT Settings")
    candidate_labels = st.sidebar.text_input(
        "Candidate Labels",
        value="Red Pottery, Black Pottery",
        help="Comma-separated list of labels for classification"
    )
    
    st.sidebar.subheader("Debug Options")
    debug_mode = st.sidebar.checkbox("Enable Debug Output", value=False)
    show_image_analysis = st.sidebar.checkbox("Show Image Analysis", value=True)
    
    st.sidebar.divider()
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This tool uses computer vision and machine learning to classify pottery colors.
    
    **Methods:**
    - LAB: Fast, threshold-based
    - K-means: Clustering approach
    - CLIP: AI-powered classification
    """)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Choose a pottery image (PNG with transparency)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image with transparent background for best results"
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            with st.spinner("Loading image..."):
                image = load_image(uploaded_file)
            
            # Display image analysis if enabled
            if show_image_analysis:
                with st.expander("📊 Image Properties", expanded=False):
                    visualize_image_properties(image)
            
            st.divider()
            st.header("🎯 Classification Results")
            
            # Run all three methods
            methods = [
                {
                    "name": "LAB Threshold",
                    "method": "lab_threshold",
                    "icon": "📊",
                    "kwargs": {}
                },
                {
                    "name": "K-means LAB",
                    "method": "kmeans_lab",
                    "icon": "🎯",
                    "kwargs": {"n_clusters": n_clusters}
                },
                {
                    "name": "CLIP ViT",
                    "method": "clip_vit",
                    "icon": "🤖",
                    "kwargs": {"candidate_labels": candidate_labels}
                }
            ]
            
            results = []
            
            # Create tabs for each method
            tabs = st.tabs([f"{m['icon']} {m['name']}" for m in methods])
            
            for tab, method_config in zip(tabs, methods):
                with tab:
                    try:
                        with st.spinner(f"Running {method_config['name']}..."):
                            start_time = time.time()
                            
                            result = classify_pottery_color(
                                image=image,
                                method=method_config['method'],
                                debug=debug_mode,
                                **method_config['kwargs']
                            )
                            
                            elapsed_time = time.time() - start_time
                        
                        # Display result
                        display_result_card(
                            method_config['name'],
                            result,
                            elapsed_time
                        )
                        
                        results.append({
                            "Method": method_config['name'],
                            "Classification": result['label'],
                            "Time (s)": f"{elapsed_time:.3f}"
                        })
                        
                    except Exception as e:
                        st.error(f"Error in {method_config['name']}: {str(e)}")
                        if debug_mode:
                            st.exception(e)
            
            # Summary comparison
            if results:
                st.divider()
                st.header("📈 Comparison Summary")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create comparison table
                    comparison_df = pd.DataFrame(results)
                    st.dataframe(comparison_df, width="stretch", hide_index=True)
                
                with col2:
                    # Count agreements
                    labels = [r['Classification'] for r in results]
                    if len(set(labels)) == 1:
                        st.success("✅ All methods agree!")
                    else:
                        st.warning("⚠️ Methods disagree")
                    
                    # Show most common classification
                    from collections import Counter
                    most_common = Counter(labels).most_common(1)[0]
                    st.metric(
                        "Consensus",
                        most_common[0],
                        f"{most_common[1]}/{len(results)} methods"
                    )
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            if debug_mode:
                st.exception(e)
    
    else:
        # Show example/instructions when no file is uploaded
        st.info("👆 Upload a pottery image to begin classification")
        
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. **Prepare your image**: Pottery should have a transparent background (RGBA format)
        2. **Upload**: Use the file uploader above
        3. **Configure**: Adjust parameters in the sidebar if needed
        4. **Review**: Compare results from all three methods
        
        #### Tips:
        - Images with clear transparency work best
        - Try different K-means cluster values (2-5)
        - Customize CLIP labels for specific pottery types
        - Enable debug mode to see detailed processing information
        """)


if __name__ == "__main__":
    main()

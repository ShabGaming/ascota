"""
Streamlit app for testing and showcasing pottery decoration classification.

This app demonstrates decoration pattern classification:
- Impressed: decorations made by pressing objects into the clay
- Incised: decorations made by cutting/carving into the clay

Uses DINOv2 ViT-L/14 model with optimized logistic regression classifier.

Run with: streamlit run decoration_classification_streamlit.py
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

from ascota_classification.decoration import classify_pottery_decoration, batch_classify_pottery_decoration


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


def display_result_card(result, elapsed_time):
    """Display a styled result card for the classification result."""
    with st.container():
        # Create columns for layout
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Display label with icon
            label = result.get('label', 'Unknown')
            if label == 'Impressed':
                st.markdown(f"<h1 style='color: #1976d2;'>👆 {label}</h1>", unsafe_allow_html=True)
                st.markdown("*Decorations made by pressing objects into clay*")
            elif label == 'Incised':
                st.markdown(f"<h1 style='color: #388e3c;'>✂️ {label}</h1>", unsafe_allow_html=True)
                st.markdown("*Decorations made by cutting/carving into clay*")
            else:
                st.markdown(f"<h1 style='color: #757575;'>❓ {label}</h1>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Processing Time", f"{elapsed_time:.3f}s")
        
        with col3:
            if 'confidence' in result:
                confidence = result['confidence']
                st.metric("Confidence", f"{confidence:.4f}")
        
        # Display confidence details if available
        if 'decision_score' in result:
            st.subheader("Classification Details")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                decision_score = result['decision_score']
                st.metric(
                    "Decision Score", 
                    f"{decision_score:.4f}",
                    help="Positive values indicate Incised, negative values indicate Impressed"
                )
            
            with col_b:
                # Create a simple gauge visualization
                import plotly.graph_objects as go
                
                # Normalize decision score for visualization (-5 to 5 range typical)
                normalized = max(-5, min(5, decision_score))
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = normalized,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Decision Boundary"},
                    gauge = {
                        'axis': {'range': [-5, 5]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-5, -1], 'color': "#bbdefb", 'name': 'Impressed'},
                            {'range': [-1, 1], 'color': "#fff9c4", 'name': 'Uncertain'},
                            {'range': [1, 5], 'color': "#c8e6c9", 'name': 'Incised'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        
        # Display method details
        with st.expander("📋 Method Details"):
            st.write(f"**Method:** {result.get('method', 'Unknown')}")
            
            if 'model_params' in result:
                st.write("**Model Parameters:**")
                params = result['model_params']
                if isinstance(params, dict):
                    params_df = pd.DataFrame([
                        {"Parameter": k, "Value": str(v)}
                        for k, v in params.items()
                    ])
                    st.dataframe(params_df, hide_index=True)
                else:
                    st.write(params)
            
            # Show all result keys
            st.write("**Full Result:**")
            st.json(result)


def visualize_image_properties(image):
    """Visualize image properties like alpha channel."""
    st.subheader("Image Analysis")
    
    img_array = np.array(image)
    
    # Create columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Alpha Channel**")
        alpha = img_array[:, :, 3]
        st.image(alpha, caption="Transparency Mask", width='stretch', clamp=True)
        
        # Stats
        pottery_pixels = np.sum(alpha > 0)
        total_pixels = alpha.size
        st.metric("Pottery Coverage", f"{pottery_pixels/total_pixels*100:.1f}%")
    
    with col2:
        st.markdown("**Original Image**")
        st.image(image, caption="Input Image", width='stretch')
        
        # Image stats
        st.metric("Image Size", f"{image.width} × {image.height}")


def batch_classification_mode():
    """Handle batch classification of multiple images."""
    st.header("📦 Batch Classification Mode")
    
    uploaded_files = st.file_uploader(
        "Choose multiple pottery images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple images with transparent backgrounds"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} images uploaded")
        
        col1, col2 = st.columns(2)
        with col1:
            return_confidence = st.checkbox("Include Confidence Scores", value=True)
        with col2:
            debug_mode = st.checkbox("Enable Debug Output", value=False)
        
        if st.button("🚀 Classify All Images", type="primary"):
            try:
                # Load all images
                with st.spinner("Loading images..."):
                    images = []
                    image_names = []
                    for uploaded_file in uploaded_files:
                        try:
                            img = load_image(uploaded_file)
                            images.append(img)
                            image_names.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {e}")
                
                if images:
                    # Run batch classification
                    with st.spinner(f"Classifying {len(images)} images..."):
                        start_time = time.time()
                        results = batch_classify_pottery_decoration(
                            images=images,
                            return_confidence=return_confidence,
                            debug=debug_mode
                        )
                        elapsed_time = time.time() - start_time
                    
                    st.success(f"✅ Classified {len(results)} images in {elapsed_time:.2f}s")
                    
                    # Create results dataframe
                    results_data = []
                    for name, result in zip(image_names, results):
                        row = {
                            "Image": name,
                            "Classification": result['label'],
                            "Method": result.get('method', 'Unknown')
                        }
                        if 'confidence' in result:
                            row['Confidence'] = f"{result['confidence']:.4f}"
                        if 'decision_score' in result:
                            row['Decision Score'] = f"{result['decision_score']:.4f}"
                        results_data.append(row)
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Display results table
                    st.subheader("📊 Results Summary")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        impressed_count = sum(1 for r in results if r['label'] == 'Impressed')
                        st.metric("Impressed", impressed_count)
                    
                    with col2:
                        incised_count = sum(1 for r in results if r['label'] == 'Incised')
                        st.metric("Incised", incised_count)
                    
                    with col3:
                        avg_time = elapsed_time / len(results)
                        st.metric("Avg Time/Image", f"{avg_time:.3f}s")
                    
                    # Visualization
                    if len(results) > 1:
                        st.subheader("📈 Distribution")
                        
                        import plotly.express as px
                        
                        # Count distribution
                        labels = [r['label'] for r in results]
                        label_counts = pd.DataFrame({
                            'Label': labels
                        }).value_counts().reset_index()
                        label_counts.columns = ['Label', 'Count']
                        
                        fig = px.pie(
                            label_counts,
                            values='Count',
                            names='Label',
                            title='Classification Distribution',
                            color='Label',
                            color_discrete_map={
                                'Impressed': '#1976d2',
                                'Incised': '#388e3c'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.subheader("💾 Export Results")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="pottery_decoration_results.csv",
                        mime="text/csv"
                    )
                    
                    # Show individual results
                    with st.expander("🔍 View Individual Results"):
                        for i, (img, name, result) in enumerate(zip(images, image_names, results)):
                            st.markdown(f"### {i+1}. {name}")
                            col_img, col_result = st.columns([1, 2])
                            
                            with col_img:
                                st.image(img, use_container_width=True)
                            
                            with col_result:
                                label = result['label']
                                if label == 'Impressed':
                                    st.markdown(f"**Classification:** 👆 {label}")
                                else:
                                    st.markdown(f"**Classification:** ✂️ {label}")
                                
                                if 'confidence' in result:
                                    st.write(f"**Confidence:** {result['confidence']:.4f}")
                                if 'decision_score' in result:
                                    st.write(f"**Decision Score:** {result['decision_score']:.4f}")
                            
                            st.divider()
            
            except Exception as e:
                st.error(f"Error during batch classification: {str(e)}")
                st.exception(e)


def single_classification_mode():
    """Handle single image classification."""
    st.header("🎯 Single Image Classification")
    
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
            
            # Sidebar options
            st.sidebar.subheader("⚙️ Classification Options")
            return_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
            debug_mode = st.sidebar.checkbox("Enable Debug Output", value=False)
            show_image_analysis = st.sidebar.checkbox("Show Image Analysis", value=True)
            
            # Display image analysis if enabled
            if show_image_analysis:
                with st.expander("📊 Image Properties", expanded=False):
                    visualize_image_properties(image)
            
            st.divider()
            
            # Run classification
            try:
                with st.spinner("Classifying decoration pattern..."):
                    start_time = time.time()
                    
                    result = classify_pottery_decoration(
                        image=image,
                        return_confidence=return_confidence,
                        debug=debug_mode
                    )
                    
                    elapsed_time = time.time() - start_time
                
                # Display result
                st.header("🎨 Classification Result")
                display_result_card(result, elapsed_time)
                
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")
                if debug_mode:
                    st.exception(e)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
    
    else:
        # Show example/instructions when no file is uploaded
        st.info("👆 Upload a pottery image to begin classification")
        
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. **Prepare your image**: Pottery should have a transparent background (RGBA format)
        2. **Upload**: Use the file uploader above
        3. **Review**: View the classification result with confidence scores
        
        #### Decoration Types:
        - **Impressed** 👆: Decorations made by pressing objects into the clay
        - **Incised** ✂️: Decorations made by cutting/carving into the clay
        
        #### Tips:
        - Enable debug mode to see detailed processing information
        - Check confidence scores to assess prediction reliability
        """)


def main():
    st.set_page_config(
        page_title="Pottery Decoration Classifier",
        page_icon="🏺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏺 Pottery Decoration Classification")
    st.markdown("""
    This app classifies pottery decoration patterns using a **DINOv2 ViT-L/14** model with optimized **logistic regression classifier**.
    
    ### Decoration Types:
    - **Impressed** 👆: Decorations made by pressing objects into the clay
    - **Incised** ✂️: Decorations made by cutting/carving into the clay
    """)
    
    # Sidebar for mode selection
    st.sidebar.header("🎯 Mode Selection")
    mode = st.sidebar.radio(
        "Choose Classification Mode:",
        ["Single Image", "Batch Processing"],
        help="Single: Classify one image with detailed analysis\nBatch: Classify multiple images efficiently"
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info("""
    This tool uses deep learning to classify pottery decoration patterns.
    
    **Model:**
    - DINOv2 ViT-L/14 for feature extraction
    - Optimized logistic regression classifier for classification
    
    **Accuracy:**
    - Trained on impressed and incised pottery samples
    - Provides confidence scores for each prediction
    """)
    
    st.sidebar.divider()
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - Ensure decorations are clearly visible
    - Higher confidence scores indicate more reliable predictions
    """)
    
    # Main content area based on mode
    st.divider()
    
    if mode == "Single Image":
        single_classification_mode()
    else:
        batch_classification_mode()


if __name__ == "__main__":
    main()

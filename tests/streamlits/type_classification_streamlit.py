"""
Streamlit app for testing and showcasing pottery type classification.

This app demonstrates pottery type classification using a multi-stage pipeline:
- Stage 1: body vs everything else (base+rim+appendages)
- Stage 2: base vs rim vs appendage (if not body)
- Stage 3: appendage subtypes using Azure OpenAI GPT-4o (optional)

Uses DINOv2 ViT-L/14 model with optimized SVM classifiers.

Run with: streamlit run type_classification_streamlit.py
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import time
import os
import dotenv
dotenv.load_dotenv()

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from ascota_classification.type import classify_pottery_type, batch_classify_pottery_type


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


def get_type_icon(label):
    """Get icon for pottery type."""
    icons = {
        'body': '🫙',
        'base': '⬇️',
        'rim': '⬆️',
        'appendage': '🔗',
        'lid': '🎩',
        'rim-handle': '👂',
        'spout': '💧',
        'rounded': '⚪',
        'body-decorated': '🎨',
        'tile': '🧱'
    }
    return icons.get(label, '❓')


def get_type_color(label):
    """Get color for pottery type."""
    colors = {
        'body': '#1976d2',
        'base': '#d32f2f',
        'rim': '#388e3c',
        'appendage': '#f57c00',
        'lid': '#7b1fa2',
        'rim-handle': '#c2185b',
        'spout': '#0097a7',
        'rounded': '#5d4037',
        'body-decorated': '#fbc02d',
        'tile': '#455a64'
    }
    return colors.get(label, '#757575')


def get_type_description(label):
    """Get description for pottery type."""
    descriptions = {
        'body': 'Main vessel body',
        'base': 'Bottom/foundation of vessel',
        'rim': 'Top edge of vessel',
        'appendage': 'Additional attached element',
        'lid': 'Cover or cap',
        'rim-handle': 'Handle attached to rim',
        'spout': 'Pouring element',
        'rounded': 'Rounded appendage',
        'body-decorated': 'Decorated body element',
        'tile': 'Flat tile piece'
    }
    return descriptions.get(label, 'Unknown type')


def display_stage_info(stage_data, stage_name):
    """Display information for a classification stage."""
    st.markdown(f"**{stage_name}**")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        label = stage_data['label']
        icon = get_type_icon(label)
        st.markdown(f"{icon} `{label}`")
    
    with col2:
        confidence = stage_data['confidence']
        st.metric("Confidence", f"{confidence:.4f}")


def display_result_card(result, elapsed_time):
    """Display a styled result card for the classification result."""
    with st.container():
        # Final classification
        label = result.get('label', 'Unknown')
        icon = get_type_icon(label)
        color = get_type_color(label)
        description = get_type_description(label)
        
        st.markdown(f"<h1 style='color: {color};'>{icon} {label.upper()}</h1>", unsafe_allow_html=True)
        st.markdown(f"*{description}*")
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processing Time", f"{elapsed_time:.3f}s")
        with col2:
            # Show final stage confidence
            if 'stage3' in result:
                final_conf = result['stage3']['confidence']
            elif 'stage2' in result:
                final_conf = result['stage2']['confidence']
            else:
                final_conf = result['stage1']['confidence']
            st.metric("Final Confidence", f"{final_conf:.4f}")
        
        st.divider()
        
        # Stage-by-stage results
        st.subheader("📊 Classification Pipeline")
        
        # Stage 1
        with st.expander("🎯 Stage 1: Body vs Everything Else", expanded=True):
            display_stage_info(result['stage1'], "Stage 1 Classification")
            
            # Visual indicator
            stage1_label = result['stage1']['label']
            if stage1_label == 'body':
                st.success("✅ Classified as **body** - Pipeline stopped at Stage 1")
            else:
                st.info("➡️ Classified as **everything_else** - Proceeding to Stage 2")
        
        # Stage 2 (if exists)
        if 'stage2' in result:
            with st.expander("🎯 Stage 2: Base vs Rim vs Appendage", expanded=True):
                display_stage_info(result['stage2'], "Stage 2 Classification")
                
                stage2_label = result['stage2']['label']
                if stage2_label == 'appendage':
                    st.info("➡️ Classified as **appendage** - Can proceed to Stage 3 (if Azure enabled)")
                else:
                    st.success(f"✅ Classified as **{stage2_label}** - Pipeline complete")
        
        # Stage 3 (if exists)
        if 'stage3' in result:
            with st.expander("🎯 Stage 3: Appendage Subtype (Azure OpenAI)", expanded=True):
                display_stage_info(result['stage3'], "Stage 3 Classification")
                st.success(f"✅ Appendage subtype identified: **{result['stage3']['label']}**")
        
        # Full result JSON
        with st.expander("📋 Full Result Details"):
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


def check_azure_credentials():
    """Check if Azure OpenAI credentials are available."""
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    return endpoint is not None and api_key is not None


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
            use_azure = st.checkbox(
                "Use Azure OpenAI for appendage subtypes",
                value=False,
                help="Enable Stage 3 classification for appendage subtypes"
            )
            if use_azure and not check_azure_credentials():
                st.warning("⚠️ Azure OpenAI credentials not found in environment variables")
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
                        results = batch_classify_pottery_type(
                            images=images,
                            use_azure_openai=use_azure,
                            debug=debug_mode
                        )
                        elapsed_time = time.time() - start_time
                    
                    st.success(f"✅ Classified {len(results)} images in {elapsed_time:.2f}s")
                    
                    # Create results dataframe
                    results_data = []
                    for name, result in zip(image_names, results):
                        row = {
                            "Image": name,
                            "Final Type": result['label'],
                            "Stage 1": result['stage1']['label'],
                            "Stage 1 Conf": f"{result['stage1']['confidence']:.4f}"
                        }
                        if 'stage2' in result:
                            row['Stage 2'] = result['stage2']['label']
                            row['Stage 2 Conf'] = f"{result['stage2']['confidence']:.4f}"
                        if 'stage3' in result:
                            row['Stage 3'] = result['stage3']['label']
                            row['Stage 3 Conf'] = f"{result['stage3']['confidence']:.4f}"
                        results_data.append(row)
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Display results table
                    st.subheader("📊 Results Summary")
                    st.dataframe(results_df, width='stretch', hide_index=True)
                    
                    # Statistics
                    st.subheader("📈 Statistics")
                    
                    # Count by type
                    type_counts = {}
                    for r in results:
                        label = r['label']
                        type_counts[label] = type_counts.get(label, 0) + 1
                    
                    # Display type counts in columns
                    cols = st.columns(min(len(type_counts), 4))
                    for i, (type_label, count) in enumerate(sorted(type_counts.items())):
                        with cols[i % len(cols)]:
                            icon = get_type_icon(type_label)
                            st.metric(f"{icon} {type_label}", count)
                    
                    # Average processing time
                    st.metric("Avg Time/Image", f"{elapsed_time / len(results):.3f}s")
                    
                    # Visualization
                    if len(results) > 1:
                        st.subheader("📊 Distribution")
                        
                        import plotly.express as px
                        
                        # Count distribution
                        labels = [r['label'] for r in results]
                        label_counts_df = pd.DataFrame({
                            'Type': labels
                        }).value_counts().reset_index()
                        label_counts_df.columns = ['Type', 'Count']
                        
                        # Create color map
                        color_map = {label: get_type_color(label) for label in label_counts_df['Type'].unique()}
                        
                        fig = px.pie(
                            label_counts_df,
                            values='Count',
                            names='Type',
                            title='Type Classification Distribution',
                            color='Type',
                            color_discrete_map=color_map
                        )
                        st.plotly_chart(fig, width='stretch')
                    
                    # Download results
                    st.subheader("💾 Export Results")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="pottery_type_results.csv",
                        mime="text/csv"
                    )
                    
                    # Show individual results
                    with st.expander("🔍 View Individual Results"):
                        for i, (img, name, result) in enumerate(zip(images, image_names, results)):
                            st.markdown(f"### {i+1}. {name}")
                            col_img, col_result = st.columns([1, 2])
                            
                            with col_img:
                                st.image(img, width='stretch')
                            
                            with col_result:
                                label = result['label']
                                icon = get_type_icon(label)
                                st.markdown(f"**Final Classification:** {icon} {label}")
                                
                                st.write(f"**Stage 1:** {result['stage1']['label']} (conf: {result['stage1']['confidence']:.4f})")
                                if 'stage2' in result:
                                    st.write(f"**Stage 2:** {result['stage2']['label']} (conf: {result['stage2']['confidence']:.4f})")
                                if 'stage3' in result:
                                    st.write(f"**Stage 3:** {result['stage3']['label']} (conf: {result['stage3']['confidence']:.4f})")
                            
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
            
            use_azure = st.sidebar.checkbox(
                "Use Azure OpenAI (Stage 3)",
                value=False,
                help="Enable Azure OpenAI GPT-4o for appendage subtype classification"
            )
            
            if use_azure:
                if check_azure_credentials():
                    st.sidebar.success("✅ Azure OpenAI credentials found")
                else:
                    st.sidebar.error("❌ Azure OpenAI credentials not found")
                    st.sidebar.info("""
                    Set environment variables:
                    - AZURE_OPENAI_ENDPOINT
                    - AZURE_OPENAI_API_KEY
                    """)
            
            debug_mode = st.sidebar.checkbox("Enable Debug Output", value=False)
            show_image_analysis = st.sidebar.checkbox("Show Image Analysis", value=True)
            
            # Display image analysis if enabled
            if show_image_analysis:
                with st.expander("📊 Image Properties", expanded=False):
                    visualize_image_properties(image)
            
            st.divider()
            
            # Run classification
            try:
                with st.spinner("Classifying pottery type..."):
                    start_time = time.time()
                    
                    result = classify_pottery_type(
                        image=image,
                        use_azure_openai=use_azure,
                        debug=debug_mode
                    )
                    
                    elapsed_time = time.time() - start_time
                
                # Display result
                st.header("🏺 Classification Result")
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
        3. **Configure**: Enable Azure OpenAI if you want appendage subtype classification
        4. **Review**: View the multi-stage classification result with confidence scores
        
        #### Classification Pipeline:
        - **Stage 1**: body vs everything else
        - **Stage 2**: base vs rim vs appendage (if not body)
        - **Stage 3**: appendage subtypes using Azure OpenAI GPT-4o (optional, if appendage)
        
        #### Pottery Types:
        - **Body** 🫙: Main vessel body
        - **Base** ⬇️: Bottom/foundation of vessel
        - **Rim** ⬆️: Top edge of vessel
        - **Appendage** 🔗: Additional attached element
        
        #### Appendage Subtypes (Stage 3):
        - **Lid** 🎩: Cover or cap
        - **Rim-handle** 👂: Handle attached to rim
        - **Spout** 💧: Pouring element
        - **Rounded** ⚪: Rounded appendage
        - **Body-decorated** 🎨: Decorated body element
        - **Tile** 🧱: Flat tile piece
        
        #### Tips:
        - Enable debug mode to see detailed processing information
        - Check confidence scores at each stage to assess prediction reliability
        - Azure OpenAI Stage 3 requires environment variables to be set
        """)


def main():
    st.set_page_config(
        page_title="Pottery Type Classifier",
        page_icon="🏺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏺 Pottery Type Classification")
    st.markdown("""
    This app classifies pottery types using a **multi-stage DINOv2 ViT-L/14** pipeline with optimized **SVM classifiers**.
    
    ### Classification Pipeline:
    1. **Stage 1**: Body vs Everything Else
    2. **Stage 2**: Base vs Rim vs Appendage (if not body)
    3. **Stage 3**: Appendage Subtypes with Azure OpenAI GPT-4o (optional)
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
    This tool uses deep learning to classify pottery types through a multi-stage pipeline.
    
    **Model:**
    - DINOv2 ViT-L/14 for feature extraction
    - Optimized SVM classifiers for Stage 1 & 2
    - Azure OpenAI GPT-4o for Stage 3 (optional)
    
    **Pipeline:**
    - Stage 1: Binary classification (body vs everything else)
    - Stage 2: 3-class classification (base vs rim vs appendage)
    - Stage 3: Appendage subtype classification
    
    **Features:**
    - Confidence scores at each stage
    - Conditional processing (early stopping if body)
    - Optional Azure OpenAI integration
    """)
    
    st.sidebar.divider()
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - Ensure pottery is clearly visible
    - Use transparent backgrounds for best results
    - Higher confidence scores indicate more reliable predictions
    - Stage 3 requires Azure OpenAI credentials
    """)
    
    # Main content area based on mode
    st.divider()
    
    if mode == "Single Image":
        single_classification_mode()
    else:
        batch_classification_mode()


if __name__ == "__main__":
    main()


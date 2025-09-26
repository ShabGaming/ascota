import streamlit as st
import numpy as np
from PIL import Image
import sys
import os
import cv2
from pathlib import Path
import glob

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ascota_core import (
    process_image_pipeline, 
    calculate_pp_cm_checker_cm,
    calculate_pp_cm_colorchecker8,
    artifact_face_size,
    setup_rmbg_pipeline,
    remove_background_and_generate_mask
)
from src.ascota_core.utils import pil_to_cv2

st.set_page_config(
    page_title="ASCOTA Scale Analysis",
    page_icon="📏",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# Sidebar navigation
st.sidebar.title("📏 ASCOTA Scale Analysis")
st.sidebar.markdown("Scale detection and artifact measurement")


# Main Analysis Page
if st.session_state.current_page == 'main':
    st.title("📏 ASCOTA Scale Analysis")
    st.markdown("Scale detection and artifact face size measurement")

    # Sidebar for controls
    st.sidebar.header("Configuration")

    # Use all available templates
    template_paths = [
        "src/ascota_core/templates/colorchecker24.png",
        "src/ascota_core/templates/checker_cm.png", 
        "src/ascota_core/templates/colorchecker8.png"
    ]

    # Get all test images
    def get_test_images():
        type_a_images = glob.glob("tests/data/type_a/jpg/*.jpg")
        type_b_images = glob.glob("tests/data/type_b/jpg/*.jpg")
        
        sample_images = {}
        for img_path in sorted(type_a_images):
            filename = os.path.basename(img_path)
            sample_images[f"Type A - {filename}"] = img_path
        
        for img_path in sorted(type_b_images):
            filename = os.path.basename(img_path)
            sample_images[f"Type B - {filename}"] = img_path
        
        return sample_images

    sample_images = get_test_images()

    # Image selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Select Test Image**")
    selected_sample = st.sidebar.selectbox("Choose an image", ["None"] + list(sample_images.keys()))

    # Scale calculation options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Scale Calculation Options**")
    enable_debug = st.sidebar.checkbox("Enable Debug Visualization", value=False)
    show_debug_images = st.sidebar.checkbox("Show Debug Images", value=True)

    # Process button
    process_btn = st.sidebar.button("🚀 Process Scale Analysis", type="primary")

    # Reset results when new image is selected
    if selected_sample != "None":
        if 'current_image' not in st.session_state or st.session_state.current_image != selected_sample:
            st.session_state.current_image = selected_sample
            st.session_state.scale_results = None
            st.session_state.scale_processed = False

    # Main content area
    if process_btn and selected_sample != "None":
        try:
            # Get image path
            image_path = sample_images[selected_sample]
            image = Image.open(image_path)
            
            # Process the image
            with st.spinner("Processing scale analysis..."):
                try:
                    # Check if template files exist
                    missing_templates = []
                    for template_path in template_paths:
                        if not os.path.exists(template_path):
                            missing_templates.append(template_path)
                    
                    if missing_templates:
                        st.warning(f"⚠️ Missing template files: {missing_templates}")
                        template_paths = [tp for tp in template_paths if os.path.exists(tp)]
                    
                    if not template_paths:
                        st.error("❌ No valid template files found!")
                        st.stop()
                    
                    # First, detect cards using imaging pipeline
                    results = process_image_pipeline(image_path, template_paths)
                    
                    # Process scale analysis for each detected card
                    scale_results = []
                    for i, (card_type, crop) in enumerate(zip(results['card_types'], results['extracted_crops'])):
                        if crop is not None:
                            # Convert crop to numpy array if needed
                            if hasattr(crop, 'shape'):  # Already numpy array
                                card_img = crop
                            else:  # PIL Image
                                card_img = pil_to_cv2(crop)
                            
                            scale_result = {
                                'card_type': card_type,
                                'card_index': i,
                                'pixels_per_cm': 0.0,
                                'debug_image': None,
                                'error': None
                            }
                            
                            try:
                                if card_type == 'checker_cm':
                                    pp_cm, debug_img = calculate_pp_cm_checker_cm(card_img, debug=enable_debug)
                                    scale_result['pixels_per_cm'] = pp_cm
                                    scale_result['debug_image'] = debug_img
                                    scale_result['method'] = 'Checker CM'
                                    
                                elif card_type == 'colorchecker8':
                                    pp_cm, debug_img = calculate_pp_cm_colorchecker8(card_img, debug=enable_debug)
                                    scale_result['pixels_per_cm'] = pp_cm
                                    scale_result['debug_image'] = debug_img
                                    scale_result['method'] = 'ColorChecker 8'
                                    
                                elif card_type == 'colorchecker24':
                                    # ColorChecker 24 doesn't have scale reference, skip
                                    scale_result['error'] = "ColorChecker 24 has no scale reference"
                                    scale_result['method'] = 'N/A'
                                    
                                else:
                                    scale_result['error'] = f"Unknown card type: {card_type}"
                                    scale_result['method'] = 'Unknown'
                                    
                            except Exception as e:
                                scale_result['error'] = str(e)
                                scale_result['method'] = 'Error'
                            
                            scale_results.append(scale_result)
                    
                    st.session_state.scale_results = scale_results
                    st.session_state.scale_processed = True
                    
                except Exception as e:
                    st.error(f"❌ Error in processing: {str(e)}")
                    st.exception(e)
                    st.stop()
            
            # Display results in a clean layout
            st.markdown("---")
            
            # Input Preview
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("📸 Input Image")
                st.image(image, width='stretch')
                st.caption(f"**File:** {os.path.basename(image_path)}")
            
            with col2:
                st.subheader("🔍 Detected Cards")
                if results['card_types']:
                    for i, card_type in enumerate(results['card_types']):
                        if card_type == 'colorchecker24':
                            st.success(f"Card {i+1}: {card_type} (24-color reference)")
                        elif card_type == 'colorchecker8':
                            st.info(f"Card {i+1}: {card_type} (8-color reference)")
                        elif card_type == 'checker_cm':
                            st.warning(f"Card {i+1}: {card_type} (scale reference)")
                        else:
                            st.error(f"Card {i+1}: {card_type} (unknown)")
                else:
                    st.warning("No color cards detected")
            
            # Card Cutouts and Scale Analysis
            if results['extracted_crops']:
                st.markdown("---")
                st.subheader("✂️ Card Cutouts & Scale Analysis")
                
                for i, (crop, card_type) in enumerate(zip(results['extracted_crops'], results['card_types'])):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        st.image(crop_pil, caption=f"Card {i+1}: {card_type}", width=300)
                    
                    with col2:
                        if i < len(st.session_state.scale_results):
                            scale_result = st.session_state.scale_results[i]
                            
                            st.markdown(f"**Scale Analysis - {scale_result['method']}**")
                            
                            if scale_result['error']:
                                st.error(f"❌ Error: {scale_result['error']}")
                            else:
                                st.success(f"✅ Pixels per cm: **{scale_result['pixels_per_cm']:.0f}**")
                                
                                # Show debug image if available and enabled
                                if scale_result['debug_image'] and show_debug_images:
                                    st.image(scale_result['debug_image'], caption="Debug Visualization", width=300)
                            
                            # Show method details
                            if scale_result['method'] == 'Checker CM':
                                st.info("Method: Analyzes white squares in checkerboard pattern")
                            elif scale_result['method'] == 'ColorChecker 8':
                                st.info("Method: Detects three circular reference points (50mm x 20mm)")
                            elif scale_result['method'] == 'N/A':
                                st.info("ColorChecker 24 has no built-in scale reference")
            
            # Artifact Face Size Analysis (if we have scale data)
            st.markdown("---")
            st.subheader("🏺 Artifact Face Size Analysis")
            
            # Find the best scale reference
            best_scale = None
            best_pp_cm = 0
            best_card_img = None
            
            if st.session_state.scale_results:
                for scale_result in st.session_state.scale_results:
                    if scale_result['pixels_per_cm'] > best_pp_cm and not scale_result['error']:
                        best_pp_cm = scale_result['pixels_per_cm']
                        best_scale = scale_result
                        # Get the corresponding card image
                        card_idx = scale_result['card_index']
                        if card_idx < len(results['extracted_crops']):
                            best_card_img = results['extracted_crops'][card_idx]
            
            if best_scale and best_card_img is not None:
                st.success(f"✅ Using scale from {best_scale['method']}: {best_pp_cm:.0f} pixels/cm")
                
                # Perform artifact segmentation using RMBG
                with st.spinner("Segmenting artifact for size calculation..."):
                    try:
                        # Setup RMBG pipeline
                        rmbg_pipe = setup_rmbg_pipeline()
                        
                        # Create cropped image with color cards removed for better RMBG processing
                        from src.ascota_core.imaging import crop_out_color_cards_for_rmbg
                        cropped_for_rmbg = crop_out_color_cards_for_rmbg(
                            pil_to_cv2(results['original_image']), 
                            results['detected_polygons'], 
                            results['card_types']
                        )
                        
                        # Create cards mask for overlap filtering
                        cards_mask = np.zeros(results['masked_image'].size[::-1], dtype=np.uint8)
                        for poly in results['detected_polygons']:
                            from src.ascota_core.utils import polygon_to_mask
                            poly_mask = polygon_to_mask(results['masked_image'].size[::-1], poly)
                            cards_mask = cv2.bitwise_or(cards_mask, poly_mask)
                        
                        # Apply RMBG to get segmented artifact
                        artifact_image, artifact_mask = remove_background_and_generate_mask(
                            results['masked_image'], 
                            rmbg_pipe,
                            cards_mask,
                            cropped_for_rmbg
                        )
                        
                        # Calculate face area using the segmented artifact
                        face_area = artifact_face_size(
                            artifact_image,  # Segmented artifact with alpha channel
                            best_card_img,   # Reference card image
                            best_scale['card_type'],  # Card type
                            debug=enable_debug
                        )
                        
                        # Display results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Segmented Artifact**")
                            st.image(artifact_image, caption="Segmented Artifact (RGBA)", width=300)
                            
                            if enable_debug and artifact_mask:
                                st.markdown("**Artifact Mask**")
                                st.image(artifact_mask, caption="Binary Mask", width=300)
                        
                        with col2:
                            st.markdown("**Face Size Calculation**")
                            st.success(f"🎯 **Face Area: {face_area:.2f} cm²**")
                            
                            # Show calculation details
                            st.info(f"""
                            **Calculation Details:**
                            - Scale: {best_pp_cm:.0f} pixels/cm
                            - Method: {best_scale['method']}
                            - Card Type: {best_scale['card_type']}
                            """)
                            
                            # Show reference card used
                            st.markdown("**Reference Card Used**")
                            ref_card_pil = Image.fromarray(cv2.cvtColor(best_card_img, cv2.COLOR_BGR2RGB))
                            st.image(ref_card_pil, caption=f"Reference: {best_scale['card_type']}", width=200)
                            
                            # Download option for segmented artifact
                            st.download_button(
                                label="📥 Download Segmented Artifact",
                                data=artifact_image.tobytes(),
                                file_name=f"artifact_{os.path.basename(image_path)}",
                                mime="image/png"
                            )
                        
                        # Additional analysis
                        st.markdown("---")
                        st.subheader("📊 Analysis Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Face Area", f"{face_area:.2f} cm²")
                        with col2:
                            st.metric("Scale Reference", f"{best_pp_cm:.0f} px/cm")
                        with col3:
                            st.metric("Method", best_scale['method'])
                        
                    except Exception as e:
                        st.error(f"❌ Error in artifact segmentation: {str(e)}")
                        st.info("Make sure transformers and required dependencies are installed for RMBG")
                        
                        # Fallback: show placeholder
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("**Artifact Image**")
                            st.image(image, caption="Original Image (segmentation failed)", width=300)
                        with col2:
                            st.markdown("**Face Size Calculation**")
                            st.warning("Artifact segmentation failed. Install RMBG dependencies to enable size calculation.")
            else:
                st.warning("⚠️ No valid scale reference found. Need either 'checker_cm' or 'colorchecker8' card.")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)

    else:
        # Welcome message when no image is selected
        st.info("💡 **Welcome!** Select an image from the sidebar and click 'Process Scale Analysis' to run the scale detection pipeline.")

# Footer
st.markdown("---")
st.markdown("**ASCOTA** - Archaeological Sherd Color, Texture & Analysis | HKU Applied AI | M. Shahab Hasan")

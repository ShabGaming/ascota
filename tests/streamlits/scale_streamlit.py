import streamlit as st
import numpy as np
from PIL import Image
import sys
import os
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ascota_core import (
    detect_color_cards,
    remove_background_mask,
    calculate_pp_cm_checker_card,
    find_circle_centers_8_hybrid_card,
    calculate_pp_cm_from_centers,
    artifact_face_size
)
from src.ascota_core.utils import pil_to_cv2, cv2_to_pil

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

    # Image upload
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Upload Image**")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        help="Upload an image containing color reference cards and an artifact"
    )

    # Scale calculation options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Scale Calculation Options**")
    enable_debug = st.sidebar.checkbox("Enable Debug Visualization", value=False)
    show_debug_images = st.sidebar.checkbox("Show Debug Images", value=True)

    # Process button
    process_btn = st.sidebar.button("🚀 Process Scale Analysis", type="primary")

    # Reset results when new image is uploaded
    if uploaded_file is not None:
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.scale_results = None
            st.session_state.scale_processed = False
            st.session_state.detected_cards = None
            st.session_state.extracted_crops = None
            st.session_state.card_types = None

    # Main content area
    if process_btn and uploaded_file is not None:
        try:
            # Load image from uploaded file
            image = Image.open(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer for potential reuse
            
            # Process the image
            with st.spinner("Processing scale analysis..."):
                try:
                    # Detect color cards using new imaging functions
                    detected_cards = detect_color_cards(image, debug=enable_debug)
                    
                    # Extract card crops and process scale analysis
                    scale_results = []
                    extracted_crops = []
                    card_types = []
                    
                    # Convert image to numpy array for cropping
                    img_array = np.array(image)
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = img_array
                    
                    for i, card in enumerate(detected_cards):
                        card_type = card['class']
                        coordinates = np.array(card['coordinates'], dtype=np.float32)
                        
                        # Extract card crop using perspective transform
                        # Get bounding box first
                        xs = coordinates[:, 0]
                        ys = coordinates[:, 1]
                        x_min, x_max = int(xs.min()), int(xs.max())
                        y_min, y_max = int(ys.min()), int(ys.max())
                        
                        # Ensure coordinates are within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img_bgr.shape[1], x_max)
                        y_max = min(img_bgr.shape[0], y_max)
                        
                        # Extract crop using bounding box (simple approach)
                        # For better results, could use perspective transform, but bounding box is simpler
                        if x_max > x_min and y_max > y_min:
                            crop = img_bgr[y_min:y_max, x_min:x_max]
                            extracted_crops.append(crop)
                            card_types.append(card_type)
                            
                            scale_result = {
                                'card_type': card_type,
                                'card_index': i,
                                'pixels_per_cm': 0.0,
                                'debug_image': None,
                                'error': None
                            }
                            
                            try:
                                if card_type == 'checker_card':
                                    pp_cm, debug_img = calculate_pp_cm_checker_card(crop, debug=enable_debug)
                                    scale_result['pixels_per_cm'] = pp_cm
                                    scale_result['debug_image'] = debug_img
                                    scale_result['method'] = 'Checker CM'
                                    
                                elif card_type == '8_hybrid_card':
                                    # Use new two-function approach for better error handling
                                    centers, centers_debug_img = find_circle_centers_8_hybrid_card(crop, debug=enable_debug)
                                    if centers is None:
                                        scale_result['error'] = "Could not detect circle centers. Try manual selection."
                                        scale_result['debug_image'] = centers_debug_img
                                        scale_result['method'] = 'ColorChecker 8 (Failed)'
                                    else:
                                        pp_cm, calc_debug_img = calculate_pp_cm_from_centers(centers, crop, debug=enable_debug)
                                        scale_result['pixels_per_cm'] = pp_cm
                                        # Use calculation debug image if available, otherwise use detection debug image
                                        scale_result['debug_image'] = calc_debug_img if calc_debug_img is not None else centers_debug_img
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
                    
                    # Store results in session state
                    st.session_state.detected_cards = detected_cards
                    st.session_state.extracted_crops = extracted_crops
                    st.session_state.card_types = card_types
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
                st.caption(f"**File:** {uploaded_file.name}")
            
            with col2:
                st.subheader("🔍 Detected Cards")
                if st.session_state.get('card_types'):
                    for i, card_type in enumerate(st.session_state.card_types):
                        if card_type == 'colorchecker24':
                            st.success(f"Card {i+1}: {card_type} (24-color reference)")
                        elif card_type == '8_hybrid_card':
                            st.info(f"Card {i+1}: {card_type} (8-color reference)")
                        elif card_type == 'checker_card':
                            st.warning(f"Card {i+1}: {card_type} (scale reference)")
                        else:
                            st.error(f"Card {i+1}: {card_type} (unknown)")
                else:
                    st.warning("No color cards detected")
            
            # Card Cutouts and Scale Analysis
            if st.session_state.get('extracted_crops'):
                st.markdown("---")
                st.subheader("✂️ Card Cutouts & Scale Analysis")
                
                for i, (crop, card_type) in enumerate(zip(st.session_state.extracted_crops, st.session_state.card_types)):
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
            
            if st.session_state.get('scale_results'):
                for scale_result in st.session_state.scale_results:
                    if scale_result['pixels_per_cm'] > best_pp_cm and not scale_result.get('error'):
                        best_pp_cm = scale_result['pixels_per_cm']
                        best_scale = scale_result
                        # Get the corresponding card image
                        card_idx = scale_result['card_index']
                        if card_idx < len(st.session_state.get('extracted_crops', [])):
                            best_card_img = st.session_state.extracted_crops[card_idx]
            
            if best_scale and best_card_img is not None:
                st.success(f"✅ Using scale from {best_scale['method']}: {best_pp_cm:.0f} pixels/cm")
                
                # Perform artifact segmentation using new RMBG function
                with st.spinner("Segmenting artifact for size calculation..."):
                    try:
                        # Use remove_background_mask with detected card coordinates
                        detected_cards = st.session_state.get('detected_cards', [])
                        
                        # Get binary mask from remove_background_mask
                        mask_result = remove_background_mask(
                            image,
                            card_coordinates=detected_cards if detected_cards else None,
                            debug=enable_debug
                        )
                        
                        # Handle debug mode return (tuple) vs normal mode (just mask)
                        if enable_debug and isinstance(mask_result, tuple):
                            binary_mask, rmbg_image, rmbg_white_bg_image = mask_result
                        else:
                            binary_mask = mask_result
                            rmbg_image = None
                            rmbg_white_bg_image = None
                        
                        # Convert binary mask (0/1) to 0/255 for image creation
                        mask_255 = (binary_mask * 255).astype(np.uint8)
                        
                        # Create transparent artifact image from mask
                        # Convert mask to RGBA where mask=1 (foreground) is opaque
                        artifact_image = image.convert('RGBA')
                        mask_pil = Image.fromarray(mask_255, mode='L')
                        artifact_image.putalpha(mask_pil)
                        
                        # Calculate face area using the binary mask and pixels per cm
                        face_area = artifact_face_size(
                            binary_mask,  # Binary mask (0/1 format)
                            best_pp_cm,   # Pixels per cm from scale calculation
                            debug=enable_debug
                        )
                        
                        # Display results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Segmented Artifact**")
                            st.image(artifact_image, caption="Segmented Artifact (RGBA)", width=300)
                            
                            if enable_debug:
                                st.markdown("**Binary Mask**")
                                mask_display = Image.fromarray(mask_255, mode='L')
                                st.image(mask_display, caption="Binary Mask (1=foreground, 0=background)", width=300)
                                
                                if rmbg_image is not None:
                                    st.markdown("**RMBG Result**")
                                    st.image(rmbg_image, caption="First RMBG Pass", width=300)
                                
                                if rmbg_white_bg_image is not None:
                                    st.markdown("**White BG Image**")
                                    st.image(rmbg_white_bg_image, caption="Input to Second RMBG Pass", width=300)
                        
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
                            from io import BytesIO
                            buf = BytesIO()
                            artifact_image.save(buf, format='PNG')
                            # Get base filename without extension for download
                            base_name = Path(uploaded_file.name).stem
                            st.download_button(
                                label="📥 Download Segmented Artifact",
                                data=buf.getvalue(),
                                file_name=f"artifact_{base_name}.png",
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
                        st.exception(e)
                        
                        # Fallback: show placeholder
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("**Artifact Image**")
                            st.image(image, caption="Original Image (segmentation failed)", width=300)
                        with col2:
                            st.markdown("**Face Size Calculation**")
                            st.warning("Artifact segmentation failed. Install RMBG dependencies to enable size calculation.")
            else:
                st.warning("⚠️ No valid scale reference found. Need either 'checker_card' or '8_hybrid_card' card.")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)

    else:
        # Welcome message when no image is uploaded
        st.info("💡 **Welcome!** Upload an image from the sidebar and click 'Process Scale Analysis' to run the scale detection pipeline.")

# Footer
st.markdown("---")
st.markdown("**ASCOTA** - Archaeological Sherd Color, Texture & Analysis | HKU Applied AI | M. Shahab Hasan")

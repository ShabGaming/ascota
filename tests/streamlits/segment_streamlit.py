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
    setup_rmbg_pipeline, 
    remove_background_and_generate_mask,
    detect_card_type
)
from src.ascota_core.utils import pil_to_cv2

st.set_page_config(
    page_title="ASCOTA Sherd Analysis",
    page_icon="🏺",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# Sidebar navigation
st.sidebar.title("🏺 ASCOTA")
st.sidebar.markdown("Archaeological Sherd Analysis")

# Main Analysis Page
if st.session_state.current_page == 'main':
    st.title("🏺 ASCOTA Archaeological Sherd Analysis")
    st.markdown("Color card detection and sherd segmentation pipeline")

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

    # Process button
    process_btn = st.sidebar.button("🚀 Process Image", type="primary")

    # Reset RMBG results when new image is selected
    if selected_sample != "None":
        if 'current_image' not in st.session_state or st.session_state.current_image != selected_sample:
            st.session_state.current_image = selected_sample
            st.session_state.rmbg_result = None
            st.session_state.rmbg_mask = None
            st.session_state.rmbg_processed = False

    # Main content area
    if process_btn and selected_sample != "None":
        try:
            # Get image path
            image_path = sample_images[selected_sample]
            image = Image.open(image_path)
            
            # Process the image
            with st.spinner("Processing sherd image..."):
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

                    results = process_image_pipeline(image_path, debug=True)
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
                st.subheader("🔍 Detected Card Types")
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
            # Card Cutouts
            if results['extracted_crops']:
                st.markdown("---")
                st.subheader("✂️ Card Cutouts")
                cols = st.columns(len(results['extracted_crops']))
                for i, crop in enumerate(results['extracted_crops']):
                    with cols[i]:
                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        card_type = results['card_types'][i] if i < len(results['card_types']) else 'unknown'
                        st.image(crop_pil, caption=f"Card {i+1}: {card_type}", width=200)
                
        
            # RMBG Sherd Segmentation
            st.markdown("---")
            st.subheader("🏺 Sherd Segmentation (RMBG-1.4)")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Masked Image (Cards Removed)**")
                st.image(results['masked_image'], width='stretch')
            
            with col2:
                st.markdown("**RMBG Background Removal**")
                
                # Auto-apply RMBG processing
                with st.spinner("Applying RMBG background removal..."):
                    try:
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
                        
                        result_image, mask_image = remove_background_and_generate_mask(
                            results['masked_image'], 
                            rmbg_pipe,
                            cards_mask,
                            cropped_for_rmbg,
                            debug=True
                        )
                        
                        # Display results
                        st.image(result_image, caption="Segmented Sherd", width='stretch')
                        
                        # Download option
                        st.download_button(
                            label="📥 Download Segmented Sherd",
                            data=result_image.tobytes(),
                            file_name=f"segmented_{os.path.basename(image_path)}",
                            mime="image/png"
                        )
                        
                        st.success("RMBG processing completed!")
                        
                    except Exception as e:
                        st.error(f"RMBG processing failed: {str(e)}")
                        st.info("Make sure transformers and required dependencies are installed")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)

    else:
        # Welcome message when no image is selected
        st.info("💡 **Welcome!** Select an image from the sidebar and click 'Process Image' to run the analysis pipeline.")

# Footer
st.markdown("---")
st.markdown("**ASCOTA** - Archaeological Sherd Color, Texture & Analysis | HKU Applied AI | M. Shahab Hasan")

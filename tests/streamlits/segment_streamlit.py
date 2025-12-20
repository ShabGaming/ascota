"""
Streamlit app for color card detection, background removal, and swatch generation.

This app allows users to upload images and run both color card detection
and background removal functions from the imaging module.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ascota_core.imaging import detect_color_cards, remove_background_mask, generate_swatch


def draw_detections_on_image(image: Image.Image, detections: list) -> Image.Image:
    """Draw detected color cards on the image with bounding boxes and labels.
    
    Args:
        image: PIL Image to draw on.
        detections: List of detection dictionaries from detect_color_cards().
    
    Returns:
        PIL Image with detection overlays drawn.
    """
    img_array = np.array(image.convert('RGB'))
    img_with_overlay = img_array.copy()
    
    # Color mapping for different card types
    color_map = {
        '24_color_card': (0, 255, 0),      # Green
        '8_hybrid_card': (255, 0, 0),      # Red
        'checker_card': (0, 0, 255),       # Blue
    }
    
    for i, det in enumerate(detections):
        coords = np.array(det['coordinates'], dtype=np.int32)
        class_name = det['class']
        confidence = det['confidence']
        color = color_map.get(class_name, (255, 255, 0))  # Yellow for unknown
        
        # Draw polygon outline
        cv2.polylines(
            img_with_overlay,
            [coords],
            isClosed=True,
            color=color,
            thickness=3
        )
        
        # Draw filled polygon with transparency
        overlay = img_with_overlay.copy()
        cv2.fillPoly(overlay, [coords], color)
        cv2.addWeighted(overlay, 0.3, img_with_overlay, 0.7, 0, img_with_overlay)
        
        # Draw label
        x_min = int(np.min(coords[:, 0]))
        y_min = int(np.min(coords[:, 1]))
        label = f"{class_name} ({confidence:.2f})"
        
        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img_with_overlay,
            (x_min, y_min - text_height - baseline - 5),
            (x_min + text_width, y_min),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img_with_overlay,
            label,
            (x_min, y_min - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    return Image.fromarray(img_with_overlay)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    """Convert binary mask to PIL Image for display.
    
    Args:
        mask: Binary mask as numpy array with values 0 or 1.
    
    Returns:
        PIL Image in grayscale mode (0 = black, 255 = white).
    """
    # Convert 0/1 mask to 0/255 for display
    mask_display = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask_display, mode='L')


def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply binary mask to image, showing foreground in color and background in gray.
    
    Args:
        image: PIL Image to apply mask to.
        mask: Binary mask as numpy array with values 0 or 1.
    
    Returns:
        PIL Image with mask applied (foreground in color, background in gray).
    """
    img_array = np.array(image.convert('RGB'))
    
    # Convert mask to 3-channel
    mask_3d = np.stack([mask, mask, mask], axis=2)
    
    # Create grayscale version of image
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray_3d = np.stack([gray, gray, gray], axis=2)
    
    # Apply mask: foreground in color, background in gray
    result = np.where(mask_3d == 1, img_array, gray_3d)
    
    return Image.fromarray(result.astype(np.uint8))


# Page configuration
st.set_page_config(
    page_title="Color Card Detection & Background Removal",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Color Card Detection & Background Removal")
st.markdown(
    "Upload an image to detect color reference cards and generate a background removal mask."
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    debug_mode = st.checkbox("Debug Mode", value=False, help="Enable debug output")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        This app uses:
        - **YOLOv8 OBB** for color card detection
        - **RMBG-1.4** for background removal
        
        Detected card types:
        - 24-color card
        - 8-hybrid card
        - Checker card
        """
    )

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Upload an image file to process"
)

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, width='stretch')
            st.caption(f"Size: {image.size[0]} × {image.size[1]} pixels")
        
        # Process image
        with st.spinner("Processing image..."):
            # Step 1: Detect color cards
            st.markdown("---")
            st.subheader("🔍 Step 1: Color Card Detection")
            
            try:
                detections = detect_color_cards(image, debug=debug_mode)
                
                if detections:
                    st.success(f"✅ Found {len(detections)} color card(s)")
                    
                    # Display detection results
                    detection_cols = st.columns(min(len(detections), 3))
                    
                    for i, det in enumerate(detections):
                        with detection_cols[i % 3]:
                            st.markdown(f"**Card {i+1}**")
                            st.write(f"**Type:** {det['class']}")
                            st.write(f"**Confidence:** {det['confidence']:.2%}")
                            st.write(f"**Class ID:** {det['class_id']}")
                    
                    # Draw detections on image
                    image_with_detections = draw_detections_on_image(image, detections)
                    
                    with col2:
                        st.subheader("🎯 Detections Overlay")
                        st.image(image_with_detections, width='stretch')
                    
                    # Show coordinates in expander
                    with st.expander("📐 View Detection Coordinates"):
                        for i, det in enumerate(detections):
                            st.markdown(f"**Card {i+1}: {det['class']}**")
                            st.json({
                                "coordinates": det['coordinates'],
                                "confidence": det['confidence'],
                                "class_id": det['class_id']
                            })
                
                else:
                    st.warning("⚠️ No color cards detected in the image")
                    detections = []
                
            except Exception as e:
                st.error(f"❌ Color card detection failed: {str(e)}")
                if debug_mode:
                    st.exception(e)
                detections = []
            
            # Step 2: Background removal
            st.markdown("---")
            st.subheader("✂️ Step 2: Background Removal")
            
            try:
                # Prepare card coordinates if available
                card_coords = detections if detections else None
                
                result = remove_background_mask(
                    image,
                    card_coordinates=card_coords,
                    debug=debug_mode
                )
                
                # Handle return value (tuple when debug=True, mask when debug=False)
                if debug_mode and isinstance(result, tuple):
                    if len(result) == 3:
                        mask, rmbg_image, rmbg_white_bg_image = result
                        rmbg_input_image = None
                    else:
                        mask, rmbg_image = result
                        rmbg_input_image = None
                        rmbg_white_bg_image = None
                else:
                    mask = result
                    rmbg_image = None
                    rmbg_input_image = None
                    rmbg_white_bg_image = None
                
                # Display results
                if debug_mode and rmbg_image is not None:
                    # Show: Input to RMBG, RMBG Result, White BG Image, and Binary Mask
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.markdown("**Input to RMBG**")
                        # Show original image (cards are cropped out internally before RMBG)
                        st.image(image, width='stretch')
                        if detections:
                            st.caption(f"Original image (cards cropped out before processing)")
                        else:
                            st.caption("Original image")
                    
                    with result_col2:
                        st.markdown("**RMBG Result**")
                        st.image(rmbg_image, width='stretch')
                        st.caption("First pass output from RMBG-1.4")
                    
                    with result_col3:
                        st.markdown("**White BG Image**")
                        if rmbg_white_bg_image is not None:
                            st.image(rmbg_white_bg_image, width='stretch')
                            st.caption("First result on white background")
                        else:
                            st.info("Not available")
                    
                    with result_col4:
                        st.markdown("**Binary Mask**")
                        mask_image = mask_to_image(mask)
                        st.image(mask_image, width='stretch')
                        st.caption("Mask from second RMBG pass")
                        
                        # Calculate statistics
                        foreground_pixels = np.sum(mask == 1)
                        total_pixels = mask.size
                        foreground_percent = 100 * foreground_pixels / total_pixels
                        
                        st.metric("Foreground Coverage", f"{foreground_percent:.1f}%")
                else:
                    # Normal display without debug
                    result_col1 = st.columns(1)[0]
                    
                    with result_col1:
                        st.markdown("**Binary Mask**")
                        mask_image = mask_to_image(mask)
                        st.image(mask_image, width='stretch')
                        
                        # Calculate statistics
                        foreground_pixels = np.sum(mask == 1)
                        total_pixels = mask.size
                        foreground_percent = 100 * foreground_pixels / total_pixels
                        
                        st.metric("Foreground Coverage", f"{foreground_percent:.1f}%")
                
                # Step 3: Generate swatch
                st.markdown("---")
                st.subheader("🎨 Step 3: Swatch Generation")
                
                try:
                    # Create transparent image from mask
                    # Convert original image to RGBA
                    img_rgba = image.convert('RGBA')
                    img_array = np.array(img_rgba)
                    
                    # Apply mask to alpha channel: foreground (mask=1) is opaque, background (mask=0) is transparent
                    img_array[:, :, 3] = (mask * 255).astype(np.uint8)
                    transparent_image = Image.fromarray(img_array, mode='RGBA')
                    
                    # Generate swatch
                    with st.spinner("Generating swatch..."):
                        swatch = generate_swatch(
                            transparent_image,
                            swatch_size=(1000, 500),
                            target_dpi=1200,
                            debug=debug_mode
                        )
                    
                    # Display swatch
                    swatch_col1 = st.columns(1)[0]
                    with swatch_col1:
                        st.markdown("**Generated Swatch**")
                        st.image(swatch, width='stretch')
                        dpi_value = swatch.info.get('dpi', (1200, 1200))
                        dpi_display = dpi_value[0] if isinstance(dpi_value, tuple) else dpi_value
                        st.caption(f"Size: {swatch.size[0]} × {swatch.size[1]} pixels, DPI: {dpi_display}")
                    
                except Exception as e:
                    st.error(f"❌ Swatch generation failed: {str(e)}")
                    if debug_mode:
                        st.exception(e)
                    swatch = None
                
                # Download options
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                if debug_mode and rmbg_image is not None:
                    # Count available downloads
                    num_cols = 0
                    num_cols += 1  # RMBG result
                    if rmbg_white_bg_image is not None:
                        num_cols += 1
                    num_cols += 1  # Mask
                    if swatch is not None:
                        num_cols += 1
                    num_cols += 1  # Detection overlay
                    
                    download_cols = st.columns(num_cols)
                    
                    col_idx = 0
                    
                    with download_cols[col_idx]:
                        # Download RMBG result
                        rmbg_bytes = io.BytesIO()
                        rmbg_image.save(rmbg_bytes, format='PNG')
                        st.download_button(
                            label="📥 Download RMBG Result",
                            data=rmbg_bytes.getvalue(),
                            file_name="rmbg_result.png",
                            mime="image/png"
                        )
                    col_idx += 1
                    
                    if rmbg_white_bg_image is not None:
                        with download_cols[col_idx]:
                            # Download white background image
                            white_bg_bytes = io.BytesIO()
                            rmbg_white_bg_image.save(white_bg_bytes, format='PNG')
                            st.download_button(
                                label="📥 Download White BG Image",
                                data=white_bg_bytes.getvalue(),
                                file_name="rmbg_white_bg.png",
                                mime="image/png"
                            )
                        col_idx += 1
                    
                    with download_cols[col_idx]:
                        # Download mask as PNG
                        mask_bytes = io.BytesIO()
                        mask_image.save(mask_bytes, format='PNG')
                        st.download_button(
                            label="📥 Download Mask",
                            data=mask_bytes.getvalue(),
                            file_name="background_mask.png",
                            mime="image/png"
                        )
                    col_idx += 1
                    
                    if swatch is not None:
                        with download_cols[col_idx]:
                            # Download swatch
                            swatch_bytes = io.BytesIO()
                            swatch.save(swatch_bytes, format='PNG')
                            st.download_button(
                                label="📥 Download Swatch",
                                data=swatch_bytes.getvalue(),
                                file_name="swatch.png",
                                mime="image/png"
                            )
                        col_idx += 1
                    
                    with download_cols[col_idx]:
                        # Download detection overlay
                        overlay_bytes = io.BytesIO()
                        if detections:
                            image_with_detections.save(overlay_bytes, format='PNG')
                            st.download_button(
                                label="📥 Download Detection Overlay",
                                data=overlay_bytes.getvalue(),
                                file_name="detections_overlay.png",
                                mime="image/png"
                            )
                        else:
                            st.info("No detections")
                else:
                    num_cols = 3 if swatch is not None else 2
                    download_cols = st.columns(num_cols)
                    
                    col_idx = 0
                    
                    with download_cols[col_idx]:
                        # Download mask as PNG
                        mask_bytes = io.BytesIO()
                        mask_image.save(mask_bytes, format='PNG')
                        st.download_button(
                            label="📥 Download Mask",
                            data=mask_bytes.getvalue(),
                            file_name="background_mask.png",
                            mime="image/png"
                        )
                    col_idx += 1
                    
                    if swatch is not None:
                        with download_cols[col_idx]:
                            # Download swatch
                            swatch_bytes = io.BytesIO()
                            swatch.save(swatch_bytes, format='PNG')
                            st.download_button(
                                label="📥 Download Swatch",
                                data=swatch_bytes.getvalue(),
                                file_name="swatch.png",
                                mime="image/png"
                            )
                        col_idx += 1
                    
                    with download_cols[col_idx]:
                        # Download detection overlay
                        overlay_bytes = io.BytesIO()
                        if detections:
                            image_with_detections.save(overlay_bytes, format='PNG')
                            st.download_button(
                                label="📥 Download Detection Overlay",
                                data=overlay_bytes.getvalue(),
                                file_name="detections_overlay.png",
                                mime="image/png"
                            )
                        else:
                            st.info("No detections to download")
                
            except Exception as e:
                st.error(f"❌ Background removal failed: {str(e)}")
                if debug_mode:
                    st.exception(e)
        
    except Exception as e:
        st.error(f"❌ Failed to process image: {str(e)}")
        if debug_mode:
            st.exception(e)

else:
    st.info("👆 Please upload an image to get started")
    
    # Show example usage
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload an image** using the file uploader above
        2. **View detections** - Color cards will be highlighted with colored boxes
        3. **View background mask** - Binary mask showing foreground/background
        4. **Download results** - Save the mask or masked image
        
        **Tips:**
        - Enable "Use Card Exclusion" to improve background removal by excluding card areas
        - Enable "Debug Mode" to see detailed processing information
        - The app works best with images containing color reference cards
        """)


"""Export service for rendering corrected images."""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import logging
import numpy as np
from PIL import Image

from app.services.models import ImageItem, CorrectionParams, ExportSummary
from app.services.correction import load_image_as_float, apply_correction_params, save_image_from_float

logger = logging.getLogger(__name__)


def resize_to_width(img: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image to target width while preserving aspect ratio.
    
    Args:
        img: RGB image as float array [0, 1]
        target_width: Target width in pixels
        
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    
    if w == target_width:
        return img
    
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    
    # Convert to PIL for high-quality resize
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Convert back to float
    return np.array(pil_img).astype(np.float32) / 255.0


def get_output_paths(
    image: ImageItem,
    overwrite: bool
) -> Dict[int, Tuple[str, str]]:
    """Generate output paths for different sizes (always JPG).
    
    Args:
        image: ImageItem with path information
        overwrite: If True, overwrite existing files; if False, add -color_correct suffix
        
    Returns:
        Dict mapping size (450/1500/3000) to (input_path, output_path) tuples
        Output paths are always .jpg
    """
    paths = {}
    
    # Use RAW file path as reference to determine base directory and name
    if not image.raw_path:
        logger.warning(f"No RAW path for image {image.id}")
        return paths
    
    raw_path_obj = Path(image.raw_path)
    base_dir = raw_path_obj.parent
    base_name = raw_path_obj.stem
    
    # Always use .jpg extension for exports
    ext = '.jpg'
    
    # Generate paths for each size (always JPG)
    sizes_config = [
        (450, base_name),
        (1500, f"{base_name}-1500"),
        (3000, f"{base_name}-3000")
    ]
    
    for size, output_stem in sizes_config:
        if overwrite:
            output_name = f"{output_stem}{ext}"
        else:
            output_name = f"{output_stem}-color_correct{ext}"
        
        output_path = str(base_dir / output_name)
        # Input path is not used for RAW export, but we need to provide it
        paths[size] = (image.raw_path, output_path)
    
    return paths


def export_image_jpeg(
    image: ImageItem,
    correction_params: CorrectionParams,
    overwrite: bool
) -> Tuple[int, List[str]]:
    """Export corrected image in JPEG mode (from proxies).
    
    Args:
        image: ImageItem with path information
        correction_params: Correction parameters to apply
        overwrite: Whether to overwrite existing files
        
    Returns:
        Tuple of (files_written_count, error_list)
    """
    errors = []
    files_written = 0
    
    try:
        # Get output paths
        output_config = get_output_paths(image, overwrite)
        
        if not output_config:
            errors.append(f"No output paths for image {image.id}")
            return files_written, errors
        
        # Load and correct the largest available proxy
        source_path = image.proxy_3000 or image.proxy_1500 or image.proxy_450 or image.primary_path
        
        if not source_path or not os.path.exists(source_path):
            errors.append(f"Source path not found: {source_path}")
            return files_written, errors
        
        logger.info(f"Loading image from {source_path}")
        img_float = load_image_as_float(source_path)
        
        if img_float is None:
            errors.append(f"Failed to load {source_path}")
            return files_written, errors
        
        # Apply corrections
        corrected = apply_correction_params(img_float, correction_params)
        
        # Export at each size
        for size, (_, output_path) in output_config.items():
            try:
                # Resize if needed
                if size == 3000:
                    sized = corrected
                elif size == 1500:
                    sized = resize_to_width(corrected, 1500)
                elif size == 450:
                    sized = resize_to_width(corrected, 450)
                else:
                    sized = corrected
                
                # Save
                save_image_from_float(sized, output_path, quality=95)
                files_written += 1
                logger.info(f"Exported {size}px -> {output_path}")
                
            except Exception as e:
                error_msg = f"Failed to export {size}px: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
    except Exception as e:
        error_msg = f"Export failed for image {image.id}: {e}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    return files_written, errors


def export_image_raw(
    image: ImageItem,
    overall_params: CorrectionParams,
    individual_params: Optional[CorrectionParams],
    overwrite: bool
) -> Tuple[int, List[str]]:
    """Export corrected image in RAW mode (process RAW file).
    
    Args:
        image: ImageItem with RAW path
        overall_params: Overall (cluster) correction parameters
        individual_params: Optional individual (image) correction parameters
        overwrite: Whether to overwrite existing files
        
    Returns:
        Tuple of (files_written_count, error_list)
    """
    errors = []
    files_written = 0
    
    try:
        import rawpy
    except ImportError:
        error_msg = "rawpy not available - cannot process RAW files"
        logger.error(error_msg)
        return files_written, [error_msg]
    
    if not image.raw_path or not os.path.exists(image.raw_path):
        error_msg = f"No RAW file for {image.id}"
        logger.error(error_msg)
        return files_written, [error_msg]
    
    try:
        # Load RAW file and process to base image (same as preview)
        logger.info(f"Loading RAW file: {image.raw_path}")
        with rawpy.imread(image.raw_path) as raw:
            # Process RAW with same settings as preview for consistency
            rgb = raw.postprocess(
                use_camera_wb=False,  # Use auto WB
                output_bps=16,
                no_auto_bright=False,  # Allow auto brightness adjustment
                bright=1.0             # Brightness multiplier (1.0 = normal)
            )
        
        # Convert to float
        img_float = rgb.astype(np.float32) / 65535.0
        
        # Apply overall corrections first
        corrected = apply_correction_params(img_float, overall_params)
        
        # Apply individual corrections on top if they exist
        if individual_params:
            corrected = apply_correction_params(corrected, individual_params)
        
        # Get output paths - always export as JPG
        output_config = get_output_paths(image, overwrite)
        
        # Export at each size (3000px, 1500px, 450px) as JPG
        for size, (_, output_path) in output_config.items():
            try:
                # Ensure output path is .jpg
                output_path_obj = Path(output_path)
                if output_path_obj.suffix.lower() not in ['.jpg', '.jpeg']:
                    output_path = str(output_path_obj.with_suffix('.jpg'))
                
                sized = resize_to_width(corrected, size)
                save_image_from_float(sized, output_path, quality=95)
                files_written += 1
                logger.info(f"Exported {size}px JPG -> {output_path}")
                
            except Exception as e:
                error_msg = f"Failed to export {size}px JPG: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
    except Exception as e:
        error_msg = f"RAW export failed for {image.id}: {e}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    return files_written, errors


def export_session_images(
    images: Dict[str, ImageItem],
    clusters_with_params: List[Tuple[List[str], CorrectionParams]],
    image_source: str,
    overwrite: bool,
    session,  # Session object to access individual corrections
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> ExportSummary:
    """Export all images with their cluster corrections.
    
    Args:
        images: Dict of image_id to ImageItem
        clusters_with_params: List of (image_ids, correction_params) tuples
        image_source: Image source mode
        overwrite: Whether to overwrite existing files
        progress_callback: Optional callback(current, total, message) for progress updates
        
    Returns:
        ExportSummary with results
    """
    # Count total images first
    total_images = sum(len(image_ids) for image_ids, _ in clusters_with_params)
    
    total_files_written = 0
    overwritten_count = 0
    new_files_count = 0
    failed_count = 0
    all_errors = []
    current_image = 0
    
    for image_ids, correction_params in clusters_with_params:
        for image_id in image_ids:
            current_image += 1
            image = images.get(image_id)
            if not image:
                all_errors.append(f"Image {image_id} not found")
                failed_count += 1
                continue
            
            # Check if files exist to determine overwrite count
            output_paths = get_output_paths(image, overwrite=False)
            existing_count = sum(1 for _, out_path in output_paths.values() if os.path.exists(out_path))
            
            # Update progress
            if progress_callback:
                progress_callback(current_image, total_images, f"Exporting image {current_image}/{total_images}")
            
            # Always use RAW mode - get individual corrections if they exist
            individual_params = session.get_individual_correction(image_id) if session else None
            
            # Export from RAW
            files_written, errors = export_image_raw(
                image,
                correction_params,  # overall params
                individual_params,  # individual params
                overwrite
            )
            
            total_files_written += files_written
            
            if errors:
                all_errors.extend(errors)
                failed_count += 1
            
            # Track overwrite vs new
            if overwrite and existing_count > 0:
                overwritten_count += min(files_written, existing_count)
                new_files_count += max(0, files_written - existing_count)
            else:
                new_files_count += files_written
    
    logger.info(f"Export complete: {total_images} images, {total_files_written} files, {failed_count} failed")
    
    return ExportSummary(
        total_images=total_images,
        total_files_written=total_files_written,
        overwritten_count=overwritten_count,
        new_files_count=new_files_count,
        failed_count=failed_count,
        errors=all_errors[:100]  # Limit error list
    )


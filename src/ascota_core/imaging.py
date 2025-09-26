"""
Segmentation, Color card detection and classification pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Union, Dict, Any
from .utils import (
    load_image_any, resize_max, polygon_to_mask, non_max_suppression_polys,
    cv2_to_pil, pil_to_cv2, create_transparent_image, contour_rect_fallback
)


class TemplateMatcher:
    """Feature-based template matcher for color card detection using ORB features."""
    
    def __init__(self, template_bgr: np.ndarray, name: str, nfeatures: int = 1500) -> None:
        """Initialize template matcher with ORB feature detection.
        
        Creates a template matcher that uses ORB (Oriented FAST and Rotated BRIEF)
        features for robust detection of color cards in images. The template is
        preprocessed to extract keypoints and descriptors for matching.
        
        Args:
            template_bgr: Template image in OpenCV BGR format as numpy array.
            name: Identifier name for this template (e.g., 'colorchecker24').
            nfeatures: Maximum number of ORB features to detect. Defaults to 1500.
            
        Raises:
            ValueError: If template_bgr is None, empty, or cannot be converted to grayscale.
            TypeError: If template_bgr is not a numpy array.
        """
        self.name = name
        # Validate template image
        if template_bgr is None:
            raise ValueError(f"Template image for '{name}' is None")
        if not isinstance(template_bgr, (np.ndarray,)):
            raise TypeError(f"Template image for '{name}' must be a numpy.ndarray, got {type(template_bgr)}")
        if template_bgr.size == 0:
            raise ValueError(f"Template image for '{name}' is empty")

        self.tmpl = template_bgr
        try:
            self.gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError(f"Failed to convert template '{name}' to grayscale: {e}")
        self.h, self.w = self.gray.shape
        self.corners = np.float32([[0,0],[self.w,0],[self.w,self.h],[0,self.h]]).reshape(-1,1,2)

        self.orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
        self.kp, self.des = self.orb.detectAndCompute(self.gray, None)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def find_instances(self, img_bgr: np.ndarray, max_instances: int = 5, debug: bool = False) -> List[np.ndarray]:
        """Find instances of the template in the input image using feature matching.
        
        Uses ORB feature matching with homography estimation to detect instances
        of the template card in the input image. Applies geometric validation
        including convexity and area checks to ensure quality detections.
        
        Args:
            img_bgr: Input image in OpenCV BGR format as numpy array.
            max_instances: Maximum number of template instances to detect. Defaults to 5.
            debug: If True, print similarity scores and detection information.
            
        Returns:
            List of detected polygon coordinates as numpy arrays with shape (4, 2).
            Each polygon represents the four corners of a detected card instance.
            Results are filtered using non-maximum suppression to remove overlaps.
        """
        found_polys = []
        img = img_bgr.copy()
        for _ in range(max_instances):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp2, des2 = self.orb.detectAndCompute(gray, None)
            if des2 is None or len(kp2) < 20:
                break

            matches = self.bf.knnMatch(self.des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) < 12:
                break

            src_pts = np.float32([ self.kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=4.0)
            if H is None:
                break

            # project template corners
            proj = cv2.perspectiveTransform(self.corners, H).reshape(4,2)
            # sanity checks: convex, area, aspect
            area = cv2.contourArea(proj.astype(np.float32))
            if area < 400:  # too small
                break
            if not cv2.isContourConvex(proj.astype(np.float32)):
                break

            # Calculate similarity for debugging
            if debug:
                try:
                    similarity = self._calculate_similarity(img_bgr, proj, H)
                    area = cv2.contourArea(proj.astype(np.float32))
                    print(f"DEBUG: TemplateMatcher '{self.name}' - Detected card similarity: {similarity:.2f}% (area: {area:.0f})")
                except ValueError as e:
                    print(f"DEBUG: TemplateMatcher '{self.name}' - {e}")
                    continue  # Skip this detection due to low similarity

            # accept
            found_polys.append(proj)

            # "erase" region to find more instances
            mask_poly = polygon_to_mask(img.shape, proj)
            img[mask_poly>0] = 255  # fill with white

        # suppress overlaps (different templates might hit same card)
        found_polys = non_max_suppression_polys(found_polys, iou_thresh=0.3)
        return found_polys

    def _calculate_similarity(self, img_bgr: np.ndarray, detected_poly: np.ndarray, 
                             homography: np.ndarray, min_similarity_threshold: float = 30.0) -> float:
        """Calculate similarity percentage between detected card region and template.
        
        Warps the detected card region to match template dimensions and computes
        a similarity score based on structural similarity and feature matching.
        Used for quality assessment of card detections.
        
        Args:
            img_bgr: Input image in BGR format containing the detected card.
            detected_poly: Detected card polygon with shape (4, 2) representing corners.
            homography: 3x3 homography matrix from template to detected region.
            min_similarity_threshold: Minimum similarity score required. Defaults to 30.0.
            
        Returns:
            Similarity percentage as float between 0.0 and 100.0.
            
        Raises:
            ValueError: If similarity is below the minimum threshold.
        
        Args:
            img_bgr: Input image in BGR format
            detected_poly: Detected polygon coordinates
            homography: Homography matrix from template to detected region
            min_similarity_threshold: Minimum similarity threshold (0-100), raises exception if below
            
        Returns:
            Similarity percentage (0-100)
            
        Raises:
            ValueError: If similarity is below the minimum threshold
        """
        try:
            # Extract the detected region from the input image
            x, y, w, h = cv2.boundingRect(detected_poly.astype(np.int32))
            
            # Ensure coordinates are within image bounds
            img_h, img_w = img_bgr.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            # Check if region is valid
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid detected region: x={x}, y={y}, w={w}, h={h}")
            
            detected_region = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            
            # Check if detected region is valid
            if detected_region.size == 0:
                raise ValueError("Detected region is empty")
            
            # Resize template to match detected region size for comparison
            template_resized = cv2.resize(self.gray, (w, h))
            
            # Calculate structural similarity using template matching
            result = cv2.matchTemplate(detected_region, template_resized, cv2.TM_CCOEFF_NORMED)
            similarity = float(result[0][0]) * 100
            
            # Additional check using histogram correlation for color similarity
            hist_template = cv2.calcHist([template_resized], [0], None, [256], [0, 256])
            hist_detected = cv2.calcHist([detected_region], [0], None, [256], [0, 256])
            hist_corr = cv2.compareHist(hist_template, hist_detected, cv2.HISTCMP_CORREL)
            
            # Combine template matching and histogram correlation
            combined_similarity = (similarity + hist_corr * 100) / 2
            final_similarity = max(0, min(100, combined_similarity))
            
            # Check if similarity is too low
            if final_similarity < min_similarity_threshold:
                raise ValueError(f"Low similarity detected: {final_similarity:.2f}% (threshold: {min_similarity_threshold}%) for template '{self.name}'. This may indicate a false positive detection.")
            
            return final_similarity
            
        except ValueError:
            # Re-raise ValueError for low similarity
            raise
        except Exception as e:
            print(f"DEBUG: _calculate_similarity failed: {e}")
            raise ValueError(f"Similarity calculation failed for template '{self.name}': {e}")


def extract_rectangular_region(img_bgr: np.ndarray, poly_xy: np.ndarray) -> Optional[np.ndarray]:
    """Extract rectangular region defined by polygon without perspective warping.
    
    Extracts a bounding rectangle around the detected polygon, with optional
    contour refinement for better edge coverage. Designed for top-down images
    where perspective correction is not needed.
    
    Args:
        img_bgr: Input image in OpenCV BGR format as numpy array.
        poly_xy: Polygon coordinates as numpy array with shape (4, 2).
        
    Returns:
        Extracted rectangular region as numpy array, or None if extraction
        fails or results in a region that is too small (< 10x10 pixels).
    """
    # Get initial bounding rectangle
    x, y, w, h = cv2.boundingRect(poly_xy.astype(np.int32))
    
    # Try to refine the bounding box using contour detection
    refined_poly = _refine_polygon_with_contours(img_bgr, poly_xy, x, y, w, h)
    if refined_poly is not None:
        x, y, w, h = cv2.boundingRect(refined_poly.astype(np.int32))
        print(f"DEBUG: extract_rectangular_region - refined bounding box: ({x},{y},{w},{h})")
    
    # Ensure coordinates are within image bounds
    img_h, img_w = img_bgr.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    # Ensure minimum size
    if w < 10 or h < 10:
        print(f"DEBUG: extract_rectangular_region - too small: {w}x{h}")
        return None
    
    # Extract rectangular region
    extracted = img_bgr[y:y+h, x:x+w]
    print(f"DEBUG: extract_rectangular_region - extracted {extracted.shape} from region ({x},{y},{w},{h})")
    
    return extracted


def _refine_polygon_with_contours(img_bgr: np.ndarray, original_poly: np.ndarray, 
                                 x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    """Refine polygon boundaries using contour detection for better edge coverage.
    
    Attempts to improve polygon detection accuracy by performing contour detection
    within a padded region around the original polygon. Uses adaptive thresholding
    to handle varying lighting conditions and selects the contour that best matches
    the original polygon area.
    
    Args:
        img_bgr: Input image in OpenCV BGR format as numpy array.
        original_poly: Original detected polygon coordinates with shape (4, 2).
        x: X coordinate of bounding rectangle.
        y: Y coordinate of bounding rectangle.
        w: Width of bounding rectangle.
        h: Height of bounding rectangle.
        
    Returns:
        Refined polygon coordinates as numpy array, or None if refinement fails
        or no suitable contour is found.
    """
    # Extract region around the detected card
    img_h, img_w = img_bgr.shape[:2]
    x = max(0, x - 20)  # Add some padding
    y = max(0, y - 20)
    w = min(w + 40, img_w - x)
    h = min(h + 40, img_h - y)
    
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the contour closest to the original polygon area
    original_area = cv2.contourArea(original_poly.astype(np.float32))
    best_contour = None
    best_area_diff = float('inf')
    
    for contour in contours:
        area = cv2.contourArea(contour)
        area_diff = abs(area - original_area)
        
        # Check if contour is roughly rectangular
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 4 and area_diff < best_area_diff:
            best_contour = contour
            best_area_diff = area_diff
    
    if best_contour is not None:
        # Convert back to original image coordinates
        refined_poly = best_contour.reshape(-1, 2) + np.array([x, y])
        return refined_poly.astype(np.float32)
    
    return None


class CardDetector:
    """Main color card detector that manages multiple template matchers.
    
    Coordinates detection across multiple template types (ColorChecker 24,
    ColorChecker 8, checker_cm) and handles template loading with fallback
    resolution for package templates.
    """
    
    def __init__(self, template_paths: List[str]) -> None:
        """Initialize card detector with template paths.
        
        Loads template images and creates TemplateMatcher instances for each
        valid template. Provides fallback resolution for relative paths using
        package template directories.
        
        Args:
            template_paths: List of paths to template image files. Supports
                absolute paths or relative names that will be resolved against
                package template directories.
                
        Raises:
            RuntimeError: If no valid templates could be loaded.
            ValueError: If template_paths contains None values.
        """
        self.matchers = []
        # Try to resolve and load each template path. Provide detailed errors and debug info.
        for p in template_paths:
            try:
                if p is None:
                    raise ValueError("Template path is None")

                # Attempt to load using provided helper. If p appears relative, try repository templates folder too.
                img = None
                try:
                    img = load_image_any(p)
                except Exception:
                    # Try to resolve relative to package templates directory
                    alt = Path(__file__).parent / "templates" / Path(p).name
                    if alt.exists():
                        if str(alt) != str(p) and __debug__:
                            print(f"DEBUG: Resolved template '{p}' -> '{alt}'")
                        img = load_image_any(str(alt))
                    else:
                        # Try workspace-level templates folder
                        alt2 = Path(__file__).parents[2] / "templates" / Path(p).name
                        if alt2.exists():
                            if __debug__:
                                print(f"DEBUG: Resolved template '{p}' -> '{alt2}'")
                            img = load_image_any(str(alt2))

                if img is None:
                    raise FileNotFoundError(f"Could not load template image for path '{p}' - tried provided path and known template folders")

                # Validate image loaded
                if not isinstance(img, np.ndarray) or img.size == 0:
                    raise ValueError(f"Template loaded from '{p}' is invalid or empty")

                self.matchers.append(TemplateMatcher(img, name=Path(p).stem))
            except Exception as e:
                # Surface useful debugging information and re-raise to fail fast
                msg = f"Failed to initialize template matcher for '{p}': {e}"
                print(f"DEBUG: {msg}")
                raise

    def detect(self, img_bgr: np.ndarray, max_per_template: int = 2, debug: bool = False) -> List[np.ndarray]:
        """Detect color cards in image with maximum limit enforcement.
        
        Runs detection across all loaded template matchers and aggregates results.
        Enforces a maximum of 2 total cards regardless of template-specific limits
        and filters out cards that are too large relative to image size.
        
        Args:
            img_bgr: Input image in OpenCV BGR format as numpy array.
            max_per_template: Maximum instances per template (currently ignored,
                system enforces max 2 total cards).
            debug: If True, print debug information during detection process.
            
        Returns:
            List of detected polygon coordinates as numpy arrays with shape (4, 2).
            Maximum of 2 polygons will be returned, filtered by size and quality.
        """
        all_polys = []
        img_area = img_bgr.shape[0] * img_bgr.shape[1]
        max_card_area = img_area * 0.55  # 55% of image area
        
        for m in self.matchers:
            polys = m.find_instances(img_bgr, max_instances=2, debug=debug)  # Max 2 per template
            
            # Filter out cards that are too large
            valid_polys = []
            for poly in polys:
                area = cv2.contourArea(poly.astype(np.float32))
                if area <= max_card_area:
                    valid_polys.append(poly)
                else:
                    print(f"DEBUG: Rejected card too large: {area:.0f} pixels (max: {max_card_area:.0f})")
            
            all_polys.extend(valid_polys)
        
        # suppress overlaps across templates too
        all_polys = non_max_suppression_polys(all_polys, iou_thresh=0.35)
        
        # If we have fewer than 2 cards, try contour-based fallback
        if len(all_polys) < 2:
            print("DEBUG: Using contour-based fallback for additional detection")
            contour_polys = self._contour_based_detection(img_bgr, all_polys, max_card_area, debug=debug)
            all_polys.extend(contour_polys)
            # Re-apply NMS to the combined results
            all_polys = non_max_suppression_polys(all_polys, iou_thresh=0.35)
        
        # Limit to maximum 2 cards total
        if len(all_polys) > 2:
            # Sort by area and keep the 2 largest
            areas = [cv2.contourArea(poly.astype(np.float32)) for poly in all_polys]
            sorted_indices = np.argsort(areas)[::-1]  # Sort descending
            all_polys = [all_polys[i] for i in sorted_indices[:2]]
        
        # Final similarity check for debugging
        if all_polys:
            if debug:
                print(f"DEBUG: CardDetector - Final validation of {len(all_polys)} detected cards")
            
            # Filter out cards with low similarity
            valid_polys = []
            for i, poly in enumerate(all_polys):
                # Find which template best matches this polygon
                best_similarity = 0
                best_template = "unknown"
                similarity_scores = {}
                
                for matcher in self.matchers:
                    try:
                        # Extract region and compare with template
                        x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                        
                        # Ensure coordinates are within image bounds
                        img_h, img_w = img_bgr.shape[:2]
                        x = max(0, min(x, img_w - 1))
                        y = max(0, min(y, img_h - 1))
                        w = min(w, img_w - x)
                        h = min(h, img_h - y)
                        
                        # Check if region is valid
                        if w <= 0 or h <= 0:
                            continue
                        
                        detected_region = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                        
                        # Check if detected region is valid
                        if detected_region.size == 0:
                            continue
                        
                        template_resized = cv2.resize(matcher.gray, (w, h))
                        
                        # Calculate similarity using template matching
                        result = cv2.matchTemplate(detected_region, template_resized, cv2.TM_CCOEFF_NORMED)
                        similarity = float(result[0][0]) * 100
                        similarity_scores[matcher.name] = similarity
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_template = matcher.name
                            
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: CardDetector - Similarity check failed for card {i} with template {matcher.name}: {e}")
                        continue
                
                if debug:
                    # Print detailed similarity scores for all templates
                    print(f"DEBUG: CardDetector - Card {i+1} similarity scores:")
                    for template_name, score in similarity_scores.items():
                        print(f"  {template_name}: {score:.2f}%")
                    print(f"  Best match: '{best_template}' with {best_similarity:.2f}% similarity")
                
                # Only keep cards with reasonable similarity (20%)
                if best_similarity >= 20.0:
                    valid_polys.append(poly)
                elif debug:
                    print(f"DEBUG: CardDetector - Rejecting card {i+1} due to low similarity: {best_similarity:.2f}%")
            
            # Update all_polys with filtered results
            all_polys = valid_polys
        
        return all_polys
    
    def _contour_based_detection(self, img_bgr: np.ndarray, existing_polys: List[np.ndarray], 
                               max_card_area: float, debug: bool = False) -> List[np.ndarray]:
        """Contour-based fallback detection for improved edge coverage.
        
        Uses contour detection as a fallback method when template matching
        doesn't find sufficient cards. Applies edge detection and contour
        analysis to identify rectangular card-like regions.
        
        Args:
            img_bgr: Input image in OpenCV BGR format as numpy array.
            existing_polys: List of already detected polygons to avoid duplicates.
            max_card_area: Maximum allowed area for detected cards in pixels.
            debug: If True, print debug information during detection.
            
        Returns:
            List of additional detected polygon coordinates as numpy arrays
            with shape (4, 2), filtered to avoid overlaps with existing detections.
        """
        # Create mask of already detected areas
        existing_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        for poly in existing_polys:
            mask = polygon_to_mask(img_bgr.shape, poly)
            existing_mask = cv2.bitwise_or(existing_mask, mask)
        
        # Use contour fallback
        contour_polys = contour_rect_fallback(img_bgr, existing_mask, min_area=1000)
        
        # Filter by size and aspect ratio
        valid_contour_polys = []
        for poly in contour_polys:
            area = cv2.contourArea(poly.astype(np.float32))
            if area > max_card_area:
                continue
                
            # Check aspect ratio (should be reasonable for color cards)
            x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
            aspect_ratio = w / h
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Too extreme
                continue
                
            # Check if it overlaps significantly with existing detections
            poly_mask = polygon_to_mask(img_bgr.shape, poly)
            overlap = cv2.bitwise_and(poly_mask, existing_mask)
            overlap_ratio = np.sum(overlap > 0) / np.sum(poly_mask > 0)
            
            if overlap_ratio < 0.3:  # Less than 30% overlap
                valid_contour_polys.append(poly)
                if debug:
                    print(f"DEBUG: Contour detection - found card: area={area:.0f}, aspect={aspect_ratio:.2f}")
        
        return valid_contour_polys

    def extract_all(self, img_bgr: np.ndarray, debug: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Extract all detected color cards without perspective warping.
        
        Detects cards in the image and extracts rectangular regions for each
        detected card. Creates a combined mask showing all detected card areas.
        Maximum of 2 cards will be processed.
        
        Args:
            img_bgr: Input image in OpenCV BGR format as numpy array.
            debug: If True, print debug information during extraction.
            
        Returns:
            Tuple containing:
                - polygons: List of valid polygon coordinates (max 2)
                - extracted_crops: List of extracted card regions as numpy arrays
                - combined_mask: Binary mask showing all detected card areas
        """
        polys = self.detect(img_bgr, debug=debug)
        crops = []
        valid_polys = []
        
        for P in polys:
            # Additional validation before extraction
            if self._validate_card_polygon(P, img_bgr.shape):
                crop = extract_rectangular_region(img_bgr, P)
                if crop is not None:
                    crops.append(crop)
                    valid_polys.append(P)
            else:
                if debug:
                    print(f"DEBUG: Rejected invalid card polygon")
        
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        for P in valid_polys:
            mask = cv2.bitwise_or(mask, polygon_to_mask(img_bgr.shape, P))
        
        return valid_polys, crops, mask
    
    def _validate_card_polygon(self, poly: np.ndarray, img_shape: Tuple[int, int, int], debug: bool = False) -> bool:
        """Validate that a detected polygon represents a reasonable color card.
        
        Performs geometric validation checks including area constraints,
        aspect ratio validation, and convexity testing to ensure detected
        polygons correspond to actual color cards rather than false positives.
        
        Args:
            poly: Polygon coordinates as numpy array with shape (4, 2).
            img_shape: Image dimensions as tuple (height, width, channels).
            debug: If True, print validation details and failure reasons.
            
        Returns:
            True if polygon passes all validation checks, False otherwise.
        """
        # Check area
        area = cv2.contourArea(poly.astype(np.float32))
        img_area = img_shape[0] * img_shape[1]
        
        if area < 1000 or area > img_area * 0.55:  # Too small or too large
            if debug == True:
                print("DEBUG: validate_card_polygon - Too small or too large")
            return False
        
        # Check if polygon is convex
        if not cv2.isContourConvex(poly.astype(np.float32)):
            if debug == True:
                print("DEBUG: validate_card_polygon - Polygon not Convex")
            return False
        
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
        aspect_ratio = w / h
        
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:  # Too extreme
            if debug == True:
                print("DEBUG: validate_card_polygon - Aspect ratio extreme")
            return False
        
        # Check if polygon is roughly rectangular
        rect_area = w * h
        if area / rect_area < 0.5:  # Too irregular
            return False
        
        return True


def detect_card_type(num_cards_detected: int, debug: bool = False) -> List[str]:
    """Determine card types based on the number of detected cards.
    
    Uses a simplified rule-based approach for card type classification:
    - Single card detection assumes ColorChecker 8
    - Dual card detection assumes ColorChecker 24 plus checker_cm scale reference
    
    Args:
        num_cards_detected: Total number of cards detected in the image.
        debug: If True, print debug information about type assignment.
        
    Returns:
        List of card type strings. For 1 card: ['colorchecker8'].
        For 2 cards: ['colorchecker24', 'checker_cm'] where the first
        entry should correspond to the larger detected card.
        Returns empty list for any other number of cards.
    """
    if num_cards_detected == 1:
        if debug:
            print("DEBUG: detect_card_type - 1 card detected -> ColorChecker 8")
        return ['colorchecker8']
    elif num_cards_detected == 2:
        if debug:
            print("DEBUG: detect_card_type - 2 cards detected -> ColorChecker 24 + checker_cm")
        return ['colorchecker24', 'checker_cm']
    else:
        if debug:
            print(f"DEBUG: detect_card_type - {num_cards_detected} cards detected -> default to ColorChecker 8")
        return ['colorchecker8']




def _assign_card_types_by_size(polygons: List[np.ndarray], debug: bool = False) -> List[str]:
    """Assign card types based on relative area when 2 cards are detected.
    
    Uses a simple size-based classification rule: the larger card is assumed
    to be ColorChecker 24 and the smaller card is assumed to be the checker_cm
    scale reference card.
    
    Args:
        polygons: List of polygon coordinates, expected to contain exactly 2 polygons.
            Each polygon should be a numpy array with shape (4, 2).
        debug: If True, print debug information about area calculations.
        
    Returns:
        List of card type strings in the same order as input polygons.
        Returns ['colorchecker8'] as fallback if not exactly 2 polygons provided.
    """
    import cv2
    import numpy as np
    
    if len(polygons) != 2:
        return ['colorchecker8']  # Fallback
    
    # Calculate areas
    areas = [cv2.contourArea(poly.astype(np.float32)) for poly in polygons]
    larger_index = 0 if areas[0] > areas[1] else 1
    smaller_index = 1 - larger_index
    
    # Create card types list in original polygon order
    card_types = ['', '']
    card_types[larger_index] = 'colorchecker24'
    card_types[smaller_index] = 'checker_cm'
    
    if debug:
        print(f"DEBUG: _assign_card_types_by_size - Larger card (area: {areas[larger_index]:.0f}) = ColorChecker 24, Smaller card (area: {areas[smaller_index]:.0f}) = checker_cm")
    
    return card_types




def create_card_masks_and_transparent(img_bgr: np.ndarray, polygons: List[np.ndarray], 
                                     debug: bool = False) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Create binary masks and transparent images for detected color cards.
    
    Generates binary masks where card regions are white (255) and background
    is black (0), then creates transparent images by applying these masks
    as alpha channels to the original image.
    
    Args:
        img_bgr: Input image in OpenCV BGR format as numpy array.
        polygons: List of detected card polygons, each with shape (4, 2).
        debug: If True, print debug information about mask creation.
        
    Returns:
        Tuple containing:
            - masks: List of PIL Images in grayscale mode ('L') representing binary masks
            - transparent_images: List of PIL Images in RGBA mode with transparency applied
    """
    masks = []
    transparent_images = []
    
    pil_img = cv2_to_pil(img_bgr)
    
    for poly in polygons:
        # Create mask for this polygon
        mask = polygon_to_mask(img_bgr.shape, poly)
        masks.append(Image.fromarray(mask, mode='L'))
        
        # Create transparent image
        transparent_img = create_transparent_image(pil_img, mask)
        transparent_images.append(transparent_img)
    
    return masks, transparent_images


def mask_out_cards(img_bgr: np.ndarray, polygons: List[np.ndarray], 
                  fill_color: Optional[Tuple[int, int, int]] = None, debug: bool = False) -> Image.Image:
    """Mask out detected color cards by filling regions with background color.
    
    Creates a copy of the input image and fills the detected card regions
    with either a specified color or automatically calculated background color.
    This removes the cards from view while preserving the rest of the image.
    
    Args:
        img_bgr: Input image in OpenCV BGR format as numpy array.
        polygons: List of detected card polygons, each with shape (4, 2).
        fill_color: Optional BGR color tuple to fill masked regions. 
            If None, automatically calculates average background color.
        debug: If True, print debug information about the masking process.
        
    Returns:
        PIL Image in RGB format with card regions filled with the specified
        or calculated background color.
    """
    img_masked = img_bgr.copy()
    
    # If no fill color specified, calculate average background color
    if fill_color is None:
        fill_color = _calculate_background_color(img_bgr, polygons)
        if debug:
            print(f"DEBUG: mask_out_cards - Calculated background color: {fill_color}")
    
    for poly in polygons:
        mask = polygon_to_mask(img_bgr.shape, poly)
        img_masked[mask > 0] = fill_color
    
    return cv2_to_pil(img_masked)


def crop_out_color_cards_for_rmbg(img_bgr: np.ndarray, polygons: List[np.ndarray], 
                                 card_types: List[str], debug: bool = False) -> Image.Image:
    """Crop out color reference cards and area above them for RMBG processing.
    
    Removes ColorChecker 24 and ColorChecker 8 cards along with the area above
    them to improve background removal processing. The checker_cm scale reference
    card is preserved as it may be needed for artifact measurement.
    
    Args:
        img_bgr: Input image in OpenCV BGR format as numpy array.
        polygons: List of detected card polygons, each with shape (4, 2).
        card_types: List of corresponding card type strings for each polygon.
            Expected types: 'colorchecker24', 'colorchecker8', 'checker_cm'.
        debug: If True, print debug information about cropping process.
        
    Returns:
        PIL Image in RGB format with specified color cards and areas above
        them cropped out (filled with background color).
    """
    img_cropped = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    
    # Find color cards (24 and 8) to crop out
    color_card_polygons = []
    for i, card_type in enumerate(card_types):
        if card_type in ['colorchecker24', 'colorchecker8'] and i < len(polygons):
            color_card_polygons.append(polygons[i])
            if debug:
                print(f"DEBUG: crop_out_color_cards_for_rmbg - Found {card_type} card to crop")
    
    if not color_card_polygons:
        if debug:
            print("DEBUG: crop_out_color_cards_for_rmbg - No color cards found, returning original image")
        return cv2_to_pil(img_cropped)
    
    # Create mask for color cards and area above them
    crop_mask = np.zeros((h, w), dtype=np.uint8)
    
    for poly in color_card_polygons:
        # Get bounding rectangle of the card
        x, y, card_w, card_h = cv2.boundingRect(poly.astype(np.int32))
        
        # Extend the area above the card (50% of card height above)
        area_above_height = int(card_h * 0.5)
        extended_y = max(0, y - area_above_height)
        extended_h = card_h + area_above_height
        
        # Create rectangle mask for this card and area above
        cv2.rectangle(crop_mask, (x, extended_y), (x + card_w, y + card_h), 255, -1)
        
        if debug:
            print(f"DEBUG: crop_out_color_cards_for_rmbg - Cropping card at ({x},{y}) size {card_w}x{card_h}, extended area: ({x},{extended_y}) to ({x+card_w},{y+card_h})")
    
    # Fill the masked areas with background color
    background_color = _calculate_background_color(img_bgr, polygons)
    img_cropped[crop_mask > 0] = background_color
    
    if debug:
        print(f"DEBUG: crop_out_color_cards_for_rmbg - Cropped out {len(color_card_polygons)} color cards and areas above them")
    
    return cv2_to_pil(img_cropped)


def _calculate_background_color(img_bgr: np.ndarray, polygons: List[np.ndarray], 
                              debug: bool = False) -> Tuple[int, int, int]:
    """Calculate average background color by sampling areas outside detected cards.
    
    Analyzes the image to determine the dominant background color by creating
    masks for detected cards, dilating them to avoid edge effects, and then
    sampling the remaining background pixels. Falls back to corner sampling
    if insufficient background area is available.
    
    Args:
        img_bgr: Input image in OpenCV BGR format as numpy array.
        polygons: List of detected card polygons to exclude from sampling.
        debug: If True, print debug information about sampling process.
        
    Returns:
        Average background color as BGR tuple (B, G, R) with uint8 values.
    """
    # Create a mask of all detected card areas
    card_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for poly in polygons:
        poly_mask = polygon_to_mask(img_bgr.shape, poly)
        card_mask = cv2.bitwise_or(card_mask, poly_mask)
    
    # Dilate the mask slightly to avoid sampling too close to card edges
    kernel = np.ones((20, 20), np.uint8)
    card_mask_dilated = cv2.dilate(card_mask, kernel, iterations=1)
    
    # Get background pixels (areas not covered by cards)
    background_mask = cv2.bitwise_not(card_mask_dilated)
    
    # Sample background pixels
    background_pixels = img_bgr[background_mask > 0]
    
    if len(background_pixels) == 0:
        # Fallback: sample from image corners if no background pixels found
        h, w = img_bgr.shape[:2]
        corner_size = min(h, w) // 10  # Sample from corners
        corners = [
            img_bgr[:corner_size, :corner_size].reshape(-1, 3),  # Top-left
            img_bgr[:corner_size, -corner_size:].reshape(-1, 3),  # Top-right
            img_bgr[-corner_size:, :corner_size].reshape(-1, 3),  # Bottom-left
            img_bgr[-corner_size:, -corner_size:].reshape(-1, 3)  # Bottom-right
        ]
        background_pixels = np.vstack(corners)
        if debug:
            print("DEBUG: _calculate_background_color - Using corner sampling fallback")
    
    # Calculate average color
    avg_color = np.mean(background_pixels, axis=0).astype(np.uint8)
    if debug:
        print(f"DEBUG: _calculate_background_color - Sampled {len(background_pixels)} background pixels")
    
    return tuple(avg_color)


def process_image_pipeline(image_path: Union[str, Path], template_paths: Optional[List[str]] = None, 
                          max_per_template: int = 2, debug: bool = False) -> Dict[str, Any]:
    """Process image to detect and extract color card information.
    
    Simplified pipeline that detects up to 2 color cards and classifies them using
    a simple rule-based approach: 1 card = ColorChecker 8, 2 cards = larger card
    is ColorChecker 24 and smaller is checker_cm scale reference.
    
    Args:
        image_path: Path to the input image file as string or Path object.
        template_paths: Optional list of template image paths for card detection.
            If None, uses default package templates for colorchecker24, checker_cm,
            and colorchecker8.
        max_per_template: Maximum instances per template (currently ignored, 
            pipeline enforces max 2 total cards).
        debug: If True, print detailed debug information during processing.
        
    Returns:
        Dictionary containing processed image data:
            - original_image: Original image as PIL Image object
            - detected_polygons: List of polygon coordinates (max 2) as numpy arrays
            - card_types: List of detected card type strings (max 2). Types are:
              'colorchecker24', 'colorchecker8', or 'checker_cm'
            - masks: List of PIL mask images (max 2) for detected cards
            - transparent_images: List of PIL images with transparency (max 2)
            - masked_image: PIL Image with detected cards masked out in white
            - extracted_crops: List of extracted card regions as OpenCV arrays (max 2)
            
    Raises:
        RuntimeError: If template loading or CardDetector initialization fails.
        ValueError: If an unexpected number of cards is detected (not 1 or 2).
        FileNotFoundError: If the input image cannot be loaded.
    """
    # Resolve package templates dir
    templates_dir = Path(__file__).parent / "templates"
    default_names = ["colorchecker24.png", "checker_cm.png", "colorchecker8.png"]

    # If caller didn't pass templates, use package templates
    if template_paths is None:
        template_paths = [str(templates_dir / n) for n in default_names]
    else:
        # Resolve provided paths: prefer existing path, otherwise try package templates by name
        resolved = []
        for p in template_paths:
            try:
                if p is None:
                    continue
                pp = Path(p)
                if pp.exists():
                    resolved.append(str(pp))
                else:
                    alt = templates_dir / pp.name
                    if alt.exists():
                        if debug:
                            print(f"DEBUG: Resolved template '{p}' -> '{alt}'")
                        resolved.append(str(alt))
                    else:
                        # keep original (will raise later) but avoid leading-root style defaults causing immediate imread warnings
                        resolved.append(str(pp))
            except Exception:
                resolved.append(str(p))
        template_paths = resolved

    # Load image
    img_bgr = load_image_any(image_path)
    original_pil = cv2_to_pil(img_bgr)

    # Check if the templates with the names "checker_cm.png", "colorchecker8.png", "colorchecker24.png" are readable
    names = ["checker_cm.png", "colorchecker8.png", "colorchecker24.png"]
    for name in names:
        if not any(name in Path(p).name for p in template_paths):
            if debug:
                print(f"DEBUG: process_image_pipeline - Warning: Template '{name}' not found in provided template paths")

    # Detect cards (max 2)
    try:
        detector = CardDetector(template_paths)
    except Exception as e:
        if debug:
            print(f"DEBUG: process_image_pipeline - Failed to initialize CardDetector with templates={template_paths}: {e}")
        # Raise a clearer error for callers
        raise RuntimeError(f"process_image_pipeline: failed to load templates or initialize CardDetector: {e}") from e
    polygons, crops, combined_mask = detector.extract_all(img_bgr, debug=debug)
    
    # Simplified card type detection based on number of cards
    num_cards = len(polygons)
    if debug:
        print(f"DEBUG: process_image_pipeline - Detected {num_cards} cards")
    
    if num_cards == 1:
        card_types = ['colorchecker8']
        if debug:
            print("DEBUG: process_image_pipeline - 1 card detected -> ColorChecker 8")
    elif num_cards == 2:
        # Assign types based on size: larger = ColorChecker 24, smaller = checker_cm
        card_types = _assign_card_types_by_size(polygons)
        if debug:
            print("DEBUG: process_image_pipeline - 2 cards detected -> ColorChecker 24 + checker_cm")
    else:
        # Fallback for 0
        card_types = ['colorchecker8'] if num_cards > 0 else []
        # Raise error
        raise ValueError(f"process_image_pipeline - {num_cards} cards detected")
    
    # Create masks and transparent images
    masks, transparent_images = create_card_masks_and_transparent(img_bgr, polygons, debug=debug)
    
    # Create masked image
    masked_image = mask_out_cards(img_bgr, polygons, debug=debug)
    
    return {
        'original_image': original_pil,
        'detected_polygons': polygons,
        'card_types': card_types,
        'masks': masks,
        'transparent_images': transparent_images,
        'masked_image': masked_image,
        'extracted_crops': crops
    }


def setup_rmbg_pipeline() -> Any:
    """Setup RMBG-1.4 pipeline for background removal.
    
    Initializes the RMBG-1.4 transformer pipeline for automated background
    removal. This is a machine learning model that can segment foreground
    objects from background in images.
    
    Returns:
        Configured RMBG pipeline object, or None if transformers library
        is not available.
        
    Raises:
        ImportError: If transformers library is not installed.
    """
    try:
        from transformers import pipeline
        pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, use_fast=True)
        return pipe
    except ImportError:
        raise ImportError("transformers library is required for RMBG pipeline. Install with: pip install transformers")


def remove_background_and_generate_mask(masked_pil_image: Image.Image, rmbg_pipe: Optional[Any] = None, 
                                       cards_mask: Optional[np.ndarray] = None, 
                                       cropped_for_rmbg: Optional[Image.Image] = None, 
                                       debug: bool = False) -> Tuple[Image.Image, Image.Image]:
    """Apply RMBG background removal with overlap filtering and double-pass processing.
    
    Performs background removal using RMBG-1.4 model on the masked image.
    Includes overlap filtering to avoid conflicts with detected cards and
    applies a two-pass approach: first pass on cropped image, second pass
    on white background for better results.
    
    Args:
        masked_pil_image: PIL Image with detected cards already masked out.
        rmbg_pipe: Optional pre-configured RMBG pipeline. If None, creates new one.
        cards_mask: Optional binary mask of card areas to avoid overlap conflicts.
        cropped_for_rmbg: Optional cropped image with color cards removed for
            improved RMBG processing.
        debug: If True, print debug information during processing.
        
    Returns:
        Tuple of (result_image, mask_image) as PIL Images where result_image
        contains the foreground object with transparent background and mask_image
        is the binary segmentation mask.
    """
    if rmbg_pipe is None:
        rmbg_pipe = setup_rmbg_pipeline()
    
    # Use cropped image for RMBG if available, otherwise use masked image
    rmbg_input = cropped_for_rmbg if cropped_for_rmbg is not None else masked_pil_image
    if debug:
        print(f"DEBUG: remove_background_and_generate_mask - Using {'cropped' if cropped_for_rmbg is not None else 'masked'} image for RMBG")
    
    # Apply RMBG to get segmentation results
    rmbg_results = rmbg_pipe(rmbg_input)
    
    # Handle different RMBG output formats
    if isinstance(rmbg_results, list) and len(rmbg_results) > 0:
        # Multiple segments detected
        if debug:
            print(f"DEBUG: apply_rmbg_to_masked_image - Found {len(rmbg_results)} RMBG segments")
        
        # Filter out segments that overlap with cards
        if cards_mask is not None:
            filtered_results = _filter_rmbg_overlaps(rmbg_results, cards_mask)
            if debug:
                print(f"DEBUG: apply_rmbg_to_masked_image - Filtered to {len(filtered_results)} non-overlapping segments")
        else:
            filtered_results = rmbg_results
        
        # Use the largest non-overlapping segment
        if filtered_results:
            best_segment = _select_best_rmbg_segment(filtered_results)
            result_image = best_segment['image']
            mask_image = best_segment['mask']
        else:
            # Fallback: use first result if no non-overlapping segments
            if debug:
                print("DEBUG: apply_rmbg_to_masked_image - No non-overlapping segments, using first result")
            result_image = rmbg_results[0]['image']
            mask_image = rmbg_results[0]['mask']
    else:
        # Single result or different format
        result_image = rmbg_results['image'] if isinstance(rmbg_results, dict) else rmbg_results
        mask_image = rmbg_results['mask'] if isinstance(rmbg_results, dict) else None

    # Ensure result_image is a PIL Image
    if hasattr(result_image, "convert"):
        pil_result = result_image.convert("RGBA")
    else:
        pil_result = Image.fromarray(result_image).convert("RGBA")

    # Create a white background
    white_bg = Image.new("RGBA", pil_result.size, (255, 255, 255, 255))
    # Composite the result image over the white background
    composited = Image.alpha_composite(white_bg, pil_result)

    # Run RMBG again on the composited image
    if debug:
        print("DEBUG: Running RMBG again on image composited over white background")
    rmbg_results2 = rmbg_pipe(composited)

    # Handle output format for the second RMBG run
    if isinstance(rmbg_results2, list) and len(rmbg_results2) > 0:
        # Use the largest segment
        best_segment2 = _select_best_rmbg_segment(rmbg_results2)
        final_result_image = best_segment2['image']
        final_mask_image = best_segment2['mask']
    elif isinstance(rmbg_results2, dict):
        final_result_image = rmbg_results2.get('image', rmbg_results2)
        final_mask_image = rmbg_results2.get('mask', None)
    else:
        final_result_image = rmbg_results2
        final_mask_image = None

    # Post-process: split cutouts, remove those touching edges or overlapping card masks
    try:
        final_result_image, final_mask_image = _split_and_filter_cutouts(
            final_result_image,
            final_mask_image,
            cards_mask=cards_mask,
            alpha_threshold=200
        )
    except Exception as e:
        if debug:
            print(f"DEBUG: _split_and_filter_cutouts failed with error: {e}")

    return final_result_image, final_mask_image


def _filter_rmbg_overlaps(rmbg_results: List[Dict[str, Any]], cards_mask: np.ndarray, 
                         debug: bool = False) -> List[Dict[str, Any]]:
    """Filter out RMBG segments that overlap significantly with card areas.
    
    Analyzes each RMBG segmentation result and removes those that have
    substantial overlap with detected color card regions to avoid
    conflicting segmentations.
    
    Args:
        rmbg_results: List of RMBG segmentation result dictionaries, each
            containing 'mask' and 'image' keys.
        cards_mask: Binary mask of card areas as numpy array where card
            regions are marked with non-zero values.
        debug: If True, print overlap statistics for each segment.
        
    Returns:
        List of filtered RMBG result dictionaries with card overlaps removed.
    """
    filtered_results = []
    
    for i, result in enumerate(rmbg_results):
        if 'mask' in result:
            # Convert mask to numpy array if it's a PIL image
            if hasattr(result['mask'], 'convert'):
                mask_array = np.array(result['mask'].convert('L'))
            else:
                mask_array = result['mask']
            
            # Ensure masks are the same size
            if mask_array.shape != cards_mask.shape:
                if debug:
                    print(f"DEBUG: _filter_rmbg_overlaps - Resizing mask {i} from {mask_array.shape} to {cards_mask.shape}")
                mask_array = cv2.resize(mask_array, (cards_mask.shape[1], cards_mask.shape[0]))
            
            # Calculate overlap with cards
            overlap = cv2.bitwise_and(mask_array, cards_mask)
            overlap_ratio = np.sum(overlap > 0) / np.sum(mask_array > 0) if np.sum(mask_array > 0) > 0 else 0
            
            if debug:
                print(f"DEBUG: _filter_rmbg_overlaps - Segment {i}: overlap ratio = {overlap_ratio:.3f}")
            
            # Keep segments with low overlap (less than 30%)
            if overlap_ratio < 0.3:
                filtered_results.append(result)
            else:
                if debug:
                    print(f"DEBUG: _filter_rmbg_overlaps - Rejected segment {i} due to high overlap ({overlap_ratio:.3f})")
        else:
            # If no mask available, keep the result
            filtered_results.append(result)
    
    return filtered_results


def _select_best_rmbg_segment(rmbg_results: List[Dict[str, Any]], debug: bool = False) -> Optional[Dict[str, Any]]:
    """Select the best RMBG segment from multiple candidates.
    
    Evaluates multiple RMBG segmentation results and selects the most suitable
    one based on size, quality metrics, and geometric properties. Prefers
    larger, more centrally located segments.
    
    Args:
        rmbg_results: List of filtered RMBG result dictionaries to evaluate.
        debug: If True, print scoring details for each candidate segment.
        
    Returns:
        Best RMBG result dictionary containing 'mask' and 'image' keys,
        or None if no results are provided.
    """
    if not rmbg_results:
        return None
    
    if len(rmbg_results) == 1:
        return rmbg_results[0]
    
    # Score segments based on size and quality
    best_score = -1
    best_result = rmbg_results[0]
    
    for result in rmbg_results:
        score = 0
        
        # Size score (prefer larger segments)
        if 'mask' in result:
            mask_array = np.array(result['mask'].convert('L')) if hasattr(result['mask'], 'convert') else result['mask']
            area = np.sum(mask_array > 0)
            score += area / 10000  # Normalize by 10k pixels
        
        # Quality score (prefer segments with higher confidence if available)
        if 'score' in result:
            score += result['score'] * 100
        
        if debug:
            print(f"DEBUG: _select_best_rmbg_segment - Segment score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_result = result
    
    if debug:
        print(f"DEBUG: _select_best_rmbg_segment - Selected segment with score: {best_score:.3f}")
    return best_result


def _split_and_filter_cutouts(result_image: Union[Image.Image, np.ndarray], 
                             mask_image: Optional[Union[Image.Image, np.ndarray]], 
                             cards_mask: Optional[np.ndarray] = None, 
                             alpha_threshold: int = 200, 
                             debug: bool = False) -> Tuple[Image.Image, Image.Image]:
    """Split foreground cutouts and filter components touching edges or overlapping cards.
    
    Performs connected component analysis on the segmented foreground to split
    multiple objects, then filters out components that touch image edges or
    overlap significantly with detected card areas. Also removes semi-transparent
    connector regions by applying alpha thresholding.
    
    Args:
        result_image: Segmented result as PIL Image or numpy array (RGBA or RGB).
        mask_image: Optional segmentation mask as PIL Image or numpy array.
            If None, mask will be derived from alpha channel.
        cards_mask: Optional binary mask of card areas with shape (H, W).
            Non-zero values indicate card regions to avoid.
        alpha_threshold: Alpha channel threshold (0-255) to remove semi-transparent
            connector regions. Defaults to 200.
        debug: If True, print detailed debug information about filtering process.
        
    Returns:
        Tuple of (filtered_result_image, filtered_mask_image) as PIL Images
        in RGBA and grayscale ('L') modes respectively.
    """
    # Normalize result image to RGBA PIL
    if hasattr(result_image, "convert"):
        res_pil = result_image.convert("RGBA")
    else:
        res_pil = Image.fromarray(result_image).convert("RGBA")

    rgba = np.array(res_pil)
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    # Normalize mask to ndarray if provided; otherwise derive from alpha
    if mask_image is not None:
        if hasattr(mask_image, "convert"):
            mask_arr = np.array(mask_image.convert('L'))
        else:
            mask_arr = mask_image
    else:
        mask_arr = alpha

    # Threshold to remove semi-transparent connectors
    fg_bin = (mask_arr.astype(np.uint8) > alpha_threshold).astype(np.uint8)

    if np.sum(fg_bin) == 0:
        # Nothing to keep
        empty_alpha = np.zeros_like(alpha, dtype=np.uint8)
        rgba[:, :, 3] = empty_alpha
        return Image.fromarray(rgba, mode="RGBA"), Image.fromarray(empty_alpha, mode='L')

    # Prepare cards mask aligned to result size, if provided
    cards_bin = None
    if cards_mask is not None:
        cards_arr = cards_mask
        if hasattr(cards_arr, "shape"):
            # Ensure 2D
            if cards_arr.ndim == 3:
                cards_arr = cards_arr[:, :, 0]
            if cards_arr.shape[0] != h or cards_arr.shape[1] != w:
                if debug:
                    print(f"DEBUG: _split_and_filter_cutouts - Resizing cards_mask from {cards_arr.shape} to {(h, w)}")
                cards_arr = cv2.resize(cards_arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            cards_bin = (cards_arr.astype(np.uint8) > 0).astype(np.uint8)

    # Connected components on foreground
    num_labels, labels = cv2.connectedComponents(fg_bin, connectivity=8)

    kept_mask = np.zeros_like(fg_bin, dtype=np.uint8)

    for label_id in range(1, num_labels):
        comp = (labels == label_id)

        # Edge-touch check
        touches_edge = comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any()
        if touches_edge:
            if debug:
                print(f"DEBUG: _split_and_filter_cutouts - Dropping component {label_id} touching edge")
            continue

        # Overlap with cards check
        if cards_bin is not None:
            if (comp & (cards_bin > 0)).any():
                if debug:
                    print(f"DEBUG: _split_and_filter_cutouts - Dropping component {label_id} overlapping cards mask")
                continue

        kept_mask[comp] = 1

    # Apply kept mask to alpha; remove semi-transparent connectors by thresholding
    new_alpha = np.where(kept_mask > 0, alpha, 0).astype(np.uint8)
    rgba[:, :, 3] = new_alpha

    filtered_res = Image.fromarray(rgba, mode="RGBA")
    filtered_mask = Image.fromarray((kept_mask * 255).astype(np.uint8), mode='L')

    if debug:
        print(f"DEBUG: _split_and_filter_cutouts - Kept {np.unique(labels[kept_mask>0]).size} components; removed {num_labels-1 - np.unique(labels[kept_mask>0]).size}")

    return filtered_res, filtered_mask

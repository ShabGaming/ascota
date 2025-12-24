"""
Artifact face area calculation using reference card for scale.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional, Union, List, Any
import math

def calculate_pp_cm_checker_card(image: np.ndarray, debug: bool = False) -> Tuple[float, Optional[Image.Image]]:
    """Calculate pixels per cm from checkerboard pattern squares.
    
    Analyzes a checkerboard image to detect white squares and calculate the
    pixels per centimeter scale factor, assuming each square is 1 cm².
    
    Args:
        image: Input image as numpy array (BGR or grayscale).
        debug: If True, return debug visualization showing detected squares.
        
    Returns:
        Tuple of (pixels_per_cm, debug_image). The debug_image is None if
        debug=False, otherwise contains visualization of detected squares.
    """
    def _calculate_checker_square_area_checker_card(image: np.ndarray, debug: bool = False) -> Tuple[float, Optional[Image.Image]]:
        """
        Calculate the average area of 'white' squares in a checker/card style image (Only works for checker_ppi styled images).
        White squares are identified by higher mean gray intensity inside the quad
        relative to the surrounding background, not by binary color.

        Returns:
            avg_area_pixels, optional debug PIL image
        """
        # --- 1) Grayscale + local contrast normalization ---
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # CLAHE helps under uneven lighting; small median blur removes salt noise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        gray_eq = cv2.medianBlur(gray_eq, 3)

        # --- 2) Full binarization (global Otsu), and try both polarities ---
        _, bin0 = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin1 = cv2.bitwise_not(bin0)

        H, W = gray.shape[:2]
        img_area = float(H * W)

        def _detect(binary_img: np.ndarray) -> Tuple[float, List[np.ndarray], List[bool]]:
            # Morphology to close tiny gaps in borders, then remove specks
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bw = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, k, iterations=2)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

            # Use TREE to keep internal contours when there are borders/frames
            contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            areas_white = []
            quads = []
            quad_is_white = []

            # Dynamic area limits (tune if needed)
            min_area = max(64.0, 0.00005 * img_area)   # 0.005% of image
            max_area = 0.30 * img_area                 # up to 30% of image

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                # Polygonal approximation + convexity
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue

                # Rotated rectangle properties (handles rotated squares)
                rect = cv2.minAreaRect(cnt)
                (w, h) = rect[1]
                if w == 0 or h == 0:
                    continue
                aspect = min(w, h) / max(w, h)
                if aspect < 0.80:  # too elongated to be a square-ish patch
                    continue

                # Rectangularity / solidity to reject jagged shapes
                rect_area = w * h
                if rect_area <= 0:
                    continue
                rectangularity = area / rect_area
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / (hull_area + 1e-6)

                if rectangularity < 0.70 or solidity < 0.90:
                    continue

                # Classify "white" using mean gray inside vs outside
                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
                mean_inside = cv2.mean(gray, mask=mask)[0]

                # Estimate local background by dilating mask a bit and subtracting inside
                dil = cv2.dilate(mask, k, iterations=5)
                ring = cv2.subtract(dil, mask)
                # Fallback: if ring empty (near edges), compare to global mean
                if cv2.countNonZero(ring) > 0:
                    mean_bg = cv2.mean(gray, mask=ring)[0]
                else:
                    mean_bg = float(np.mean(gray))

                is_white = mean_inside > mean_bg

                quads.append(approx)
                quad_is_white.append(is_white)
                if is_white:
                    areas_white.append(area)

            avg_area = float(np.mean(areas_white)) if len(areas_white) > 0 else 0.0
            return avg_area, quads, quad_is_white

        avg0, quads0, flags0 = _detect(bin0)
        avg1, quads1, flags1 = _detect(bin1)

        # Choose the polarity with more detected quads; tie-break by higher avg
        if len(quads1) > len(quads0) or (len(quads1) == len(quads0) and avg1 > avg0):
            avg_area = avg1
            quads, flags = quads1, flags1
            chosen_bin = bin1
        else:
            avg_area = avg0
            quads, flags = quads0, flags0
            chosen_bin = bin0

        debug_image = None
        if debug:
            # Draw detections (green = white square, red = non-white)
            vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(vis)
            draw = ImageDraw.Draw(pil)
            for quad, is_white in zip(quads, flags):
                pts = [(int(p[0][0]), int(p[0][1])) for p in quad]
                color = "lime" if is_white else "red"
                draw.polygon(pts, outline=color, width=2)
            # Optional: overlay small preview of the chosen binary map in a corner
            try:
                small = cv2.resize(chosen_bin, (min(200, W//4), min(200, H//4)))
                small_rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
                small_pil = Image.fromarray(small_rgb)
                pil.paste(small_pil, (10, 10))
            except Exception:
                pass
            debug_image = pil

        return avg_area, debug_image

    pixels_per_cm, debug_image = _calculate_checker_square_area_checker_card(image, debug=debug)

    pp_cm = np.sqrt(pixels_per_cm)
    if debug:
        print(f'DEBUG: "calculate_pp_cm_checker_card" - Raw value: {pixels_per_cm} | calculate_pp_cm_checker_card')
    # Round to nearest 10 (if 5 or more round up else round down)
    pp_cm = math.floor(pp_cm / 10) * 10 if pp_cm % 10 < 5 else math.ceil(pp_cm / 10) * 10

    return pp_cm, debug_image

def find_circle_centers_8_hybrid_card(image: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
    """Find the center points of three circles on an '8 hybrid card'.
    
    Detects three circular reference points on a ColorChecker 8 card that form
    a 50mm x 20mm rectangle. Returns the center coordinates of these circles,
    which can be used for manual correction if automatic detection fails.
    
    Args:
        image: Input BGR image as numpy array.
        debug: If True, return debug visualization showing detected circles
            and selected reference points with center markers.
            
    Returns:
        Tuple of (centers, debug_image) where:
        - centers: NumPy array of shape (3, 2) with (x, y) center coordinates,
        or None if detection fails
        - debug_image: PIL Image with visualization, or None if debug=False
    """
    
    def _validate_circle(gray: np.ndarray, cx: int, cy: int, r: int) -> bool:
        """Validate that a detected circle is actually a good circular target."""
        H, W = gray.shape[:2]
        
        # Check bounds
        if cx - r < 0 or cx + r >= W or cy - r < 0 or cy + r >= H:
            return False
        
        if r < 3:
            return False
        
        # Extract ROI
        x1, y1 = max(0, cx - r - 2), max(0, cy - r - 2)
        x2, y2 = min(W, cx + r + 2), min(H, cy + r + 2)
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Create circular mask
        center_x, center_y = cx - x1, cy - y1
        Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Check edge strength at circle boundary
        edge_mask = (np.abs(dist_from_center - r) < 2).astype(np.uint8)
        if np.sum(edge_mask) < r * 2:  # Need sufficient edge pixels
            return False
        
        edge_pixels = roi[edge_mask > 0]
        if len(edge_pixels) == 0:
            return False
        
        # Edge should have good contrast (high variance or strong gradient)
        edge_variance = np.var(edge_pixels)
        edge_mean = np.mean(edge_pixels)
        
        # Check inside vs outside contrast
        inside_mask = (dist_from_center < r * 0.7).astype(np.uint8)
        outside_mask = ((dist_from_center > r * 1.3) & (dist_from_center < r * 2.0)).astype(np.uint8)
        
        if np.sum(inside_mask) > 0 and np.sum(outside_mask) > 0:
            inside_mean = np.mean(roi[inside_mask > 0])
            outside_mean = np.mean(roi[outside_mask > 0])
            contrast = abs(inside_mean - outside_mean)
            
            # Good circle should have reasonable contrast and edge strength
            if contrast > 20 and edge_variance > 100:
                return True
        
        return False
    
    def find_circles_robust(gray: np.ndarray) -> np.ndarray:
        """Find circles using improved preprocessing and filtering."""
        H, W = gray.shape[:2]
        img_diag = math.sqrt(H * H + W * W)
        
        # Estimate circle radius based on image size (targets are typically 2-5% of image diagonal)
        minR = max(5, int(0.015 * img_diag))
        maxR = max(minR + 1, int(0.05 * img_diag))
        minDist = int(0.10 * min(H, W))  # Minimum distance between circle centers
        
        all_circles = []
        
        # Strategy 1: Use Canny edge detection with HoughCircles (most accurate)
        for clip_limit in [2.0, 3.0]:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            g_eq = clahe.apply(gray)
            
            # Light blur to reduce noise
            g_blur = cv2.GaussianBlur(g_eq, (5, 5), 0)
            
            # Canny edge detection for better circle detection
            edges = cv2.Canny(g_blur, 50, 150)
            
            # Try HoughCircles with different parameters
            for param2 in [20, 25, 30, 35]:
                circles = cv2.HoughCircles(
                    edges, cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=minDist,
                    param1=100,  # Upper threshold for Canny
                    param2=param2,  # Accumulator threshold (lower = more circles)
                    minRadius=minR,
                    maxRadius=maxR
                )
                
                if circles is not None:
                    circles_float = circles[0].astype(np.float32)
                    for x, y, r in circles_float:
                        # Validate circle quality
                        if _validate_circle(gray, int(x), int(y), int(r)):
                            all_circles.append([x, y, r])
        
        # Strategy 2: Direct HoughCircles on enhanced grayscale (fallback)
        if len(all_circles) < 3:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g_eq = clahe.apply(gray)
            g_blur = cv2.GaussianBlur(g_eq, (5, 5), 0)
            
            for param2 in [25, 30, 35]:
                circles = cv2.HoughCircles(
                    g_blur, cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=minDist,
                    param1=100,
                    param2=param2,
                    minRadius=minR,
                    maxRadius=maxR
                )
                
                if circles is not None:
                    circles_float = circles[0].astype(np.float32)
                    for x, y, r in circles_float:
                        if _validate_circle(gray, int(x), int(y), int(r)):
                            all_circles.append([x, y, r])
        
        # Strategy 3: Try inverted image (for light circles on dark background)
        if len(all_circles) < 3:
            gray_inv = cv2.bitwise_not(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g_eq = clahe.apply(gray_inv)
            g_blur = cv2.GaussianBlur(g_eq, (5, 5), 0)
            edges = cv2.Canny(g_blur, 50, 150)
            
            circles = cv2.HoughCircles(
                edges, cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=minDist,
                param1=100,
                param2=25,
                minRadius=minR,
                maxRadius=maxR
            )
            
            if circles is not None:
                circles_float = circles[0].astype(np.float32)
                for x, y, r in circles_float:
                    if _validate_circle(gray, int(x), int(y), int(r)):
                        all_circles.append([x, y, r])
        
        # Remove duplicates and merge similar circles
        if len(all_circles) > 0:
            circles_array = np.array(all_circles, dtype=np.float32)
            unique_circles = []
            used = set()
            
            # Sort by radius (prefer medium-sized circles)
            radii = circles_array[:, 2]
            sorted_indices = np.argsort(np.abs(radii - np.median(radii)))
            
            for idx in sorted_indices:
                if idx in used:
                    continue
                
                x, y, r = circles_array[idx]
                unique_circles.append([x, y, r])
                used.add(idx)
                
                # Mark nearby circles as duplicates
                for j in range(len(circles_array)):
                    if j in used:
                        continue
                    x2, y2, r2 = circles_array[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    # If centers are very close and radii are similar, merge them
                    if dist < max(r, r2) * 0.3 and abs(r - r2) < max(r, r2) * 0.2:
                        used.add(j)
            
            return np.array(unique_circles, dtype=np.float32) if unique_circles else np.zeros((0, 3), dtype=np.float32)
        
        return np.zeros((0, 3), dtype=np.float32)
    
    def select_three_by_geometry(circles: np.ndarray) -> Optional[np.ndarray]:
        """Select three circles that best match the expected geometry.
        
        Args:
            circles: Array of detected circles with shape (N, 3) for (x, y, radius).
            
        Returns:
            Array of shape (3, 2) for the best three circle centers, or None if no
            suitable triplet is found.
        """
        n = len(circles)
        if n < 3:
            return None

        best = None
        best_err = 1e9
        ratio_ms = 2.5                 # width/height = 50mm / 20mm
        ratio_lm = math.sqrt(29) / 5   # diag/width = sqrt(29)/5 ≈ 1.077

        # Try every triplet (N is small after Hough)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    P = circles[[i, j, k], :3]  # (x, y, r)

                    # Radii of the three should be similar (within 20% of median)
                    r_med = np.median(P[:, 2])
                    if r_med <= 0:
                        continue
                    r_variation = np.max(np.abs(P[:, 2] - r_med)) / r_med
                    if r_variation > 0.20:  # Stricter: was 0.25
                        continue

                    # Pairwise center distances
                    C = P[:, :2]
                    d = np.array([
                        np.linalg.norm(C[0] - C[1]),
                        np.linalg.norm(C[1] - C[2]),
                        np.linalg.norm(C[0] - C[2]),
                    ])
                    d.sort()
                    s, m, l = d  # s≈2cm, m≈5cm, l≈sqrt(29)≈5.385cm
                    
                    if s < 1 or m < 1 or l < 1:
                        continue
                    
                    # Ensure reasonable size relationships (s < m < l)
                    if not (s < m < l):
                        continue

                    # Check expected ratios with tighter tolerances
                    e1 = abs(m / s - ratio_ms) / ratio_ms  # width/height error
                    e2 = abs(l / m - ratio_lm) / ratio_lm   # diag/width error
                    
                    # Reject if ratios are too far off (more than 30% error)
                    if e1 > 0.30 or e2 > 0.30:
                        continue

                    # Find the right angle (should be close to 90 degrees)
                    angles = []
                    for t in range(3):
                        v1 = C[(t + 1) % 3] - C[t]
                        v2 = C[(t + 2) % 3] - C[t]
                        den = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
                        cosang = float(np.clip(np.dot(v1, v2) / den, -1.0, 1.0))
                        angles.append(math.degrees(math.acos(cosang)))
                    
                    # Find the angle closest to 90 degrees
                    ang_err = min(abs(a - 90) for a in angles)
                    
                    # Reject if no angle is reasonably close to 90 degrees (within 25 degrees)
                    if ang_err > 25:
                        continue

                    # Combined error (weighted)
                    err = e1 + e2 + 0.02 * ang_err
                    
                    if err < best_err:
                        best_err = err
                        best = C.copy()  # Return only centers, not full circle data

        # Only return if error is acceptable (within 20% total error)
        if best is not None and best_err < 0.20:
            return best
        
        return None

    # --- preprocess ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Try both polarities
    circles = find_circles_robust(gray)
    if len(circles) == 0:
        circles = find_circles_robust(cv2.bitwise_not(gray))

    chosen_centers = select_three_by_geometry(circles)
    debug_img = None
    
    if debug:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else \
              cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        
        # Draw all detected circle candidates with center points
        for x, y, r in circles:
            # Draw circle outline
            draw.ellipse([x - r, y - r, x + r, y + r], outline="#88FFFF", width=1)
            # Draw center point with crosshairs
            size = max(5, int(r * 0.2))
            draw.line([x - size, y, x + size, y], fill="#88FFFF", width=2)
            draw.line([x, y - size, x, y + size], fill="#88FFFF", width=2)
            # Small filled circle at center
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="#88FFFF")
        
        # Highlight the selected triplet with prominent markers
        if chosen_centers is not None:
            for i, (x, y) in enumerate(chosen_centers):
                # Find corresponding circle for radius
                r = 10  # Default if not found
                for cx, cy, cr in circles:
                    if abs(cx - x) < 1 and abs(cy - y) < 1:
                        r = cr
                        break
                
                # Draw circle outline
                draw.ellipse([x - r, y - r, x + r, y + r], outline="lime", width=3)
                # Draw prominent center crosshairs
                size = max(8, int(r * 0.3))
                draw.line([x - size, y, x + size, y], fill="red", width=3)
                draw.line([x, y - size, x, y + size], fill="red", width=3)
                # Filled circle at center
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="red")
                # Label with coordinates
                label = f"({int(x)}, {int(y)})"
                draw.text((int(x) + 5, int(y) - 10), label, fill="yellow")
        
        debug_img = pil
    
    return chosen_centers, debug_img

def calculate_pp_cm_from_centers(centers: np.ndarray, image: Optional[np.ndarray] = None, 
                                  debug: bool = False) -> Tuple[float, Optional[Image.Image]]:
    """Calculate pixels per cm from center points of three circles.
    
    Takes the center coordinates of three circles from an 8 hybrid card and
    calculates the pixels-per-centimeter scale factor based on the expected
    geometry (50mm x 20mm rectangle).
    
    Args:
        centers: NumPy array of shape (3, 2) with (x, y) center coordinates.
        image: Optional image for debug visualization. If None and debug=True,
            visualization will be minimal.
        debug: If True, return debug visualization showing the calculation.
            
    Returns:
        Tuple of (pixels_per_cm, debug_image). Returns 0.0 for pixels_per_cm
        if calculation fails. Debug_image is None if debug=False.
        
    Raises:
        ValueError: If centers array has incorrect shape.
    """
    if centers is None or centers.shape != (3, 2):
        raise ValueError("centers must be a numpy array of shape (3, 2)")
    
    # Compute px/cm from width, height, and diagonal; robust aggregate via median
    dists = np.array([
        np.linalg.norm(centers[0] - centers[1]),
        np.linalg.norm(centers[1] - centers[2]),
        np.linalg.norm(centers[0] - centers[2]),
    ])
    s, m, l = np.sort(dists)                # s≈2cm, m≈5cm, l≈sqrt(29)≈5.385cm
    px_per_cm_candidates = np.array([
        m / 5.0,
        s / 2.0,
        l / math.sqrt(29.0)
    ])
    px_per_cm = float(np.median(px_per_cm_candidates))
    
    debug_img = None
    if debug and image is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else \
              cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        
        # Draw the three centers with connecting lines
        for i, (x, y) in enumerate(centers):
            # Draw center point
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="lime", outline="red", width=2)
            # Draw crosshairs
            draw.line([x - 10, y, x + 10, y], fill="red", width=2)
            draw.line([x, y - 10, x, y + 10], fill="red", width=2)
            # Label
            label = f"P{i+1}"
            draw.text((int(x) + 8, int(y) - 15), label, fill="yellow")
        
        # Draw connecting lines
        for i in range(3):
            x1, y1 = centers[i]
            x2, y2 = centers[(i + 1) % 3]
            draw.line([(x1, y1), (x2, y2)], fill="cyan", width=2)
        
        # Label px/cm
        label = f"px/cm ~ {px_per_cm:.2f}"
        w = 8 * len(label)
        draw.rectangle([10, 10, 10 + w, 35], fill=(0, 0, 0, 150))
        draw.text((14, 14), label, fill="white")
        
        debug_img = pil
    
    if debug:
        print(f'DEBUG: "calculate_pp_cm_from_centers" - Raw value: {px_per_cm} | calculate_pp_cm_from_centers')
    
    # Round to nearest 10 (if 5 or more round up else round down)
    px_per_cm = math.floor(px_per_cm / 10) * 10 if px_per_cm % 10 < 5 else math.ceil(px_per_cm / 10) * 10

    return px_per_cm, debug_img

def artifact_face_size(binary_mask: Union[np.ndarray, Image.Image], pixels_per_cm: float, 
                      debug: bool = False) -> float:
    """Calculate the face area of an artifact from a binary mask using a scale factor.
    
    Determines the area of an artifact by counting foreground pixels in a binary mask
    and converting to square centimeters using the provided pixels-per-centimeter scale factor.
    
    Args:
        binary_mask: Binary mask image where foreground pixels are non-zero.
            Can be PIL Image or numpy array. Supports:
            - Binary mask (0/1 or 0/255): Foreground pixels are non-zero
            - Grayscale: Foreground pixels are non-zero
            The mask should be a clean binary mask from imaging.py's remove_background_mask.
        pixels_per_cm: Scale factor in pixels per centimeter. Must be > 0.
        debug: If True, print detailed debug information about processing steps.
        
    Returns:
        Face area of the artifact in square centimeters (cm²).
        
    Raises:
        ValueError: If pixels_per_cm is invalid or mask format is unsupported.
    """
    # Validate pixels_per_cm
    if pixels_per_cm <= 0:
        raise ValueError(f"pixels_per_cm must be > 0, got {pixels_per_cm}")

    # Convert PIL Image to numpy array if needed
    if hasattr(binary_mask, 'convert'):  # PIL Image
        mask_array = np.array(binary_mask)
        if debug:
            print(f'DEBUG: "artifact_face_size" - Converted PIL Image to numpy array, shape: {mask_array.shape}')
    else:
        mask_array = binary_mask  # Already numpy array
        if debug:
            print(f'DEBUG: "artifact_face_size" - Using existing numpy array, shape: {mask_array.shape}')
    
    # Ensure mask is 2D (grayscale/binary)
    if mask_array.ndim == 3:
        # Multi-channel image: convert to grayscale first
        if mask_array.shape[2] == 4:
            # RGBA: use alpha channel
            mask_array = mask_array[:, :, 3]
            if debug:
                print('DEBUG: "artifact_face_size" - Extracted alpha channel from RGBA image')
        elif mask_array.shape[2] == 3:
            # RGB: convert to grayscale
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
            if debug:
                print('DEBUG: "artifact_face_size" - Converted RGB to grayscale')
        else:
            raise ValueError(f"Unsupported image shape: {mask_array.shape}. Expected 2D (binary/grayscale) or 3D (RGB/RGBA).")
    elif mask_array.ndim != 2:
        raise ValueError(f"Unsupported image shape: {mask_array.shape}. Expected 2D (binary/grayscale) or 3D (RGB/RGBA).")
    
    # Create binary mask (0 or 1) - handle both 0/1 and 0/255 formats
    if mask_array.dtype == np.uint8:
        # Normalize to 0/1: if max is > 1, assume 0/255 format
        if mask_array.max() > 1:
            mask = (mask_array > 0).astype(np.uint8)
        else:
            mask = mask_array.astype(np.uint8)
    else:
        # For other dtypes, just check if > 0
        mask = (mask_array > 0).astype(np.uint8)
    
    if debug:
        print(f'DEBUG: "artifact_face_size" - Binary mask: {np.sum(mask > 0)} foreground pixels (out of {mask.size} total)')
            
    # Count foreground pixels in the binary mask
    artifact_pixels = cv2.countNonZero(mask)

    # Convert pixels to cm² using the provided pixels per cm
    # pixels_per_cm is pixels per cm, so pixels_per_cm² is pixels per cm²
    if debug:
        print(f'DEBUG: "artifact_face_size" - Artifact pixels: {artifact_pixels}, px/cm: {pixels_per_cm:.2f}')
    face_area_cm2 = artifact_pixels / (pixels_per_cm * pixels_per_cm)

    return face_area_cm2

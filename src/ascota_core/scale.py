"""
Artifact face area calculation using reference card for scale.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional, Union, List, Any
import math

def calculate_pp_cm_checker_cm(image: np.ndarray, debug: bool = False) -> Tuple[float, Optional[Image.Image]]:
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
    def _calculate_checker_square_area_checker_cm(image: np.ndarray, debug: bool = False) -> Tuple[float, Optional[Image.Image]]:
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

    pixels_per_cm, debug_image = _calculate_checker_square_area_checker_cm(image, debug=debug)

    pp_cm = np.sqrt(pixels_per_cm)
    if debug:
        print(f'DEBUG: "calculate_pp_cm_checker_cm" - Raw value: {pixels_per_cm} | calculate_pp_cm_checker_cm')
    # Round to nearest 10 (if 5 or more round up else round down)
    pp_cm = math.floor(pp_cm / 10) * 10 if pp_cm % 10 < 5 else math.ceil(pp_cm / 10) * 10

    return pp_cm, debug_image

def calculate_pp_cm_colorchecker8(image: np.ndarray, debug: bool = False) -> Tuple[float, Optional[Image.Image]]:
    """Calculate pixels per cm from ColorChecker 8 card reference points.
    
    Detects three circular reference points on a ColorChecker 8 card that form
    a 50mm x 20mm rectangle and calculates the pixels-per-centimeter scale factor.
    The reference points should form a right triangle with specific dimensional
    relationships.
    
    Args:
        image: Input BGR image as numpy array.
        debug: If True, return debug visualization showing detected circles
            and selected reference points.
            
    Returns:
        Tuple of (pixels_per_cm, debug_image). Returns 0.0 for pixels_per_cm
        if detection fails. Debug_image is None if debug=False.
    """

    def find_circles(gray: np.ndarray) -> np.ndarray:
        # Contrast normalize + light blur helps Hough
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        g = cv2.GaussianBlur(g, (5, 5), 0)

        H, W = gray.shape[:2]
        minR = max(6, int(0.012 * min(H, W)))           # scale with image size
        maxR = max(minR + 1, int(0.06 * min(H, W)))     # exclude very large circles

        circles = cv2.HoughCircles(
            g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(0.08 * min(H, W)),
            param1=120, param2=25, minRadius=minR, maxRadius=maxR
        )
        if circles is None:
            return np.zeros((0, 3), dtype=np.float32)
        return circles[0].astype(np.float32)  # shape (N, 3) columns x,y,r

    def select_three_by_geometry(circles: np.ndarray) -> Optional[np.ndarray]:
        """Select three circles that best match the expected geometry.
        
        Args:
            circles: Array of detected circles with shape (N, 3) for (x, y, radius).
            
        Returns:
            Array of shape (3, 3) for the best three circles, or None if no
            suitable triplet is found.
        """
        # Need at least 3 candidates
        n = len(circles)
        if n < 3:
            return None

        best = None
        best_err = 1e9
        ratio_ms = 2.5                 # width/height = 50mm / 20mm
        ratio_lm = math.sqrt(29) / 5   # diag/width

        # Try every triplet (N is small after Hough)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    P = circles[[i, j, k], :3]  # (x, y, r)

                    # Radii of the three should be similar
                    r_med = np.median(P[:, 2])
                    if r_med <= 0:
                        continue
                    if np.max(np.abs(P[:, 2] - r_med)) / r_med > 0.25:
                        continue

                    # Pairwise center distances
                    C = P[:, :2]
                    d = np.array([
                        np.linalg.norm(C[0] - C[1]),
                        np.linalg.norm(C[1] - C[2]),
                        np.linalg.norm(C[0] - C[2]),
                    ])
                    d.sort()
                    s, m, l = d  # s=height(2cm), m=width(5cm), l=diag(sqrt(29)cm)
                    if s < 1 or m < 1:
                        continue

                    # Check expected ratios
                    e1 = abs(m / s - ratio_ms) / ratio_ms
                    e2 = abs(l / m - ratio_lm) / ratio_lm

                    # Encourage near-right angle (approx) at the corner where height & width meet
                    angles = []
                    for t in range(3):
                        v1 = C[(t + 1) % 3] - C[t]
                        v2 = C[(t + 2) % 3] - C[t]
                        den = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
                        cosang = float(np.clip(np.dot(v1, v2) / den, -1.0, 1.0))
                        angles.append(math.degrees(math.acos(cosang)))
                    ang_err = min(abs(a - 90) for a in angles)

                    err = e1 + e2 + 0.01 * ang_err
                    if err < best_err:
                        best_err = err
                        best = P
        return best

    # --- preprocess ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Try both polarities (just in case)
    circles = find_circles(gray)
    if len(circles) == 0:
        circles = find_circles(cv2.bitwise_not(gray))

    chosen = select_three_by_geometry(circles)
    debug_img = None
    if chosen is None:
        if debug:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else \
                  cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            d = ImageDraw.Draw(pil)
            for x, y, r in circles:
                d.ellipse([x - r, y - r, x + r, y + r], outline="orange", width=2)
            debug_img = pil
        return 0.0, debug_img

    # Compute px/cm from width, height, and diagonal; robust aggregate via median
    C = chosen[:, :2]
    dists = np.array([
        np.linalg.norm(C[0] - C[1]),
        np.linalg.norm(C[1] - C[2]),
        np.linalg.norm(C[0] - C[2]),
    ])
    s, m, l = np.sort(dists)                # s≈2cm, m≈5cm, l≈sqrt(29)≈5.385cm
    px_per_cm_candidates = np.array([
        m / 5.0,
        s / 2.0,
        l / math.sqrt(29.0)
    ])
    px_per_cm = float(np.median(px_per_cm_candidates))

    if debug:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else \
              cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        # Lightly draw all circle candidates
        for x, y, r in circles:
            draw.ellipse([x - r, y - r, x + r, y + r], outline="#88FFFF", width=1)
        # Highlight the selected triplet
        for x, y, r in chosen:
            draw.ellipse([x - r, y - r, x + r, y + r], outline="lime", width=3)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="red")
        # Label px/cm
        label = f"px/cm ~ {px_per_cm:.2f}"
        w = 8 * len(label)
        draw.rectangle([10, 10, 10 + w, 35], fill=(0, 0, 0, 150))
        draw.text((14, 14), label, fill="white")
        debug_img = pil

    if debug:
        print(f'DEBUG: "calculate_pp_cm_colorchecker8" - Raw value: {px_per_cm} | calculate_pp_cm_colorchecker8')
    # Round to nearest 10 (if 5 or more round up else round down)
    px_per_cm = math.floor(px_per_cm / 10) * 10 if px_per_cm % 10 < 5 else math.ceil(px_per_cm / 10) * 10

    return px_per_cm, debug_img

def artifact_face_size(img: Union[np.ndarray, Image.Image], card_img: np.ndarray, 
                      card_type: str, debug: bool = False) -> float:
    """Calculate the face area of an artifact using a reference card for scale.
    
    Determines the area of an artifact image by counting non-transparent pixels
    and converting to square centimeters using a reference card for scale calibration.
    Handles both PIL Images and numpy arrays with various channel configurations.
    
    Args:
        img: Artifact image with transparent background. Can be PIL Image or
            numpy array. For arrays, supports RGBA (4-channel), RGB (3-channel),
            or grayscale (2D) formats.
        card_img: Reference card image as numpy array for scale calculation.
        card_type: Type of reference card used for scale. Must be either
            'colorchecker8' or 'checker_cm'.
        debug: If True, print detailed debug information about processing steps.
        
    Returns:
        Face area of the artifact in square centimeters (cm²).
        
    Raises:
        ValueError: If card_type is invalid, scale cannot be determined from
            the reference card, or if image format is unsupported.
    """
    if card_type == 'colorchecker8':
        pp_cm, _ = calculate_pp_cm_colorchecker8(card_img, debug=False)
    elif card_type == 'checker_cm':
        pp_cm, _ = calculate_pp_cm_checker_cm(card_img, debug=False)
    else:
        raise ValueError(f"Invalid card type: {card_type}")

    # Ensure we have a valid pixels per cm value
    if pp_cm <= 0:
        raise ValueError("Could not determine scale from reference card")

    # Convert PIL Image to numpy array if needed
    if hasattr(img, 'convert'):  # PIL Image
        img_array = np.array(img)
        if debug:
            print(f'DEBUG: "artifact_face_size" - Converted PIL Image to numpy array, shape: {img_array.shape}, ndim: {img_array.ndim}')
    else:
        img_array = img  # Already numpy array
        if debug:
            print(f'DEBUG: "artifact_face_size" - Using existing numpy array, shape: {img_array.shape}, ndim: {img_array.ndim}')
    
    # Handle transparent background: create binary mask for artifact pixels
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        # RGBA image, use alpha channel as mask
        if debug:
            print('DEBUG: "artifact_face_size" - Using RGBA branch (4 channels)')
        alpha = img_array[:, :, 3]
        if debug:
            print(f'DEBUG: "artifact_face_size" - Alpha channel stats - min: {alpha.min()}, max: {alpha.max()}, mean: {alpha.mean():.2f}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 0 pixels: {np.sum(alpha > 0)}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 128 pixels: {np.sum(alpha > 128)}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 200 pixels: {np.sum(alpha > 200)}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 240 pixels: {np.sum(alpha > 240)}')
            print(f'DEBUG: "artifact_face_size" - Alpha = 255 pixels: {np.sum(alpha == 255)}')
        
        # Try a very high threshold first, then fall back if needed
        if np.sum(alpha > 240) > 0:
            mask = (alpha > 240).astype(np.uint8)  # Very strict threshold
            if debug:
                print('DEBUG: "artifact_face_size" - Using alpha > 240 threshold')
        elif np.sum(alpha > 200) > 0:
            mask = (alpha > 200).astype(np.uint8)  # High threshold
            if debug:
                print('DEBUG: "artifact_face_size" - Using alpha > 200 threshold')
        else:
            # If all pixels have high alpha, try using RGB content instead
            if debug:
                print('DEBUG: "artifact_face_size" - All pixels have high alpha, using RGB content for mask')
            rgb = img_array[:, :, :3]
            # Create mask based on RGB content - assume black/transparent areas are background
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mask = (gray > 30).astype(np.uint8)  # Pixels brighter than 30
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        # RGB image, treat all nonzero as artifact (legacy fallback)
        if debug:
            print('DEBUG: "artifact_face_size" - Using RGB branch (3 channels)')
        mask = (np.any(img_array > 0, axis=2)).astype(np.uint8)
    elif img_array.ndim == 2:
        # Single channel, treat nonzero as artifact
        if debug:
            print('DEBUG: "artifact_face_size" - Using 2D array branch (grayscale)')
        mask = (img_array > 0).astype(np.uint8)
    elif img_array.ndim == 4:
        # RGBA image, use alpha channel as mask
        alpha = img_array[:, :, 3]
        if debug:
            print(f'DEBUG: "artifact_face_size" - Alpha channel stats - min: {alpha.min()}, max: {alpha.max()}, mean: {alpha.mean():.2f}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 0 pixels: {np.sum(alpha > 0)}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 128 pixels: {np.sum(alpha > 128)}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 200 pixels: {np.sum(alpha > 200)}')
            print(f'DEBUG: "artifact_face_size" - Alpha > 240 pixels: {np.sum(alpha > 240)}')
            print(f'DEBUG: "artifact_face_size" - Alpha = 255 pixels: {np.sum(alpha == 255)}')
        
        # Try a very high threshold first, then fall back if needed
        if np.sum(alpha > 240) > 0:
            mask = (alpha > 240).astype(np.uint8)  # Very strict threshold
            if debug:
                print('DEBUG: "artifact_face_size" - Using alpha > 240 threshold')
        elif np.sum(alpha > 200) > 0:
            mask = (alpha > 200).astype(np.uint8)  # High threshold
            if debug:
                print('DEBUG: "artifact_face_size" - Using alpha > 200 threshold')
        else:
            # If all pixels have high alpha, try using RGB content instead
            if debug:
                print('DEBUG: "artifact_face_size" - All pixels have high alpha, using RGB content for mask')
            rgb = img_array[:, :, :3]
            # Create mask based on RGB content - assume black/transparent areas are background
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mask = (gray > 30).astype(np.uint8)  # Pixels brighter than 30
    else:
        raise ValueError("Unsupported image shape for artifact image")
            
    # Count only the white/foreground pixels in the binary mask
    artifact_pixels = cv2.countNonZero(mask)

    # Convert pixels to cm² using the calculated pixels per cm
    # pp_cm is pixels per cm, so pp_cm² is pixels per cm²
    if debug:
        print(f'DEBUG: "artifact_face_size" - Artifact pixels: {artifact_pixels}')
    face_area_cm2 = artifact_pixels / (pp_cm * pp_cm)

    return face_area_cm2

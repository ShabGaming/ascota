"""
Utility functions for image processing and color card detection.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, List, Optional
import io


def load_image_any(path: Union[str, Path]) -> np.ndarray:
    """Load image from various formats using OpenCV.
    
    Loads images in common formats (JPG, JPEG, PNG, CR2, CR3) and returns
    them in OpenCV BGR format for further processing.
    
    Args:
        path: Path to the image file as string or Path object.
        
    Returns:
        OpenCV image array in BGR color format with shape (height, width, 3).
        
    Raises:
        FileNotFoundError: If the image file cannot be loaded or does not exist.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def resize_max(img: np.ndarray, max_side: int = 2000) -> Tuple[np.ndarray, float]:
    """Resize image to fit within maximum side length constraint.
    
    Proportionally resizes an image so that its largest dimension (width or height)
    does not exceed the specified maximum. If the image is already smaller, it
    remains unchanged.
    
    Args:
        img: Input OpenCV image array.
        max_side: Maximum allowed dimension in pixels. Defaults to 2000.
        
    Returns:
        Tuple of (resized_image, scale_factor). The scale_factor is 1.0 if no
        resizing was needed, otherwise it's the ratio of new_size/original_size.
    """
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side: 
        return img, 1.0
    scale = max_side / s
    img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img_small, scale


def polygon_to_mask(img_shape: Tuple[int, int, int], poly_xy: np.ndarray) -> np.ndarray:
    """Convert polygon coordinates to a binary mask.
    
    Creates a binary mask where pixels inside the polygon are set to 255
    and pixels outside are set to 0. Uses OpenCV's fillConvexPoly for
    efficient polygon filling.
    
    Args:
        img_shape: Shape tuple of the target image as (height, width, channels).
        poly_xy: Polygon vertices as numpy array with shape (N, 2) where
            N is the number of vertices and each row is (x, y) coordinates.
        
    Returns:
        Binary mask as uint8 numpy array with shape (height, width).
        Pixels inside the polygon have value 255, outside pixels have value 0.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly_xy.astype(np.int32), 255)
    return mask


def non_max_suppression_polys(polys: List[np.ndarray], iou_thresh: float = 0.2) -> List[np.ndarray]:
    """Apply non-maximum suppression to remove overlapping polygons.
    
    Filters a list of polygons by removing those that have high intersection
    over union (IoU) with larger polygons. This helps eliminate duplicate
    detections of the same object.
    
    Args:
        polys: List of polygon arrays, each with shape (4, 2) representing
            four corner coordinates as (x, y) pairs.
        iou_thresh: IoU threshold for suppression. Polygons with IoU above
            this threshold will be suppressed. Defaults to 0.2.
        
    Returns:
        List of filtered polygon arrays after non-maximum suppression.
        Polygons are returned in order of decreasing area.
    """
    if not polys: 
        return []
    
    # use bounding boxes for IoU
    boxes = []
    for P in polys:
        xs, ys = P[:,0], P[:,1]
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    boxes = np.array(boxes, dtype=np.float32)

    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    order = np.argsort(-areas)
    keep = []

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate intersection over union between two bounding boxes.
        
        Args:
            a: First bounding box as [x1, y1, x2, y2].
            b: Second bounding box as [x1, y1, x2, y2].
            
        Returns:
            IoU value between 0 and 1.
        """
        xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
        w = max(0, xx2-xx1); h = max(0, yy2-yy1)
        inter = w*h
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return 0 if union <= 0 else inter/union

    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        rest_keep = []
        for j in rest:
            if iou(boxes[i], boxes[j]) <= iou_thresh:
                rest_keep.append(j)
        order = np.array(rest_keep, dtype=int)
    return [polys[i] for i in keep]


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV image from BGR to PIL Image in RGB format.
    
    Performs color space conversion from BGR (Blue-Green-Red) used by OpenCV
    to RGB (Red-Green-Blue) used by PIL/Pillow and creates a PIL Image object.
    
    Args:
        cv2_img: OpenCV image array in BGR format with shape (height, width, 3).
        
    Returns:
        PIL Image object in RGB color format.
    """
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image from RGB to OpenCV image in BGR format.
    
    Converts a PIL Image object to a numpy array and performs color space
    conversion from RGB (Red-Green-Blue) to BGR (Blue-Green-Red) format
    used by OpenCV.
    
    Args:
        pil_img: PIL Image object in RGB color format.
        
    Returns:
        OpenCV image array in BGR format with shape (height, width, 3).
    """
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def create_transparent_image(pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create transparent image by applying a binary mask as alpha channel.
    
    Takes a PIL Image and a binary mask, then creates a new RGBA image where
    the mask determines pixel transparency. White areas in the mask (255) become
    fully opaque, while black areas (0) become fully transparent.
    
    Args:
        pil_img: Input PIL Image in any color mode.
        mask: Binary mask as numpy array with shape (height, width).
            Values should be 0 (transparent) or 255 (opaque).
        
    Returns:
        PIL Image in RGBA mode with transparency applied according to the mask.
    """
    # Convert mask to PIL Image
    mask_pil = Image.fromarray(mask, mode='L')
    
    # Create RGBA image
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    
    # Apply mask as alpha channel
    pil_img.putalpha(mask_pil)
    return pil_img


def contour_rect_fallback(img_bgr: np.ndarray, tried_mask: Optional[np.ndarray] = None, 
                         min_area: int = 4000) -> List[np.ndarray]:
    """Fallback rectangle detection using contour analysis and edge detection.
    
    Detects rectangular shapes using edge detection and contour approximation
    when other detection methods fail. Applies filtering based on area,
    convexity, and rectangularity to ensure quality detections.
    
    Args:
        img_bgr: Input OpenCV image in BGR format.
        tried_mask: Optional binary mask of previously attempted regions to exclude.
            Areas where mask > 0 will be filled with white to avoid re-detection.
        min_area: Minimum contour area in pixels for a valid rectangle detection.
            Defaults to 4000.
        
    Returns:
        List of detected rectangle polygons, each as numpy array with shape (4, 2)
        representing the four corner coordinates as (x, y) pairs.
    """
    img = img_bgr.copy()
    if tried_mask is not None:
        img[tried_mask>0] = 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: 
            continue
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # rectangularity check
            rect_area = cv2.contourArea(approx)
            x,y,w,h = cv2.boundingRect(approx)
            if rect_area/(w*h) < 0.6:
                continue
            quads.append(approx.reshape(4,2).astype(np.float32))
    return quads

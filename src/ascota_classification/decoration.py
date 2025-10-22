"""
Decoration classification module for pottery images with transparent backgrounds.

This module classifies pottery decoration patterns into two categories:
- Impressed: decorations made by pressing objects into the clay
- Incised: decorations made by cutting/carving into the clay

The classification uses a pre-trained DINOv2 ViT-L/14 model with optimized logistic regression classifier.
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
from transformers import AutoModel

# Default model path (relative to module)
DEFAULT_MODEL_PATH = "models/decoration_dinov2_large_logistic_regression.pkl"


def _load_dino_model(device: torch.device) -> AutoModel:
    """
    Load the DINOv2 ViT-L/14 model for feature extraction.
    
    Args:
        device: PyTorch device (cpu or cuda)
        
    Returns:
        Loaded DINOv2 model in eval mode
    """
    try:
        model = AutoModel.from_pretrained('facebook/dinov2-large')
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load DINOv2 model: {e}")


def _extract_features(
    image: Image.Image,
    model: AutoModel,
    device: torch.device
) -> np.ndarray:
    """
    Extract DINO features from pottery image.
    
    Args:
        image: PIL Image with transparent background (RGBA)
        model: Pre-loaded DINOv2 model
        device: PyTorch device
        
    Returns:
        Feature vector as numpy array
    """
    # Convert RGBA to RGB with white background
    if image.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Standard DINO preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        outputs = model(image_tensor)
        # Use pooler output if available, otherwise mean of last hidden state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
    
    return features.cpu().numpy().flatten()


def _load_classifier(model_path: Path) -> Tuple[any, Optional[Dict]]:
    """
    Load the trained logistic regression classifier and its parameters.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Tuple of (classifier, parameters dict or None)
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure the trained model is available at this location."
        )
    
    try:
        classifier = joblib.load(model_path)
        
        # Try to load parameters file if it exists
        params_path = model_path.parent / model_path.name.replace('_optimized.pkl', '_params.pkl')
        params = None
        if params_path.exists():
            params = joblib.load(params_path)
        
        return classifier, params
    except Exception as e:
        raise RuntimeError(f"Failed to load classifier: {e}")


def classify_pottery_decoration(
    image: Image.Image,
    model_path: Optional[Path] = None,
    return_confidence: bool = False,
    debug: bool = False
) -> Dict[str, any]:
    """
    Classify pottery decoration pattern from an image with transparent background.
    
    Args:
        image: PIL Image with transparent background (RGBA or RGB format)
        model_path: Path to trained model file. If None, uses default model path.
        return_confidence: If True, include decision function scores in output
        debug: If True, print debug information
        
    Returns:
        Dictionary containing:
            - "label": Classification result ("Impressed" or "Incised")
            - "method": "DINOv2 + Logistic Regression"
            - "confidence": (Optional) Decision function score if return_confidence=True
            - "model_params": (Optional) Model hyperparameters if available
            
    Raises:
        FileNotFoundError: If model file is not found
        RuntimeError: If model loading or inference fails
        
    Examples:
        >>> from PIL import Image
        >>> img = Image.open("pottery_decoration.png")
        >>> result = classify_pottery_decoration(img, debug=True)
        >>> print(result["label"])
        'Impressed'
        
        >>> result = classify_pottery_decoration(img, return_confidence=True)
        >>> print(f"Label: {result['label']}, Confidence: {result['confidence']:.4f}")
        Label: Impressed, Confidence: 0.8532
    """
    # Use default model path if not provided
    if model_path is None:
        # Convert default string path to Path object relative to module location
        model_path = Path(__file__).parent / DEFAULT_MODEL_PATH
    elif isinstance(model_path, str):
        # Convert string to Path object if needed
        model_path = Path(model_path)
    
    if debug:
        print(f"Using model: {model_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        print(f"Using device: {device}")
    
    try:
        # Load DINOv2 model
        if debug:
            print("Loading DINOv2 model...")
        dino_model = _load_dino_model(device)
        
        # Extract features
        if debug:
            print("Extracting features...")
        features = _extract_features(image, dino_model, device)
        
        if debug:
            print(f"Feature shape: {features.shape}")
        
        # Load classifier
        if debug:
            print("Loading classifier...")
        classifier, params = _load_classifier(model_path)
        
        # Make prediction
        if debug:
            print("Making prediction...")
        
        # Reshape features for prediction (classifier expects 2D array)
        features_2d = features.reshape(1, -1)
        prediction = classifier.predict(features_2d)[0]
        
        # Map numeric prediction to label
        # Assuming 0 = impressed, 1 = incised based on sklearn.preprocessing.LabelEncoder
        label_map = {0: "Impressed", 1: "Incised"}
        label = label_map.get(prediction, "Unknown")
        
        result = {
            "label": label,
            "method": "DINOv2 + Logistic Regression"
        }
        
        # Add confidence score if requested
        if return_confidence:
            if hasattr(classifier, 'decision_function'):
                decision_score = classifier.decision_function(features_2d)[0]
                # For binary classification, convert to probability-like confidence
                # Higher absolute value means higher confidence
                confidence = abs(decision_score)
                result["confidence"] = float(confidence)
                result["decision_score"] = float(decision_score)
                
                if debug:
                    print(f"Decision score: {decision_score:.4f}")
                    print(f"Confidence: {confidence:.4f}")
        
        # Add model parameters if available
        if params is not None:
            result["model_params"] = params
            if debug:
                print(f"Model parameters: {params}")
        
        if debug:
            print(f"Classification result: {label}")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Classification failed: {e}")


def batch_classify_pottery_decoration(
    images: list[Image.Image],
    model_path: Optional[Path] = None,
    return_confidence: bool = False,
    debug: bool = False
) -> list[Dict[str, any]]:
    """
    Classify multiple pottery decoration images efficiently.
    
    This function loads the models once and reuses them for all images,
    making it more efficient than calling classify_pottery_decoration repeatedly.
    
    Args:
        images: List of PIL Images with transparent backgrounds
        model_path: Path to trained model file. If None, uses default model path.
        return_confidence: If True, include confidence scores in output
        debug: If True, print debug information
        
    Returns:
        List of classification result dictionaries, one per image
        
    Examples:
        >>> from PIL import Image
        >>> images = [Image.open(f"pottery_{i}.png") for i in range(5)]
        >>> results = batch_classify_pottery_decoration(images)
        >>> for i, result in enumerate(results):
        ...     print(f"Image {i}: {result['label']}")
    """
    # Use default model path if not provided
    if model_path is None:
        # Convert default string path to Path object relative to module location
        model_path = Path(__file__).parent / DEFAULT_MODEL_PATH
    elif isinstance(model_path, str):
        # Convert string to Path object if needed
        model_path = Path(model_path)
    
    if debug:
        print(f"Using model: {model_path}")
        print(f"Processing {len(images)} images...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        print(f"Using device: {device}")
    
    try:
        # Load models once
        if debug:
            print("Loading DINOv2 model...")
        dino_model = _load_dino_model(device)
        
        if debug:
            print("Loading classifier...")
        classifier, params = _load_classifier(model_path)
        
        # Process all images
        results = []
        for i, image in enumerate(images):
            if debug and (i + 1) % 10 == 0:
                print(f"Processing image {i + 1}/{len(images)}...")
            
            # Extract features
            features = _extract_features(image, dino_model, device)
            
            # Make prediction
            features_2d = features.reshape(1, -1)
            prediction = classifier.predict(features_2d)[0]
            
            # Map to label
            label_map = {0: "Impressed", 1: "Incised"}
            label = label_map.get(prediction, "Unknown")
            
            result = {
                "label": label,
                "method": "DINOv2 + Logistic Regression"
            }
            
            # Add confidence if requested
            if return_confidence and hasattr(classifier, 'decision_function'):
                decision_score = classifier.decision_function(features_2d)[0]
                confidence = abs(decision_score)
                result["confidence"] = float(confidence)
                result["decision_score"] = float(decision_score)
            
            # Add model parameters to first result only
            if i == 0 and params is not None:
                result["model_params"] = params
            
            results.append(result)
        
        if debug:
            print(f"✓ Processed {len(images)} images successfully!")
            # Print summary
            impressed_count = sum(1 for r in results if r["label"] == "Impressed")
            incised_count = sum(1 for r in results if r["label"] == "Incised")
            print(f"Summary: {impressed_count} Impressed, {incised_count} Incised")
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Batch classification failed: {e}")

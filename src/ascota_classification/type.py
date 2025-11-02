"""
Pottery type classification module for pottery images with transparent backgrounds.

This module classifies pottery types using a multi-stage pipeline:
- Stage 1: body vs everything else (base+rim+appendages)
- Stage 2: base vs rim vs appendage (if not body)
- Stage 3: appendage subtypes using Azure OpenAI GPT-4o (optional, if appendage)

The classification uses pre-trained DINOv2 ViT-L/14 model for feature extraction with
optimized classifiers for each stage.
"""

import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
from transformers import AutoModel

# Azure OpenAI imports (optional)
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

import base64
from io import BytesIO

# Default model paths (relative to module)
DEFAULT_MODEL_PATH_STAGE1 = "models/type_1_dinov2_large_svm_rbf.pkl"
DEFAULT_MODEL_PATH_STAGE2 = "models/type_2_dinov2_large_svm_rbf.pkl"


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
    Load a trained classifier and its parameters.
    
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
        params_path = model_path.parent / model_path.name.replace('.pkl', '_params.pkl')
        params = None
        if params_path.exists():
            params = joblib.load(params_path)
        
        return classifier, params
    except Exception as e:
        raise RuntimeError(f"Failed to load classifier: {e}")


def _classify_stage1(
    features: np.ndarray,
    classifier: any,
    debug: bool = False
) -> Tuple[str, float]:
    """
    Stage 1: Classify body vs everything else.
    
    Args:
        features: Feature vector from DINOv2
        classifier: Trained Stage 1 classifier
        debug: If True, print debug information
        
    Returns:
        Tuple of (label, confidence)
    """
    # Reshape features for prediction
    features_2d = features.reshape(1, -1)
    
    # Get prediction probabilities
    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(features_2d)[0]
        prediction = classifier.predict(features_2d)[0]
        confidence = float(np.max(proba))
    else:
        prediction = classifier.predict(features_2d)[0]
        confidence = 1.0  # Default if no probability available
    
    # Map prediction: 0 = everything_else, 1 = body
    label = "body" if prediction == 1 else "everything_else"
    
    if debug:
        print(f"_classify_stage1: Stage 1: {label} (confidence: {confidence:.4f})")
    
    return label, confidence


def _classify_stage2(
    features: np.ndarray,
    classifier: any,
    debug: bool = False
) -> Tuple[str, float]:
    """
    Stage 2: Classify base vs rim vs appendage.
    
    Args:
        features: Feature vector from DINOv2
        classifier: Trained Stage 2 classifier
        debug: If True, print debug information
        
    Returns:
        Tuple of (label, confidence)
    """
    # Reshape features for prediction
    features_2d = features.reshape(1, -1)
    
    # Get prediction probabilities
    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(features_2d)[0]
        prediction = classifier.predict(features_2d)[0]
        confidence = float(np.max(proba))
        
        # LabelEncoder classes for Stage 2: ['appendage', 'base', 'rim']
        # So: 0=appendage, 1=base, 2=rim
        class_names = ['appendage', 'base', 'rim']
        if prediction < len(class_names):
            label = class_names[prediction]
        else:
            # Fallback if prediction index is out of range
            label = class_names[np.argmax(proba)]
    else:
        prediction = classifier.predict(features_2d)[0]
        class_names = ['appendage', 'base', 'rim']
        label = class_names[prediction] if prediction < len(class_names) else 'appendage'
        confidence = 1.0
    
    if debug:
        print(f"_classify_stage2: Stage 2: {label} (confidence: {confidence:.4f})")
    
    return label, confidence


def _prepare_image_for_gpt4o(image: Image.Image, max_size: int = 512) -> str:
    """
    Resize image and convert to base64 for GPT-4o.
    
    Args:
        image: PIL Image
        max_size: Maximum size for thumbnail
        
    Returns:
        Base64 encoded image string
    """
    # Resize while maintaining aspect ratio
    image_copy = image.copy()
    image_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = BytesIO()
    image_copy.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64


def _classify_appendage_with_gpt4o(
    image: Image.Image,
    debug: bool = False
) -> Tuple[Optional[str], Optional[float]]:
    """
    Stage 3: Classify appendage subtype using Azure OpenAI GPT-4o.
    
    Args:
        image: PIL Image of appendage
        debug: If True, print debug information
        
    Returns:
        Tuple of (label, confidence) or (None, None) if failed
    """
    if not AZURE_OPENAI_AVAILABLE:
        if debug:
            print("_classify_appendage_with_gpt4o: Azure OpenAI SDK not available. Skipping GPT-4o classification.")
        return None, None
    
    # Get credentials from environment variables
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    
    if azure_endpoint is None or azure_api_key is None:
        if debug:
            print("_classify_appendage_with_gpt4o: Azure OpenAI credentials not found in environment variables.")
            print("_classify_appendage_with_gpt4o: Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        return None, None
    
    try:
        # Prepare image
        img_base64 = _prepare_image_for_gpt4o(image, max_size=512)
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version="2024-02-15-preview"
        )
        
        # Create prompt
        prompt = """Classify this pottery sherd appendage into one of the following categories:
- lid
- rim-handle
- spout
- rounded
- body-decorated
- tile

Please respond with:
1. The classification label (one of the categories above)
2. Your confidence level (0.0 to 1.0)
3. A brief explanation of why you chose this classification

Format your response as:
Label: <category>
Confidence: <number>
Explanation: <brief explanation>"""
        
        # Call GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        # Extract label and confidence
        label = None
        confidence = None
        
        for line in response_text.split('\n'):
            if line.startswith('Label:'):
                label = line.split('Label:')[1].strip().lower()
            elif line.startswith('Confidence:'):
                try:
                    confidence = float(line.split('Confidence:')[1].strip())
                except:
                    confidence = 0.5  # Default
        
        # Validate label
        valid_labels = ['lid', 'rim-handle', 'spout', 'rounded', 'body-decorated', 'tile']
        if label not in valid_labels:
            # Try to find label in response
            for valid_label in valid_labels:
                if valid_label in response_text.lower():
                    label = valid_label
                    break
        
        # If still not found, default to first valid label
        if label not in valid_labels:
            label = valid_labels[0]
            if debug:
                print(f"_classify_appendage_with_gpt4o: Warning: Could not parse label from GPT-4o response, using default: {label}")
        
        if confidence is None:
            confidence = 0.5  # Default confidence
        
        if debug:
            print(f"_classify_appendage_with_gpt4o: Stage 3 (GPT-4o): {label} (confidence: {confidence:.4f})")
        
        return label, confidence
        
    except Exception as e:
        if debug:
            print(f"_classify_appendage_with_gpt4o: Error calling GPT-4o: {e}")
        return None, None


def classify_pottery_type(
    image: Image.Image,
    model_path_stage1: Optional[Path] = None,
    model_path_stage2: Optional[Path] = None,
    use_azure_openai: bool = False,
    debug: bool = False
) -> Dict[str, any]:
    """
    Classify pottery type from an image with transparent background.
    
    Uses a multi-stage pipeline:
    1. Stage 1: body vs everything else
    2. Stage 2: base vs rim vs appendage (if not body)
    3. Stage 3: appendage subtypes using Azure OpenAI (optional, if appendage)
    
    Args:
        image: PIL Image with transparent background (RGBA or RGB format)
        model_path_stage1: Path to Stage 1 model. If None, uses default path.
        model_path_stage2: Path to Stage 2 model. If None, uses default path.
        use_azure_openai: If True, use Azure OpenAI GPT-4o for appendage subtypes
        debug: If True, print debug information
        
    Returns:
        Dictionary with conditional structure:
        - If Stage 1 = "body": {"label": "body", "stage1": {...}}
        - If Stage 2 = "base" or "rim": {"label": "...", "stage1": {...}, "stage2": {...}}
        - If Stage 2 = "appendage": {"label": "...", "stage1": {...}, "stage2": {...}, "stage3": {...}?}
        - If Azure disabled and appendage: no stage3 field
        
    Raises:
        FileNotFoundError: If model file is not found
        RuntimeError: If model loading or inference fails
        
    Examples:
        >>> from PIL import Image
        >>> img = Image.open("pottery_type.png")
        >>> result = classify_pottery_type(img, debug=True)
        >>> print(result["label"])
        'body'
        
        >>> result = classify_pottery_type(img, use_azure_openai=True)
        >>> print(result["label"])  # Could be appendage subtype if appendage
        'spout'
    """
    # Use default model paths if not provided
    if model_path_stage1 is None:
        model_path_stage1 = Path(__file__).parent / DEFAULT_MODEL_PATH_STAGE1
    elif isinstance(model_path_stage1, str):
        model_path_stage1 = Path(model_path_stage1)
    
    if model_path_stage2 is None:
        model_path_stage2 = Path(__file__).parent / DEFAULT_MODEL_PATH_STAGE2
    elif isinstance(model_path_stage2, str):
        model_path_stage2 = Path(model_path_stage2)
    
    if debug:
        print(f"classify_pottery_type: Using Stage 1 model: {model_path_stage1}")
        print(f"classify_pottery_type: Using Stage 2 model: {model_path_stage2}")
        print(f"classify_pottery_type: Azure OpenAI: {use_azure_openai}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        print(f"classify_pottery_type: Using device: {device}")
    
    try:
        # Load DINOv2 model
        if debug:
            print("classify_pottery_type: Loading DINOv2 model...")
        dino_model = _load_dino_model(device)
        
        # Extract features
        if debug:
            print("classify_pottery_type: Extracting features...")
        features = _extract_features(image, dino_model, device)
        
        if debug:
            print(f"classify_pottery_type: Feature shape: {features.shape}")
        
        # Stage 1: body vs everything else
        if debug:
            print("classify_pottery_type: Stage 1: Classifying body vs everything else...")
        classifier_stage1, _ = _load_classifier(model_path_stage1)
        stage1_label, stage1_confidence = _classify_stage1(features, classifier_stage1, debug)
        
        result = {
            "label": stage1_label,
            "stage1": {
                "label": stage1_label,
                "confidence": stage1_confidence
            }
        }
        
        # If Stage 1 is "body", return early
        if stage1_label == "body":
            if debug:
                print("classify_pottery_type: Result: body (Stage 1 only)")
            return result
        
        # Stage 2: base vs rim vs appendage
        if debug:
            print("classify_pottery_type: Stage 2: Classifying base vs rim vs appendage...")
        classifier_stage2, _ = _load_classifier(model_path_stage2)
        stage2_label, stage2_confidence = _classify_stage2(features, classifier_stage2, debug)
        
        result["label"] = stage2_label
        result["stage2"] = {
            "label": stage2_label,
            "confidence": stage2_confidence
        }
        
        # If Stage 2 is "appendage" and Azure OpenAI is enabled, do Stage 3
        if stage2_label == "appendage" and use_azure_openai:
            if debug:
                print("classify_pottery_type: Stage 3: Classifying appendage subtype with Azure OpenAI...")
            stage3_label, stage3_confidence = _classify_appendage_with_gpt4o(image, debug)
            
            if stage3_label is not None and stage3_confidence is not None:
                result["label"] = stage3_label
                result["stage3"] = {
                    "label": stage3_label,
                    "confidence": stage3_confidence
                }
                if debug:
                    print(f"classify_pottery_type: Result: {stage3_label} (with Stage 3)")
            else:
                # Azure OpenAI failed, keep appendage label (no stage3 field)
                if debug:
                    print("classify_pottery_type: Azure OpenAI failed, returning appendage (no Stage 3)")
        # If Stage 2 is "appendage" but Azure is disabled, label stays "appendage" (no stage3 field)
        elif stage2_label == "appendage":
            if debug:
                print("classify_pottery_type: Result: appendage (Azure OpenAI disabled, no Stage 3)")
        
        if debug:
            print(f"classify_pottery_type: Final result: {result['label']}")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Classification failed: {e}")


def batch_classify_pottery_type(
    images: list[Image.Image],
    model_path_stage1: Optional[Path] = None,
    model_path_stage2: Optional[Path] = None,
    use_azure_openai: bool = False,
    debug: bool = False
) -> list[Dict[str, any]]:
    """
    Classify multiple pottery type images efficiently.
    
    This function loads the models once and reuses them for all images,
    making it more efficient than calling classify_pottery_type repeatedly.
    
    Args:
        images: List of PIL Images with transparent backgrounds
        model_path_stage1: Path to Stage 1 model. If None, uses default path.
        model_path_stage2: Path to Stage 2 model. If None, uses default path.
        use_azure_openai: If True, use Azure OpenAI GPT-4o for appendage subtypes
        debug: If True, print debug information
        
    Returns:
        List of classification result dictionaries, one per image
        
    Examples:
        >>> from PIL import Image
        >>> images = [Image.open(f"pottery_{i}.png") for i in range(5)]
        >>> results = batch_classify_pottery_type(images)
        >>> for i, result in enumerate(results):
        ...     print(f"Image {i}: {result['label']}")
    """
    # Use default model paths if not provided
    if model_path_stage1 is None:
        model_path_stage1 = Path(__file__).parent / DEFAULT_MODEL_PATH_STAGE1
    elif isinstance(model_path_stage1, str):
        model_path_stage1 = Path(model_path_stage1)
    
    if model_path_stage2 is None:
        model_path_stage2 = Path(__file__).parent / DEFAULT_MODEL_PATH_STAGE2
    elif isinstance(model_path_stage2, str):
        model_path_stage2 = Path(model_path_stage2)
    
    if debug:
        print(f"batch_classify_pottery_type: Using Stage 1 model: {model_path_stage1}")
        print(f"batch_classify_pottery_type: Using Stage 2 model: {model_path_stage2}")
        print(f"Processing {len(images)} images...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        print(f"batch_classify_pottery_type: Using device: {device}")
    
    try:
        # Load models once
        if debug:
            print("batch_classify_pottery_type: Loading DINOv2 model...")
        dino_model = _load_dino_model(device)
        
        if debug:
            print("batch_classify_pottery_type: Loading Stage 1 classifier...")
        classifier_stage1, _ = _load_classifier(model_path_stage1)
        
        if debug:
            print("batch_classify_pottery_type: Loading Stage 2 classifier...")
        classifier_stage2, _ = _load_classifier(model_path_stage2)
        
        # Process all images
        results = []
        for i, image in enumerate(images):
            if debug and (i + 1) % 10 == 0:
                print(f"batch_classify_pottery_type: Processing image {i + 1}/{len(images)}...")
            
            # Extract features
            features = _extract_features(image, dino_model, device)
            
            # Stage 1: body vs everything else
            stage1_label, stage1_confidence = _classify_stage1(features, classifier_stage1, False)
            
            result = {
                "label": stage1_label,
                "stage1": {
                    "label": stage1_label,
                    "confidence": stage1_confidence
                }
            }
            
            # If Stage 1 is "body", continue to next image
            if stage1_label == "body":
                results.append(result)
                continue
            
            # Stage 2: base vs rim vs appendage
            stage2_label, stage2_confidence = _classify_stage2(features, classifier_stage2, False)
            
            result["label"] = stage2_label
            result["stage2"] = {
                "label": stage2_label,
                "confidence": stage2_confidence
            }
            
            # If Stage 2 is "appendage" and Azure OpenAI is enabled, do Stage 3
            if stage2_label == "appendage" and use_azure_openai:
                stage3_label, stage3_confidence = _classify_appendage_with_gpt4o(image, False)
                
                if stage3_label is not None and stage3_confidence is not None:
                    result["label"] = stage3_label
                    result["stage3"] = {
                        "label": stage3_label,
                        "confidence": stage3_confidence
                    }
            
            results.append(result)
        
        if debug:
            print(f"batch_classify_pottery_type: ✓ Processed {len(images)} images successfully!")
            # Print summary
            label_counts = {}
            for r in results:
                label = r["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"batch_classify_pottery_type: Summary: {label_counts}")
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Batch classification failed: {e}")


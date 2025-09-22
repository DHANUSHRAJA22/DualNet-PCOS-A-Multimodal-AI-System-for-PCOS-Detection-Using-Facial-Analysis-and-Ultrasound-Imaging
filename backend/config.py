"""
Configuration settings for PCOS Analyzer Backend

Centralized configuration with environment variable support and automatic model discovery.
Supports ensemble inference with multiple models per modality.
"""

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FACE_MODELS_DIR = MODELS_DIR / "face"
XRAY_MODELS_DIR = MODELS_DIR / "xray"
YOLO_MODELS_DIR = MODELS_DIR / "yolo"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "5000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # Ensemble configuration
    USE_ENSEMBLE: bool = os.getenv("USE_ENSEMBLE", "true").lower() == "true"
    FUSION_MODE: str = os.getenv("FUSION_MODE", "threshold")  # "threshold" or "discrete"
    
    # CORS configuration
    ALLOWED_ORIGINS: List[str] = [
        origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ] if os.getenv("ALLOWED_ORIGINS") != "*" else ["*"]
    
    # File upload configuration
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "5"))
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    STATIC_TTL_SECONDS: int = int(os.getenv("STATIC_TTL_SECONDS", "3600"))
    
    # Model configuration
    GENDER_MODEL: str = "gender_classifier.h5"
    YOLO_MODEL: str = "bestv8.pt"
    
    # Default ensemble weights (can be overridden by environment variables)
    DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
        "vgg16_pcos": 0.2,
        "resnet50_pcos": 0.2,
        "efficientnetb0_pcos": 0.2,
        "efficientnetb1_pcos": 0.2,
        "efficientnetb2_pcos": 0.1,
        "efficientnetb3_pcos": 0.1
    }
    
    # Risk thresholds
    RISK_LOW_THRESHOLD: float = 0.33
    RISK_HIGH_THRESHOLD: float = 0.66
    
    # Image proxy whitelist
    ALLOWED_PROXY_HOSTS: List[str] = [
        "images.pexels.com",
        "images.unsplash.com",
        "picsum.photos",
        "via.placeholder.com",
        "localhost"
    ]

# Global settings instance
settings = Settings()

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FACE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
XRAY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
YOLO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
FACE_MODELS = {
    "VGG16": "vgg16_pcos.h5",
    "ResNet50": "resnet50_pcos.h5", 
    "EfficientNetB0": "efficientnetb0_pcos.h5",
    "EfficientNetB1": "efficientnetb1_pcos.h5",
    "EfficientNetB2": "efficientnetb2_pcos.h5",
    "EfficientNetB3": "efficientnetb3_pcos.h5"
}

XRAY_MODELS = {
    "VGG16": "vgg16_xray.h5",
    "ResNet50": "resnet50_xray.h5",
    "EfficientNetB0": "efficientnetb0_xray.h5"
}

# Best single models (fallback when ensemble disabled)
BEST_FACE_MODEL = "VGG16"
BEST_XRAY_MODEL = "VGG16"

# Model ensemble weights (can be overridden by environment variables)
FACE_ENSEMBLE_WEIGHTS = {
    "VGG16": 0.2,
    "ResNet50": 0.2,
    "EfficientNetB0": 0.2,
    "EfficientNetB1": 0.2,
    "EfficientNetB2": 0.1,
    "EfficientNetB3": 0.1
}

XRAY_ENSEMBLE_WEIGHTS = {
    "VGG16": 0.4,
    "ResNet50": 0.35,
    "EfficientNetB0": 0.25
}

# Class labels
FACE_LABELS = ["non_pcos", "pcos"]
XRAY_LABELS = ["normal", "pcos"]

# Model input sizes
FACE_IMAGE_SIZE = (224, 224)
XRAY_IMAGE_SIZE = (224, 224)
XRAY_YOLO_SIZE = (640, 640)
XRAY_VIT_SIZE = (224, 224)

# Image processing
IMAGE_QUALITY = 90

def get_risk_level(score: float) -> str:
    """Classify risk level based on probability score"""
    if score < settings.RISK_LOW_THRESHOLD:
        return "low"
    elif score < settings.RISK_HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"

def get_available_face_models() -> Dict[str, Path]:
    """
    Get available face models from the face models directory
    
    Returns:
        Dictionary mapping model names to file paths for available models
    """
    available = {}
    for model_name, filename in FACE_MODELS.items():
        model_path = FACE_MODELS_DIR / filename
        if model_path.exists():
            available[model_name] = model_path
    return available

def get_available_xray_models() -> Dict[str, Path]:
    """
    Get available X-ray models from the xray models directory
    
    Returns:
        Dictionary mapping model names to file paths for available models
    """
    available = {}
    for model_name, filename in XRAY_MODELS.items():
        model_path = XRAY_MODELS_DIR / filename
        if model_path.exists():
            available[model_name] = model_path
    return available

def get_model_labels(model_type: str) -> List[str]:
    """
    Get class labels for a model type
    
    Args:
        model_type: Type of model ('face' or 'xray')
        
    Returns:
        List of class labels
    """
    if model_type == 'face':
        return FACE_LABELS
    elif model_type == 'xray':
        return XRAY_LABELS
    else:
        return ["class_0", "class_1"]

def get_ensemble_weights(model_type: str) -> Dict[str, float]:
    """
    Get ensemble weights for a model type from environment variables or defaults
    
    Args:
        model_type: Type of model ('face' or 'xray')
    
    Returns:
        Dictionary mapping model names to weights
    """
    if model_type == 'face':
        weights = FACE_ENSEMBLE_WEIGHTS.copy()
        prefix = "FACE_ENSEMBLE_WEIGHT_"
    elif model_type == 'xray':
        weights = XRAY_ENSEMBLE_WEIGHTS.copy()
        prefix = "XRAY_ENSEMBLE_WEIGHT_"
    else:
        return {}
    
    # Override with environment variables if provided
    for model_name in weights.keys():
        env_var = f"{prefix}{model_name.upper()}"
        if env_var in os.environ:
            try:
                weights[model_name] = float(os.environ[env_var])
            except ValueError:
                pass  # Keep default weight
    
    return weights

def normalize_weights(weights: Dict[str, float], available_models: List[str]) -> Dict[str, float]:
    """
    Normalize weights for available models to sum to 1.0
    
    Args:
        weights: Original weights dictionary
        available_models: List of actually available model names
        
    Returns:
        Normalized weights dictionary
    """
    # Filter weights for available models
    available_weights = {name: weights.get(name, 1.0) for name in available_models}
    
    # Normalize to sum to 1.0
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        return {name: weight / total_weight for name, weight in available_weights.items()}
    else:
        # Equal weights if no weights specified
        equal_weight = 1.0 / len(available_models) if available_models else 0.0
        return {name: equal_weight for name in available_models}
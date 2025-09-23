"""
Configuration settings for PCOS Analyzer Backend

Centralized configuration with environment variable support and automatic model discovery.
Supports ensemble inference with multiple models per modality.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional
# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"

STATIC_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

XRAY_IMAGE_SIZE = (224, 224)  # or (100, 100) depending on your model
FACE_IMAGE_SIZE = (100, 100)

MAX_UPLOAD_MB = 10
STATIC_TTL_SECONDS = 3600
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

ALLOWED_ORIGINS = ["*"]

RISK_LOW_THRESHOLD = 0.3
RISK_HIGH_THRESHOLD = 0.7

FUSION_MODE = "average"
USE_ENSEMBLE = True

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FACE_MODELS_DIR = MODELS_DIR / "face"
XRAY_MODELS_DIR = MODELS_DIR / "xray"
YOLO_MODELS_DIR = MODELS_DIR / "yolo"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FACE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
XRAY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
YOLO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

class Settings:
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
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080", 
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # File upload configuration
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "5"))
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    STATIC_TTL_SECONDS: int = int(os.getenv("STATIC_TTL_SECONDS", "3600"))
    
    # Model configuration - exact filenames
    GENDER_MODEL: str = "gender_classifier.h5"
    YOLO_MODEL: str = "bestv8.pt"
    
    # Risk thresholds
    RISK_LOW_THRESHOLD: float = 0.33
    RISK_HIGH_THRESHOLD: float = 0.66
    
    # Image sizes
    FACE_IMAGE_SIZE: tuple = (224, 224)
    GENDER_IMAGE_SIZE: tuple = (249, 249)  # Gender model expects 249x249
    XRAY_IMAGE_SIZE: tuple = (224, 224)
    
    # Image proxy whitelist
    ALLOWED_PROXY_HOSTS: List[str] = [
        "images.pexels.com",
        "images.unsplash.com",
        "picsum.photos",
        "via.placeholder.com"
    ]

# Global settings instance
settings = Settings()

def get_available_face_models() -> Dict[str, Path]:
    """
    Auto-discover available face PCOS models using glob pattern
    
    Returns:
        Dictionary mapping model names to file paths for available models
    """
    available = {}
    
    # Use glob to find all pcos_*.h5 files in face directory
    pattern = str(FACE_MODELS_DIR / "pcos_*.h5")
    for model_path in glob.glob(pattern):
        model_path = Path(model_path)
        if model_path.exists() and model_path.is_file():
            try:
                # Validate that the file is readable
                with open(model_path, 'rb') as f:
                    # Read first few bytes to check if it's a valid HDF5 file
                    header = f.read(8)
                    if header.startswith(b'\x89HDF'):
                        # Extract model name from filename (remove pcos_ prefix and .h5 suffix)
                        model_name = model_path.stem.replace('pcos_', '')
                        available[model_name] = model_path
                    else:
                        logger.warning(f"Invalid HDF5 file detected: {model_path}")
            except Exception as e:
                logger.warning(f"Could not validate model file {model_path}: {str(e)}")
    
    return available

def get_available_xray_models() -> Dict[str, Path]:
    """
    Auto-discover available X-ray PCOS models using glob pattern
    
    Returns:
        Dictionary mapping model names to file paths for available models
    """
    available = {}
    
    # Use glob to find all pcos_*.h5 files in xray directory
    pattern = str(XRAY_MODELS_DIR / "pcos_*.h5")
    for model_path in glob.glob(pattern):
        model_path = Path(model_path)
        if model_path.exists() and model_path.is_file():
            try:
                # Validate that the file is readable
                with open(model_path, 'rb') as f:
                    # Read first few bytes to check if it's a valid HDF5 file
                    header = f.read(8)
                    if header.startswith(b'\x89HDF'):
                        # Extract model name from filename (remove pcos_ prefix and .h5 suffix)
                        model_name = model_path.stem.replace('pcos_', '')
                        available[model_name] = model_path
                    else:
                        logger.warning(f"Invalid HDF5 file detected: {model_path}")
            except Exception as e:
                logger.warning(f"Could not validate model file {model_path}: {str(e)}")
    
    return available

def get_model_labels(model_type: str) -> List[str]:
    """
    Get default class labels for model type
    
    Args:
        model_type: Type of model ('face' or 'xray')
        
    Returns:
        List of class labels
    """
    return ["non_pcos", "pcos"]

def load_model_labels(model_path: Path) -> List[str]:
    """
    Load class labels for a model from corresponding .labels.txt file
    
    Args:
        model_path: Path to the model file
        
    Returns:
        List of class labels, defaults to ["non_pcos", "pcos"] if file not found
    """
    labels_path = model_path.with_suffix('.labels.txt')
    
    try:
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                import json
                labels = json.load(f)
                if isinstance(labels, list):
                    return labels
    except Exception:
        pass
    
    # Default labels
    return ["non_pcos", "pcos"]

def get_risk_level(score: float) -> str:
    """Classify risk level based on probability score"""
    if score < settings.RISK_LOW_THRESHOLD:
        return "low"
    elif score < settings.RISK_HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"

def get_ensemble_weights(model_type: str) -> Dict[str, float]:
    """
    Get ensemble weights for a model type from environment variables or defaults
    
    Args:
        model_type: Type of model ('face' or 'xray')
    
    Returns:
        Dictionary mapping model names to weights (defaults to 1.0 for each)
    """
    if model_type == 'face':
        available_models = get_available_face_models()
        prefix = "ENSEMBLE_WEIGHT_"
    elif model_type == 'xray':
        available_models = get_available_xray_models()
        prefix = "ENSEMBLE_WEIGHT_"
    else:
        return {}
    
    # Initialize with default weights (1.0 for each available model)
    weights = {model_name: 1.0 for model_name in available_models.keys()}
    
    # Override with environment variables if provided
    for model_name in weights.keys():
        env_var = f"{prefix}{model_name.upper()}"
        if env_var in os.environ:
            try:
                weights[model_name] = float(os.environ[env_var])
            except ValueError:
                pass  # Keep default weight
    
    return weights

def normalize_weights(weights: Dict[str, float], available_models: List[str] = None) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0
    
    Args:
        weights: Original weights dictionary
        available_models: List of available model names to filter weights
        
    Returns:
        Normalized weights dictionary
    """
    if available_models:
        # Filter weights to only include available models
        filtered_weights = {name: weights.get(name, 1.0) for name in available_models}
    else:
        filtered_weights = weights.copy()
    
    total_weight = sum(filtered_weights.values())
    if total_weight > 0:
        return {name: weight / total_weight for name, weight in filtered_weights.items()}
    else:
        # Equal weights if no weights specified
        equal_weight = 1.0 / len(filtered_weights) if filtered_weights else 0.0
        return {name: equal_weight for name in filtered_weights.keys()}
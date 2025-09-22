"""
Configuration settings for PCOS Analyzer Backend

Centralized configuration with environment variable support and automatic model discovery.
All model filenames are discovered dynamically from the models directories.
"""

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List

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

def get_risk_level(score: float) -> str:
    """Classify risk level based on probability score"""
    if score < settings.RISK_LOW_THRESHOLD:
        return "low"
    elif score < settings.RISK_HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"

def discover_models(models_dir: Path) -> Dict[str, Path]:
    """
    Discover all .h5 model files in a directory
    
    Args:
        models_dir: Directory to search for models
        
    Returns:
        Dictionary mapping model names to file paths
    """
    models = {}
    if models_dir.exists():
        for model_file in models_dir.glob("*.h5"):
            model_name = model_file.stem
            models[model_name] = model_file
    return models

def load_model_labels(model_path: Path) -> List[str]:
    """
    Load class labels for a model from corresponding .labels.txt file
    
    Args:
        model_path: Path to the .h5 model file
        
    Returns:
        List of class labels
    """
    labels_path = model_path.with_suffix('.labels.txt')
    
    try:
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                content = f.read().strip()
                
                # Try JSON format first
                if content.startswith('[') and content.endswith(']'):
                    import json
                    labels = json.loads(content)
                else:
                    # Plain text format, one label per line
                    labels = [line.strip() for line in content.split('\n') if line.strip()]
                
                return labels
        else:
            # Default labels if file doesn't exist
            return ["non_pcos", "pcos"]
            
    except Exception as e:
        print(f"Warning: Could not load labels for {model_path}: {e}")
        return ["non_pcos", "pcos"]

def get_ensemble_weights() -> Dict[str, float]:
    """
    Get ensemble weights from environment variables or defaults
    
    Returns:
        Dictionary mapping model names to weights
    """
    weights = settings.DEFAULT_ENSEMBLE_WEIGHTS.copy()
    
    # Override with environment variables if provided
    for model_name in weights.keys():
        env_var = f"ENSEMBLE_WEIGHT_{model_name.upper()}"
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
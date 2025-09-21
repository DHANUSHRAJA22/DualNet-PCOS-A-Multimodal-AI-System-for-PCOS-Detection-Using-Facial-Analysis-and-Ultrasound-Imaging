"""
Configuration settings for PCOS Analyzer Backend

Centralized configuration with environment variable support and model path management.
All model filenames are hardcoded to match the exact files you will provide.
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
    
    # Ensemble weights for face models
    FACE_ENSEMBLE_WEIGHTS: Dict[str, float] = {
        "vgg16": float(os.getenv("FACE_VGG16_WEIGHT", "0.33")),
        "resnet50": float(os.getenv("FACE_RESNET50_WEIGHT", "0.33")),
        "efficientnetb0": float(os.getenv("FACE_EFFICIENTNET_WEIGHT", "0.34"))
    }
    
    # Face model files
    FACE_MODELS: Dict[str, str] = {
        "vgg16": "face_model_vgg16.h5",
        "resnet50": "face_model_resnet50.h5",
        "efficientnetb0": "face_model_efficientnetb0.h5"
    }
    
    # Ensemble weights for X-ray models
    XRAY_ENSEMBLE_WEIGHTS: Dict[str, float] = {
        "vgg16": float(os.getenv("XRAY_VGG16_WEIGHT", "0.33")),
        "resnet50": float(os.getenv("XRAY_RESNET50_WEIGHT", "0.33")),
        "efficientnetb0": float(os.getenv("XRAY_EFFICIENTNET_WEIGHT", "0.34"))
    }
    
    # X-ray model files
    XRAY_MODELS: Dict[str, str] = {
        "vgg16": "xray_model_vgg16.h5",
        "resnet50": "xray_model_resnet50.h5",
        "efficientnetb0": "xray_model_efficientnetb0.h5"
    }
    
    # Best single models (when USE_ENSEMBLE=False)
    BEST_FACE_MODEL: str = "efficientnetb0"
    BEST_XRAY_MODEL: str = "efficientnetb0"
    
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
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
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # CORS configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8080",
        "http://127.0.0.1:8080", 
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    
    # File upload limits
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    # Model filenames (DO NOT CHANGE - these match your provided files)
    GENDER_MODEL: str = "gender_classifier.h5"
    
    FACE_PCOS_MODELS: Dict[str, str] = {
        "vgg16": "pcos_vgg16.h5",
        "resnet50": "pcos_resnet50.h5",
        "efficientnet_b0": "pcos_efficientnet_b0.h5",
    }
    
    XRAY_PCOS_MODELS: Dict[str, str] = {
        "vgg16": "pcos_vgg16.h5",
        "resnet50": "pcos_resnet50.h5",
        "detector_158": "pcos_detector_158.h5",
    }
    
    YOLO_MODEL: str = "bestv8.pt"
    
    # Risk thresholds
    RISK_LOW_THRESHOLD: float = 0.33
    RISK_HIGH_THRESHOLD: float = 0.66
    
    # Image proxy whitelist
    ALLOWED_PROXY_HOSTS: List[str] = [
        "images.pexels.com",
        "example.com",
        "localhost"
    ]

# Global settings instance
settings = Settings()

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FACE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
XRAY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
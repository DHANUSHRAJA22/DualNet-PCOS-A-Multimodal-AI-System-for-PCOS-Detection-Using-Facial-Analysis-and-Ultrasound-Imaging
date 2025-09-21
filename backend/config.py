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
    
    # CORS configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    # File upload configuration
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "5"))
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    STATIC_TTL_SECONDS: int = int(os.getenv("STATIC_TTL_SECONDS", "3600"))
    
    # Model filenames (DO NOT CHANGE - these match your provided files)
    GENDER_MODEL: str = "gender_classifier.h5"
    
    FACE_PCOS_MODELS: Dict[str, Dict] = {
        "vgg16": {
            "path": "face_model_A.h5",
            "input_size": [224, 224],
            "weight": 0.5
        },
        "resnet50": {
            "path": "face_model_B.h5", 
            "input_size": [224, 224],
            "weight": 0.5
        }
    }
    
    XRAY_PCOS_MODELS: Dict[str, Dict] = {
        "xray_a": {
            "path": "xray_model_A.h5",
            "input_size": [224, 224],
            "weight": 0.5
        },
        "xray_b": {
            "path": "xray_model_B.h5",
            "input_size": [224, 224], 
            "weight": 0.5
        }
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
YOLO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
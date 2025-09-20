"""
Pydantic response models for structured API documentation

Defines all request and response schemas for the PCOS Analyzer API
with comprehensive field documentation and validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class ModelStatus(BaseModel):
    """Status information for a single model"""
    status: str = Field(..., description="Model status: 'loaded', 'not_loaded', 'error'")
    file_exists: bool = Field(..., description="Whether the model file exists on disk")
    path: Optional[str] = Field(None, description="Path to model file")
    error: Optional[str] = Field(None, description="Error message if failed to load")
    version: Optional[str] = Field(None, description="Model version if available")

class HealthResponse(BaseModel):
    """Health check response with detailed model status"""
    ok: bool = Field(..., description="Overall system health status")
    success: bool = Field(..., description="Success flag for compatibility")
    overall_status: str = Field(..., description="System status: 'healthy', 'degraded', 'unhealthy'")
    total_models_configured: int = Field(..., description="Total number of configured models")
    total_models_loaded: int = Field(..., description="Number of successfully loaded models")
    models: Dict[str, ModelStatus] = Field(..., description="Detailed status for each model")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Health check timestamp")

class GenderPrediction(BaseModel):
    """Gender classification results"""
    male: float = Field(..., description="Probability of male classification (0.0-1.0)")
    female: float = Field(..., description="Probability of female classification (0.0-1.0)")
    label: str = Field(..., description="Predicted gender label: 'male', 'female', or 'unknown'")

class EnsembleResult(BaseModel):
    """Ensemble prediction results"""
    method: str = Field(..., description="Ensemble method used")
    score: float = Field(..., description="Ensemble PCOS probability score (0.0-1.0)")
    models_used: int = Field(..., description="Number of models contributing to ensemble")
    weights_used: Optional[Dict[str, float]] = Field(None, description="Weights used for each model")

class Detection(BaseModel):
    """YOLO detection result"""
    box: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    conf: float = Field(..., description="Detection confidence (0.0-1.0)")
    label: str = Field(..., description="Detected object class label")

class ROIResult(BaseModel):
    """Region of Interest classification results"""
    roi_id: int = Field(..., description="ROI identifier")
    per_model: Dict[str, float] = Field(..., description="Individual model predictions for this ROI")
    ensemble: EnsembleResult = Field(..., description="Ensemble result for this ROI")

class ModalityResult(BaseModel):
    """Results for a single modality (face or X-ray)"""
    modality: str = Field(..., description="Modality type: 'face' or 'xray'")
    
    # Face-specific fields
    gender: Optional[GenderPrediction] = Field(None, description="Gender classification (face only)")
    
    # X-ray-specific fields
    detections: Optional[List[Detection]] = Field(None, description="YOLO detections (X-ray only)")
    per_roi: Optional[List[ROIResult]] = Field(None, description="Per-ROI results (X-ray only)")
    yolo_vis_url: Optional[str] = Field(None, description="YOLO visualization image URL")
    
    # Common fields
    per_model: Optional[Dict[str, float]] = Field(None, description="Individual model predictions")
    ensemble: Optional[EnsembleResult] = Field(None, description="Ensemble prediction results")

class FinalResult(BaseModel):
    """Final combined prediction results"""
    risk_score: float = Field(..., description="Final PCOS risk score (0.0-1.0)")
    risk_label: str = Field(..., description="Risk classification: 'PCOS-positive', 'PCOS-negative', 'unknown'")

class PredictionResponse(BaseModel):
    """Rich prediction response for new frontend"""
    api_version: str = Field(default="1.0.0", description="API version")
    modalities: List[ModalityResult] = Field(..., description="Results for each processed modality")
    final: FinalResult = Field(..., description="Final combined prediction")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    message: str = Field(default="ok", description="Processing status message")
    debug: Dict[str, Any] = Field(default_factory=dict, description="Debug information")

class LegacyPredictionResponse(BaseModel):
    """Legacy prediction response for current frontend compatibility"""
    ok: bool = Field(default=True, description="Success status")
    
    # Face results
    face_pred: Optional[str] = Field(None, description="Face prediction: 'pcos', 'non_pcos', or null")
    face_scores: Optional[List[float]] = Field(None, description="Face model scores")
    face_img: Optional[str] = Field(None, description="Face image URL")
    face_risk: Optional[str] = Field(None, description="Face risk level")
    
    # X-ray results
    xray_pred: Optional[str] = Field(None, description="X-ray prediction: 'pcos', 'normal', or null")
    xray_img: Optional[str] = Field(None, description="X-ray image URL")
    yolo_vis: Optional[str] = Field(None, description="YOLO visualization URL")
    found_labels: Optional[List[str]] = Field(None, description="Detected object labels")
    xray_risk: Optional[str] = Field(None, description="X-ray risk level")
    
    # Combined results
    combined: Optional[str] = Field(None, description="Combined prediction summary")
    overall_risk: Optional[str] = Field(None, description="Overall risk level")
    message: str = Field(default="ok", description="Processing message")

class ErrorResponse(BaseModel):
    """Error response model"""
    ok: bool = Field(default=False, description="Success status")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
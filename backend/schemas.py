"""
Pydantic response models for structured API documentation

Defines all request and response schemas for the PCOS Analyzer API
with comprehensive field documentation and validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class StandardResponse(BaseModel):
    """Standard API response format"""
    ok: bool = Field(..., description="Success status")
    message: Optional[str] = Field(None, description="Human-readable message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")

class ModelStatus(BaseModel):
    """Status information for a single model"""
    status: str = Field(..., description="Model status: 'loaded', 'not_loaded', 'error'")
    file_exists: bool = Field(..., description="Whether the model file exists on disk")
    lazy_loadable: bool = Field(..., description="Whether the model can be lazy loaded")
    path: Optional[str] = Field(None, description="Path to model file")
    error: Optional[str] = Field(None, description="Error message if failed to load")
    version: Optional[str] = Field(None, description="Model version if available")

class EnhancedHealthResponse(BaseModel):
    """Health check response with detailed model status"""
    status: str = Field(..., description="System status: 'healthy', 'degraded', 'unhealthy'")
    models: Dict[str, ModelStatus] = Field(..., description="Detailed status for each model")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="API version")

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
    box: List[float] = Field(..., description="ROI bounding box [x1, y1, x2, y2]")
    per_model: Dict[str, float] = Field(..., description="Individual model predictions for this ROI")
    ensemble: EnsembleResult = Field(..., description="Ensemble result for this ROI")

class ModalityResult(BaseModel):
    """Results for a single modality (face or X-ray)"""
    type: str = Field(..., description="Modality type: 'face' or 'xray'")
    label: str = Field(..., description="Human-readable prediction label")
    scores: List[float] = Field(..., description="Raw prediction scores")
    risk: str = Field(..., description="Risk level: 'low', 'moderate', 'high', 'unknown'")
    original_img: Optional[str] = Field(None, description="URL to original uploaded image")
    visualization: Optional[str] = Field(None, description="URL to visualization image (X-ray only)")
    found_labels: Optional[List[str]] = Field(None, description="Detected object labels (X-ray only)")
    
    # Face-specific fields
    gender: Optional[GenderPrediction] = Field(None, description="Gender classification (face only)")
    
    # X-ray-specific fields
    detections: Optional[List[Detection]] = Field(None, description="YOLO detections (X-ray only)")
    per_roi: Optional[List[ROIResult]] = Field(None, description="Per-ROI results (X-ray only)")
    
    # Common fields
    per_model: Optional[Dict[str, float]] = Field(None, description="Individual model predictions")
    ensemble: Optional[EnsembleResult] = Field(None, description="Ensemble prediction results")

class FinalResult(BaseModel):
    """Final combined prediction results"""
    risk: str = Field(..., description="Overall risk level: 'low', 'moderate', 'high', 'unknown'")
    confidence: float = Field(..., description="Overall confidence score (0.0-1.0)")
    explanation: str = Field(..., description="Human-readable explanation of the results")
    fusion_mode: str = Field(..., description="Fusion mode used: 'threshold' or 'discrete'")

class StructuredPredictionResponse(BaseModel):
    """Rich prediction response for new frontend"""
    ok: bool = Field(..., description="Success status")
    modalities: List[ModalityResult] = Field(..., description="Results for each processed modality")
    final: FinalResult = Field(..., description="Final combined prediction")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
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
"""
PCOS Analyzer FastAPI Backend - Production Ready

Enhanced version with structured responses, proper validation, ROI processing,
and comprehensive error handling for production deployment.

Author: DHANUSH RAJA (21MIC0158)
Version: 2.0.0
"""

import os
import logging
import traceback
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import httpx
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import settings, STATIC_DIR, UPLOADS_DIR
from managers.face_manager import FaceManager
from managers.xray_manager import XrayManager
from ensemble import EnsembleManager
from utils.validators import validate_request_files, validate_proxy_url, validate_file_size
from schemas import (
    StructuredPredictionResponse, 
    LegacyPredictionResponse, 
    EnhancedHealthResponse,
    ErrorResponse,
    StandardResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PCOS Analyzer API",
    description="AI-powered PCOS screening using ensemble deep learning models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware with environment-based origins
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_str == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize managers
face_manager = FaceManager()
xray_manager = XrayManager()
ensemble_manager = EnsembleManager()

# Track startup time
startup_time = datetime.now()

def cleanup_old_files():
    """Clean up old uploaded files"""
    try:
        current_time = time.time()
        max_age = settings.STATIC_TTL_SECONDS
        
        if not UPLOADS_DIR.exists():
            return
            
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age:
                    try:
                        file_path.unlink()
                        logger.debug(f"Cleaned up old file: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove old file {file_path.name}: {str(e)}")
                        
    except Exception as e:
        logger.error(f"File cleanup failed: {str(e)}")

def classify_risk(score: float) -> str:
    """Classify risk level based on probability score using thresholds"""
    if score < settings.RISK_LOW_THRESHOLD:
        return "low"
    elif score < settings.RISK_HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"

def validate_uploaded_file(file: UploadFile) -> None:
    """Validate uploaded file size and type"""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size (read content to get actual size)
    content = file.file.read()
    file.file.seek(0)  # Reset file pointer
    
    max_size = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({len(content) / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({settings.MAX_UPLOAD_MB}MB)"
        )
    
    # Check MIME type
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed types: {', '.join(settings.ALLOWED_MIME_TYPES)}"
        )

def ensure_json_serializable(obj: Any) -> Any:
    """Convert NumPy types to JSON-serializable Python types"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj

@app.get("/health", response_model=EnhancedHealthResponse)
async def enhanced_health_check():
    """
    Enhanced health check with lazy loading validation
    
    Tests actual model loading capability, not just file existence
    """
    try:
        uptime = (datetime.now() - startup_time).total_seconds()
        
        # Get detailed model status with lazy loading test
        face_status = face_manager.get_model_status()
        xray_status = xray_manager.get_model_status()
        
        # Test lazy loading capability
        models_status = {}
        
        # Face models
        from schemas import ModelStatus
        
        models_status["gender"] = ModelStatus(
            status="loaded" if face_status.get("gender", {}).get("loaded", False) else "not_loaded",
            file_exists=face_status.get("gender", {}).get("available", False),
            lazy_loadable=face_manager.can_lazy_load_gender(),
            error=face_status.get("gender", {}).get("error")
        )
        
        models_status["face"] = ModelStatus(
            status="loaded" if face_status.get("face", {}).get("loaded", False) else "not_loaded",
            file_exists=face_status.get("face", {}).get("available", False),
            lazy_loadable=face_manager.can_lazy_load_pcos(),
            error=face_status.get("face", {}).get("error")
        )
        
        models_status["yolo"] = ModelStatus(
            status="loaded" if xray_status.get("yolo", {}).get("loaded", False) else "not_loaded",
            file_exists=xray_status.get("yolo", {}).get("available", False),
            lazy_loadable=xray_manager.can_lazy_load_yolo(),
            error=xray_status.get("yolo", {}).get("error")
        )
        
        models_status["xray"] = ModelStatus(
            status="loaded" if xray_status.get("xray", {}).get("loaded", False) else "not_loaded",
            file_exists=xray_status.get("xray", {}).get("available", False),
            lazy_loadable=xray_manager.can_lazy_load_pcos(),
            error=xray_status.get("xray", {}).get("error")
        )
        
        # Determine overall status
        loadable_count = sum(1 for model in models_status.values() if model.lazy_loadable)
        total_models = len(models_status)
        
        if loadable_count == 0:
            overall_status = "unhealthy"
        elif loadable_count < total_models:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return EnhancedHealthResponse(
            status=overall_status,
            models=models_status,
            uptime_seconds=uptime,
            version="2.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        from schemas import ModelStatus
        return EnhancedHealthResponse(
            status="error",
            models={
                "gender": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e)),
                "face": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e)),
                "yolo": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e)),
                "xray": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e))
            },
            uptime_seconds=0.0,
            version="2.0.0"
        )

@app.post("/predict", response_model=StructuredPredictionResponse)
async def structured_predict(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None)
):
    """
    Enhanced prediction endpoint with structured response format
    
    Returns comprehensive analysis with modalities array and final assessment
    """
    start_time = datetime.now()
    
    try:
        # Clean up old files
        cleanup_old_files()
        
        # Validate request
        validate_request_files(face_img, xray_img)
        
        # Validate individual files
        if face_img:
            validate_uploaded_file(face_img)
        if xray_img:
            validate_uploaded_file(xray_img)
        
        from schemas import ModalityResult, FinalResult
        
        modalities = []
        warnings = []
        face_score = None
        xray_score = None
        debug_info = {}
        
        # Process face image if provided
        if face_img:
            logger.info("Processing face image")
            try:
                face_result = await face_manager.process_face_image(face_img)
                debug_info["face_processing"] = {
                    "filename": face_img.filename,
                    "models_used": face_result.get("models_used", []),
                    "gender_detected": face_result.get("gender", {}).get("label"),
                    "weights_used": face_result.get("ensemble", {}).get("weights_used") if face_result.get("ensemble") else None
                }
                
                # Extract data for modality result
                face_risk = classify_risk(face_result.get("ensemble_score", 0))
                
                modality = ModalityResult(
                    type="face",
                    label=face_result.get("face_pred", "Analysis failed"),
                    scores=face_result.get("face_scores", []),
                    risk=face_risk,
                    original_img=face_result.get("face_img"),
                    gender=face_result.get("gender"),
                    per_model=face_result.get("per_model"),
                    ensemble=face_result.get("ensemble")
                )
                modalities.append(modality)
                
                # Store score for final fusion
                face_score = face_result.get("ensemble_score")
                
                # Check for male face warning
                if face_result.get("gender", {}).get("label") == "male":
                    warnings.append("Male face detected - PCOS analysis may not be applicable")
                
            except Exception as e:
                logger.error(f"Face processing failed: {str(e)}")
                warnings.append(f"Face analysis failed: {str(e)}")
                
                # Add failed modality
                modality = ModalityResult(
                    type="face",
                    label="Analysis failed",
                    scores=[],
                    risk="unknown"
                )
                modalities.append(modality)
        
        # Process X-ray image if provided
        if xray_img:
            logger.info("Processing X-ray image")
            try:
                xray_result = await xray_manager.process_xray_image(xray_img)
                
                # Enhanced debug info with ROI details
                roi_details = []
                if xray_result.get("per_roi"):
                    for roi in xray_result["per_roi"]:
                        roi_details.append({
                            "roi_id": roi.roi_id,
                            "box": roi.box,
                            "per_model": roi.per_model,
                            "ensemble": roi.ensemble.dict() if roi.ensemble else None
                        })
                
                debug_info["xray_processing"] = {
                    "filename": xray_img.filename,
                    "detections_count": len(xray_result.get("detections", [])),
                    "roi_count": len(xray_result.get("per_roi", [])),
                    "models_used": xray_result.get("models_used", []),
                    "weights_used": xray_result.get("ensemble", {}).get("weights_used") if xray_result.get("ensemble") else None,
                    "roi_details": roi_details,
                    "yolo_confidences": [det.conf for det in xray_result.get("detections", [])]
                }
                
                # Extract data for modality result
                xray_risk = classify_risk(xray_result.get("ensemble_score", 0))
                
                modality = ModalityResult(
                    type="xray",
                    label=xray_result.get("xray_pred", "Analysis failed"),
                    scores=[xray_result.get("ensemble_score", 0)],  # X-ray returns single ensemble score
                    risk=xray_risk,
                    original_img=xray_result.get("xray_img"),
                    visualization=xray_result.get("yolo_vis"),
                    found_labels=xray_result.get("found_labels", []),
                    detections=xray_result.get("detections"),
                    per_roi=xray_result.get("per_roi"),
                    per_model=xray_result.get("per_model"),
                    ensemble=xray_result.get("ensemble")
                )
                modalities.append(modality)
                
                # Store score for final fusion
                xray_score = xray_result.get("ensemble_score")
                
            except Exception as e:
                logger.error(f"X-ray processing failed: {str(e)}")
                warnings.append(f"X-ray analysis failed: {str(e)}")
                
                # Add failed modality
                modality = ModalityResult(
                    type="xray",
                    label="Analysis failed",
                    scores=[],
                    risk="unknown"
                )
                modalities.append(modality)
        
        # Generate final combined result
        final_result = ensemble_manager.combine_modalities(face_score, xray_score)
        debug_info["final_fusion"] = {
            "face_score": face_score,
            "xray_score": xray_score,
            "fusion_method": final_result.get("fusion_method", "weighted_average"),
            "modalities_used": final_result.get("modalities_used", []),
            "thresholds_used": {
                "low": settings.RISK_LOW_THRESHOLD,
                "high": settings.RISK_HIGH_THRESHOLD
            }
        }
        
        # Create final assessment
        final = FinalResult(
            overall_risk=final_result["overall_risk"],
            confidence=final_result.get("final_score", 0.0),
            explanation=final_result["combined"]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Structured prediction completed in {processing_time:.2f}ms")
        
        # Ensure JSON serializable
        response_data = {
            "ok": True,
            "modalities": [ensure_json_serializable(m.dict()) for m in modalities],
            "final": ensure_json_serializable(final.dict()),
            "warnings": warnings,
            "processing_time_ms": processing_time,
            "debug": ensure_json_serializable(debug_info)
        }
        
        return StructuredPredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return StructuredPredictionResponse(
            ok=False,
            modalities=[],
            final=FinalResult(
                overall_risk="unknown",
                confidence=0.0,
                explanation="Analysis failed due to internal error"
            ),
            warnings=[f"Internal error: {str(e)}"],
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            debug={"error": str(e)}
        )

@app.post("/predict-legacy", response_model=LegacyPredictionResponse)
async def legacy_predict(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None)
):
    """
    Legacy prediction endpoint for backward compatibility
    
    Returns results in the flat format expected by existing frontends
    """
    try:
        # Get structured response first
        structured_response = await structured_predict(face_img, xray_img)
        
        # Convert to legacy format
        legacy_response = LegacyPredictionResponse(ok=structured_response.ok)
        
        if structured_response.ok:
            # Extract face data
            face_modality = next((m for m in structured_response.modalities if m["type"] == "face"), None)
            if face_modality:
                legacy_response.face_pred = face_modality["label"]
                legacy_response.face_scores = face_modality["scores"]
                legacy_response.face_img = face_modality.get("original_img")
                legacy_response.face_risk = face_modality["risk"]
            
            # Extract X-ray data
            xray_modality = next((m for m in structured_response.modalities if m["type"] == "xray"), None)
            if xray_modality:
                legacy_response.xray_pred = xray_modality["label"]
                legacy_response.xray_img = xray_modality.get("original_img")
                legacy_response.yolo_vis = xray_modality.get("visualization")
                legacy_response.found_labels = xray_modality.get("found_labels")
                legacy_response.xray_risk = xray_modality["risk"]
            
            # Final results
            legacy_response.combined = structured_response.final["explanation"]
            legacy_response.overall_risk = structured_response.final["overall_risk"]
            legacy_response.message = "ok"
        else:
            # Handle error case
            legacy_response.message = "; ".join(structured_response.warnings) if structured_response.warnings else "Analysis failed"
        
        return legacy_response
        
    except Exception as e:
        logger.error(f"Legacy prediction failed: {str(e)}")
        return LegacyPredictionResponse(
            ok=False,
            message=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-file", response_model=StandardResponse)
async def predict_file(
    file: UploadFile = File(...),
    type: str = Query("face", description="Analysis type: 'face' or 'xray'")
):
    """
    Single file upload endpoint for backward compatibility
    
    Accepts a single file and routes to appropriate analysis pipeline
    """
    try:
        if type == "face":
            result = await structured_predict(face_img=file, xray_img=None)
        elif type == "xray":
            result = await structured_predict(face_img=None, xray_img=file)
        else:
            raise HTTPException(status_code=400, detail="Type must be 'face' or 'xray'")
        
        return StandardResponse(
            ok=result.ok,
            message="Analysis completed successfully" if result.ok else "Analysis failed",
            data=ensure_json_serializable(result.dict())
        )
        
    except Exception as e:
        logger.error(f"File prediction failed: {str(e)}")
        return StandardResponse(
            ok=False,
            message=f"Analysis failed: {str(e)}"
        )

@app.get("/img-proxy")
async def image_proxy(url: str = Query(..., description="Image URL to proxy")):
    """
    Safe CORS image proxy for external images
    
    Fetches images from whitelisted hosts and streams them back with proper
    headers to avoid CORS issues in the frontend.
    """
    if not validate_proxy_url(url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL not allowed for proxy"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            
            # Determine content type
            content_type = response.headers.get("content-type", "image/jpeg")
            
            return StreamingResponse(
                iter([response.content]),
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
    except httpx.HTTPError as e:
        logger.error(f"Image proxy fetch failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not fetch image"
        )
    except Exception as e:
        logger.error(f"Image proxy error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Proxy error"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "ok": False,
            "details": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    """
    Development server entry point
    
    For production deployment:
    - Use gunicorn or uvicorn with multiple workers
    - Set host to 0.0.0.0 for container deployment
    - Configure environment variables
    - Enable access logs and structured logging
    """
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
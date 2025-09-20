"""
PCOS Analyzer FastAPI Backend

Production-ready API for AI-powered PCOS screening using ensemble deep learning models.
Supports facial recognition and X-ray analysis with flexible ensemble prediction logic.

Author: DHANUSH RAJA (21MIC0158)
Version: 1.0.0
"""

import os
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
import httpx
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings, STATIC_DIR
from schemas import (
    HealthResponse, PredictionResponse, LegacyPredictionResponse, 
    ErrorResponse, ModelStatus
)
from managers.face_manager import FaceManager
from managers.xray_manager import XrayManager
from ensemble import EnsembleManager
from utils.validators import ValidationError, validate_proxy_url

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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    
    Returns detailed status of all models including load status,
    file paths, and any error messages.
    
    Returns:
        HealthResponse: Complete system health information
    """
    try:
        # Collect model status from all managers
        all_models = {}
        all_models.update(face_manager.get_model_status())
        all_models.update(xray_manager.get_model_status())
        
        # Convert to ModelStatus objects
        models_status = {
            name: ModelStatus(**status) for name, status in all_models.items()
        }
        
        # Calculate overall statistics
        total_configured = len(models_status)
        total_loaded = sum(1 for status in models_status.values() if status.status == "loaded")
        
        # Determine overall status
        if total_loaded == 0:
            overall_status = "unhealthy"
        elif total_loaded < total_configured:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return HealthResponse(
            ok=True,
            success=True,
            overall_status=overall_status,
            total_models_configured=total_configured,
            total_models_loaded=total_loaded,
            models=models_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return HealthResponse(
            ok=False,
            success=False,
            overall_status="error",
            total_models_configured=0,
            total_models_loaded=0,
            models={}
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_rich(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Rich prediction endpoint with detailed ensemble results
    
    Accepts facial images and/or X-ray images for comprehensive PCOS analysis
    using ensemble deep learning models.
    
    Args:
        face_img: Optional facial image file (JPEG/PNG/WebP, max 5MB)
        xray_img: Optional X-ray image file (JPEG/PNG/WebP, max 5MB)  
        threshold: Confidence threshold for predictions (0.0-1.0)
        
    Returns:
        PredictionResponse: Detailed analysis results with ensemble predictions
        
    Raises:
        HTTPException: 400 for validation errors, 500 for processing errors
    """
    start_time = datetime.now()
    
    if not face_img and not xray_img:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image (face_img or xray_img) must be provided"
        )
    
    try:
        modalities = []
        warnings = []
        debug_info = {}
        
        # Process face image if provided
        if face_img:
            logger.info("Processing face image for rich prediction")
            try:
                face_result = await face_manager.process_face_image(face_img)
                modalities.append(face_result)
                debug_info["face_img_url"] = f"/static/uploads/face-{face_img.filename}"
            except Exception as e:
                logger.error(f"Face processing failed: {str(e)}")
                warnings.append(f"Face analysis failed: {str(e)}")
        
        # Process X-ray image if provided
        if xray_img:
            logger.info("Processing X-ray image for rich prediction")
            try:
                xray_result = await xray_manager.process_xray_image(xray_img)
                modalities.append(xray_result)
                debug_info["xray_img_url"] = f"/static/uploads/xray-{xray_img.filename}"
            except Exception as e:
                logger.error(f"X-ray processing failed: {str(e)}")
                warnings.append(f"X-ray analysis failed: {str(e)}")
        
        # Combine modality results
        face_ensemble = None
        xray_ensemble = None
        
        for modality in modalities:
            if modality["modality"] == "face" and modality.get("ensemble"):
                face_ensemble = modality["ensemble"]
            elif modality["modality"] == "xray" and modality.get("ensemble"):
                xray_ensemble = modality["ensemble"]
        
        # Generate final combined result
        final_result = ensemble_manager.combine_modalities(face_ensemble, xray_ensemble)
        
        # Convert final result to expected format
        final_formatted = {
            "risk_score": final_result["risk_score"],
            "risk_label": "PCOS-positive" if final_result["risk_score"] >= 0.5 else "PCOS-negative"
        }
        
        return PredictionResponse(
            modalities=modalities,
            final=final_formatted,
            warnings=warnings,
            debug=debug_info
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/predict-legacy", response_model=LegacyPredictionResponse)
async def predict_legacy(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None)
):
    """
    Legacy prediction endpoint for current frontend compatibility
    
    Returns results in the exact format expected by the existing frontend,
    ensuring seamless integration without frontend changes.
    
    Args:
        face_img: Optional facial image file
        xray_img: Optional X-ray image file
        
    Returns:
        LegacyPredictionResponse: Results in legacy format
    """
    if not face_img and not xray_img:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image must be provided"
        )
    
    try:
        # Initialize legacy response
        response = LegacyPredictionResponse()
        
        # Process face image
        if face_img:
            try:
                face_result = await face_manager.process_face_image(face_img)
                
                # Convert to legacy format
                if face_result.get("ensemble") and face_result["ensemble"].get("score") is not None:
                    ensemble_score = face_result["ensemble"]["score"]
                    response.face_pred = "pcos" if ensemble_score > 0.5 else "non_pcos"
                    response.face_scores = list(face_result["per_model"].values()) if face_result.get("per_model") else []
                    response.face_risk = ensemble_manager._classify_risk(ensemble_score)
                elif face_result.get("warnings") and "male" in str(face_result["warnings"]).lower():
                    response.face_pred = None  # Male detected
                    response.face_scores = []
                    response.face_risk = "unknown"
                
                # Always provide image URL
                response.face_img = f"/static/uploads/face-{face_img.filename}"
                
            except Exception as e:
                logger.error(f"Legacy face processing failed: {str(e)}")
                response.message = f"Face analysis failed: {str(e)}"
        
        # Process X-ray image
        if xray_img:
            try:
                xray_result = await xray_manager.process_xray_image(xray_img)
                
                # Convert to legacy format
                if xray_result.get("ensemble") and xray_result["ensemble"].get("score") is not None:
                    ensemble_score = xray_result["ensemble"]["score"]
                    response.xray_pred = "pcos" if ensemble_score > 0.5 else "normal"
                    response.xray_risk = ensemble_manager._classify_risk(ensemble_score)
                    
                    # Extract found labels from detections
                    if xray_result.get("detections"):
                        response.found_labels = [det["label"] for det in xray_result["detections"]]
                    
                    # YOLO visualization
                    if xray_result.get("yolo_vis_url"):
                        response.yolo_vis = xray_result["yolo_vis_url"]
                
                # Always provide image URL
                response.xray_img = f"/static/uploads/xray-{xray_img.filename}"
                
            except Exception as e:
                logger.error(f"Legacy X-ray processing failed: {str(e)}")
                response.message = f"X-ray analysis failed: {str(e)}"
        
        # Generate combined assessment
        risk_scores = []
        if response.face_pred == "pcos":
            risk_scores.append(0.7)  # Approximate from face
        elif response.face_pred == "non_pcos":
            risk_scores.append(0.3)
            
        if response.xray_pred == "pcos":
            risk_scores.append(0.7)  # Approximate from X-ray
        elif response.xray_pred == "normal":
            risk_scores.append(0.3)
        
        if risk_scores:
            combined_score = sum(risk_scores) / len(risk_scores)
            response.overall_risk = ensemble_manager._classify_risk(combined_score)
            response.combined = f"Final risk score {combined_score:.2f} → {'PCOS-positive' if combined_score > 0.5 else 'PCOS-negative'}"
        else:
            response.overall_risk = "unknown"
            response.combined = "Insufficient data for combined assessment"
        
        return response
        
    except Exception as e:
        logger.error(f"Legacy prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/img-proxy")
async def image_proxy(url: str = Query(..., description="Image URL to proxy")):
    """
    Safe CORS image proxy for external images
    
    Fetches images from whitelisted hosts and streams them back with proper
    headers to avoid CORS issues in the frontend.
    
    Args:
        url: Image URL to fetch and proxy
        
    Returns:
        StreamingResponse: Proxied image with appropriate headers
        
    Raises:
        HTTPException: 400 for invalid URLs, 404 for fetch failures
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
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred"
        ).dict()
    )

if __name__ == "__main__":
    """
    Development server entry point
    
    For production deployment:
    - Use gunicorn or uvicorn with multiple workers
    - Set host to 0.0.0.0 for container deployment
    - Configure environment variables
    - Enable access logs and structured logging
    
    Example production command:
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
    """
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
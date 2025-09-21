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
import time
from datetime import datetime
from typing import Optional, Dict, Any
import httpx
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings, STATIC_DIR, UPLOADS_DIR
from managers.face_manager import FaceManager
from managers.xray_manager import XrayManager
from ensemble import EnsembleManager
from utils.validators import validate_request_files, validate_proxy_url

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

# Response models
class HealthResponse(BaseModel):
    status: str
    models: Dict[str, bool]

class PredictionResponse(BaseModel):
    ok: bool
    face_pred: Optional[str] = None
    face_scores: Optional[list] = None
    face_img: Optional[str] = None
    face_risk: Optional[str] = None
    xray_pred: Optional[str] = None
    xray_img: Optional[str] = None
    yolo_vis: Optional[str] = None
    found_labels: Optional[list] = None
    xray_risk: Optional[str] = None
    combined: Optional[str] = None
    overall_risk: Optional[str] = None
    message: str = "ok"

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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with model status
    
    Returns detailed status of all models including load status.
    Never crashes on missing models - reports availability instead.
    
    Returns:
        HealthResponse: System health information
    """
    try:
        face_status = face_manager.get_model_status()
        xray_status = xray_manager.get_model_status()
        
        models_status = {
            "gender": face_status.get("gender", {}).get("loaded", False),
            "face": face_status.get("face", {}).get("loaded", False),
            "yolo": xray_status.get("yolo", {}).get("loaded", False),
            "xray": xray_status.get("xray", {}).get("loaded", False)
        }
        
        # Determine overall status
        loaded_count = sum(models_status.values())
        if loaded_count == 0:
            overall_status = "unhealthy"
        elif loaded_count < len(models_status):
            overall_status = "degraded"
        else:
            overall_status = "ok"
        
        return HealthResponse(
            status=overall_status,
            models=models_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return HealthResponse(
            status="error",
            models={
                "gender": False,
                "face": False,
                "yolo": False,
                "xray": False
            }
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None)
):
    """
    Main prediction endpoint with rich response format
    
    Accepts facial images and/or X-ray images for comprehensive PCOS analysis
    using ensemble deep learning models.
    
    Args:
        face_img: Optional facial image file (JPEG/PNG/WebP, max 5MB)
        xray_img: Optional X-ray image file (JPEG/PNG/WebP, max 5MB)
        
    Returns:
        PredictionResponse: Detailed analysis results
        
    Raises:
        HTTPException: 400 for validation errors, 500 for processing errors
    """
    start_time = datetime.now()
    
    # Clean up old files
    cleanup_old_files()
    
    # Validate request
    validate_request_files(face_img, xray_img)
    
    try:
        result = PredictionResponse(ok=True)
        face_score = None
        xray_score = None
        
        # Process face image if provided
        if face_img:
            logger.info("Processing face image")
            try:
                face_result = await face_manager.process_face_image(face_img)
                
                result.face_pred = face_result.get("face_pred")
                result.face_scores = face_result.get("face_scores", [])
                result.face_img = face_result.get("face_img")
                result.face_risk = face_result.get("face_risk", "unknown")
                
                # Get ensemble score for final combination
                face_score = face_result.get("ensemble_score")
                
            except Exception as e:
                logger.error(f"Face processing failed: {str(e)}")
                result.face_pred = f"Face analysis failed: {str(e)}"
                result.face_risk = "unknown"
        
        # Process X-ray image if provided
        if xray_img:
            logger.info("Processing X-ray image")
            try:
                xray_result = await xray_manager.process_xray_image(xray_img)
                
                result.xray_pred = xray_result.get("xray_pred")
                result.xray_img = xray_result.get("xray_img")
                result.yolo_vis = xray_result.get("yolo_vis")
                result.found_labels = xray_result.get("found_labels", [])
                result.xray_risk = xray_result.get("xray_risk", "unknown")
                
                # Get ensemble score for final combination
                xray_score = xray_result.get("ensemble_score")
                
            except Exception as e:
                logger.error(f"X-ray processing failed: {str(e)}")
                result.xray_pred = f"X-ray analysis failed: {str(e)}"
                result.xray_risk = "unknown"
        
        # Generate final combined result
        final_result = ensemble_manager.combine_modalities(face_score, xray_score)
        result.overall_risk = final_result["overall_risk"]
        result.combined = final_result["combined"]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Prediction completed in {processing_time:.2f}ms")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/predict-legacy", response_model=PredictionResponse)
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
        PredictionResponse: Results in legacy format
    """
    # Use the same logic as /predict
    return await predict(face_img, xray_img)

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
        content={
            "ok": False,
            "message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
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
    
    Example production command:
    uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
    """
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
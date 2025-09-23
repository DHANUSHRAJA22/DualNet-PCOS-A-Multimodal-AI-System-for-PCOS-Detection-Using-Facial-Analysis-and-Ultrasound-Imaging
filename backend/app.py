"""
PCOS Analyzer FastAPI Backend - Ensemble Production Ready

Enhanced version with automatic model discovery, ensemble inference,
structured responses, and comprehensive error handling for production deployment.

Author: DHANUSH RAJA (21MIC0158)
Version: 3.0.0
"""

import os
import logging
import traceback
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import httpx
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query, Request
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
    StandardResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pcos-backend")

# Initialize FastAPI app
app = FastAPI(
    title="PCOS Analyzer API",
    description="AI-powered PCOS screening with automatic model discovery and ensemble inference",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enhanced CORS middleware with environment-based origins
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
            detail=(
                f"File size ({len(content) / 1024 / 1024:.1f}MB) exceeds maximum allowed size "
                f"({settings.MAX_UPLOAD_MB}MB)"
            ),
        )

    # Check MIME type
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed types: {', '.join(settings.ALLOWED_MIME_TYPES)}",
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


def get_risk_level(score: float) -> str:
    """Helper function to determine risk level from score"""
    if score < settings.RISK_LOW_THRESHOLD:
        return "low"
    elif score < settings.RISK_HIGH_THRESHOLD:
        return "moderate"
    else:
        return "high"


@app.get("/health", response_model=EnhancedHealthResponse)
async def enhanced_health_check():
    """Enhanced health check with detailed model discovery status"""
    try:
        uptime = (datetime.now() - startup_time).total_seconds()

        # Get model status
        face_status = face_manager.get_model_status()
        xray_status = xray_manager.get_model_status()

        models_status = {}

        from schemas import ModelStatus

        # Face models
        models_status["gender"] = ModelStatus(
            status="loaded" if face_status.get("gender", {}).get("loaded", False) else "not_loaded",
            file_exists=face_status.get("gender", {}).get("available", False),
            lazy_loadable=face_manager.can_lazy_load_gender(),
            error=face_status.get("gender", {}).get("error"),
        )

        models_status["face"] = ModelStatus(
            status="loaded" if face_status.get("face", {}).get("loaded", False) else "not_loaded",
            file_exists=face_status.get("face", {}).get("available", False),
            lazy_loadable=face_manager.can_lazy_load_pcos(),
            error=face_status.get("face", {}).get("error"),
        )

        # X-ray/YOLO models
        models_status["yolo"] = ModelStatus(
            status="loaded" if xray_status.get("yolo", {}).get("loaded", False) else "not_loaded",
            file_exists=xray_status.get("yolo", {}).get("available", False),
            lazy_loadable=xray_manager.can_lazy_load_yolo(),
            error=xray_status.get("yolo", {}).get("error"),
        )

        models_status["xray"] = ModelStatus(
            status="loaded" if xray_status.get("xray", {}).get("loaded", False) else "not_loaded",
            file_exists=xray_status.get("xray", {}).get("available", False),
            lazy_loadable=xray_manager.can_lazy_load_pcos(),
            error=xray_status.get("xray", {}).get("error"),
        )

        # Add individual model details (if provided by managers)
        if "pcos_models" in face_status:
            for model_name, model_info in face_status["pcos_models"].items():
                models_status[f"face_{model_name}"] = ModelStatus(
                    status="loaded",
                    file_exists=True,
                    lazy_loadable=True,
                    path=model_info.get("path"),
                    version=f"weight_{model_info.get('weight', 0):.2f}",
                )

        if "pcos_models" in xray_status:
            for model_name, model_info in xray_status["pcos_models"].items():
                models_status[f"xray_{model_name}"] = ModelStatus(
                    status="loaded",
                    file_exists=True,
                    lazy_loadable=True,
                    path=model_info.get("path"),
                    version=f"weight_{model_info.get('weight', 0):.2f}",
                )

        # Determine overall status
        loadable_count = sum(1 for model in models_status.values() if getattr(model, "lazy_loadable", False))
        total_models = len(models_status)

        if loadable_count == 0:
            overall_status = "unhealthy"
        elif loadable_count < total_models:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        response_data = {
            status=overall_status,
            models=models_status,
            uptime_seconds=uptime,
            version="3.0.0",
            # Add configuration info
            "config": {
                "fusion_mode": settings.FUSION_MODE,
                "use_ensemble": settings.USE_ENSEMBLE,
                "risk_thresholds": {
                    "low": settings.RISK_LOW_THRESHOLD,
                    "high": settings.RISK_HIGH_THRESHOLD
                },
                "max_upload_mb": settings.MAX_UPLOAD_MB
            }
        }
        
        return EnhancedHealthResponse(**response_data)

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.debug(traceback.format_exc())

        from schemas import ModelStatus

        return EnhancedHealthResponse(
            status="error",
            models={
                "gender": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e)),
                "face_ensemble": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e)),
                "yolo": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e)),
                "xray_ensemble": ModelStatus(status="error", file_exists=False, lazy_loadable=False, error=str(e)),
            },
            uptime_seconds=0.0,
            version="3.0.0",
        )


@app.post("/predict", response_model=StructuredPredictionResponse)
async def structured_predict(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None),
) -> Union[StructuredPredictionResponse, JSONResponse]:
    """Enhanced prediction endpoint with ensemble inference and structured response"""
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

        modalities: List[ModalityResult] = []
        warnings: List[str] = []
        face_score: Optional[float] = None
        xray_score: Optional[float] = None

        debug_info: Dict[str, Any] = {
            "filenames": [],
            "models_used": [],
            "weights": {},
            "roi_boxes": [],
            "fusion_mode": settings.FUSION_MODE,
            "use_ensemble": settings.USE_ENSEMBLE,
        }

        # Process face image if provided
        if face_img:
            logger.info("Processing face image")
            debug_info["filenames"].append(face_img.filename)

            try:
                face_result = await face_manager.process_face_image(face_img)

                modality = ModalityResult(
                    type="face",
                    label=face_result.get("face_pred", "Analysis failed"),
                    scores=face_result.get("face_scores", []),
                    risk=face_result.get("face_risk", "unknown"),
                    original_img=face_result.get("face_img"),
                    gender=face_result.get("gender"),
                    per_model=face_result.get("per_model"),
                    ensemble=face_result.get("ensemble"),
                )
                modalities.append(modality)

                face_score = face_result.get("ensemble_score")

                debug_info["models_used"].extend(face_result.get("models_used", []))

                if face_result.get("ensemble") and hasattr(face_result["ensemble"], "weights_used"):
                    debug_info["weights"]["face"] = face_result["ensemble"].weights_used

                # Heuristic warning for gender
                gender_info = face_result.get("gender") or {}
                if gender_info.get("label") == "male":
                    warnings.append("Male face detected - PCOS analysis may not be applicable")

            except Exception as e:
                logger.error(f"Face processing failed: {str(e)}")
                logger.debug(traceback.format_exc())
                warnings.append(f"Face analysis failed: {str(e)}")
                modality = ModalityResult(type="face", label="Analysis failed", scores=[], risk="unknown")
                modalities.append(modality)

        # Process X-ray image if provided
        if xray_img:
            logger.info("Processing X-ray image")
            debug_info["filenames"].append(xray_img.filename)

            try:
                xray_result = await xray_manager.process_xray_image(xray_img)

                modality = ModalityResult(
                    type="xray",
                    label=xray_result.get("xray_pred", "Analysis failed"),
                    scores=[xray_result.get("ensemble_score", 0)],
                    risk=xray_result.get("xray_risk", "unknown"),
                    original_img=xray_result.get("xray_img"),
                    visualization=xray_result.get("yolo_vis"),
                    found_labels=xray_result.get("found_labels", []),
                    detections=xray_result.get("detections"),
                    per_roi=xray_result.get("per_roi"),
                    per_model=xray_result.get("per_model"),
                    ensemble=xray_result.get("ensemble"),
                )
                modalities.append(modality)

                xray_score = xray_result.get("ensemble_score")

                debug_info["models_used"].extend(xray_result.get("models_used", []))
                if xray_result.get("ensemble") and hasattr(xray_result["ensemble"], "weights_used"):
                    debug_info["weights"]["xray"] = xray_result["ensemble"].weights_used

                if xray_result.get("per_roi"):
                    for roi in xray_result["per_roi"]:
                        debug_info["roi_boxes"].append(
                            {
                                "roi_id": getattr(roi, "roi_id", None),
                                "box": getattr(roi, "box", None),
                                "confidence": getattr(roi, "confidence", 0.0),
                            }
                        )

            except Exception as e:
                logger.error(f"X-ray processing failed: {str(e)}")
                logger.debug(traceback.format_exc())
                warnings.append(f"X-ray analysis failed: {str(e)}")
                modality = ModalityResult(type="xray", label="Analysis failed", scores=[], risk="unknown")
                modalities.append(modality)

        # Generate final combined result
        if face_score is not None and xray_score is not None:
            combined_score = (face_score + xray_score) / 2
            combined_risk = get_risk_level(combined_score)
            explanation = f"Combined analysis indicates {combined_risk} PCOS risk based on both facial and X-ray analysis"
        elif face_score is not None:
            combined_score = face_score
            combined_risk = get_risk_level(combined_score)
            explanation = f"Facial analysis indicates {combined_risk} PCOS risk"
        elif xray_score is not None:
            combined_score = xray_score
            combined_risk = get_risk_level(combined_score)
            explanation = f"X-ray analysis indicates {combined_risk} PCOS risk"
        else:
            combined_score = 0.0
            combined_risk = "unknown"
            explanation = "No analysis results available"

        final = FinalResult(
            risk=combined_risk,
            confidence=combined_score,
            explanation=explanation,
            fusion_mode=settings.FUSION_MODE,
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000.0
        logger.info(f"Structured prediction completed in {processing_time:.2f}ms")

        response_data = {
            "ok": True,
            "modalities": [ensure_json_serializable(m.dict()) for m in modalities],
            "final": ensure_json_serializable(final.dict()),
            "warnings": warnings,
            "processing_time_ms": processing_time,
            "debug": ensure_json_serializable(debug_info),
        }

        return StructuredPredictionResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured prediction failed: {str(e)}")
        logger.debug(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={"ok": False, "details": str(e) if settings.DEBUG else "Internal server error"},
        )


@app.post("/predict-legacy", response_model=LegacyPredictionResponse)
async def legacy_predict(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None),
):
    """Legacy prediction endpoint for backward compatibility"""
    try:
        structured = await structured_predict(face_img, xray_img)

        # If the structured endpoint returned a JSONResponse (error), pass it through
        if isinstance(structured, JSONResponse):
            return structured

        legacy_response = LegacyPredictionResponse(ok=structured.ok)

        if structured.ok:
            face_modality = next((m for m in structured.modalities if m.type == "face"), None)
            if face_modality:
                legacy_response.face_pred = face_modality.label
                legacy_response.face_scores = face_modality.scores
                legacy_response.face_img = face_modality.original_img
                legacy_response.face_risk = face_modality.risk

            xray_modality = next((m for m in structured.modalities if m.type == "xray"), None)
            if xray_modality:
                legacy_response.xray_pred = xray_modality.label
                legacy_response.xray_img = xray_modality.original_img
                legacy_response.yolo_vis = xray_modality.visualization
                legacy_response.found_labels = xray_modality.found_labels
                legacy_response.xray_risk = xray_modality.risk

            legacy_response.combined = structured.final.explanation
            legacy_response.overall_risk = structured.final.risk
            legacy_response.message = "ok"
        else:
            legacy_response.message = "error"
            legacy_response.overall_risk = "unknown"
            legacy_response.combined = "Analysis failed"

        return legacy_response

    except Exception as e:
        logger.error(f"Legacy prediction failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"ok": False, "details": str(e) if settings.DEBUG else "Internal server error"},
        )


@app.post("/predict-file", response_model=StandardResponse)
async def predict_file(
    file: UploadFile = File(...),
    type: str = Query("auto", description="Analysis type: 'face', 'xray', or 'auto'"),
):
    """Single file upload endpoint with auto-detection"""
    try:
        if type == "auto":
            filename = (file.filename or "").lower()
            if any(keyword in filename for keyword in ["face", "portrait", "selfie"]):
                type = "face"
            elif any(keyword in filename for keyword in ["xray", "x-ray", "scan", "ultrasound"]):
                type = "xray"
            else:
                type = "face"

        if type == "face":
            result = await structured_predict(face_img=file, xray_img=None)
        elif type == "xray":
            result = await structured_predict(face_img=None, xray_img=file)
        else:
            raise HTTPException(status_code=400, detail="Type must be 'face', 'xray', or 'auto'")

        if isinstance(result, JSONResponse):
            # Bubble up the error JSON (from structured_predict)
            return result

        return StandardResponse(
            ok=result.ok,
            message="Analysis completed successfully" if result.ok else "Analysis failed",
            data=ensure_json_serializable(result.dict()),
        )

    except Exception as e:
        logger.error(f"File prediction failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"ok": False, "details": str(e) if settings.DEBUG else "Internal server error"},
        )


@app.get("/img-proxy")
async def image_proxy(url: str = Query(..., description="Image URL to proxy")):
    """Safe CORS image proxy for external images"""
    if not validate_proxy_url(url):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URL not allowed for proxy")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "image/jpeg")

            return StreamingResponse(
                iter([response.content]),
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*",
                },
            )

    except httpx.HTTPError as e:
        logger.error(f"Image proxy fetch failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not fetch image")
    except Exception as e:
        logger.error(f"Image proxy error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Proxy error")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.debug(traceback.format_exc())

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"ok": False, "details": str(exc) if settings.DEBUG else "An unexpected error occurred"},
    )


if __name__ == "__main__":
    """Development server entry point"""
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
    )

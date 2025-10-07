"""
PCOS Analyzer FastAPI Backend - Ensemble Production Ready

Enhanced version with automatic model discovery, ensemble inference,
structured responses, and comprehensive error handling for production deployment.

Author: DHANUSH RAJA (21MIC0158)
Version: 3.0.0
"""

from __future__ import annotations

import logging
import traceback
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import settings, STATIC_DIR, UPLOADS_DIR
from managers.face_manager import FaceManager
from managers.xray_manager import XrayManager
from ensemble import EnsembleManager
from utils.validators import validate_request_files, validate_proxy_url
from schemas import (
    StructuredPredictionResponse,
    LegacyPredictionResponse,
    EnhancedHealthResponse,
    ErrorResponse,       # imported for completeness
    StandardResponse,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pcos-backend")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# FastAPI app (declare BEFORE using decorators)
# -----------------------------------------------------------------------------
app = FastAPI(
    title="PCOS Analyzer API",
    description="AI-powered PCOS screening with automatic model discovery and ensemble inference",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -----------------------------------------------------------------------------
# Managers
# -----------------------------------------------------------------------------
face_manager = FaceManager()
xray_manager = XrayManager()
ensemble_manager = EnsembleManager()  # kept for future fusion needs

startup_time = datetime.now()

# -----------------------------------------------------------------------------
# Startup warmup (after managers are created)
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def _warmup_models() -> None:
    try:
        await face_manager.warmup()
    except Exception as e:
        logger.warning(f"Face warmup skipped: {e}")
    try:
        await xray_manager.warmup()
    except Exception as e:
        logger.warning(f"X-ray warmup skipped: {e}")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def cleanup_old_files() -> None:
    """Clean up old uploaded files under /static/uploads based on TTL."""
    try:
        current_time = time.time()
        max_age = settings.STATIC_TTL_SECONDS
        if not UPLOADS_DIR.exists():
            return
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.is_file():
                if current_time - file_path.stat().st_mtime > max_age:
                    try:
                        file_path.unlink()
                        logger.debug(f"Cleaned up old file: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove old file {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")

def validate_uploaded_file(file: UploadFile) -> None:
    """Validate uploaded file size and mime."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    # read to measure size
    content = file.file.read()
    file.file.seek(0)
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({len(content)/1024/1024:.1f}MB) exceeds max {settings.MAX_UPLOAD_MB}MB",
        )
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        allowed = ", ".join(settings.ALLOWED_MIME_TYPES)
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file.content_type}'. Allowed: {allowed}")

def ensure_json_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-safe python primitives."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_json_serializable(x) for x in obj]
    return obj

def get_risk_level(score: float) -> str:
    if score < settings.RISK_LOW_THRESHOLD:
        return "low"
    elif score < settings.RISK_HIGH_THRESHOLD:
        return "moderate"
    return "high"

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health", response_model=EnhancedHealthResponse)
async def enhanced_health_check():
    """Enhanced health check with detailed model discovery status."""
    try:
        uptime = (datetime.now() - startup_time).total_seconds()

        face_status = face_manager.get_model_status()
        xray_status = xray_manager.get_model_status()

        from schemas import ModelStatus

        models_status: Dict[str, ModelStatus] = {}

        # Face models
        models_status["gender"] = ModelStatus(
            status="loaded" if face_status.get("gender", {}).get("loaded", False) else "not_loaded",
            file_exists=face_status.get("gender", {}).get("available", False),
            lazy_loadable=face_manager.can_lazy_load_gender(),
            error=face_status.get("gender", {}).get("error"),
        )
        models_status["face_ensemble"] = ModelStatus(
            status="loaded" if face_status.get("face", {}).get("loaded", False) else "not_loaded",
            file_exists=face_status.get("face", {}).get("available", False),
            lazy_loadable=face_manager.can_lazy_load_pcos(),
            error=face_status.get("face", {}).get("error"),
        )

        # X-ray / YOLO
        models_status["yolo"] = ModelStatus(
            status="loaded" if xray_status.get("yolo", {}).get("loaded", False) else "not_loaded",
            file_exists=xray_status.get("yolo", {}).get("available", False),
            lazy_loadable=getattr(xray_manager, "can_lazy_load_yolo", lambda: False)(),
            error=xray_status.get("yolo", {}).get("error"),
        )
        # For xray ensemble, some versions of XrayManager may not expose a "can_lazy_load_*" helper.
        models_status["xray_ensemble"] = ModelStatus(
            status="loaded" if xray_status.get("xray", {}).get("loaded", False) else "not_loaded",
            file_exists=xray_status.get("xray", {}).get("available", False),
            lazy_loadable=bool(xray_status.get("xray", {}).get("available", False)),
            error=xray_status.get("xray", {}).get("error"),
        )

        # List individual models (if provided)
        if "pcos_models" in face_status:
            for model_name, info in face_status["pcos_models"].items():
                models_status[f"face_{model_name}"] = ModelStatus(
                    status="loaded", file_exists=True, lazy_loadable=True,
                    path=info.get("path"), version=f"weight_{info.get('weight', 0):.2f}"
                )
        if "pcos_models" in xray_status:
            for model_name, info in xray_status["pcos_models"].items():
                models_status[f"xray_{model_name}"] = ModelStatus(
                    status="loaded", file_exists=True, lazy_loadable=True,
                    path=info.get("path"), version=f"weight_{info.get('weight', 0):.2f}"
                )

        loadable_count = sum(1 for m in models_status.values() if getattr(m, "lazy_loadable", False))
        overall_status = "healthy" if loadable_count == len(models_status) else ("degraded" if loadable_count > 0 else "unhealthy")

        return EnhancedHealthResponse(
            status=overall_status,
            models=models_status,
            uptime_seconds=uptime,
            version="3.0.0",
            config={
                "fusion_mode": settings.FUSION_MODE,
                "use_ensemble": settings.USE_ENSEMBLE,
                "risk_thresholds": {"low": settings.RISK_LOW_THRESHOLD, "high": settings.RISK_HIGH_THRESHOLD},
                "max_upload_mb": settings.MAX_UPLOAD_MB,
            },
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
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
    """Enhanced prediction endpoint with ensemble inference and structured response."""
    start_time = datetime.now()
    try:
        cleanup_old_files()
        validate_request_files(face_img, xray_img)
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

        # Face
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

                # FIX: ensemble is a dict, not an object
                ens_face = face_result.get("ensemble") or {}
                if isinstance(ens_face, dict) and "weights_used" in ens_face:
                    debug_info["weights"]["face"] = ens_face["weights_used"]

                gender_info = face_result.get("gender") or {}
                if gender_info.get("label") == "male":
                    warnings.append("Male face detected - PCOS analysis may not be applicable")
                warnings.extend(face_manager.get_loading_warnings())
            except Exception as e:
                logger.error(f"Face processing failed: {e}")
                logger.debug(traceback.format_exc())
                warnings.append(f"Face analysis failed: {e}")
                modalities.append(ModalityResult(type="face", label="Analysis failed", scores=[], risk="unknown"))

        # X-ray
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

                # FIX: ensemble is a dict, not an object
                ens_x = xray_result.get("ensemble") or {}
                if isinstance(ens_x, dict) and "weights_used" in ens_x:
                    debug_info["weights"]["xray"] = ens_x["weights_used"]

                # FIX: be robust if per_roi is a list of dicts
                for roi in xray_result.get("per_roi", []) or []:
                    if isinstance(roi, dict):
                        debug_info["roi_boxes"].append(
                            {
                                "roi_id": roi.get("roi_id"),
                                "box": roi.get("box"),
                                "confidence": float(roi.get("confidence", 0.0)),
                            }
                        )
                warnings.extend(xray_manager.get_loading_warnings())
            except Exception as e:
                logger.error(f"X-ray processing failed: {e}")
                logger.debug(traceback.format_exc())
                warnings.append(f"X-ray analysis failed: {e}")
                modalities.append(ModalityResult(type="xray", label="Analysis failed", scores=[], risk="unknown"))

        # Final fusion
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
            explanation = "Analysis unavailable - no models could process the uploaded images"

        from schemas import FinalResult
        final = FinalResult(
            risk=combined_risk, confidence=combined_score, explanation=explanation, fusion_mode=settings.FUSION_MODE
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000.0
        logger.info(f"Structured prediction completed in {processing_time:.2f}ms with {len(warnings)} warnings")

        return StructuredPredictionResponse(
            ok=True,
            modalities=[ensure_json_serializable(m.dict()) for m in modalities],
            final=ensure_json_serializable(final.dict()),
            warnings=warnings,
            processing_time_ms=processing_time,
            debug=ensure_json_serializable(debug_info),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured prediction failed: {e}")
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
    """Legacy prediction endpoint for backward compatibility."""
    try:
        structured = await structured_predict(face_img, xray_img)
        if isinstance(structured, JSONResponse):
            return structured

        legacy = LegacyPredictionResponse(ok=structured.ok)
        if structured.ok:
            face_mod = next((m for m in structured.modalities if m["type"] == "face"), None)
            if face_mod:
                legacy.face_pred = face_mod.get("label")
                legacy.face_scores = face_mod.get("scores", [])
                legacy.face_img = face_mod.get("original_img")
                legacy.face_risk = face_mod.get("risk")
            xray_mod = next((m for m in structured.modalities if m["type"] == "xray"), None)
            if xray_mod:
                legacy.xray_pred = xray_mod.get("label")
                legacy.xray_img = xray_mod.get("original_img")
                legacy.yolo_vis = xray_mod.get("visualization")
                legacy.found_labels = xray_mod.get("found_labels", [])
                legacy.xray_risk = xray_mod.get("risk")
            legacy.combined = structured.final["explanation"]
            legacy.overall_risk = structured.final["risk"]
            legacy.message = "ok"
        else:
            legacy.message = "error"
            legacy.overall_risk = "unknown"
            legacy.combined = "Analysis failed"
        return legacy
    except Exception as e:
        logger.error(f"Legacy prediction failed: {e}")
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
    """Single file upload endpoint with auto-detection."""
    try:
        if type == "auto":
            name = (file.filename or "").lower()
            if any(k in name for k in ["xray", "x-ray", "scan", "ultrasound"]):
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
            return result

        return StandardResponse(
            ok=result.ok,
            message="Analysis completed successfully" if result.ok else "Analysis failed",
            data=ensure_json_serializable(result.dict()),
        )
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"ok": False, "details": str(e) if settings.DEBUG else "Internal server error"},
        )

@app.get("/img-proxy")
async def image_proxy(url: str = Query(..., description="Image URL to proxy")):
    """Safe CORS image proxy for external images."""
    if not validate_proxy_url(url):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URL not allowed for proxy")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "image/jpeg")
            return StreamingResponse(
                iter([resp.content]),
                media_type=ctype,
                headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"},
            )
    except httpx.HTTPError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not fetch image")
    except Exception as e:
        logger.error(f"Image proxy error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Proxy error")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    logger.debug(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"ok": False, "details": str(exc) if settings.DEBUG else "An unexpected error occurred"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
    )

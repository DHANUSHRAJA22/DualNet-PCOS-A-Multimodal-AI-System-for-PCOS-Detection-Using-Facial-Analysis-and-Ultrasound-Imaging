"""
X-ray analysis manager for YOLO detection and PCOS classification

Handles automatic discovery and loading of all X-ray analysis models with
YOLO object detection and dynamic ensemble inference.
"""

import os
import uuid
import logging
import json
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile

# Import TensorFlow and suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ultralytics YOLO not available")

from config import (
    settings, XRAY_MODELS_DIR, YOLO_MODELS_DIR, UPLOADS_DIR, get_risk_level,
    get_available_xray_models, get_model_labels, get_ensemble_weights, normalize_weights,
    XRAY_MODELS, BEST_XRAY_MODEL, XRAY_ENSEMBLE_WEIGHTS
)
from utils.validators import validate_image, get_safe_filename

logger = logging.getLogger(__name__)

class XrayManager:
    """
    Manages ensemble inference of X-ray analysis models
    
    Loads multiple X-ray models and YOLO for detection, performs ensemble inference
    with configurable weights and ROI processing.
    """
    
    def __init__(self):
        """Initialize X-ray manager and load models"""
        self.yolo_model = None
        self.pcos_models = {}  # Dict[str, Dict[str, Any]]
        self.can_detect_objects = False
        self.ensemble_weights = get_ensemble_weights('xray')
        
        # Model status tracking
        self.model_status = {
            "yolo": {"loaded": False, "available": False, "error": None},
            "xray": {"loaded": False, "available": False, "error": None}
        }
        
        self._load_models()
    
    def can_lazy_load_yolo(self) -> bool:
        """Check if YOLO model can be lazy loaded"""
        if not YOLO_AVAILABLE:
            return False
        yolo_path = YOLO_MODELS_DIR / settings.YOLO_MODEL
        return yolo_path.exists() and yolo_path.is_file()
    
    def can_lazy_load_pcos(self) -> bool:
        """Check if any PCOS models can be lazy loaded"""
        available_models = get_available_xray_models()
        return len(available_models) > 0
    
    def _load_models(self) -> None:
        """Load all X-ray analysis models"""
        logger.info("Loading X-ray analysis models...")
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Load PCOS classification models
        self._load_pcos_models()
        
        logger.info(f"X-ray manager initialized. YOLO detection: {self.can_detect_objects}, "
                   f"PCOS models loaded: {len(self.pcos_models)}")
    
    def _load_yolo_model(self) -> None:
        """Load YOLO object detection model"""
        if not YOLO_AVAILABLE:
            self.model_status["yolo"]["error"] = "Ultralytics not available"
            return
            
        yolo_path = YOLO_MODELS_DIR / settings.YOLO_MODEL
        
        self.model_status["yolo"]["available"] = yolo_path.exists()
        
        try:
            if yolo_path.exists():
                self.yolo_model = YOLO(str(yolo_path))
                self.can_detect_objects = True
                self.model_status["yolo"]["loaded"] = True
                logger.info(f"Loaded YOLO model: {yolo_path}")
            else:
                logger.warning(f"YOLO model not found: {yolo_path}")
                self.model_status["yolo"]["error"] = "File not found"
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model_status["yolo"]["error"] = str(e)
    
    def _load_pcos_models(self) -> None:
        """Load PCOS classification models with ensemble support"""
        # Get available models
        available_models = get_available_xray_models()
        
        logger.info(f"Available X-ray models: {list(available_models.keys())}")
        
        loaded_count = 0
        
        # Load models based on USE_ENSEMBLE setting
        if settings.USE_ENSEMBLE:
            models_to_load = available_models
        else:
            # Load only the best single model
            if BEST_XRAY_MODEL in available_models:
                models_to_load = {BEST_XRAY_MODEL: available_models[BEST_XRAY_MODEL]}
            else:
                # Fallback to first available model
                if available_models:
                    first_model = next(iter(available_models.items()))
                    models_to_load = {first_model[0]: first_model[1]}
                else:
                    models_to_load = {}
        
        for model_name, model_path in models_to_load.items():
            try:
                # Load model
                model = tf.keras.models.load_model(str(model_path), compile=False)
                
                # Get weight for this model
                weight = self.ensemble_weights.get(model_name, 1.0)
                
                self.pcos_models[model_name] = {
                    "model": model,
                    "path": str(model_path),
                    "weight": weight
                }
                loaded_count += 1
                logger.info(f"Loaded X-ray model {model_name}: {model_path} (weight: {weight})")
                    
            except Exception as e:
                logger.error(f"Failed to load PCOS model {model_name}: {str(e)}")
                continue
        
        # Normalize weights for loaded models
        if self.pcos_models:
            model_names = list(self.pcos_models.keys())
            normalized_weights = normalize_weights(self.ensemble_weights, model_names)
            
            for model_name, weight in normalized_weights.items():
                if model_name in self.pcos_models:
                    self.pcos_models[model_name]["weight"] = weight
        
        self.model_status["xray"]["loaded"] = loaded_count > 0
        self.model_status["xray"]["available"] = len(available_models) > 0
        
        if loaded_count > 0:
            logger.info(f"Successfully loaded {loaded_count} X-ray models with normalized weights")
    
    def _preprocess_image(self, image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Args:
            image_bytes: Raw image bytes
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize with high-quality resampling
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0,1]
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    async def detect_objects(self, image_bytes: bytes) -> Tuple[List[Dict], str]:
        """
        Run YOLO object detection on X-ray image
        
        Args:
            image_bytes: X-ray image bytes
            
        Returns:
            Tuple of (detections_list, visualization_path)
        """
        detections = []
        vis_path = ""
        
        if not self.can_detect_objects or self.yolo_model is None:
            logger.warning("YOLO model not available, skipping object detection")
            return detections, vis_path
        
        try:
            # Convert bytes to PIL Image for YOLO
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run YOLO detection
            results = self.yolo_model(image, verbose=False)
            
            # Extract detections
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        detection = {
                            "box": [float(x) for x in boxes.xyxy[i].tolist()],  # [x1, y1, x2, y2]
                            "conf": float(boxes.conf[i]),
                            "label": result.names[int(boxes.cls[i])]
                        }
                        detections.append(detection)
            
            # Save visualization if we have detections
            if detections and results:
                vis_array = results[0].plot()
                vis_image = Image.fromarray(vis_array)
                
                # Generate unique filename for visualization
                vis_filename = f"yolo-{uuid.uuid4().hex[:8]}.jpg"
                vis_full_path = UPLOADS_DIR / vis_filename
                vis_image.save(vis_full_path, 'JPEG', quality=90)
                vis_path = f"/static/uploads/{vis_filename}"
            
            logger.info(f"YOLO detection completed: {len(detections)} objects found")
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}")
            # Return empty results on failure
            
        return detections, vis_path
    
    async def predict_pcos_ensemble_roi(self, image_bytes: bytes, roi_box: List[float]) -> Dict[str, Any]:
        """
        Predict PCOS from a specific ROI using ensemble of all loaded models
        
        Args:
            image_bytes: X-ray image bytes
            roi_box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with per-model predictions and ensemble result
        """
        per_model_predictions = {}
        per_model_scores = {}
        
        try:
            # Open image and crop ROI
            image = Image.open(io.BytesIO(image_bytes))
            roi = image.crop(roi_box)
            
            # Convert ROI back to bytes for preprocessing
            roi_bytes = io.BytesIO()
            roi.save(roi_bytes, format='JPEG')
            roi_bytes = roi_bytes.getvalue()
            
            for model_name, model_data in self.pcos_models.items():
                try:
                    model = model_data["model"]
                    
                    # Use standard input size (224x224 for most models)
                    input_size = (224, 224)
                    
                    # Preprocess ROI
                    roi_array = self._preprocess_image(roi_bytes, input_size)
                    
                    # Run prediction
                    prediction = model.predict(roi_array, verbose=0)
                    
                    # Extract probabilities
                    if prediction.shape[1] == 1:
                        # Single output (sigmoid)
                        pcos_prob = float(prediction[0][0])
                        probs = [1.0 - pcos_prob, pcos_prob]
                    else:
                        # Multiple outputs (softmax)
                        probs = [float(p) for p in prediction[0]]
                        pcos_prob = probs[1]
                    
                    # Store per-model results
                    per_model_predictions[model_name] = probs
                    per_model_scores[model_name] = pcos_prob
                    
                    logger.debug(f"X-ray {model_name} ROI prediction: {probs}")
                    
                except Exception as e:
                    logger.error(f"PCOS ROI prediction failed for {model_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"ROI processing failed: {str(e)}")
        
        if not per_model_predictions:
            return {
                "per_model": {},
                "per_model_scores": {},
                "ensemble": {"prob": 0.0, "label": "unknown", "weights": {}},
                "labels": get_model_labels('xray')
            }
        
        # Compute ensemble prediction
        ensemble_result = self._compute_ensemble(per_model_predictions)
        
        return {
            "per_model": per_model_predictions,
            "per_model_scores": per_model_scores,
            "ensemble": ensemble_result,
            "labels": get_model_labels('xray')
        }
    
    async def predict_pcos_ensemble_full_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict PCOS from full X-ray image using ensemble of all loaded models
        
        Args:
            image_bytes: X-ray image bytes
            
        Returns:
            Dictionary with per-model predictions and ensemble result
        """
        per_model_predictions = {}
        per_model_scores = {}
        
        for model_name, model_data in self.pcos_models.items():
            try:
                model = model_data["model"]
                
                # Use standard input size (224x224 for most models)
                input_size = (224, 224)
                
                # Preprocess full image
                image_array = self._preprocess_image(image_bytes, input_size)
                
                # Run prediction
                prediction = model.predict(image_array, verbose=0)
                
                # Extract probabilities
                if prediction.shape[1] == 1:
                    # Single output (sigmoid)
                    pcos_prob = float(prediction[0][0])
                    probs = [1.0 - pcos_prob, pcos_prob]
                else:
                    # Multiple outputs (softmax)
                    probs = [float(p) for p in prediction[0]]
                    pcos_prob = probs[1]
                
                # Store per-model results
                per_model_predictions[model_name] = probs
                per_model_scores[model_name] = pcos_prob
                
                logger.debug(f"X-ray {model_name} full image prediction: {probs}")
                
            except Exception as e:
                logger.error(f"PCOS full image prediction failed for {model_name}: {str(e)}")
                continue
        
        if not per_model_predictions:
            return {
                "per_model": {},
                "per_model_scores": {},
                "ensemble": {"prob": 0.0, "label": "unknown", "weights": {}},
                "labels": get_model_labels('xray')
            }
        
        # Compute ensemble prediction
        ensemble_result = self._compute_ensemble(per_model_predictions)
        
        return {
            "per_model": per_model_predictions,
            "per_model_scores": per_model_scores,
            "ensemble": ensemble_result,
            "labels": get_model_labels('xray')
        }
    
    def _compute_ensemble(self, per_model_predictions: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Compute weighted ensemble prediction from per-model results
        
        Args:
            per_model_predictions: Dictionary mapping model names to probability lists
            
        Returns:
            Dictionary with ensemble results
        """
        if not per_model_predictions:
            return {"prob": 0.0, "label": "unknown", "weights": {}}
        
        # Get weights for available models
        available_models = list(per_model_predictions.keys())
        weights = {name: self.pcos_models[name]["weight"] for name in available_models}
        
        # Compute weighted average for each class
        num_classes = len(next(iter(per_model_predictions.values())))
        ensemble_probs = [0.0] * num_classes
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            for model_name, probs in per_model_predictions.items():
                weight = weights[model_name]
                for i, prob in enumerate(probs):
                    ensemble_probs[i] += prob * weight / total_weight
        else:
            # Equal weighting fallback
            for probs in per_model_predictions.values():
                for i, prob in enumerate(probs):
                    ensemble_probs[i] += prob / len(per_model_predictions)
        
        # Determine final prediction
        max_prob_idx = np.argmax(ensemble_probs)
        max_prob = ensemble_probs[max_prob_idx]
        
        # Use first model's labels as reference
        labels = get_model_labels('xray')
        predicted_label = labels[max_prob_idx] if max_prob_idx < len(labels) else "unknown"
        
        return {
            "prob": float(max_prob),
            "probs": ensemble_probs,
            "label": predicted_label,
            "weights": weights,
            "models_used": len(available_models)
        }
    
    async def process_xray_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded X-ray image and run full analysis pipeline
        
        Args:
            file: Uploaded X-ray image file
            
        Returns:
            Dictionary with complete X-ray analysis results
        """
        # Validate uploaded file
        image_bytes = await validate_image(file, max_mb=settings.MAX_UPLOAD_MB)
        
        # Generate unique filename and save
        file_id = str(uuid.uuid4())[:8]
        safe_filename = get_safe_filename(file.filename)
        name, ext = os.path.splitext(safe_filename)
        filename = f"xray-{file_id}-{name}.jpg"
        file_path = UPLOADS_DIR / filename
        
        # Save uploaded file
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        try:
            # Run YOLO detection
            detections, yolo_vis_url = await self.detect_objects(image_bytes)
            
            # Initialize result structure
            result = {
                "xray_img": f"/static/uploads/{filename}",
                "yolo_vis": yolo_vis_url,
                "found_labels": [],
                "xray_pred": None,
                "xray_risk": "unknown",
                "xray_models": {},
                "per_model": {},
                "ensemble_score": 0.0
            }
            
            # Extract found labels from detections
            if detections:
                result["found_labels"] = [det["label"] for det in detections]
            
            # Process ROIs if detections found
            if detections:
                all_roi_predictions = {}
                all_roi_scores = {}
                
                for roi_id, detection in enumerate(detections):
                    roi_results = await self.predict_pcos_ensemble_roi(image_bytes, detection["box"])
                    if roi_results["per_model"]:
                        # Accumulate predictions across ROIs
                        for model_name, probs in roi_results["per_model"].items():
                            if model_name not in all_roi_predictions:
                                all_roi_predictions[model_name] = []
                            all_roi_predictions[model_name].append(probs)
                        
                        for model_name, score in roi_results["per_model_scores"].items():
                            if model_name not in all_roi_scores:
                                all_roi_scores[model_name] = []
                            all_roi_scores[model_name].append(score)
                
                # Ensemble ROI predictions
                if all_roi_predictions:
                    # Average predictions across all ROIs for each model
                    averaged_predictions = {}
                    averaged_scores = {}
                    
                    for model_name, roi_probs_list in all_roi_predictions.items():
                        # Average probabilities across ROIs
                        num_classes = len(roi_probs_list[0])
                        avg_probs = [0.0] * num_classes
                        for probs in roi_probs_list:
                            for i, prob in enumerate(probs):
                                avg_probs[i] += prob / len(roi_probs_list)
                        averaged_predictions[model_name] = avg_probs
                    
                    for model_name, roi_scores_list in all_roi_scores.items():
                        # Average scores across ROIs
                        averaged_scores[model_name] = sum(roi_scores_list) / len(roi_scores_list)
                    
                    # Store per-model predictions
                    result["xray_models"] = averaged_predictions
                    result["per_model"] = averaged_scores
                    
                    # Compute ensemble
                    ensemble_result = self._compute_ensemble(averaged_predictions)
                    ensemble_score = ensemble_result["prob"]
                    
                    result["ensemble_score"] = ensemble_score
                    
                    # Determine prediction label and risk
                    risk_level = get_risk_level(ensemble_score)
                    result["xray_risk"] = risk_level
                    
                    if risk_level == "high":
                        result["xray_pred"] = "PCOS symptoms detected in X-ray"
                    elif risk_level == "moderate":
                        result["xray_pred"] = "Moderate PCOS indicators in X-ray"
                    else:
                        result["xray_pred"] = "No PCOS symptoms detected in X-ray"
                else:
                    result["xray_pred"] = "No PCOS models available for ROI analysis"
            
            else:
                # No detections - classify full image
                full_image_results = await self.predict_pcos_ensemble_full_image(image_bytes)
                
                if full_image_results["per_model"]:
                    # Store per-model predictions
                    result["xray_models"] = full_image_results["per_model"]
                    result["per_model"] = full_image_results["per_model_scores"]
                    
                    # Get ensemble results
                    ensemble = full_image_results["ensemble"]
                    ensemble_score = ensemble["prob"]
                    
                    result["ensemble_score"] = ensemble_score
                    
                    # Determine prediction label and risk
                    risk_level = get_risk_level(ensemble_score)
                    result["xray_risk"] = risk_level
                    
                    if risk_level == "high":
                        result["xray_pred"] = "PCOS symptoms detected in X-ray"
                    elif risk_level == "moderate":
                        result["xray_pred"] = "Moderate PCOS indicators in X-ray"
                    else:
                        result["xray_pred"] = "No PCOS symptoms detected in X-ray"
                else:
                    result["xray_pred"] = "No PCOS models available for analysis"
            
            return result
            
        except Exception as e:
            logger.error(f"X-ray processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all X-ray models"""
        status = self.model_status.copy()
        
        # Add detailed model information
        status["pcos_models"] = {}
        for model_name, model_data in self.pcos_models.items():
            status["pcos_models"][model_name] = {
                "loaded": True,
                "path": model_data["path"],
                "weight": model_data["weight"]
            }
        
        return status
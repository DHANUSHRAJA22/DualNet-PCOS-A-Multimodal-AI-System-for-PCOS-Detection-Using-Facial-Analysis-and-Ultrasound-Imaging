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
    discover_models, load_model_labels, get_ensemble_weights, normalize_weights
)
from utils.validators import validate_image, get_safe_filename

logger = logging.getLogger(__name__)

class XrayManager:
    """
    Manages automatic discovery and inference of X-ray analysis models
    
    Automatically discovers all .h5 models in xray directory, loads YOLO for detection,
    and supports both single model and ensemble inference with ROI processing.
    """
    
    def __init__(self):
        """Initialize X-ray manager and load models"""
        self.yolo_model = None
        self.pcos_models = {}  # Dict[str, Dict[str, Any]]
        self.can_detect_objects = False
        self.ensemble_weights = get_ensemble_weights()
        
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
        discovered_models = discover_models(XRAY_MODELS_DIR)
        return len(discovered_models) > 0
    
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
        """Discover and load all PCOS classification models"""
        # Discover all .h5 models in xray directory
        discovered_models = discover_models(XRAY_MODELS_DIR)
        
        logger.info(f"Discovered X-ray models: {list(discovered_models.keys())}")
        
        loaded_count = 0
        
        # Load models based on USE_ENSEMBLE setting
        if settings.USE_ENSEMBLE:
            models_to_load = discovered_models
        else:
            # Load only the first available model for single model mode
            if discovered_models:
                first_model = next(iter(discovered_models.items()))
                models_to_load = {first_model[0]: first_model[1]}
            else:
                models_to_load = {}
        
        for model_name, model_path in models_to_load.items():
            try:
                # Load model
                model = tf.keras.models.load_model(str(model_path), compile=False)
                
                # Load corresponding labels
                labels = load_model_labels(model_path)
                
                # Get weight for this model
                weight = self.ensemble_weights.get(model_name, 1.0)
                
                self.pcos_models[model_name] = {
                    "model": model,
                    "labels": labels,
                    "path": str(model_path),
                    "weight": weight
                }
                loaded_count += 1
                logger.info(f"Loaded X-ray model {model_name}: {model_path} (weight: {weight})")
                    
            except Exception as e:
                logger.error(f"Failed to load PCOS model {model_name}: {str(e)}")
        
        # Normalize weights for loaded models
        if self.pcos_models:
            model_names = list(self.pcos_models.keys())
            normalized_weights = normalize_weights(self.ensemble_weights, model_names)
            
            for model_name, weight in normalized_weights.items():
                if model_name in self.pcos_models:
                    self.pcos_models[model_name]["weight"] = weight
        
        self.model_status["xray"]["loaded"] = loaded_count > 0
        self.model_status["xray"]["available"] = len(discovered_models) > 0
        
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
    
    async def predict_pcos_roi(self, image_bytes: bytes, roi_box: List[float]) -> Dict[str, Any]:
        """
        Predict PCOS from a specific ROI using ensemble of all loaded models
        
        Args:
            image_bytes: X-ray image bytes
            roi_box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with per-model predictions and ensemble result
        """
        per_model_predictions = {}
        per_model_labels = {}
        
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
                    labels = model_data["labels"]
                    
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
                    
                    # Store per-model results
                    per_model_predictions[model_name] = probs
                    per_model_labels[model_name] = labels
                    
                    logger.debug(f"X-ray {model_name} ROI prediction: {probs}")
                    
                except Exception as e:
                    logger.error(f"PCOS ROI prediction failed for {model_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"ROI processing failed: {str(e)}")
        
        if not per_model_predictions:
            return {
                "per_model": {},
                "ensemble": {"prob": 0.0, "label": "unknown", "weights": {}},
                "labels": ["normal", "pcos"]
            }
        
        # Compute ensemble prediction
        ensemble_result = self._compute_ensemble(per_model_predictions)
        
        return {
            "per_model": per_model_predictions,
            "per_model_labels": per_model_labels,
            "ensemble": ensemble_result,
            "labels": list(per_model_labels.values())[0] if per_model_labels else ["normal", "pcos"]
        }
    
    async def predict_pcos_full_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict PCOS from full X-ray image using ensemble of all loaded models
        
        Args:
            image_bytes: X-ray image bytes
            
        Returns:
            Dictionary with per-model predictions and ensemble result
        """
        per_model_predictions = {}
        per_model_labels = {}
        
        for model_name, model_data in self.pcos_models.items():
            try:
                model = model_data["model"]
                labels = model_data["labels"]
                
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
                
                # Store per-model results
                per_model_predictions[model_name] = probs
                per_model_labels[model_name] = labels
                
                logger.debug(f"X-ray {model_name} full image prediction: {probs}")
                
            except Exception as e:
                logger.error(f"PCOS full image prediction failed for {model_name}: {str(e)}")
                continue
        
        if not per_model_predictions:
            return {
                "per_model": {},
                "ensemble": {"prob": 0.0, "label": "unknown", "weights": {}},
                "labels": ["normal", "pcos"]
            }
        
        # Compute ensemble prediction
        ensemble_result = self._compute_ensemble(per_model_predictions)
        
        return {
            "per_model": per_model_predictions,
            "per_model_labels": per_model_labels,
            "ensemble": ensemble_result,
            "labels": list(per_model_labels.values())[0] if per_model_labels else ["normal", "pcos"]
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
        first_model = next(iter(self.pcos_models.values()))
        labels = first_model["labels"]
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
                "detections": [],
                "per_roi": [],
                "per_model": {},
                "ensemble": None,
                "models_used": []
            }
            
            # Extract found labels from detections
            if detections:
                result["found_labels"] = [det["label"] for det in detections]
                
                # Convert detections to schema format
                from schemas import Detection
                result["detections"] = [
                    Detection(
                        box=det["box"],
                        conf=det["conf"],
                        label=det["label"]
                    ) for det in detections
                ]
            
            # Process ROIs if detections found
            if detections:
                roi_predictions_list = []
                roi_results = []
                
                for roi_id, detection in enumerate(detections):
                    roi_predictions = await self.predict_pcos_roi(image_bytes, detection["box"])
                    if roi_predictions:
                        roi_predictions_list.append(roi_predictions)
                        
                        # Create ROI result with ensemble
                        weights = {name: self.pcos_models[name]["weight"] for name in roi_predictions.keys()}
                        roi_ensemble = self.ensemble_manager.combine_xray_models(roi_predictions, weights)
                        
                        from schemas import ROIResult, EnsembleResult
                        roi_result = ROIResult(
                            roi_id=roi_id,
                            box=detection["box"],
                            per_model=roi_predictions,
                            ensemble=EnsembleResult(
                                method=roi_ensemble["method"],
                                score=roi_ensemble["score"],
                                models_used=roi_ensemble["models_used"],
                                weights_used=roi_ensemble.get("weights_used")
                            )
                        )
                        roi_results.append(roi_result)
                
                result["per_roi"] = roi_results
                
                # Ensemble ROI predictions
                if roi_predictions_list:
                    # Average predictions across all ROIs
                    all_models = set()
                    for roi_pred in roi_predictions_list:
                        all_models.update(roi_pred.keys())
                    
                    averaged_predictions = {}
                    for model_name in all_models:
                        scores = [roi_pred.get(model_name, 0) for roi_pred in roi_predictions_list if model_name in roi_pred]
                        if scores:
                            averaged_predictions[model_name] = sum(scores) / len(scores)
                    
                    # Store per-model predictions
                    result["per_model"] = averaged_predictions
                    result["models_used"] = list(averaged_predictions.keys())
                    
                    # Extract weights for ensemble (only for loaded models)
                    weights = {name: self.pcos_models[name]["weight"] for name in averaged_predictions.keys()}
                    
                    # Run ensemble
                    ensemble_result = self.ensemble_manager.combine_xray_models(averaged_predictions, weights)
                    final_score = ensemble_result["score"]
                    
                    # Store ensemble metadata
                    from schemas import EnsembleResult
                    result["ensemble"] = EnsembleResult(
                        method=ensemble_result["method"],
                        score=final_score,
                        models_used=ensemble_result["models_used"],
                        weights_used=ensemble_result.get("weights_used")
                    )
                    
                    # Determine prediction label and risk
                    risk_level = get_risk_level(final_score)
                    result["xray_risk"] = risk_level
                    
                    if risk_level == "high":
                        result["xray_pred"] = "PCOS symptoms detected in X-ray"
                    elif risk_level == "moderate":
                        result["xray_pred"] = "Moderate PCOS indicators in X-ray"
                    else:
                        result["xray_pred"] = "No PCOS symptoms detected in X-ray"
                    
                    result["ensemble_score"] = final_score
                else:
                    result["xray_pred"] = "No PCOS models available for ROI analysis"
            
            else:
                # No detections - classify full image
                full_image_predictions = await self.predict_pcos_full_image(image_bytes)
                
                if full_image_predictions:
                    # Store per-model predictions
                    result["per_model"] = full_image_predictions
                    result["models_used"] = list(full_image_predictions.keys())
                    
                    # Extract weights for ensemble (only for loaded models)
                    weights = {name: self.pcos_models[name]["weight"] for name in full_image_predictions.keys()}
                    
                    # Run ensemble
                    ensemble_result = self.ensemble_manager.combine_xray_models(full_image_predictions, weights)
                    final_score = ensemble_result["score"]
                    
                    # Store ensemble metadata
                    from schemas import EnsembleResult
                    result["ensemble"] = EnsembleResult(
                        method=ensemble_result["method"],
                        score=final_score,
                        models_used=ensemble_result["models_used"],
                        weights_used=ensemble_result.get("weights_used")
                    )
                    
                    # Determine prediction label and risk
                    risk_level = get_risk_level(final_score)
                    result["xray_risk"] = risk_level
                    
                    if risk_level == "high":
                        result["xray_pred"] = "PCOS symptoms detected in X-ray"
                    elif risk_level == "moderate":
                        result["xray_pred"] = "Moderate PCOS indicators in X-ray"
                    else:
                        result["xray_pred"] = "No PCOS symptoms detected in X-ray"
                    
                    result["ensemble_score"] = final_score
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
                "weight": model_data["weight"],
                "labels": model_data["labels"]
            }
        
        return status
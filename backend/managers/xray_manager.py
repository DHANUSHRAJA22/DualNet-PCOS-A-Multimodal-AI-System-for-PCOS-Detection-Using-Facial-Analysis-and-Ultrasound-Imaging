"""
X-ray analysis manager for YOLO detection and PCOS classification

Handles loading and inference of X-ray analysis models including YOLO object
detection and PCOS ensemble prediction with proper error handling.
"""

import os
import uuid
import logging
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
    logger.warning("Ultralytics YOLO not available")

from config import settings, XRAY_MODELS_DIR, UPLOADS_DIR
from utils.validators import validate_image, get_safe_filename
from ensemble import EnsembleManager

logger = logging.getLogger(__name__)

class XrayManager:
    """
    Manages X-ray analysis including YOLO detection and PCOS classification
    
    Loads and manages YOLO model for object detection and multiple TensorFlow
    models for PCOS classification with ensemble prediction capabilities.
    """
    
    def __init__(self):
        """Initialize X-ray manager and load models"""
        self.yolo_model = None
        self.pcos_models = {}
        self.can_detect_objects = False
        self.ensemble_manager = EnsembleManager()
        
        # Model status tracking
        self.model_status = {
            "yolo_v8": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
            "xray_vgg16": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
            "xray_resnet50": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
            "xray_detector_158": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
        }
        
        self._load_models()
    
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
            self.model_status["yolo_v8"]["error"] = "Ultralytics not available"
            return
            
        yolo_path = XRAY_MODELS_DIR / settings.YOLO_MODEL
        
        self.model_status["yolo_v8"]["file_exists"] = yolo_path.exists()
        self.model_status["yolo_v8"]["path"] = str(yolo_path)
        
        try:
            if yolo_path.exists():
                self.yolo_model = YOLO(str(yolo_path))
                self.can_detect_objects = True
                
                self.model_status["yolo_v8"]["status"] = "loaded"
                logger.info(f"Loaded YOLO model: {yolo_path}")
            else:
                logger.warning(f"YOLO model not found: {yolo_path}")
                self.model_status["yolo_v8"]["error"] = "File not found"
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model_status["yolo_v8"]["status"] = "error"
            self.model_status["yolo_v8"]["error"] = str(e)
    
    def _load_pcos_models(self) -> None:
        """Load PCOS classification models"""
        for model_name, filename in settings.XRAY_PCOS_MODELS.items():
            model_path = XRAY_MODELS_DIR / filename
            status_key = f"xray_{model_name}"
            
            self.model_status[status_key]["file_exists"] = model_path.exists()
            self.model_status[status_key]["path"] = str(model_path)
            
            try:
                if model_path.exists():
                    model = tf.keras.models.load_model(str(model_path), compile=False)
                    self.pcos_models[model_name] = model
                    
                    self.model_status[status_key]["status"] = "loaded"
                    logger.info(f"Loaded PCOS model {model_name}: {model_path}")
                else:
                    logger.warning(f"PCOS model {model_name} not found: {model_path}")
                    self.model_status[status_key]["error"] = "File not found"
                    
            except Exception as e:
                logger.error(f"Failed to load PCOS model {model_name}: {str(e)}")
                self.model_status[status_key]["status"] = "error"
                self.model_status[status_key]["error"] = str(e)
    
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
                            "box": boxes.xyxy[i].tolist(),  # [x1, y1, x2, y2]
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
    
    async def predict_pcos_roi(self, image_bytes: bytes, roi_box: List[float]) -> Dict[str, float]:
        """
        Predict PCOS from a specific ROI using ensemble of models
        
        Args:
            image_bytes: X-ray image bytes
            roi_box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary mapping model names to PCOS probability scores
        """
        predictions = {}
        
        try:
            # Open image and crop ROI
            image = Image.open(io.BytesIO(image_bytes))
            roi = image.crop(roi_box)
            
            # Convert ROI back to bytes for preprocessing
            roi_bytes = io.BytesIO()
            roi.save(roi_bytes, format='JPEG')
            roi_bytes = roi_bytes.getvalue()
            
            for model_name, model in self.pcos_models.items():
                try:
                    # Preprocess ROI
                    roi_array = self._preprocess_image(roi_bytes, (224, 224))
                    
                    # Run prediction
                    prediction = model.predict(roi_array, verbose=0)
                    
                    # Extract PCOS probability
                    if prediction.shape[1] == 1:
                        # Single output (sigmoid)
                        pcos_prob = float(prediction[0][0])
                    else:
                        # Two outputs (softmax) - assume [non_pcos, pcos]
                        pcos_prob = float(prediction[0][1])
                    
                    predictions[model_name] = pcos_prob
                    logger.debug(f"X-ray {model_name} ROI prediction: {pcos_prob:.3f}")
                    
                except Exception as e:
                    logger.error(f"PCOS ROI prediction failed for {model_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"ROI processing failed: {str(e)}")
        
        return predictions
    
    async def predict_pcos_full_image(self, image_bytes: bytes) -> Dict[str, float]:
        """
        Predict PCOS from full X-ray image using ensemble of models
        
        Args:
            image_bytes: X-ray image bytes
            
        Returns:
            Dictionary mapping model names to PCOS probability scores
        """
        predictions = {}
        
        for model_name, model in self.pcos_models.items():
            try:
                # Preprocess full image
                image_array = self._preprocess_image(image_bytes, (224, 224))
                
                # Run prediction
                prediction = model.predict(image_array, verbose=0)
                
                # Extract PCOS probability
                if prediction.shape[1] == 1:
                    # Single output (sigmoid)
                    pcos_prob = float(prediction[0][0])
                else:
                    # Two outputs (softmax) - assume [non_pcos, pcos]
                    pcos_prob = float(prediction[0][1])
                
                predictions[model_name] = pcos_prob
                logger.debug(f"X-ray {model_name} full image prediction: {pcos_prob:.3f}")
                
            except Exception as e:
                logger.error(f"PCOS full image prediction failed for {model_name}: {str(e)}")
                continue
        
        return predictions
    
    async def process_xray_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded X-ray image and run full analysis pipeline
        
        Args:
            file: Uploaded X-ray image file
            
        Returns:
            Dictionary with complete X-ray analysis results
        """
        # Validate uploaded file
        image_bytes = await validate_image(file)
        
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
                "modality": "xray",
                "detections": detections,
                "per_roi": [],
                "yolo_vis_url": yolo_vis_url,
                "per_model": {},
                "ensemble": {},
                "warnings": []
            }
            
            # Process ROIs if detections found
            if detections:
                roi_results = []
                all_roi_predictions = {}
                
                for i, detection in enumerate(detections):
                    roi_predictions = await self.predict_pcos_roi(image_bytes, detection["box"])
                    
                    if roi_predictions:
                        roi_ensemble = self.ensemble_manager.combine_xray_models(roi_predictions)
                        
                        roi_result = {
                            "roi_id": i,
                            "per_model": roi_predictions,
                            "ensemble": roi_ensemble
                        }
                        roi_results.append(roi_result)
                        
                        # Accumulate predictions for overall ensemble
                        for model_name, score in roi_predictions.items():
                            if model_name not in all_roi_predictions:
                                all_roi_predictions[model_name] = []
                            all_roi_predictions[model_name].append(score)
                
                result["per_roi"] = roi_results
                
                # Average ROI predictions for overall score
                if all_roi_predictions:
                    averaged_predictions = {
                        model_name: sum(scores) / len(scores)
                        for model_name, scores in all_roi_predictions.items()
                    }
                    result["per_model"] = averaged_predictions
                    result["ensemble"] = self.ensemble_manager.combine_xray_models(averaged_predictions)
            
            else:
                # No detections - classify full image
                full_image_predictions = await self.predict_pcos_full_image(image_bytes)
                result["per_model"] = full_image_predictions
                
                if full_image_predictions:
                    result["ensemble"] = self.ensemble_manager.combine_xray_models(full_image_predictions)
                else:
                    result["warnings"].append("No PCOS models available for classification")
            
            return result
            
        except Exception as e:
            logger.error(f"X-ray processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all X-ray models"""
        return self.model_status.copy()
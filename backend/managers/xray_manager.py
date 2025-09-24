"""
X-ray analysis manager for YOLO detection and PCOS classification

Handles automatic discovery and loading of all X-ray analysis models with
robust fallback for Keras version incompatibilities and graceful degradation.
"""

import os
import uuid
import logging
import json
import h5py
import json
import h5py
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json

from config import (
    settings, XRAY_MODELS_DIR, YOLO_MODELS_DIR, UPLOADS_DIR, get_risk_level,
    get_available_xray_models, load_model_labels, get_ensemble_weights, normalize_weights
)
from config_runtime import XRAY_COMPILE, XRAY_ALLOW_FALLBACK
from ensemble import robust_weighted_ensemble
from utils.validators import validate_image, get_safe_filename

logger = logging.getLogger(__name__)

def _safe_load_h5(path, compile_model=False):
    """
    Robust H5 model loader with fallbacks for batch_shape and config issues
    
    Args:
        path: Path to .h5 model file
        compile_model: Whether to compile the model
        
    Returns:
        Loaded model or None if all methods fail
    """
    try:
        return load_model(path, compile=compile_model)
    except Exception as e1:
        logger.warning(f"[xray] load_model failed for {path}: {e1}")
        
        # Try reading model_config from H5 attrs
        try:
            with h5py.File(path, "r") as f:
                if "model_config" in f.attrs:
                    cfg = f.attrs["model_config"].decode("utf-8")
                    try:
                        model = model_from_json(cfg)
                        model.load_weights(path, by_name=True, skip_mismatch=True)
                        logger.info(f"[xray] Successfully loaded via model_from_json: {path}")
                        return model
                    except Exception as e2:
                        logger.warning(f"[xray] model_from_json fallback failed: {e2}")
        except Exception as e3:
            logger.warning(f"[xray] h5py open failed: {e3}")
        
        # Final fallback: try with custom objects
        try:
            custom_objects = {"GlorotUniform": tf.keras.initializers.glorot_uniform()}
            model = load_model(path, compile=compile_model, custom_objects=custom_objects)
            logger.info(f"[xray] Successfully loaded with custom_objects: {path}")
            return model
        except Exception as e4:
            logger.error(f"[xray] All load fallbacks failed for {path}: {e4}")
            return None

# Compiled prediction function to avoid retracing
@tf.function(reduce_retracing=True)
def compiled_predict_xray(model, input_data):
    """Compiled prediction function for X-ray models to avoid TensorFlow retracing warnings"""
    return model(input_data, training=False)

def _try_load_full_model(path: str):
    """Try to load model normally, return None if Keras version mismatch"""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except TypeError as e:
        # Keras 3/2 serialization mismatch
        if "Unrecognized keyword arguments" in str(e) or "batch_shape" in str(e):
            logger.warning(f"Keras version mismatch for {path}, will try weights-only fallback")
            return None
        raise
    except Exception as e:
        logger.warning(f"Failed to load model {path}: {str(e)}")
        return None

def _build_resnet50_xray(input_shape=(100, 100, 3), num_classes=2):
    """Build ResNet50 architecture for X-ray analysis"""
    base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

def _build_vgg16_xray(input_shape=(100, 100, 3), num_classes=2):
    """Build VGG16 architecture for X-ray analysis"""
    base = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=input_shape)
    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

def _build_efficientnet_xray(input_shape=(100, 100, 3), num_classes=2):
    """Build EfficientNet architecture for X-ray analysis"""
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

def _build_detector_158(input_shape=(100, 100, 3), num_classes=2):
    """Build custom detector architecture for X-ray analysis"""
    base = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=input_shape)
    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

def load_with_weights_fallback(model_path: str, arch: str, input_shape=(100, 100, 3), num_classes=2):
    """
    Load X-ray model with fallback to weights-only loading for Keras version mismatches
    
    Args:
        model_path: Path to model file
        arch: Architecture name (resnet50, vgg16, efficientnet, detector_158)
        input_shape: Input shape for model
        num_classes: Number of output classes
        
    Returns:
        Loaded model or None if failed
    """
    # 1) Try full model loading first
    model = _try_load_full_model(model_path)
    if model is not None:
        logger.info(f"Successfully loaded full X-ray model: {model_path}")
        return model
    
    # 2) Rebuild architecture and load weights only
    logger.info(f"Attempting weights-only fallback for X-ray model {model_path} with architecture {arch}")
    
    try:
        arch_lower = arch.lower()
        if "resnet" in arch_lower:
            model = _build_resnet50_xray(input_shape, num_classes)
        elif "vgg" in arch_lower:
            model = _build_vgg16_xray(input_shape, num_classes)
        elif "efficientnet" in arch_lower:
            model = _build_efficientnet_xray(input_shape, num_classes)
        elif "detector" in arch_lower:
            model = _build_detector_158(input_shape, num_classes)
        else:
            # Default fallback architecture
            logger.warning(f"Unknown X-ray architecture {arch}, using VGG16 fallback")
            model = _build_vgg16_xray(input_shape, num_classes)
        
        # Load weights with skip_mismatch for robustness
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        logger.info(f"Successfully loaded weights-only X-ray model: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Weights-only fallback failed for X-ray model {model_path}: {str(e)}")
        return None

class XrayManager:
    """
    Manages ensemble inference of X-ray analysis models
    
    Loads YOLO detection model and multiple X-ray classification models,
    performs ensemble inference with graceful degradation when models fail.
    """
    
    def __init__(self):
        """Initialize X-ray manager and load models"""
        self.yolo_model = None
        self.pcos_models = {}  # Dict[str, Dict[str, Any]]
        self.can_detect_objects = False
        self.models_unavailable = []  # Track failed models
        self.ensemble_weights = {}
        
        # Model status tracking
        self.model_status = {
            "yolo": {"loaded": False, "available": False, "error": None},
            "xray": {"loaded": False, "available": False, "error": None}
        }
        
        self.loading_warnings = []
        self._load_models()
    
    def can_lazy_load_yolo(self) -> bool:
        """Check if YOLO model can be lazy loaded"""
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
        yolo_path = YOLO_MODELS_DIR / settings.YOLO_MODEL
        
        self.model_status["yolo"]["available"] = yolo_path.exists()
        
        try:
            if yolo_path.exists():
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(yolo_path))
                self.can_detect_objects = True
                self.model_status["yolo"]["loaded"] = True
                logger.info(f"Loaded YOLO model: {yolo_path}")
            else:
                logger.warning(f"YOLO model not found: {yolo_path}")
                self.model_status["yolo"]["error"] = "File not found"
                self.loading_warnings.append(f"YOLO model not found: {yolo_path}")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model_status["yolo"]["error"] = str(e)
            self.loading_warnings.append(f"YOLO model failed to load: {str(e)}")
    
    def _load_pcos_models(self) -> None:
        """Load PCOS classification models with ensemble support"""
        loaded_count = 0
        
        # Get available models using auto-discovery
        available_models = get_available_xray_models()
        logger.info(f"Available X-ray PCOS models: {list(available_models.keys())}")
        
        if not available_models:
            logger.warning("No X-ray PCOS models found")
            self.model_status["xray"]["available"] = False
            self.loading_warnings.append("No X-ray PCOS models found")
            return
        
        # Get ensemble weights
        self.ensemble_weights = get_ensemble_weights('xray')
        
        # Load each available model with robust fallback
        for model_name, model_path in available_models.items():
            try:
                logger.info(f"Attempting to load X-ray model: {model_name} from {model_path}")
                
                # Use robust loader
                model = _safe_load_h5(str(model_path), compile_model=XRAY_COMPILE)
                
                if model is not None:
                    # Validate model by running a test prediction
                    if not self._validate_model(model, model_path):
                        logger.error(f"Model validation failed for {model_name}")
                        self.models_unavailable.append({
                            "name": model_name, 
                            "path": str(model_path), 
                            "reason": "validation_failed"
                        })
                        self.loading_warnings.append(f"X-ray model {model_name} validation failed")
                        continue
                    
                    weight = self.ensemble_weights.get(model_name, 1.0)
                    labels = load_model_labels(model_path)
                    
                    self.pcos_models[model_name] = {
                        "model": model,
                        "path": str(model_path),
                        "weight": weight,
                        "labels": labels,
                        "input_shape": settings.XRAY_IMAGE_SIZE
                    }
                    loaded_count += 1
                    logger.info(f"✅ Successfully loaded X-ray PCOS model {model_name}: {model_path} (weight: {weight})")
                else:
                    logger.error(f"❌ Failed to load X-ray model {model_name} - all methods exhausted")
                    self.models_unavailable.append({
                        "name": model_name, 
                        "path": str(model_path), 
                        "reason": "load_failed"
                    })
                    self.loading_warnings.append(f"X-ray model {model_name} failed to load")
                    
            except Exception as e:
                logger.error(f"❌ Exception loading X-ray PCOS model {model_name}: {str(e)}")
                self.models_unavailable.append({
                    "name": model_name, 
                    "path": str(model_path), 
                    "reason": f"exception: {str(e)}"
                })
                continue
        
        # Normalize weights for loaded models
        if self.pcos_models:
            model_names = list(self.pcos_models.keys())
            current_weights = {name: self.pcos_models[name]["weight"] for name in model_names}
            normalized_weights = normalize_weights(current_weights, model_names)
            
            for model_name, weight in normalized_weights.items():
                if model_name in self.pcos_models:
                    self.pcos_models[model_name]["weight"] = weight
            
            logger.info(f"Normalized X-ray model weights: {normalized_weights}")
        
        self.model_status["xray"]["loaded"] = loaded_count > 0
        self.model_status["xray"]["available"] = len(available_models) > 0
        
        if loaded_count > 0:
            logger.info(f"✅ Successfully loaded {loaded_count}/{len(available_models)} X-ray PCOS models")
        else:
            logger.warning("⚠️  No X-ray PCOS models could be loaded - X-ray analysis will be unavailable")
            logger.info("YOLO detection will still work for object detection in X-rays")
            self.model_status["xray"]["error"] = "No models loaded successfully"
            self.loading_warnings.append("No X-ray PCOS models could be loaded - X-ray analysis will be unavailable")
    
    def _reconstruct_xray_model(self, model_path: Path, model_name: str):
        """
        Reconstruct X-ray model architecture and load weights to bypass batch_shape issues
        
        Args:
            model_path: Path to model file
            model_name: Model name for architecture detection
            
        Returns:
            Reconstructed model or None if failed
        """
        try:
            logger.info(f"Reconstructing X-ray model architecture for {model_name}")
            
            # Detect architecture from model name
            model_name_lower = model_name.lower()
            input_shape = settings.XRAY_IMAGE_SIZE + (3,)  # (100, 100, 3)
            
            if "resnet50" in model_name_lower:
                logger.debug(f"Building ResNet50 architecture for {model_name}")
                base_model = tf.keras.applications.ResNet50(
                    weights=None,
                    include_top=False,
                    input_shape=input_shape
                )
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(2, activation='softmax')  # Binary classification
                ])
            elif "vgg16" in model_name_lower:
                logger.debug(f"Building VGG16 architecture for {model_name}")
                base_model = tf.keras.applications.VGG16(
                    weights=None,
                    include_top=False,
                    input_shape=input_shape
                )
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(2, activation='softmax')
                ])
            elif "detector" in model_name_lower:
                logger.debug(f"Building custom detector architecture for {model_name}")
                # Custom architecture for detector_158
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(2, activation='softmax')
                ])
            else:
                logger.warning(f"Unknown architecture for {model_name}, using generic CNN")
                # Generic CNN fallback
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(2, activation='softmax')
                ])
            
            # Try to load weights
            try:
                model.load_weights(str(model_path))
                logger.info(f"✅ Successfully reconstructed and loaded weights for {model_name}")
                return model
            except Exception as weights_error:
                logger.error(f"Failed to load weights for reconstructed {model_name}: {str(weights_error)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to reconstruct X-ray model {model_name}: {str(e)}")
            return None
    
    def _detect_architecture(self, model_name: str) -> str:
        """Detect architecture from model name"""
        model_name_lower = model_name.lower()
        if "resnet" in model_name_lower:
            return "resnet50"
        elif "vgg" in model_name_lower:
            return "vgg16"
        elif "efficientnet" in model_name_lower:
            return "efficientnet"
        elif "detector" in model_name_lower:
            return "detector_158"
        else:
            return "vgg16"  # Default fallback
    
    def _validate_model(self, model, model_path: Path) -> bool:
        """
        Validate model by running a test prediction
        
        Args:
            model: Loaded TensorFlow model
            model_path: Path to model file
            
        Returns:
            True if model is valid, False if corrupted
        """
        try:
            # Create dummy input data with consistent shape
            dummy_input = np.random.random((1, *settings.XRAY_IMAGE_SIZE, 3)).astype(np.float32)
            
            # Try to run prediction
            prediction = model.predict(dummy_input, verbose=0)
            
            # Check if prediction has expected shape
            if prediction is None or len(prediction.shape) != 2:
                logger.error(f"X-ray model {model_path} returned invalid prediction shape")
                return False
            
            # Check if prediction contains valid probabilities
            if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
                logger.error(f"X-ray model {model_path} returned NaN or Inf values")
                return False
            
            logger.debug(f"X-ray model validation successful for {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"X-ray model validation failed for {model_path}: {str(e)}")
            return False
    
    def _preprocess_image(self, image_bytes: bytes, target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Preprocess image for model inference with consistent shape
        
        Args:
            image_bytes: Raw image bytes
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize with high-quality resampling (PIL expects width, height)
            pil_size = (target_size[1], target_size[0])  # Convert to (width, height)
            image = image.resize(pil_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0,1]
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension - always (1, height, width, 3) for consistency
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"X-ray image preprocessing failed: {str(e)}")
            raise
    
    async def detect_objects(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect objects in X-ray image using YOLO
        
        Args:
            image_bytes: Preprocessed image bytes
            
        Returns:
            Dictionary with detection results
        """
        if not self.can_detect_objects or self.yolo_model is None:
            return {
                "detections": [],
                "found_labels": [],
                "yolo_vis": None
            }
        
        try:
            # Save temporary file for YOLO processing
            temp_id = str(uuid.uuid4())[:8]
            temp_path = UPLOADS_DIR / f"temp_yolo_{temp_id}.jpg"
            
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            # Run YOLO detection
            results = self.yolo_model(str(temp_path))
            
            detections = []
            found_labels = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        if hasattr(result, 'names') and cls in result.names:
                            label = result.names[cls]
                        else:
                            label = f"class_{cls}"
                        
                        detections.append({
                            "box": box.tolist(),
                            "conf": conf,
                            "label": label
                        })
                        
                        if label not in found_labels:
                            found_labels.append(label)
                
                # Save visualization
                vis_path = UPLOADS_DIR / f"yolo_vis_{temp_id}.jpg"
                if hasattr(result, 'save'):
                    result.save(str(vis_path))
                    yolo_vis = f"/static/uploads/yolo_vis_{temp_id}.jpg"
                else:
                    yolo_vis = None
            else:
                yolo_vis = None
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
            return {
                "detections": detections,
                "found_labels": found_labels,
                "yolo_vis": yolo_vis
            }
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}")
            return {
                "detections": [],
                "found_labels": [],
                "yolo_vis": None
            }
    
    async def predict_pcos_ensemble(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict PCOS from X-ray using ensemble of all loaded models
        
        Args:
            image_bytes: Preprocessed image bytes
            
        Returns:
            Dictionary with per-model predictions and ensemble result
        """
        if not self.pcos_models:
            return {
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "labels": ["non_pcos", "pcos"]
            }
        
        per_model_scores = {}
        successful_predictions = 0
        
        for model_name, model_data in self.pcos_models.items():
            try:
                model = model_data["model"]
                input_shape = model_data.get("input_shape", settings.XRAY_IMAGE_SIZE)
                
                # Preprocess image with correct size for this model
                image_array = self._preprocess_image(image_bytes, input_shape)
                
                # Run prediction with compiled function to avoid retracing
                try:
                    prediction = compiled_predict_xray(model, image_array)
                    prediction = prediction.numpy()  # Convert to numpy if needed
                except Exception:
                    # Fallback to regular predict if compiled version fails
                    prediction = model.predict(image_array, verbose=0)
                
                # Extract PCOS probability (assuming binary classification)
                if prediction.shape[1] == 1:
                    # Single output (sigmoid)
                    pcos_prob = float(prediction[0][0])
                else:
                    # Two outputs (softmax) - take second class (PCOS)
                    pcos_prob = float(prediction[0][1])
                
                per_model_scores[model_name] = pcos_prob
                successful_predictions += 1
                logger.debug(f"X-ray {model_name} prediction: {pcos_prob:.3f}")
                
            except Exception as e:
                logger.error(f"X-ray PCOS prediction failed for {model_name}: {str(e)}")
                self.loading_warnings.append(f"X-ray model {model_name} prediction failed: {str(e)}")
                # Continue with other models
                continue
        
        if not per_model_scores:
            self.loading_warnings.append("No X-ray PCOS models available for prediction")
            return {
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "labels": ["non_pcos", "pcos"]
            }
        
        # Calculate weighted ensemble score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_name, score in per_model_scores.items():
            weight = self.pcos_models[model_name]["weight"]
            weighted_sum += score * weight
            total_weight += weight
        
        ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            "per_model": per_model_scores,
            "ensemble_score": float(ensemble_score),
            "ensemble": {
                "method": "weighted_average",
                "score": float(ensemble_score),
                "models_used": successful_predictions,
                "weights_used": {name: self.pcos_models[name]["weight"] for name in per_model_scores.keys()}
            },
            "labels": ["non_pcos", "pcos"]
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
            # Initialize result structure
            result = {
                "xray_img": f"/static/uploads/{filename}",
                "detections": [],
                "found_labels": [],
                "yolo_vis": None,
                "xray_pred": None,
                "xray_risk": "unknown",
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "models_unavailable": self.models_unavailable,
                "models_used": []
            }
            
            # Run YOLO detection
            detection_results = await self.detect_objects(image_bytes)
            result.update(detection_results)
            
            # Run PCOS classification if models available
            if self.pcos_models:
                pcos_results = await self.predict_pcos_ensemble(image_bytes)
                
                if pcos_results["per_model"]:
                    # Store detailed results
                    result["per_model"] = pcos_results["per_model"]
                    result["models_used"] = list(pcos_results["per_model"].keys())
                    result["ensemble"] = pcos_results["ensemble"]
                    
                    # Get ensemble results
                    ensemble_score = pcos_results["ensemble_score"]
                    result["ensemble_score"] = ensemble_score
                    
                    # Determine prediction label and risk
                    risk_level = get_risk_level(ensemble_score)
                    result["xray_risk"] = risk_level
                    
                    if risk_level == "high":
                        result["xray_pred"] = "PCOS symptoms detected in X-ray"
                    elif risk_level == "moderate":
                        result["xray_pred"] = "Moderate PCOS indicators in X-ray"
                    else:
                        result["xray_pred"] = "No significant PCOS symptoms detected in X-ray"
                else:
                    # Fallback to YOLO-based assessment
                    result["xray_pred"] = self._assess_from_yolo(result["found_labels"])
                    result["xray_risk"] = self._get_yolo_risk(result["found_labels"])
            else:
                # No PCOS models available - use YOLO only
                result["xray_pred"] = self._assess_from_yolo(result["found_labels"])
                result["xray_risk"] = self._get_yolo_risk(result["found_labels"])
                result["models_unavailable"] = self.models_unavailable
                self.loading_warnings.append("No X-ray PCOS models available - using YOLO detection only")
            
            return result
            
        except Exception as e:
            logger.error(f"X-ray processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def _assess_from_yolo(self, found_labels: List[str]) -> str:
        """Generate assessment based on YOLO detections only"""
        if not found_labels:
            return "No significant structures detected in X-ray"
        
        # Simple heuristic based on detected objects
        pcos_indicators = ['cyst', 'enlarged_ovary', 'multiple_follicles', 'polycystic']
        detected_indicators = [label for label in found_labels if any(indicator in label.lower() for indicator in pcos_indicators)]
        
        if detected_indicators:
            return f"PCOS-related structures detected: {', '.join(detected_indicators)}"
        else:
            return f"Anatomical structures detected: {', '.join(found_labels)}"
    
    def _get_yolo_risk(self, found_labels: List[str]) -> str:
        """Determine risk level based on YOLO detections"""
        if not found_labels:
            return "unknown"
        
        # Simple heuristic based on detected objects
        pcos_indicators = ['cyst', 'enlarged_ovary', 'multiple_follicles', 'polycystic']
        detected_indicators = [label for label in found_labels if any(indicator in label.lower() for indicator in pcos_indicators)]
        
        if len(detected_indicators) >= 2:
            return "high"
        elif len(detected_indicators) == 1:
            return "moderate"
        else:
            return "low"
    
    def get_loading_warnings(self) -> List[str]:
        """Get any warnings from model loading"""
        return self.loading_warnings
    
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
                "input_shape": model_data.get("input_shape")
            }
        
        return status
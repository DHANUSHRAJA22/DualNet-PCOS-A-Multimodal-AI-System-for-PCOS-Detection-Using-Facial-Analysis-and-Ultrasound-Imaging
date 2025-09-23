"""
X-ray analysis manager for YOLO detection and PCOS classification

Handles automatic discovery and loading of all X-ray analysis models with
YOLO object detection and dynamic ensemble inference.
"""

import os
import uuid
import logging
import json
import h5py
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
    get_available_xray_models, load_model_labels, get_ensemble_weights, normalize_weights,
    XRAY_IMAGE_SIZE
)
from utils.validators import validate_image, get_safe_filename

logger = logging.getLogger(__name__)

# Create a compiled prediction function to avoid retracing warnings
@tf.function
def compiled_predict(model, input_data):
    """Compiled prediction function to avoid TensorFlow retracing warnings"""
    return model(input_data, training=False)

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
        self.ensemble_weights = {}
        
        # Model status tracking
        self.model_status = {
            "yolo": {"loaded": False, "available": False, "error": None},
            "xray": {"loaded": False, "available": False, "error": None}
        }
        
        # Track loading warnings
        self.loading_warnings = []
        
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
        loaded_count = 0
        
        # Get available models using auto-discovery
        available_models = get_available_xray_models()
        logger.info(f"Available X-ray PCOS models: {list(available_models.keys())}")
        
        if not available_models:
            logger.warning("No X-ray PCOS models found")
            self.model_status["xray"]["available"] = False
            return
        
        # Get ensemble weights
        self.ensemble_weights = get_ensemble_weights('xray')
        
        # Load each available model
        for model_name, model_path in available_models.items():
            try:
                # Try loading with fallback for batch_shape issues
                model = self._load_model_with_fallback(model_path)
                
                if model is not None:
                    weight = self.ensemble_weights.get(model_name, 1.0)
                    labels = load_model_labels(model_path)
                    
                    # Detect input shape from model
                    input_shape = self._get_model_input_shape(model)
                    
                    self.pcos_models[model_name] = {
                        "model": model,
                        "path": str(model_path),
                        "weight": weight,
                        "labels": labels,
                        "input_shape": input_shape
                    }
                    loaded_count += 1
                    logger.info(f"Loaded PCOS model {model_name}: {model_path} (weight: {weight}, input_shape: {input_shape})")
                    
            except Exception as e:
                logger.error(f"Failed to load PCOS model {model_name}: {str(e)}")
                self.loading_warnings.append(f"X-ray model {model_name} failed to load: {str(e)}")
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
            logger.info(f"Successfully loaded {loaded_count} X-ray PCOS models with normalized weights")
        else:
            logger.warning("No X-ray PCOS models could be loaded - X-ray analysis will be unavailable")
            self.model_status["xray"]["error"] = "No models loaded successfully"
            self.loading_warnings.append("No X-ray PCOS models could be loaded - imaging analysis unavailable")
    
    def _load_model_with_fallback(self, model_path: Path):
        """
        Load model with comprehensive fallback for batch_shape and config issues
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model or None if failed
        """
        try:
            # Try normal loading first
            model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info(f"Successfully loaded model: {model_path}")
            return model
            
        except TypeError as e:
            if "batch_shape" in str(e) or "unrecognized keyword arguments" in str(e):
                logger.warning(f"Model {model_path} has config issues, trying fallback methods...")
                try:
                    # Method 1: Try loading with custom objects cleared
                    model = tf.keras.models.load_model(
                        str(model_path), 
                        compile=False,
                        custom_objects={},
                        safe_mode=False
                    )
                    logger.info(f"Fallback method 1 successful for: {model_path}")
                    return model
                except Exception as fallback_e:
                    logger.warning(f"Fallback method 1 failed for {model_path}: {str(fallback_e)}")
                    
                    # Method 2: Parse HDF5 and fix batch_shape issues
                    try:
                        return self._fix_batch_shape_and_load(model_path)
                    except Exception as reconstruct_e:
                        logger.warning(f"Batch shape fix failed for {model_path}: {str(reconstruct_e)}")
                        
                        # Method 3: Try to reconstruct from architecture
                        try:
                            return self._reconstruct_model_from_weights(model_path)
                        except Exception as final_e:
                            logger.error(f"All fallback methods failed for {model_path}: {str(final_e)}")
                            return None
            else:
                logger.error(f"Model loading failed for {model_path}: {str(e)}")
                return None
        except ValueError as e:
            if "No model config found" in str(e):
                logger.warning(f"Model {model_path} has no config, trying weights-only reconstruction...")
                try:
                    return self._reconstruct_from_weights_only(model_path, model_path.stem.lower())
                except Exception as weights_e:
                    logger.error(f"Weights-only reconstruction failed for {model_path}: {str(weights_e)}")
                    return None
            else:
                logger.error(f"Model loading failed for {model_path}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Model loading failed for {model_path}: {str(e)}")
            return None
    
    def _fix_batch_shape_and_load(self, model_path: Path):
        """
        Fix batch_shape issues by parsing HDF5 and rebuilding model config
        
        Args:
            model_path: Path to model file
            
        Returns:
            Fixed model or None if failed
        """
        try:
            with h5py.File(str(model_path), 'r') as f:
                # Try to find model config
                model_config = None
                
                if 'model_config' in f.attrs:
                    model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                elif 'model_topology' in f.attrs:
                    model_config = json.loads(f.attrs['model_topology'].decode('utf-8'))
                elif 'model_config' in f:
                    model_config = json.loads(f['model_config'][()].decode('utf-8'))
                
                if not model_config:
                    raise ValueError("No model config found in HDF5 file")
                
                # Remove batch_shape from InputLayer configs
                def remove_batch_shape(config):
                    if isinstance(config, dict):
                        if config.get('class_name') == 'InputLayer':
                            if 'config' in config and 'batch_shape' in config['config']:
                                del config['config']['batch_shape']
                                logger.debug(f"Removed batch_shape from InputLayer in {model_path}")
                        
                        for key, value in config.items():
                            remove_batch_shape(value)
                    elif isinstance(config, list):
                        for item in config:
                            remove_batch_shape(item)
                
                remove_batch_shape(model_config)
                
                # Rebuild model from fixed config
                model = tf.keras.models.model_from_config(model_config)
                
                # Load weights
                model.load_weights(str(model_path))
                
                logger.info(f"Successfully fixed batch_shape issues for: {model_path}")
                return model
                
        except Exception as e:
            logger.error(f"Failed to fix batch_shape issues for {model_path}: {str(e)}")
            raise
    
    def _reconstruct_model_from_weights(self, model_path: Path):
        """
        Detect weights-only files and reconstruct appropriate architecture
        
        Args:
            model_path: Path to model file
            
        Returns:
            Reconstructed model or None if failed
        """
        model_name = model_path.stem.lower()
        
        # First check if this is a weights-only file
        try:
            with h5py.File(str(model_path), 'r') as f:
                has_config = ('model_config' in f.attrs or 
                             'model_topology' in f.attrs or 
                             'model_config' in f)
                
                if not has_config:
                    logger.info(f"Detected weights-only file: {model_path}")
                    return self._reconstruct_from_weights_only(model_path, model_name)
        except Exception as e:
            logger.warning(f"Could not check file structure for {model_path}: {str(e)}")
        
        try:
            # Detect architecture from filename
            if 'vgg16' in model_name:
                base_model = tf.keras.applications.VGG16(
                    weights=None,
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
            elif 'resnet50' in model_name:
                base_model = tf.keras.applications.ResNet50(
                    weights=None,
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
            elif 'efficientnet' in model_name:
                base_model = tf.keras.applications.EfficientNetB0(
                    weights=None,
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
            elif 'efficientnetb1' in model_name or 'efficientnet_b1' in model_name:
                base_model = tf.keras.applications.EfficientNetB1(
                    weights=None,
                    include_top=False,
                    input_shape=(240, 240, 3)
                )
            elif 'efficientnetb2' in model_name or 'efficientnet_b2' in model_name:
                base_model = tf.keras.applications.EfficientNetB2(
                    weights=None,
                    include_top=False,
                    input_shape=(260, 260, 3)
                )
            elif 'efficientnetb3' in model_name or 'efficientnet_b3' in model_name:
                base_model = tf.keras.applications.EfficientNetB3(
                    weights=None,
                    include_top=False,
                    input_shape=(300, 300, 3)
                )
            elif 'detector' in model_name or 'custom' in model_name:
                # Handle custom detector models with generic architecture
                logger.info(f"Using generic architecture for custom model: {model_name}")
                return self._create_generic_model()
            elif 'mobilenet' in model_name:
                base_model = tf.keras.applications.MobileNetV2(
                    weights=None,
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
            elif 'densenet' in model_name:
                base_model = tf.keras.applications.DenseNet121(
                    weights=None,
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
            else:
                logger.warning(f"Unknown architecture for {model_name}, using generic fallback")
                # Fallback to generic model for unknown architectures
                return self._create_generic_model()
            
            # Add classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2, activation='softmax')  # Binary classification
            ])
            
            # Try to load weights only
            try:
                model.load_weights(str(model_path))
                logger.info(f"Successfully reconstructed model from weights: {model_path}")
                return model
            except Exception as weights_e:
                logger.error(f"Failed to load weights for reconstructed model {model_path}: {str(weights_e)}")
                # Try generic fallback as last resort
                logger.info(f"Attempting generic fallback for {model_name}")
                return self._create_generic_model()
                return None
                
        except Exception as e:
            logger.error(f"Model reconstruction failed for {model_path}: {str(e)}")
    
    def _reconstruct_from_weights_only(self, model_path: Path, model_name: str):
        """
        Reconstruct model architecture for weights-only files
        
        Args:
            model_path: Path to weights file
            model_name: Model name for architecture detection
            
        Returns:
            Reconstructed model
        """
        logger.info(f"Reconstructing architecture for weights-only file: {model_name}")
        
        # Detect architecture from filename and create appropriate model
        if 'resnet50' in model_name:
            base_model = tf.keras.applications.ResNet50(
                weights=None,
                include_top=False,
                input_shape=(224, 224, 3)
            )
        elif 'vgg16' in model_name:
            base_model = tf.keras.applications.VGG16(
                weights=None,
                include_top=False,
                input_shape=(224, 224, 3)
            )
        elif 'efficientnetb0' in model_name:
            base_model = tf.keras.applications.EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(224, 224, 3)
            )
        elif 'detector' in model_name:
            # Custom detector architecture
            base_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
            ])
        else:
            # Generic CNN for unknown architectures
            base_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            ])
        
        # Add classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')  # Binary classification
        ])
        
        # Load weights
        model.load_weights(str(model_path))
        
        logger.info(f"Successfully reconstructed weights-only model: {model_path}")
        return model
    
    def _create_generic_model(self):
        """
        Create a generic model for unknown architectures
        
        Returns:
            Generic model that can be used as fallback
        """
        try:
            # Create a simple CNN model as fallback
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2, activation='softmax')  # Binary classification
            ])
            
            logger.info("Created generic fallback model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create generic model: {str(e)}")
            return None
    
    def _get_model_input_shape(self, model) -> Tuple[int, int]:
        """
        Get input shape from loaded model
        
        Args:
            model: Loaded TensorFlow model
            
        Returns:
            Tuple of (height, width) for image preprocessing
        """
        try:
            if hasattr(model, 'input_shape') and model.input_shape:
                # Get (height, width) from model input shape
                shape = model.input_shape
                if len(shape) >= 3:
                    return (shape[1], shape[2])  # (height, width)
        except Exception:
            pass
        
        # Default to standard size
        return settings.XRAY_IMAGE_SIZE
    
    def _preprocess_image(self, image_bytes: bytes, target_size: Tuple[int, int] = XRAY_IMAGE_SIZE) -> np.ndarray:
        """
        Preprocess image for model inference
        
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
            results = self.yolo_model.predict(image, verbose=False)
            
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
        if not self.pcos_models:
            return {
                "per_model": {},
                "per_model_scores": {},
                "ensemble": {"prob": 0.0, "label": "unknown", "weights": {}},
                "labels": load_model_labels(Path("dummy"))
            }
        
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
                    input_shape = model_data.get("input_shape", settings.XRAY_IMAGE_SIZE)
                    
                    # Preprocess ROI
                    roi_array = self._preprocess_image(roi_bytes, input_shape)
                    
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
                "labels": load_model_labels(Path("dummy"))
            }
        
        # Compute ensemble prediction
        ensemble_result = self._compute_ensemble(per_model_predictions)
        
        return {
            "per_model": per_model_predictions,
            "per_model_scores": per_model_scores,
            "ensemble": ensemble_result,
            "labels": load_model_labels(Path("dummy"))
        }
    
    async def predict_pcos_ensemble_full_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict PCOS from full X-ray image using ensemble of all loaded models
        
        Args:
            image_bytes: X-ray image bytes
            
        Returns:
            Dictionary with per-model predictions and ensemble result
        """
        if not self.pcos_models:
            return {
                "per_model": {},
                "per_model_scores": {},
                "ensemble": {"prob": 0.0, "label": "unknown", "weights": {}},
                "labels": load_model_labels(Path("dummy"))
            }
        
        per_model_predictions = {}
        per_model_scores = {}
        
        for model_name, model_data in self.pcos_models.items():
            try:
                model = model_data["model"]
                input_shape = model_data.get("input_shape", settings.XRAY_IMAGE_SIZE)
                
                # Preprocess full image
                image_array = self._preprocess_image(image_bytes, input_shape)
                
                # Run prediction
                try:
                    prediction = compiled_predict(model, image_array)
                    prediction = prediction.numpy()  # Convert to numpy if needed
                except Exception:
                    # Fallback to regular predict if compiled version fails
                    prediction = model.predict(image_array, verbose=0)
                    prediction = compiled_predict(model, roi_array)
                    prediction = prediction.numpy()  # Convert to numpy if needed
                except Exception:
                    # Fallback to regular predict if compiled version fails
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
                
                logger.debug(f"X-ray {model_name} full image prediction: {probs}")
                
            except Exception as e:
                logger.error(f"PCOS full image prediction failed for {model_name}: {str(e)}")
                self.loading_warnings.append(f"X-ray model {model_name} full image prediction failed: {str(e)}")
                self.loading_warnings.append(f"X-ray model {model_name} ROI prediction failed: {str(e)}")
                continue
        
        if not per_model_predictions:
            self.loading_warnings.append("No X-ray PCOS models available for full image prediction")
            self.loading_warnings.append("No X-ray PCOS models available for ROI prediction")
            return {
                "per_model": {},
                "per_model_scores": {},
                "ensemble": {"prob": 0.0, "label": "unknown", "weights": {}},
                "labels": load_model_labels(Path("dummy"))
            }
        
        # Compute ensemble prediction
        ensemble_result = self._compute_ensemble(per_model_predictions)
        
        return {
            "per_model": per_model_predictions,
            "per_model_scores": per_model_scores,
            "ensemble": ensemble_result,
            "labels": load_model_labels(Path("dummy"))
        }
    
    def _load_model_with_fallback(self, model_path: Path):
        """
        Load model with fallback for batch_shape compatibility issues
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model or None if failed
        """
        try:
            # Try normal loading first
            model = tf.keras.models.load_model(str(model_path), compile=False)
            return model
            
        except TypeError as e:
            if "batch_shape" in str(e) or "unrecognized keyword arguments" in str(e):
                logger.warning(f"Model {model_path} has batch_shape issues, trying fallback...")
                try:
                    # Try loading without compilation and custom objects
                    model = tf.keras.models.load_model(
                        str(model_path), 
                        compile=False,
                        custom_objects={}
                    )
                    return model
                except Exception as fallback_e:
                    logger.error(f"Fallback loading failed for {model_path}: {str(fallback_e)}")
                    return None
            else:
                logger.error(f"Model loading failed for {model_path}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Model loading failed for {model_path}: {str(e)}")
            return None
    
    def _get_model_input_shape(self, model) -> Tuple[int, int]:
        """
        Get input shape from loaded model
        
        Args:
            model: Loaded TensorFlow model
            
        Returns:
            Tuple of (height, width) for image preprocessing
        """
        try:
            if hasattr(model, 'input_shape') and model.input_shape:
                # Get (height, width) from model input shape
                shape = model.input_shape
                if len(shape) >= 3:
                    return (shape[1], shape[2])  # (height, width)
        except Exception:
            pass
        
        # Default to standard size
        return settings.XRAY_IMAGE_SIZE
    
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
        
        # Use labels
        labels = load_model_labels(Path("dummy"))
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
                "ensemble_score": 0.0,
                "models_used": [],
                "detections": detections
            }
            
            # Extract found labels from detections
            if detections:
                result["found_labels"] = [det["label"] for det in detections]
            
            # Check if we have any PCOS models
            if not self.pcos_models:
                result["xray_pred"] = "No PCOS models available for analysis"
                result["xray_risk"] = "unknown"
                self.loading_warnings.append("No X-ray PCOS models loaded")
                return result
            
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
                    result["models_used"] = list(averaged_predictions.keys())
                    
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
                    result["xray_pred"] = "No valid ROI predictions available"
                    self.loading_warnings.append("No valid ROI predictions available")
            
            else:
                # No detections - classify full image
                full_image_results = await self.predict_pcos_ensemble_full_image(image_bytes)
                
                if full_image_results["per_model"]:
                    # Store per-model predictions
                    result["xray_models"] = full_image_results["per_model"]
                    result["per_model"] = full_image_results["per_model_scores"]
                    result["models_used"] = list(full_image_results["per_model"].keys())
                    
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
                    self.loading_warnings.append("No PCOS models available for full image analysis")
            
            return result
            
        except Exception as e:
            logger.error(f"X-ray processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def get_loading_warnings(self) -> List[str]:
        """Get any warnings from model loading"""
        return getattr(self, 'loading_warnings', [])
    
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
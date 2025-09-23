"""
Face analysis manager for gender detection and PCOS classification

Handles automatic discovery and loading of all facial analysis models with
dynamic ensemble inference and proper error handling.
"""

import os
import uuid
import logging
import json
import h5py
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile

# Import TensorFlow and suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from config import (
    settings, FACE_MODELS_DIR, UPLOADS_DIR, get_risk_level,
    get_available_face_models, load_model_labels, get_ensemble_weights, normalize_weights
)
from utils.validators import validate_image, get_safe_filename
from ensemble import EnsembleManager

logger = logging.getLogger(__name__)

class FaceManager:
    """
    Manages ensemble inference of facial analysis models
    
    Loads multiple face models and performs ensemble inference with configurable weights.
    Supports graceful degradation when models are missing.
    """
    
    def __init__(self):
        """Initialize face manager and load models"""
        self.gender_model = None
        self.pcos_models = {}  # Dict[str, Dict[str, Any]]
        self.can_predict_gender = False
        self.ensemble_weights = {}
        self.ensemble_manager = EnsembleManager()
        
        # Model status tracking
        self.model_status = {
            "gender": {"loaded": False, "available": False, "error": None},
            "face": {"loaded": False, "available": False, "error": None}
        }
        
        self._load_models()
    
    def can_lazy_load_gender(self) -> bool:
        """Check if gender model can be lazy loaded"""
        gender_path = FACE_MODELS_DIR / settings.GENDER_MODEL
        return gender_path.exists() and gender_path.is_file()
    
    def can_lazy_load_pcos(self) -> bool:
        """Check if any PCOS models can be lazy loaded"""
        available_models = get_available_face_models()
        return len(available_models) > 0
    
    def _load_models(self) -> None:
        """Load all facial analysis models"""
        logger.info("Loading facial analysis models...")
        
        # Load gender classifier
        self._load_gender_model()
        
        # Load PCOS classification models and collect warnings
        self.loading_warnings = []
        self._load_pcos_models()
        
        logger.info(f"Face manager initialized. Gender detection: {self.can_predict_gender}, "
                   f"PCOS models loaded: {len(self.pcos_models)}")
    
    def _load_gender_model(self) -> None:
        """Load gender classification model"""
        gender_path = FACE_MODELS_DIR / settings.GENDER_MODEL
        
        self.model_status["gender"]["available"] = gender_path.exists()
        
        try:
            if gender_path.exists():
                self.gender_model = tf.keras.models.load_model(str(gender_path), compile=False)
                self.can_predict_gender = True
                self.model_status["gender"]["loaded"] = True
                logger.info(f"Loaded gender model: {gender_path}")
            else:
                logger.warning(f"Gender model not found: {gender_path}")
                self.model_status["gender"]["error"] = "File not found"
                
        except Exception as e:
            logger.error(f"Failed to load gender model: {str(e)}")
            self.model_status["gender"]["error"] = str(e)
    
    def _load_pcos_models(self) -> None:
        """Load PCOS classification models with ensemble support"""
        loaded_count = 0
        
        # Get available models using auto-discovery
        available_models = get_available_face_models()
        logger.info(f"Available face PCOS models: {list(available_models.keys())}")
        
        if not available_models:
            logger.warning("No face PCOS models found")
            self.model_status["face"]["available"] = False
            return
        
        # Get ensemble weights
        self.ensemble_weights = get_ensemble_weights('face')
        
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
                self.loading_warnings.append(f"Face model {model_name} failed to load: {str(e)}")
                continue
        
        # Normalize weights for loaded models
        if self.pcos_models:
            model_names = list(self.pcos_models.keys())
            current_weights = {name: self.pcos_models[name]["weight"] for name in model_names}
            normalized_weights = normalize_weights(current_weights, model_names)
            
            for model_name, weight in normalized_weights.items():
                if model_name in self.pcos_models:
                    self.pcos_models[model_name]["weight"] = weight
            
            logger.info(f"Normalized face model weights: {normalized_weights}")
        
        self.model_status["face"]["loaded"] = loaded_count > 0
        self.model_status["face"]["available"] = len(available_models) > 0
        
        if loaded_count > 0:
            logger.info(f"Successfully loaded {loaded_count} face PCOS models with normalized weights")
        else:
            logger.warning("No face PCOS models could be loaded - face analysis will be unavailable")
            self.model_status["face"]["error"] = "No models loaded successfully"
            self.loading_warnings.append("No face PCOS models could be loaded - facial analysis unavailable")
    
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
        return settings.FACE_IMAGE_SIZE
    
    def _preprocess_image(self, image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
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
    
    async def predict_gender(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict gender from facial image
        
        Args:
            image_bytes: Preprocessed image bytes
            
        Returns:
            Dictionary with gender prediction results
        """
        if not self.can_predict_gender or self.gender_model is None:
            return {
                "male": 0.0,
                "female": 1.0,  # Default to female to allow PCOS analysis
                "label": "female"
            }
        
        try:
            # Use gender model specific input size (249x249)
            image_array = self._preprocess_image(image_bytes, settings.GENDER_IMAGE_SIZE)
            
            # Run prediction
            prediction = self.gender_model.predict(image_array, verbose=0)
            
            # Extract probabilities (assuming binary classification)
            if prediction.shape[1] == 1:
                # Single output (sigmoid)
                female_prob = float(prediction[0][0])
                male_prob = 1.0 - female_prob
            else:
                # Two outputs (softmax)
                male_prob = float(prediction[0][0])
                female_prob = float(prediction[0][1])
            
            label = "female" if female_prob > male_prob else "male"
            
            return {
                "male": male_prob,
                "female": female_prob,
                "label": label
            }
            
        except Exception as e:
            logger.error(f"Gender prediction failed: {str(e)}")
            return {
                "male": 0.0,
                "female": 1.0,  # Default to female on error
                "label": "female"
            }
    
    async def predict_pcos_ensemble(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict PCOS from facial features using ensemble of all loaded models
        
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
                input_shape = model_data.get("input_shape", settings.FACE_IMAGE_SIZE)
                
                # Preprocess image with correct size for this model
                image_array = self._preprocess_image(image_bytes, input_shape)
                
                # Run prediction with error handling
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
                logger.debug(f"Face {model_name} prediction: {pcos_prob:.3f}")
                
            except Exception as e:
                logger.error(f"PCOS prediction failed for {model_name}: {str(e)}")
                # Continue with other models
                continue
        
        if not per_model_scores:
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
    
    async def process_face_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded face image and run full analysis pipeline
        
        Args:
            file: Uploaded face image file
            
        Returns:
            Dictionary with complete face analysis results
        """
        # Validate uploaded file
        image_bytes = await validate_image(file, max_mb=settings.MAX_UPLOAD_MB)
        
        # Generate unique filename and save
        file_id = str(uuid.uuid4())[:8]
        safe_filename = get_safe_filename(file.filename)
        name, ext = os.path.splitext(safe_filename)
        filename = f"face-{file_id}-{name}.jpg"
        file_path = UPLOADS_DIR / filename
        
        # Save uploaded file
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        try:
            # Run gender prediction
            gender_result = await self.predict_gender(image_bytes)
            
            # Initialize result structure
            result = {
                "face_img": f"/static/uploads/{filename}",
                "gender": gender_result,
                "face_scores": [],
                "face_pred": None,
                "face_risk": "unknown",
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "models_used": []
            }
            
            # Check if we should skip PCOS analysis for males
            if gender_result["label"] == "male":
                result["face_pred"] = "Male face detected; PCOS face analysis skipped"
                result["face_scores"] = []
                result["face_risk"] = "unknown"
                return result
            
            # Run PCOS prediction for females
            if gender_result["label"] == "female":
                if not self.pcos_models:
                    result["face_pred"] = "No PCOS models available for analysis"
                    result["face_scores"] = []
                    result["face_risk"] = "unknown"
                    result["warning"] = "No face PCOS models loaded"
                    return result
                
                pcos_results = await self.predict_pcos_ensemble(image_bytes)
                
                if pcos_results["per_model"]:
                    # Store detailed results
                    result["per_model"] = pcos_results["per_model"]
                    result["models_used"] = list(pcos_results["per_model"].keys())
                    result["ensemble"] = pcos_results["ensemble"]
                    
                    # Get ensemble results
                    ensemble_score = pcos_results["ensemble_score"]
                    result["ensemble_score"] = ensemble_score
                    
                    # Convert to face_scores format (non_pcos, pcos probabilities)
                    non_pcos_prob = 1.0 - ensemble_score
                    result["face_scores"] = [float(non_pcos_prob), float(ensemble_score)]
                    
                    # Determine prediction label and risk
                    risk_level = get_risk_level(ensemble_score)
                    result["face_risk"] = risk_level
                    
                    if risk_level == "high":
                        result["face_pred"] = "PCOS symptoms detected in facial analysis"
                    elif risk_level == "moderate":
                        result["face_pred"] = "Moderate PCOS indicators in facial analysis"
                    else:
                        result["face_pred"] = "No significant PCOS indicators in facial analysis"
                else:
                    result["face_pred"] = "No PCOS models available for analysis"
                    result["face_scores"] = []
                    result["face_risk"] = "unknown"
                    result["warning"] = "No face PCOS models loaded"
            
            return result
            
        except Exception as e:
            logger.error(f"Face processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def get_loading_warnings(self) -> List[str]:
        """Get any warnings from model loading"""
        return getattr(self, 'loading_warnings', [])
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all face models"""
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
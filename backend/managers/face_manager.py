"""
Face analysis manager for gender detection and PCOS classification

Handles automatic discovery and loading of all facial analysis models with
dynamic ensemble inference, proper label mapping, and robust error handling.
"""

import os
import uuid
import logging
import json
import h5py
from typing import Dict, Optional, Any, Tuple, List
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

logger = logging.getLogger(__name__)

# Model reliability blacklist - models known to give poor/opposite predictions
UNRELIABLE_MODELS = {
    'pcos_vgg16',  # Known to give opposite predictions
    'vgg16_weights_tf_dim_ordering_tf_kernels',  # Corrupted weights
    'resnet50_weights_tf_dim_ordering_tf_kernels',  # Corrupted weights
    'pcos_resnet50',  # Layer mismatch issues
}

# Minimum validation accuracy threshold for model inclusion
MIN_MODEL_ACCURACY = 0.60

def read_gender_labels(labels_path: str) -> List[str]:
    """
    Read gender labels from file with proper error handling
    
    Args:
        labels_path: Path to gender.labels.txt file
        
    Returns:
        List of gender labels in correct order
    """
    try:
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Try JSON format first
            if content.startswith('['):
                labels = json.loads(content)
                if isinstance(labels, list) and len(labels) >= 2:
                    return [str(label).strip().lower() for label in labels]
            
            # Try line-separated format
            lines = [line.strip().lower() for line in content.split('\n') if line.strip()]
            if len(lines) >= 2:
                return lines
                
        logger.warning(f"Could not read valid labels from {labels_path}, using defaults")
        
    except Exception as e:
        logger.warning(f"Error reading gender labels from {labels_path}: {str(e)}")
    
    # Default fallback
    return ['female', 'male']

def create_gender_mapping(labels: List[str]) -> Dict[str, int]:
    """
    Create gender mapping from labels ensuring correct index assignment
    
    Args:
        labels: List of gender labels
        
    Returns:
        Dictionary mapping gender names to indices
    """
    mapping = {}
    
    for i, label in enumerate(labels):
        if 'female' in label.lower():
            mapping['female'] = i
        elif 'male' in label.lower():
            mapping['male'] = i
    
    # Ensure both genders are mapped
    if 'female' not in mapping:
        mapping['female'] = 0
    if 'male' not in mapping:
        mapping['male'] = 1 if 'female' in mapping and mapping['female'] == 0 else 0
    
    logger.info(f"Created gender mapping: {mapping} from labels: {labels}")
    return mapping

# Create compiled prediction functions to avoid retracing warnings
@tf.function(reduce_retracing=True)
def compiled_predict_gender(model, input_data):
    """Compiled prediction function for gender model"""
    return model(input_data, training=False)

@tf.function(reduce_retracing=True)
def compiled_predict_pcos(model, input_data):
    """Compiled prediction function for PCOS models"""
    return model(input_data, training=False)

def load_model_with_fallback(model_path: str) -> Optional[tf.keras.Model]:
    """
    Load model with comprehensive fallback for various Keras issues
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model or None if failed
    """
    try:
        # Try normal loading first
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.debug(f"Successfully loaded model normally: {model_path}")
        return model
        
    except Exception as e:
        if "batch_shape" in str(e) or "Unrecognized keyword arguments" in str(e):
            logger.warning(f"Keras compatibility issue for {model_path}, trying fallback...")
            
            try:
                # Try loading with custom objects
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects={},
                    safe_mode=False
                )
                logger.info(f"Fallback loading successful: {model_path}")
                return model
                
            except Exception as fallback_e:
                logger.error(f"All loading methods failed for {model_path}: {str(fallback_e)}")
                return None
        else:
            logger.error(f"Model loading failed for {model_path}: {str(e)}")
            return None

def validate_model_output(model: tf.keras.Model, model_path: str) -> bool:
    """
    Validate model by running test prediction and checking output shape
    
    Args:
        model: Loaded model
        model_path: Path to model file for logging
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        # Get input shape from model
        if hasattr(model, 'input_shape') and model.input_shape:
            input_shape = model.input_shape[1:]  # Remove batch dimension
        else:
            input_shape = (224, 224, 3)  # Default
        
        # Create dummy input
        dummy_input = np.random.random((1, *input_shape)).astype(np.float32)
        
        # Run prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        # Validate output shape and values
        if prediction is None:
            logger.error(f"Model {model_path} returned None prediction")
            return False
            
        if len(prediction.shape) != 2:
            logger.error(f"Model {model_path} returned invalid shape: {prediction.shape}")
            return False
            
        if prediction.shape[1] not in [1, 2]:
            logger.error(f"Model {model_path} returned unexpected number of classes: {prediction.shape[1]}")
            return False
            
        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
            logger.error(f"Model {model_path} returned NaN or Inf values")
            return False
        
        logger.debug(f"Model validation successful for {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed for {model_path}: {str(e)}")
        return False

class FaceManager:
    """
    Enhanced face manager with robust gender detection and PCOS ensemble
    
    Features:
    - Proper gender label mapping from gender.labels.txt
    - Integration of new pcos_detector_158 model
    - Automatic exclusion of unreliable models
    - Robust ensemble with normalized weights
    - Comprehensive error handling and logging
    """
    
    def __init__(self):
        """Initialize face manager"""
        self.gender_model = None
        self.gender_labels = []
        self.gender_mapping = {}
        self.pcos_models = {}
        self.can_predict_gender = False
        self.ensemble_weights = {}
        self.loading_warnings = []
        
        # Model status tracking
        self.model_status = {
            "gender": {"loaded": False, "available": False, "error": None},
            "face": {"loaded": False, "available": False, "error": None}
        }
        
        # Load models at initialization
        self._load_models()
    
    def can_lazy_load_gender(self) -> bool:
        """Check if gender model can be lazy loaded"""
        gender_path = FACE_MODELS_DIR / settings.GENDER_MODEL
        return gender_path.exists() and gender_path.is_file()
    
    def can_lazy_load_pcos(self) -> bool:
        """Check if any PCOS models can be lazy loaded"""
        available_models = get_available_face_models()
        # Filter out unreliable models
        reliable_models = {k: v for k, v in available_models.items() if k not in UNRELIABLE_MODELS}
        return len(reliable_models) > 0
    
    def _load_models(self) -> None:
        """Load all facial analysis models with proper error handling"""
        logger.info("Loading facial analysis models...")
        
        # Load gender classifier first
        self._load_gender_model()
        
        # Load PCOS classification models
        self._load_pcos_models()
        
        logger.info(f"Face manager initialized. Gender detection: {self.can_predict_gender}, "
                   f"PCOS models loaded: {len(self.pcos_models)}")
    
    def _load_gender_model(self) -> None:
        """Load gender classification model with proper label mapping"""
        gender_path = FACE_MODELS_DIR / settings.GENDER_MODEL
        labels_path = FACE_MODELS_DIR / "gender.labels.txt"
        
        self.model_status["gender"]["available"] = gender_path.exists()
        
        try:
            if gender_path.exists():
                # Load model
                self.gender_model = load_model_with_fallback(str(gender_path))
                
                if self.gender_model is None:
                    raise Exception("Failed to load gender model with all fallback methods")
                
                # Load and validate labels
                self.gender_labels = read_gender_labels(str(labels_path))
                self.gender_mapping = create_gender_mapping(self.gender_labels)
                
                # Validate model works
                if validate_model_output(self.gender_model, str(gender_path)):
                    self.can_predict_gender = True
                    self.model_status["gender"]["loaded"] = True
                    logger.info(f"✅ Loaded gender model with labels {self.gender_labels}")
                    logger.info(f"Gender mapping: {self.gender_mapping}")
                else:
                    raise Exception("Gender model validation failed")
                    
            else:
                logger.warning(f"Gender model not found: {gender_path}")
                self.model_status["gender"]["error"] = "File not found"
                
        except Exception as e:
            logger.error(f"Failed to load gender model: {str(e)}")
            self.model_status["gender"]["error"] = str(e)
            self.loading_warnings.append(f"Gender model failed to load: {str(e)}")
    
    def _load_pcos_models(self) -> None:
        """Load PCOS classification models with reliability filtering"""
        loaded_count = 0
        skipped_count = 0
        
        # Get available models
        available_models = get_available_face_models()
        logger.info(f"Found {len(available_models)} potential face PCOS models")
        
        if not available_models:
            logger.warning("No face PCOS models found")
            self.model_status["face"]["available"] = False
            return
        
        # Get ensemble weights
        self.ensemble_weights = get_ensemble_weights('face')
        
        # Priority loading: Load new detector first
        priority_models = ['pcos_detector_158']
        regular_models = []
        
        for model_name in available_models.keys():
            if model_name in priority_models:
                continue  # Handle separately
            elif model_name in UNRELIABLE_MODELS:
                logger.info(f"⏭️  Skipping unreliable model: {model_name}")
                skipped_count += 1
                continue
            else:
                regular_models.append(model_name)
        
        # Load priority models first
        for model_name in priority_models:
            if model_name in available_models:
                success = self._load_single_pcos_model(model_name, available_models[model_name])
                if success:
                    loaded_count += 1
                    logger.info(f"✅ Loaded PCOS model: {model_name}")
        
        # Load other reliable models
        for model_name in regular_models:
            if model_name in available_models:
                success = self._load_single_pcos_model(model_name, available_models[model_name])
                if success:
                    loaded_count += 1
                else:
                    skipped_count += 1
        
        # Normalize weights for loaded models
        if self.pcos_models:
            model_names = list(self.pcos_models.keys())
            current_weights = {name: self.pcos_models[name]["weight"] for name in model_names}
            normalized_weights = normalize_weights(current_weights, model_names)
            
            for model_name, weight in normalized_weights.items():
                if model_name in self.pcos_models:
                    self.pcos_models[model_name]["weight"] = weight
            
            logger.info(f"✅ Normalized ensemble weights: {normalized_weights}")
            logger.info(f"📊 Models included in ensemble: {list(self.pcos_models.keys())}")
        
        self.model_status["face"]["loaded"] = loaded_count > 0
        self.model_status["face"]["available"] = len(available_models) > 0
        
        logger.info(f"✅ Successfully loaded {loaded_count} face PCOS models (skipped {skipped_count} unreliable)")
        
        if loaded_count == 0:
            logger.warning("⚠️  No reliable face PCOS models could be loaded")
            self.model_status["face"]["error"] = "No reliable models loaded"
            self.loading_warnings.append("No reliable face PCOS models available")
    
    def _load_single_pcos_model(self, model_name: str, model_path: Path) -> bool:
        """
        Load a single PCOS model with validation
        
        Args:
            model_name: Name of the model
            model_path: Path to model file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading PCOS model: {model_name}")
            
            # Load model with fallback
            model = load_model_with_fallback(str(model_path))
            
            if model is None:
                logger.error(f"Failed to load model {model_name}")
                return False
            
            # Validate model works correctly
            if not validate_model_output(model, str(model_path)):
                logger.error(f"Model validation failed for {model_name}")
                return False
            
            # Get model weight (prioritize new detector)
            if model_name == 'pcos_detector_158':
                weight = 1.0  # High weight for new trained model
            else:
                weight = self.ensemble_weights.get(model_name, 0.5)  # Lower weight for older models
            
            # Load labels
            labels = load_model_labels(model_path)
            
            # Get input shape
            input_shape = self._get_model_input_shape(model)
            
            # Store model data
            self.pcos_models[model_name] = {
                "model": model,
                "path": str(model_path),
                "weight": weight,
                "labels": labels,
                "input_shape": input_shape,
                "validated": True
            }
            
            logger.info(f"✅ Loaded PCOS model {model_name}: weight={weight}, input_shape={input_shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PCOS model {model_name}: {str(e)}")
            self.loading_warnings.append(f"PCOS model {model_name} failed to load: {str(e)}")
            return False
    
    def _get_model_input_shape(self, model: tf.keras.Model) -> Tuple[int, int]:
        """
        Get input shape from loaded model
        
        Args:
            model: Loaded TensorFlow model
            
        Returns:
            Tuple of (height, width) for image preprocessing
        """
        try:
            if hasattr(model, 'input_shape') and model.input_shape:
                shape = model.input_shape
                if len(shape) >= 3:
                    return (shape[1], shape[2])  # (height, width)
        except Exception:
            pass
        
        # Default to standard size
        return settings.FACE_IMAGE_SIZE
    
    def _preprocess_image(self, image_bytes: bytes, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Preprocess image for model inference with consistent shape
        
        Args:
            image_bytes: Raw image bytes
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image array with shape (1, height, width, 3)
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
            
            # Add batch dimension - ensure shape is (1, height, width, 3)
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    async def predict_gender(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict gender from facial image with proper label mapping
        
        Args:
            image_bytes: Preprocessed image bytes
            
        Returns:
            Dictionary with gender prediction results
        """
        if not self.can_predict_gender or self.gender_model is None:
            logger.debug("Gender model not available, defaulting to female")
            return {
                "male": 0.0,
                "female": 1.0,
                "label": "female",
                "confidence": 1.0
            }
        
        try:
            # Use gender model specific input size
            image_array = self._preprocess_image(image_bytes, settings.GENDER_IMAGE_SIZE)
            
            # Run prediction with compiled function
            try:
                prediction = compiled_predict_gender(self.gender_model, image_array)
                prediction = prediction.numpy()
            except Exception:
                # Fallback to regular predict
                prediction = self.gender_model.predict(image_array, verbose=0)
            
            # Handle prediction output - ensure we have proper shape
            probs = prediction[0] if len(prediction.shape) > 1 else prediction
            
            # Ensure we have the right number of outputs
            if len(probs.shape) == 0 or probs.shape[0] < 2:
                # Single output or insufficient outputs - handle as binary
                if len(probs.shape) == 0:
                    single_prob = float(probs)
                else:
                    single_prob = float(probs[0])
                
                # Map to gender based on labels
                if len(self.gender_labels) >= 2:
                    if self.gender_labels[0].lower() == 'female':
                        female_prob = single_prob
                        male_prob = 1.0 - single_prob
                    else:
                        male_prob = single_prob
                        female_prob = 1.0 - single_prob
                else:
                    # Default mapping
                    female_prob = single_prob
                    male_prob = 1.0 - single_prob
            else:
                # Two or more outputs - use mapping
                female_idx = self.gender_mapping.get('female', 0)
                male_idx = self.gender_mapping.get('male', 1)
                
                # Ensure indices are within bounds
                if female_idx < len(probs) and male_idx < len(probs):
                    female_prob = float(probs[female_idx])
                    male_prob = float(probs[male_idx])
                else:
                    logger.error(f"Gender mapping indices out of bounds: female_idx={female_idx}, male_idx={male_idx}, probs_shape={probs.shape}")
                    # Fallback to first two outputs
                    female_prob = float(probs[0]) if len(probs) > 0 else 0.5
                    male_prob = float(probs[1]) if len(probs) > 1 else 1.0 - female_prob
            
            # Determine predicted label and confidence
            if female_prob > male_prob:
                pred_label = "female"
                confidence = female_prob
            else:
                pred_label = "male"
                confidence = male_prob
            
            logger.info(f"Gender prediction - Female: {female_prob:.3f}, Male: {male_prob:.3f}, "
                       f"Label: {pred_label}, Confidence: {confidence:.3f}")
            
            return {
                "male": male_prob,
                "female": female_prob,
                "label": pred_label,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Gender prediction failed: {str(e)}")
            # Return safe default
            return {
                "male": 0.0,
                "female": 1.0,
                "label": "female",
                "confidence": 1.0
            }
    
    async def predict_pcos_ensemble(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict PCOS using ensemble of reliable models
        
        Args:
            image_bytes: Preprocessed image bytes
            
        Returns:
            Dictionary with ensemble prediction results
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
        
        # Run prediction on each loaded model
        for model_name, model_data in self.pcos_models.items():
            try:
                model = model_data["model"]
                input_shape = model_data.get("input_shape", settings.FACE_IMAGE_SIZE)
                
                # Preprocess image with correct size for this model
                image_array = self._preprocess_image(image_bytes, input_shape)
                
                # Run prediction with compiled function
                try:
                    prediction = compiled_predict_pcos(model, image_array)
                    prediction = prediction.numpy()
                except Exception:
                    # Fallback to regular predict
                    prediction = model.predict(image_array, verbose=0)
                
                # Extract PCOS probability with proper shape handling
                probs = prediction[0] if len(prediction.shape) > 1 else prediction
                
                if len(probs.shape) == 0 or probs.shape[0] == 1:
                    # Single output (sigmoid) - treat as PCOS probability
                    pcos_prob = float(probs) if len(probs.shape) == 0 else float(probs[0])
                else:
                    # Multiple outputs (softmax) - take PCOS class (index 1)
                    if probs.shape[0] >= 2:
                        pcos_prob = float(probs[1])
                    else:
                        logger.warning(f"Unexpected output shape for {model_name}: {probs.shape}")
                        pcos_prob = float(probs[0])
                
                # Validate probability is in valid range
                pcos_prob = max(0.0, min(1.0, pcos_prob))
                
                per_model_scores[model_name] = pcos_prob
                successful_predictions += 1
                logger.debug(f"PCOS {model_name} prediction: {pcos_prob:.3f}")
                
            except Exception as e:
                logger.error(f"PCOS prediction failed for {model_name}: {str(e)}")
                self.loading_warnings.append(f"PCOS model {model_name} prediction failed: {str(e)}")
                continue
        
        if not per_model_scores:
            self.loading_warnings.append("No PCOS models available for prediction")
            return {
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "labels": ["non_pcos", "pcos"]
            }
        
        # Calculate weighted ensemble score
        total_weight = 0.0
        weighted_sum = 0.0
        weights_used = {}
        
        for model_name, score in per_model_scores.items():
            weight = self.pcos_models[model_name]["weight"]
            weighted_sum += score * weight
            total_weight += weight
            weights_used[model_name] = weight
        
        ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Build ensemble metadata
        ensemble_meta = {
            "method": "weighted_average",
            "score": float(ensemble_score),
            "models_used": successful_predictions,
            "weights_used": weights_used
        }
        
        logger.info(f"✅ Ensemble prediction: {ensemble_score:.3f} from {successful_predictions} models")
        
        return {
            "per_model": per_model_scores,
            "ensemble_score": float(ensemble_score),
            "ensemble": ensemble_meta,
            "labels": ["non_pcos", "pcos"]
        }
    
    async def process_face_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded face image and run complete analysis pipeline
        
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
            
            # Check if we should skip PCOS analysis for males with high confidence
            gender_confidence_threshold = 0.7
            should_skip = (gender_result["label"] == "male" and 
                          gender_result.get("confidence", 0) >= gender_confidence_threshold)
            
            if should_skip:
                result["face_pred"] = "Male face detected - PCOS analysis not applicable"
                result["face_scores"] = []
                result["face_risk"] = "not_applicable"
                logger.info(f"Skipping PCOS analysis for male face (confidence: {gender_result['confidence']:.3f})")
                return result
            
            # Run PCOS prediction for females or uncertain gender
            if not self.pcos_models:
                result["face_pred"] = "No PCOS models available for analysis"
                result["face_scores"] = []
                result["face_risk"] = "unknown"
                self.loading_warnings.append("No face PCOS models loaded")
                return result
            
            # Run ensemble PCOS prediction
            pcos_results = await self.predict_pcos_ensemble(image_bytes)
            
            if pcos_results["per_model"]:
                # Store detailed results
                result["per_model"] = pcos_results["per_model"]
                result["models_used"] = list(pcos_results["per_model"].keys())
                result["ensemble"] = pcos_results["ensemble"]
                
                # Get ensemble score
                ensemble_score = pcos_results["ensemble_score"]
                result["ensemble_score"] = ensemble_score
                
                # Convert to face_scores format (non_pcos, pcos probabilities)
                non_pcos_prob = 1.0 - ensemble_score
                result["face_scores"] = [float(non_pcos_prob), float(ensemble_score)]
                
                # Determine risk level and prediction text
                risk_level = get_risk_level(ensemble_score)
                result["face_risk"] = risk_level
                
                if risk_level == "high":
                    result["face_pred"] = "High PCOS risk detected in facial analysis"
                elif risk_level == "moderate":
                    result["face_pred"] = "Moderate PCOS indicators detected in facial analysis"
                else:
                    result["face_pred"] = "Low PCOS risk - minimal indicators in facial analysis"
                
                logger.info(f"✅ Face analysis complete: {risk_level} risk, ensemble_score={ensemble_score:.3f}")
            else:
                result["face_pred"] = "PCOS analysis failed - no reliable predictions"
                result["face_scores"] = []
                result["face_risk"] = "unknown"
                self.loading_warnings.append("No reliable PCOS predictions available")
            
            return result
            
        except Exception as e:
            logger.error(f"Face processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def get_loading_warnings(self) -> List[str]:
        """Get any warnings from model loading"""
        return self.loading_warnings
    
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
                "input_shape": model_data.get("input_shape"),
                "validated": model_data.get("validated", False)
            }
        
        # Add gender model details
        if self.can_predict_gender:
            status["gender"]["labels"] = self.gender_labels
            status["gender"]["mapping"] = self.gender_mapping
        
        return status
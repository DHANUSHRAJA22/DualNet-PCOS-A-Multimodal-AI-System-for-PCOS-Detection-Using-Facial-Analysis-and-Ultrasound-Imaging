"""
Face analysis manager for gender detection and PCOS classification

Handles automatic discovery and loading of all facial analysis models with
dynamic ensemble inference and proper error handling.
"""

import os
import uuid
import logging
import json
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
    get_available_face_models, get_model_labels, get_ensemble_weights, normalize_weights,
    FACE_MODELS, BEST_FACE_MODEL, FACE_ENSEMBLE_WEIGHTS
)
from utils.validators import validate_image, get_safe_filename

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
        self.ensemble_weights = get_ensemble_weights('face')
        
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
        
        # Load PCOS classification models
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
        
        # Get available models
        available_models = get_available_face_models()
        
        if settings.USE_ENSEMBLE:
            models_to_load = available_models
        else:
            # Load only the best single model
            if BEST_FACE_MODEL in available_models:
                models_to_load = {BEST_FACE_MODEL: available_models[BEST_FACE_MODEL]}
            else:
                # Fallback to first available model
                if available_models:
                    first_model = next(iter(available_models.items()))
                    models_to_load = {first_model[0]: first_model[1]}
                else:
                    models_to_load = {}
        
        for model_name, model_path in models_to_load.items():
            
            try:
                model = tf.keras.models.load_model(str(model_path), compile=False)
                weight = self.ensemble_weights.get(model_name, 1.0)
                
                self.pcos_models[model_name] = {
                    "model": model,
                    "path": str(model_path),
                    "weight": weight
                }
                loaded_count += 1
                logger.info(f"Loaded PCOS model {model_name}: {model_path} (weight: {weight})")
                    
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
        
        self.model_status["face"]["loaded"] = loaded_count > 0
        self.model_status["face"]["available"] = len(available_models) > 0
        
        if loaded_count > 0:
            logger.info(f"Successfully loaded {loaded_count} face models with normalized weights")
    
    def _preprocess_image(self, image_bytes: bytes, target_size: Tuple[int, int]) -> np.ndarray:
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
            # Preprocess image for gender model (assuming 224x224 input)
            image_array = self._preprocess_image(image_bytes, (224, 224))
            
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
        per_model_predictions = {}
        per_model_scores = {}
        
        for model_name, model_data in self.pcos_models.items():
            try:
                model = model_data["model"]
                
                # Use standard input size (224x224 for most models)
                input_size = (224, 224)
                
                # Preprocess image
                image_array = self._preprocess_image(image_bytes, input_size)
                
                # Run prediction
                prediction = model.predict(image_array, verbose=0)
                
                # Extract PCOS probability (assuming binary classification)
                if prediction.shape[1] == 1:
                    # Single output (sigmoid)
                    pcos_prob = float(prediction[0][0])
                    probs = [1.0 - pcos_prob, pcos_prob]
                    # Two outputs (softmax)
                    probs = [float(p) for p in prediction[0]]
                    pcos_prob = probs[1]
                
                per_model_predictions[model_name] = probs
                per_model_scores[model_name] = pcos_prob
                logger.debug(f"Face {model_name} prediction: {pcos_prob:.3f}")
                
            except Exception as e:
                logger.error(f"PCOS prediction failed for {model_name}: {str(e)}")
                # Don't include failed models in ensemble
                continue
        
        if not per_model_predictions:
            return {
                "per_model": {},
                "per_model_scores": {},
                "ensemble": {"prob": 0.0, "probs": [0.5, 0.5], "label": "unknown", "weights": {}},
                "labels": get_model_labels('face')
            }
        
        # Compute ensemble prediction
        ensemble_result = self._compute_ensemble(per_model_predictions)
        
        return {
            "per_model": per_model_predictions,
            "per_model_scores": per_model_scores,
            "ensemble": ensemble_result,
            "labels": get_model_labels('face')
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
            return {"prob": 0.0, "probs": [0.5, 0.5], "label": "unknown", "weights": {}}
        
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
        
        labels = get_model_labels('face')
        predicted_label = labels[max_prob_idx] if max_prob_idx < len(labels) else "unknown"
        
        return {
            "prob": float(max_prob),
            "probs": ensemble_probs,
            "label": predicted_label,
            "weights": weights,
            "models_used": len(available_models)
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
                "face_models": {},
                "per_model": {},
                "ensemble_score": 0.0
            }
            
            # Check if we should skip PCOS analysis for males
            if gender_result["label"] == "male":
                result["face_pred"] = "Male face detected; PCOS face analysis skipped"
                result["face_scores"] = []
                result["face_risk"] = "unknown"
                return result
            
            # Run PCOS prediction for females
            if gender_result["label"] == "female":
                pcos_results = await self.predict_pcos_ensemble(image_bytes)
                
                if pcos_results["per_model"]:
                    # Store detailed results
                    result["face_models"] = pcos_results["per_model"]
                    result["per_model"] = pcos_results["per_model_scores"]
                    
                    # Get ensemble results
                    ensemble = pcos_results["ensemble"]
                    ensemble_probs = ensemble["probs"]
                    ensemble_score = ensemble["prob"]
                    
                    # Store ensemble results
                    result["face_scores"] = ensemble_probs
                    result["ensemble_score"] = ensemble_score
                    
                    # Determine prediction label
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
            
            return result
            
        except Exception as e:
            logger.error(f"Face processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all face models"""
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
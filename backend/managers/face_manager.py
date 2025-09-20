"""
Face analysis manager for gender detection and PCOS classification

Handles loading and inference of facial analysis models including gender
classification and PCOS ensemble prediction with proper error handling.
"""

import os
import uuid
import logging
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import UploadFile

# Import TensorFlow and suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras

from config import settings, FACE_MODELS_DIR, UPLOADS_DIR
from utils.validators import validate_image, get_safe_filename
from ensemble import EnsembleManager

logger = logging.getLogger(__name__)

class FaceManager:
    """
    Manages facial analysis including gender detection and PCOS classification
    
    Loads and manages multiple TensorFlow models for comprehensive facial analysis
    with ensemble prediction capabilities.
    """
    
    def __init__(self):
        """Initialize face manager and load models"""
        self.gender_model = None
        self.pcos_models = {}
        self.can_predict_gender = False
        self.ensemble_manager = EnsembleManager()
        
        # Model status tracking
        self.model_status = {
            "gender_detector": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
            "face_vgg16": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
            "face_resnet50": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
            "face_efficientnet_b0": {"status": "not_loaded", "file_exists": False, "path": None, "error": None},
        }
        
        self._load_models()
    
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
        
        self.model_status["gender_detector"]["file_exists"] = gender_path.exists()
        self.model_status["gender_detector"]["path"] = str(gender_path)
        
        try:
            if gender_path.exists():
                self.gender_model = tf.keras.models.load_model(str(gender_path), compile=False)
                self.can_predict_gender = True
                
                self.model_status["gender_detector"]["status"] = "loaded"
                logger.info(f"Loaded gender model: {gender_path}")
            else:
                logger.warning(f"Gender model not found: {gender_path}")
                self.model_status["gender_detector"]["error"] = "File not found"
                
        except Exception as e:
            logger.error(f"Failed to load gender model: {str(e)}")
            self.model_status["gender_detector"]["status"] = "error"
            self.model_status["gender_detector"]["error"] = str(e)
    
    def _load_pcos_models(self) -> None:
        """Load PCOS classification models"""
        for model_name, filename in settings.FACE_PCOS_MODELS.items():
            model_path = FACE_MODELS_DIR / filename
            status_key = f"face_{model_name}"
            
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
    
    def _get_input_size(self, model_name: str) -> Tuple[int, int]:
        """Get input size for specific model"""
        if model_name in ["vgg16", "resnet50"]:
            return (100, 100)
        elif model_name == "efficientnet_b0":
            return (224, 224)
        else:
            return (224, 224)  # Default
    
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
            # Preprocess image for gender model (assuming 100x100 input)
            image_array = self._preprocess_image(image_bytes, (100, 100))
            
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
    
    async def predict_pcos(self, image_bytes: bytes) -> Dict[str, float]:
        """
        Predict PCOS from facial features using ensemble of models
        
        Args:
            image_bytes: Preprocessed image bytes
            
        Returns:
            Dictionary mapping model names to PCOS probability scores
        """
        predictions = {}
        
        for model_name, model in self.pcos_models.items():
            try:
                # Get appropriate input size for this model
                input_size = self._get_input_size(model_name)
                
                # Preprocess image
                image_array = self._preprocess_image(image_bytes, input_size)
                
                # Run prediction
                prediction = model.predict(image_array, verbose=0)
                
                # Extract PCOS probability (assuming binary classification)
                if prediction.shape[1] == 1:
                    # Single output (sigmoid)
                    pcos_prob = float(prediction[0][0])
                else:
                    # Two outputs (softmax) - assume [non_pcos, pcos]
                    pcos_prob = float(prediction[0][1])
                
                predictions[model_name] = pcos_prob
                logger.debug(f"Face {model_name} prediction: {pcos_prob:.3f}")
                
            except Exception as e:
                logger.error(f"PCOS prediction failed for {model_name}: {str(e)}")
                # Don't include failed models in ensemble
                continue
        
        return predictions
    
    async def process_face_image(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded face image and run full analysis pipeline
        
        Args:
            file: Uploaded face image file
            
        Returns:
            Dictionary with complete face analysis results
        """
        # Validate uploaded file
        image_bytes = await validate_image(file)
        
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
                "modality": "face",
                "gender": gender_result,
                "per_model": {},
                "ensemble": {},
                "warnings": []
            }
            
            # Check if we should skip PCOS analysis for males
            if gender_result["label"] == "male":
                result["warnings"].append("Face appears male; skipping PCOS inference.")
                return result
            
            # Run PCOS prediction for females
            if gender_result["label"] == "female":
                pcos_predictions = await self.predict_pcos(image_bytes)
                result["per_model"] = pcos_predictions
                
                # Run ensemble if we have predictions
                if pcos_predictions:
                    ensemble_result = self.ensemble_manager.combine_face_models(pcos_predictions)
                    result["ensemble"] = ensemble_result
            
            return result
            
        except Exception as e:
            logger.error(f"Face processing failed: {str(e)}")
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all face models"""
        return self.model_status.copy()
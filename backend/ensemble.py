"""
Ensemble logic for combining multiple model predictions

Handles weighted averaging across face and X-ray modalities with automatic
weight normalization and missing model handling.
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EnsembleManager:
    """
    Manages ensemble prediction logic for both face and X-ray modalities
    
    Provides weighted averaging with automatic normalization and graceful
    handling of missing or failed models.
    """
    
    def __init__(self):
        """Initialize ensemble manager with default weights"""
        # Tune these weights as needed - they're normalized automatically
        self.weights = {
            "face": {
                "vgg16": 0.35,
                "resnet50": 0.35,
                "efficientnet_b0": 0.30
            },
            "xray": {
                "vgg16": 0.30,
                "resnet50": 0.30,
                "detector_158": 0.40
            }
        }
    
    def combine_face_models(self, per_model: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine face model predictions using weighted average
        
        Args:
            per_model: Dictionary mapping model names to PCOS probability scores
            
        Returns:
            Dictionary with ensemble method and combined score
        """
        return self._combine_models(per_model, "face")
    
    def combine_xray_models(self, per_model: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine X-ray model predictions using weighted average
        
        Args:
            per_model: Dictionary mapping model names to PCOS probability scores
            
        Returns:
            Dictionary with ensemble method and combined score
        """
        return self._combine_models(per_model, "xray")
    
    def _combine_models(self, per_model: Dict[str, float], modality: str) -> Dict[str, Any]:
        """
        Internal method to combine model predictions with weighted averaging
        
        Args:
            per_model: Model predictions dictionary
            modality: Either "face" or "xray"
            
        Returns:
            Dictionary with ensemble results
        """
        if not per_model:
            return {"method": "weighted_average", "score": 0.0, "models_used": 0}
        
        weight_map = self.weights.get(modality, {})
        
        # Keep only models that have numeric scores and configured weights
        valid_items = []
        for model_name, score in per_model.items():
            if model_name in weight_map and isinstance(score, (int, float)):
                valid_items.append((model_name, max(0.0, min(1.0, float(score)))))
        
        if not valid_items:
            logger.warning(f"No valid models for {modality} ensemble")
            return {"method": "weighted_average", "score": 0.0, "models_used": 0}
        
        # Calculate weighted average with normalization
        total_weight = sum(weight_map[model_name] for model_name, _ in valid_items)
        if total_weight == 0:
            # Fallback to simple average
            ensemble_score = sum(score for _, score in valid_items) / len(valid_items)
            method = "simple_average"
        else:
            # Weighted average
            weighted_sum = sum(weight_map[model_name] * score for model_name, score in valid_items)
            ensemble_score = weighted_sum / total_weight
            method = "weighted_average"
        
        result = {
            "method": method,
            "score": float(ensemble_score),
            "models_used": len(valid_items),
            "weights_used": {name: weight_map.get(name, 0) for name, _ in valid_items}
        }
        
        logger.info(f"{modality.title()} ensemble: {result}")
        return result
    
    def combine_modalities(self, face_ensemble: Optional[Dict], xray_ensemble: Optional[Dict]) -> Dict[str, Any]:
        """
        Combine face and X-ray ensemble results into final prediction
        
        Args:
            face_ensemble: Face ensemble results (can be None if skipped/failed)
            xray_ensemble: X-ray ensemble results (can be None if not provided)
            
        Returns:
            Dictionary with final risk assessment
        """
        available_scores = []
        modalities_used = []
        
        if face_ensemble and face_ensemble.get("score") is not None:
            available_scores.append(face_ensemble["score"])
            modalities_used.append("face")
        
        if xray_ensemble and xray_ensemble.get("score") is not None:
            available_scores.append(xray_ensemble["score"])
            modalities_used.append("xray")
        
        if not available_scores:
            return {
                "risk_score": 0.0,
                "risk_label": "unknown",
                "confidence": "low",
                "modalities_used": [],
                "method": "no_data"
            }
        
        # Simple average of available modality scores
        final_score = sum(available_scores) / len(available_scores)
        risk_label = self._classify_risk(final_score)
        confidence = self._calculate_confidence(available_scores, len(modalities_used))
        
        return {
            "risk_score": float(final_score),
            "risk_label": risk_label,
            "confidence": confidence,
            "modalities_used": modalities_used,
            "method": "modality_average"
        }
    
    def _classify_risk(self, score: float) -> str:
        """
        Classify risk level based on probability score
        
        Args:
            score: PCOS probability (0.0 to 1.0)
            
        Returns:
            Risk level: "low", "moderate", or "high"
        """
        from config import settings
        
        if score < settings.RISK_LOW_THRESHOLD:
            return "low"
        elif score < settings.RISK_HIGH_THRESHOLD:
            return "moderate"
        else:
            return "high"
    
    def _calculate_confidence(self, scores: List[float], num_modalities: int) -> str:
        """
        Calculate confidence level based on score consistency and modality coverage
        
        Args:
            scores: List of modality scores
            num_modalities: Number of modalities used
            
        Returns:
            Confidence level: "low", "medium", or "high"
        """
        if not scores:
            return "low"
        
        # Calculate score variance (consistency)
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            consistency = 1.0 - min(1.0, variance * 4)  # Scale variance to 0-1
        else:
            consistency = 0.8  # Single modality gets medium consistency
        
        # Factor in modality coverage
        coverage_factor = min(1.0, num_modalities / 2.0)  # Max benefit from 2 modalities
        
        # Combined confidence
        confidence_score = (consistency + coverage_factor) / 2.0
        
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def update_weights(self, modality: str, new_weights: Dict[str, float]) -> None:
        """
        Update ensemble weights for a specific modality
        
        Args:
            modality: "face" or "xray"
            new_weights: New weight configuration
        """
        if modality in self.weights:
            self.weights[modality].update(new_weights)
            logger.info(f"Updated {modality} weights: {self.weights[modality]}")
        else:
            logger.warning(f"Unknown modality for weight update: {modality}")
    
    def get_weights(self) -> Dict[str, Dict[str, float]]:
        """Get current ensemble weights configuration"""
        return self.weights.copy()
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
        """Initialize ensemble manager"""
        pass
    
    def combine_face_models(self, per_model: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine face model predictions using weighted average
        
        Args:
            per_model: Dictionary mapping model names to PCOS probability scores
            weights: Dictionary mapping model names to weights
            
        Returns:
            Dictionary with ensemble method and combined score
        """
        return self._combine_models(per_model, weights, "face")
    
    def combine_xray_models(self, per_model: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine X-ray model predictions using weighted average
        
        Args:
            per_model: Dictionary mapping model names to PCOS probability scores
            weights: Dictionary mapping model names to weights
            
        Returns:
            Dictionary with ensemble method and combined score
        """
        return self._combine_models(per_model, weights, "xray")
    
    def _combine_models(self, per_model: Dict[str, float], weights: Dict[str, float], modality: str) -> Dict[str, Any]:
        """
        Internal method to combine model predictions with weighted averaging
        
        Args:
            per_model: Model predictions dictionary
            weights: Model weights dictionary
            modality: Either "face" or "xray"
            
        Returns:
            Dictionary with ensemble results
        """
        if not per_model:
            return {"method": "weighted_average", "score": 0.0, "models_used": 0}
        
        # Keep only models that have numeric scores and configured weights
        valid_items = []
        for model_name, score in per_model.items():
            if model_name in weights and isinstance(score, (int, float)):
                # Ensure score is between 0 and 1
                clamped_score = max(0.0, min(1.0, float(score)))
                valid_items.append((model_name, clamped_score))
        
        if not valid_items:
            logger.warning(f"No valid models for {modality} ensemble")
            return {"method": "weighted_average", "score": 0.0, "models_used": 0}
        
        # Calculate weighted average with normalization
        total_weight = sum(weights[model_name] for model_name, _ in valid_items)
        if total_weight == 0:
            # Fallback to simple average
            ensemble_score = sum(score for _, score in valid_items) / len(valid_items)
            method = "simple_average"
        else:
            # Weighted average
            weighted_sum = sum(weights[model_name] * score for model_name, score in valid_items)
            ensemble_score = weighted_sum / total_weight
            method = "weighted_average"
        
        result = {
            "method": method,
            "score": float(ensemble_score),
            "models_used": len(valid_items),
            "weights_used": {name: weights.get(name, 0) for name, _ in valid_items}
        }
        
        logger.info(f"{modality.title()} ensemble: {result}")
        return result
    
    def combine_modalities(self, face_score: Optional[float], xray_score: Optional[float]) -> Dict[str, Any]:
        """
        Combine face and X-ray scores into final risk assessment
        
        Args:
            face_score: Face ensemble score (can be None if skipped/failed)
            xray_score: X-ray ensemble score (can be None if not provided)
            
        Returns:
            Dictionary with final risk assessment
        """
        available_scores = []
        modalities_used = []
        
        if face_score is not None:
            available_scores.append(float(face_score))
            modalities_used.append("face")
        
        if xray_score is not None:
            available_scores.append(float(xray_score))
            modalities_used.append("xray")
        
        if not available_scores:
            return {
                "overall_risk": "unknown",
                "combined": "No valid predictions available for risk assessment",
                "modalities_used": [],
                "final_score": 0.0,
                "fusion_method": "none"
            }
        
        # Enhanced fusion logic with thresholds
        high_risk_count = sum(1 for score in available_scores if score >= 0.66)
        moderate_risk_count = sum(1 for score in available_scores if 0.33 <= score < 0.66)
        
        # Discrete fusion rules
        if len(available_scores) == 2:  # Both modalities
            if high_risk_count == 2:
                risk_level = "high"
                explanation = "High risk: Both facial and X-ray analysis indicate PCOS symptoms."
            elif high_risk_count == 1:
                risk_level = "moderate"
                explanation = "Moderate risk: One modality shows high risk indicators."
            elif moderate_risk_count >= 1:
                risk_level = "moderate"
                explanation = "Moderate risk: Mixed indicators across modalities."
            else:
                risk_level = "low"
                explanation = "Low risk: Both modalities show minimal PCOS indicators."
        else:  # Single modality
            single_score = available_scores[0]
            if single_score >= 0.66:
                risk_level = "high"
            elif single_score >= 0.33:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            modality_name = "facial analysis" if "face" in modalities_used else "X-ray analysis"
            explanation = f"{risk_level.title()} risk based on {modality_name}."
        
        # Calculate final score as weighted average
        final_score = sum(available_scores) / len(available_scores)
        
        return {
            "overall_risk": risk_level,
            "combined": explanation,
            "modalities_used": modalities_used,
            "final_score": float(final_score),
            "fusion_method": "discrete_rules",
            "high_risk_count": high_risk_count,
            "thresholds_used": {"low": 0.33, "high": 0.66}
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
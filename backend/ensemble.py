"""
Ensemble fusion logic for combining face and X-ray modality predictions

Supports multiple fusion modes including threshold-based and discrete rules
with configurable risk thresholds and detailed explanations.
"""

import logging
from typing import Dict, Optional, Any
from config import settings, get_risk_level

logger = logging.getLogger(__name__)

class EnsembleManager:
    """
    Manages fusion of predictions from multiple modalities
    
    Combines face and X-ray ensemble results using configurable fusion modes
    with detailed risk assessment and explanations.
    """
    
    def __init__(self):
        """Initialize ensemble manager"""
        self.fusion_mode = settings.FUSION_MODE
        logger.info(f"Initialized EnsembleManager with fusion_mode={self.fusion_mode}")
    
    def combine_modalities(self, face_score: Optional[float], xray_score: Optional[float]) -> Dict[str, Any]:
        """
        Combine face and X-ray scores into final risk assessment
        
        Args:
            face_score: Face ensemble probability (0-1) or None if not available
            xray_score: X-ray ensemble probability (0-1) or None if not available
            
        Returns:
            Dictionary with final risk assessment and metadata
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
                "score": 0.0,
                "fusion_method": "none",
                "confidence": 0.0
            }
        
        # Apply fusion logic based on mode
        if self.fusion_mode == "discrete":
            return self._discrete_fusion(face_score, xray_score, modalities_used)
        else:
            return self._threshold_fusion(available_scores, modalities_used, face_score, xray_score)
    
    def _threshold_fusion(self, scores: list, modalities_used: list, face_score: Optional[float], xray_score: Optional[float]) -> Dict[str, Any]:
        """
        Threshold-based fusion using probability bands
        
        Args:
            scores: List of available scores
            modalities_used: List of modality names
            face_score: Face score for detailed explanation
            xray_score: X-ray score for detailed explanation
            
        Returns:
            Dictionary with fusion results
        """
        # Calculate final score as weighted average
        final_score = sum(scores) / len(scores)
        
        # Classify risk using thresholds
        risk_level = get_risk_level(final_score)
        
        # Generate detailed explanation
        explanation = self._generate_threshold_explanation(
            risk_level, final_score, modalities_used, face_score, xray_score
        )
        
        return {
            "overall_risk": risk_level,
            "combined": explanation,
            "modalities_used": modalities_used,
            "final_score": float(final_score),
            "score": float(final_score),
            "fusion_method": "threshold",
            "confidence": float(final_score),
            "thresholds_used": {
                "low": settings.RISK_LOW_THRESHOLD,
                "high": settings.RISK_HIGH_THRESHOLD
            }
        }
    
    def _discrete_fusion(self, face_score: Optional[float], xray_score: Optional[float], modalities_used: list) -> Dict[str, Any]:
        """
        Discrete fusion rules with specific logic for multi-modal assessment
        
        Rules:
        - High risk: Both modalities ≥ 0.66 OR any single ≥ 0.8
        - Moderate risk: Exactly one ≥ 0.66 OR average ≥ 0.33
        - Low risk: All modalities < 0.33
        
        Args:
            face_score: Face score (can be None)
            xray_score: X-ray score (can be None)
            modalities_used: List of modality names
            
        Returns:
            Dictionary with fusion results
        """
        # Count high and moderate risk modalities
        high_count = 0
        moderate_count = 0
        very_high_count = 0
        
        scores = []
        
        if face_score is not None:
            scores.append(face_score)
            if face_score >= 0.8:
                very_high_count += 1
            elif face_score >= settings.RISK_HIGH_THRESHOLD:
                high_count += 1
            elif face_score >= settings.RISK_LOW_THRESHOLD:
                moderate_count += 1
        
        if xray_score is not None:
            scores.append(xray_score)
            if xray_score >= 0.8:
                very_high_count += 1
            elif xray_score >= settings.RISK_HIGH_THRESHOLD:
                high_count += 1
            elif xray_score >= settings.RISK_LOW_THRESHOLD:
                moderate_count += 1
        
        # Apply discrete rules
        if len(modalities_used) == 2:  # Both modalities available
            if very_high_count >= 1 or high_count == 2:
                risk_level = "high"
                explanation = "High risk: Strong PCOS indicators detected across modalities"
            elif high_count == 1 or moderate_count >= 1:
                risk_level = "moderate"
                explanation = "Moderate risk: Mixed indicators across facial and X-ray analysis"
            else:
                risk_level = "low"
                explanation = "Low risk: Minimal PCOS indicators in both modalities"
        else:  # Single modality
            single_score = face_score if face_score is not None else xray_score
            risk_level = get_risk_level(single_score)
            modality_name = "facial analysis" if "face" in modalities_used else "X-ray analysis"
            
            if risk_level == "high":
                explanation = f"High risk: Strong PCOS indicators in {modality_name}"
            elif risk_level == "moderate":
                explanation = f"Moderate risk: Some PCOS indicators in {modality_name}"
            else:
                explanation = f"Low risk: Minimal PCOS indicators in {modality_name}"
        
        # Calculate final score for confidence
        final_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "overall_risk": risk_level,
            "combined": explanation,
            "modalities_used": modalities_used,
            "final_score": float(final_score),
            "score": float(final_score),
            "fusion_method": "discrete",
            "confidence": float(final_score),
            "high_risk_count": high_count,
            "very_high_risk_count": very_high_count,
            "moderate_risk_count": moderate_count,
            "thresholds_used": {
                "low": settings.RISK_LOW_THRESHOLD,
                "high": settings.RISK_HIGH_THRESHOLD,
                "very_high": 0.8
            }
        }
    
    def _generate_threshold_explanation(self, risk_level: str, final_score: float, modalities_used: list, face_score: Optional[float], xray_score: Optional[float]) -> str:
        """
        Generate detailed explanation for threshold-based fusion
        
        Args:
            risk_level: Computed risk level
            final_score: Final ensemble score
            modalities_used: List of modalities used
            face_score: Face score for context
            xray_score: X-ray score for context
            
        Returns:
            Detailed explanation string
        """
        confidence_pct = final_score * 100
        
        if len(modalities_used) == 2:
            # Both modalities available
            face_risk = get_risk_level(face_score) if face_score is not None else "unknown"
            xray_risk = get_risk_level(xray_score) if xray_score is not None else "unknown"
            
            if risk_level == "high":
                if face_risk == "high" and xray_risk == "high":
                    return f"High risk ({confidence_pct:.1f}%): Both facial and X-ray analysis show strong PCOS indicators"
                elif face_risk == "high" or xray_risk == "high":
                    dominant = "facial" if face_risk == "high" else "X-ray"
                    return f"High risk ({confidence_pct:.1f}%): {dominant} analysis shows strong PCOS indicators"
                else:
                    return f"High risk ({confidence_pct:.1f}%): Combined analysis indicates elevated PCOS probability"
            elif risk_level == "moderate":
                return f"Moderate risk ({confidence_pct:.1f}%): Mixed indicators across facial and X-ray modalities"
            else:
                return f"Low risk ({confidence_pct:.1f}%): Both modalities show minimal PCOS indicators"
        
        else:
            # Single modality
            modality_name = "facial analysis" if "face" in modalities_used else "X-ray analysis"
            
            if risk_level == "high":
                return f"High risk ({confidence_pct:.1f}%): {modality_name} shows strong PCOS indicators"
            elif risk_level == "moderate":
                return f"Moderate risk ({confidence_pct:.1f}%): {modality_name} shows some PCOS indicators"
            else:
                return f"Low risk ({confidence_pct:.1f}%): {modality_name} shows minimal PCOS indicators"
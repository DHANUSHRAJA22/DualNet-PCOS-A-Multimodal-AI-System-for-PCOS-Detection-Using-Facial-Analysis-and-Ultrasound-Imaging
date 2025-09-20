"""
Ensemble prediction logic for combining multiple model outputs

Supports flexible ensemble methods including soft voting, hard voting,
and weighted averaging with configurable model weights.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    Flexible ensemble predictor for combining multiple model outputs
    
    Supports various ensemble methods and configurable weights for different
    modalities (face analysis, X-ray analysis) and individual models.
    """
    
    def __init__(self, method: str = None, weights: Dict[str, float] = None):
        """
        Initialize ensemble predictor
        
        Args:
            method: Ensemble method ('soft_voting', 'hard_voting', 'weighted_average')
            weights: Weights for different modalities
        """
        self.method = method or settings.ENSEMBLE_METHOD
        self.weights = weights or settings.ENSEMBLE_WEIGHTS.copy()
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        logger.info(f"Initialized EnsemblePredictor with method={self.method}, weights={self.weights}")
    
    def combine_predictions(
        self, 
        face_results: Optional[Dict[str, Any]], 
        xray_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combine predictions from face and X-ray analysis using ensemble logic
        
        Args:
            face_results: Results dictionary from facial analysis ensemble
            xray_results: Results dictionary from X-ray analysis ensemble
            
        Returns:
            Dictionary containing:
                - final_prediction: Human-readable combined prediction
                - combined_probability: Ensemble probability score
                - confidence: Overall confidence score
                - risk_level: Classified risk level (low/moderate/high)
                - weights: Weights used in ensemble
                - method: Ensemble method used
                - individual_contributions: Breakdown of each modality's contribution
                
        Raises:
            ValueError: If no valid predictions provided
        """
        
        if not face_results and not xray_results:
            raise ValueError("At least one prediction result must be provided")
        
        start_time = datetime.now()
        
        # Extract probabilities (assuming [non_pcos_prob, pcos_prob] format)
        face_prob = face_results["scores"][1] if face_results else 0.0
        xray_prob = xray_results["scores"][1] if xray_results else 0.0
        
        face_confidence = face_results["confidence"] if face_results else 0.0
        xray_confidence = xray_results["confidence"] if xray_results else 0.0
        
        # Apply ensemble method
        if self.method == "soft_voting":
            combined_prob, overall_confidence = self._soft_voting_ensemble(
                face_prob, xray_prob, face_confidence, xray_confidence,
                face_results is not None, xray_results is not None
            )
        elif self.method == "weighted_average":
            combined_prob, overall_confidence = self._weighted_average_ensemble(
                face_prob, xray_prob, face_confidence, xray_confidence,
                face_results is not None, xray_results is not None
            )
        else:
            # Default to soft voting
            combined_prob, overall_confidence = self._soft_voting_ensemble(
                face_prob, xray_prob, face_confidence, xray_confidence,
                face_results is not None, xray_results is not None
            )
        
        # Classify risk level
        risk_level = settings.get_risk_level(combined_prob)
        
        # Generate human-readable prediction
        prediction_text = self._generate_prediction_text(
            face_results, xray_results, combined_prob, risk_level
        )
        
        # Calculate individual contributions
        contributions = self._calculate_contributions(
            face_prob, xray_prob, face_results is not None, xray_results is not None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = {
            "final_prediction": prediction_text,
            "combined_probability": float(combined_prob),
            "confidence": float(overall_confidence),
            "risk_level": risk_level,
            "weights": self.weights,
            "method": self.method,
            "individual_contributions": contributions,
            "processing_time_ms": processing_time
        }
        
        logger.info(f"Ensemble prediction completed: risk={risk_level}, confidence={overall_confidence:.3f}")
        
        return result
    
    def _soft_voting_ensemble(
        self, 
        face_prob: float, 
        xray_prob: float,
        face_conf: float,
        xray_conf: float,
        has_face: bool,
        has_xray: bool
    ) -> Tuple[float, float]:
        """
        Soft voting ensemble with confidence-weighted averaging
        
        Args:
            face_prob: Face analysis PCOS probability
            xray_prob: X-ray analysis PCOS probability
            face_conf: Face analysis confidence
            xray_conf: X-ray analysis confidence
            has_face: Whether face analysis was performed
            has_xray: Whether X-ray analysis was performed
            
        Returns:
            Tuple of (combined_probability, overall_confidence)
        """
        
        if has_face and has_xray:
            # Weight by both modality weights and individual confidence
            face_weight = self.weights['face'] * face_conf
            xray_weight = self.weights['xray'] * xray_conf
            
            total_weight = face_weight + xray_weight
            
            if total_weight > 0:
                combined_prob = (face_prob * face_weight + xray_prob * xray_weight) / total_weight
                overall_confidence = (face_conf * self.weights['face'] + xray_conf * self.weights['xray'])
            else:
                combined_prob = (face_prob + xray_prob) / 2
                overall_confidence = (face_conf + xray_conf) / 2
                
        elif has_face:
            combined_prob = face_prob
            overall_confidence = face_conf
        else:
            combined_prob = xray_prob
            overall_confidence = xray_conf
        
        return combined_prob, overall_confidence
    
    def _weighted_average_ensemble(
        self,
        face_prob: float,
        xray_prob: float, 
        face_conf: float,
        xray_conf: float,
        has_face: bool,
        has_xray: bool
    ) -> Tuple[float, float]:
        """
        Simple weighted average ensemble
        
        Args:
            face_prob: Face analysis PCOS probability
            xray_prob: X-ray analysis PCOS probability
            face_conf: Face analysis confidence
            xray_conf: X-ray analysis confidence
            has_face: Whether face analysis was performed
            has_xray: Whether X-ray analysis was performed
            
        Returns:
            Tuple of (combined_probability, overall_confidence)
        """
        
        if has_face and has_xray:
            combined_prob = (
                face_prob * self.weights['face'] + 
                xray_prob * self.weights['xray']
            )
            overall_confidence = (
                face_conf * self.weights['face'] + 
                xray_conf * self.weights['xray']
            )
        elif has_face:
            combined_prob = face_prob
            overall_confidence = face_conf
        else:
            combined_prob = xray_prob
            overall_confidence = xray_conf
        
        return combined_prob, overall_confidence
    
    def _generate_prediction_text(
        self,
        face_results: Optional[Dict[str, Any]],
        xray_results: Optional[Dict[str, Any]], 
        probability: float,
        risk_level: str
    ) -> str:
        """
        Generate human-readable prediction text based on ensemble results
        
        Args:
            face_results: Face analysis results
            xray_results: X-ray analysis results
            probability: Combined PCOS probability
            risk_level: Classified risk level
            
        Returns:
            Human-readable prediction text
        """
        
        modalities = []
        if face_results:
            modalities.append("facial")
        if xray_results:
            modalities.append("imaging")
        
        modality_text = " and ".join(modalities)
        
        if risk_level == 'low':
            return (
                f"Combined analysis indicates low PCOS risk (confidence: {probability:.1%}). "
                f"The {modality_text} analysis shows minimal concerning indicators. "
                f"Regular monitoring and healthy lifestyle practices are recommended."
            )
        elif risk_level == 'moderate':
            return (
                f"Combined analysis suggests moderate PCOS risk (confidence: {probability:.1%}). "
                f"Some indicators present in {modality_text} analysis warrant further medical evaluation. "
                f"Consider consulting with a healthcare professional for comprehensive assessment."
            )
        else:  # high risk
            return (
                f"Combined analysis indicates high PCOS risk (confidence: {probability:.1%}). "
                f"Multiple concerning indicators detected across {modality_text} analysis. "
                f"Professional medical consultation is strongly recommended for proper diagnosis and treatment planning."
            )
    
    def _calculate_contributions(
        self,
        face_prob: float,
        xray_prob: float,
        has_face: bool,
        has_xray: bool
    ) -> Dict[str, float]:
        """
        Calculate individual modality contributions to final prediction
        
        Args:
            face_prob: Face analysis probability
            xray_prob: X-ray analysis probability
            has_face: Whether face analysis was performed
            has_xray: Whether X-ray analysis was performed
            
        Returns:
            Dictionary with contribution percentages
        """
        
        contributions = {}
        
        if has_face and has_xray:
            total_weighted = (
                face_prob * self.weights['face'] + 
                xray_prob * self.weights['xray']
            )
            
            if total_weighted > 0:
                contributions['face'] = (face_prob * self.weights['face']) / total_weighted
                contributions['xray'] = (xray_prob * self.weights['xray']) / total_weighted
            else:
                contributions['face'] = self.weights['face']
                contributions['xray'] = self.weights['xray']
                
        elif has_face:
            contributions['face'] = 1.0
        else:
            contributions['xray'] = 1.0
        
        return contributions
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update ensemble weights with validation
        
        Args:
            new_weights: New weight configuration
            
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.weights.update(new_weights)
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get current ensemble configuration information
        
        Returns:
            Dictionary with ensemble metadata
        """
        return {
            "method": self.method,
            "weights": self.weights,
            "confidence_threshold": self.confidence_threshold,
            "risk_thresholds": settings.RISK_THRESHOLDS
        }
"""
Ensemble utilities and multi-modal fusion for PCOS detection.

This module provides:
1) robust_weighted_ensemble(...)  -> combine per-model scores (face/X-ray model lists)
2) EnsembleManager               -> combine FACE vs XRAY modality scores

It is careful about:
- Missing/NaN/None scores
- Optional per-model weights
- Outlier suppression
- Clear metadata (kept/excluded models, normalized weights, method used)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger("ensemble")

# thresholds & defaults (kept self-contained to avoid import loops)
DEFAULT_LOW = 0.33
DEFAULT_HIGH = 0.66


def _risk_level(p: Optional[float], low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH) -> str:
    if p is None:
        return "unknown"
    try:
        p = float(p)
    except Exception:
        return "unknown"
    if p >= high:
        return "high"
    if p >= low:
        return "moderate"
    return "low"


# ---------- Per-model ensemble (within a modality) ----------

def robust_weighted_ensemble(
    items: List[Dict[str, Any]],
    *,
    min_score: float = 0.0,
    max_score: float = 1.0,
    zscore_clip: float = 2.5,
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Combine multiple model outputs into a single score using a robust weighted average.

    items: list of { "name": str, "score": float|None, "weight": float=1.0 }
    returns: (final_score, kept_items_with_used_weight, excluded_with_reason)
    """
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []

    # 1) validate
    for it in items:
        name = it.get("name", "unknown")
        raw_score = it.get("score", None)
        raw_weight = it.get("weight", 1.0)

        if raw_score is None:
            excluded.append({**it, "reason": "score_none"})
            continue

        try:
            s = float(raw_score)
        except Exception:
            excluded.append({**it, "reason": f"score_not_float({raw_score})"})
            continue

        try:
            w = float(raw_weight)
        except Exception:
            w = 1.0

        if not np.isfinite(s):
            excluded.append({**it, "reason": "score_not_finite"})
            continue
        if s < min_score or s > max_score:
            excluded.append({**it, "reason": f"score_out_of_range({s})"})
            continue
        if w <= 0 or not np.isfinite(w):
            excluded.append({**it, "reason": f"bad_weight({raw_weight})"})
            continue

        kept.append({"name": name, "score": s, "weight": w})

    if not kept:
        logger.warning("robust_weighted_ensemble: no valid items to keep")
        return 0.0, [], excluded

    # 2) optional outlier suppression
    if zscore_clip and len(kept) >= 3:
        scores = np.array([k["score"] for k in kept], dtype=float)
        mu = float(np.mean(scores))
        sd = float(np.std(scores)) or 1e-9
        filtered: List[Dict[str, Any]] = []
        for k in kept:
            z = abs((k["score"] - mu) / sd)
            if z > zscore_clip:
                excluded.append({**k, "reason": f"zscore_outlier({z:.2f})"})
            else:
                filtered.append(k)
        kept = filtered

    if not kept:
        logger.warning("robust_weighted_ensemble: all items excluded after z-clip")
        return 0.0, [], excluded

    # 3) normalize weights
    total_w = float(sum(k["weight"] for k in kept))
    if total_w <= 0:
        for k in kept:
            k["used_weight"] = 1.0 / len(kept)
    else:
        for k in kept:
            k["used_weight"] = k["weight"] / total_w

    # 4) weighted average
    final_score = float(sum(k["score"] * k["used_weight"] for k in kept))
    final_score = max(min(final_score, 1.0), 0.0)

    return final_score, kept, excluded


# ---------- Cross-modality fusion (face vs x-ray) ----------

class EnsembleManager:
    """
    Combines FACE and XRAY modality ensemble scores into a final risk assessment.
    Supports 'threshold' (default) and 'discrete' fusion modes.

    Configure via env if desired:
      - FUSION_MODE: "threshold" | "discrete"
      - RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD
    """

    def __init__(self, fusion_mode: str = "threshold", low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH) -> None:
        self.fusion_mode = fusion_mode
        self.low = float(low)
        self.high = float(high)
        logger.info(f"Initialized EnsembleManager with fusion_mode={self.fusion_mode}")

    def combine_modalities(
        self,
        face_score: Optional[float],
        xray_score: Optional[float],
    ) -> Dict[str, Any]:
        avail_scores: List[float] = []
        used_modalities: List[str] = []

        if face_score is not None:
            avail_scores.append(float(face_score))
            used_modalities.append("face")

        if xray_score is not None:
            avail_scores.append(float(xray_score))
            used_modalities.append("xray")

        if not avail_scores:
            return {
                "overall_risk": "unknown",
                "combined": "No valid predictions available for risk assessment",
                "modalities_used": [],
                "final_score": 0.0,
                "score": 0.0,
                "fusion_method": "none",
                "confidence": 0.0,
            }

        if self.fusion_mode == "discrete":
            return self._discrete_fusion(face_score, xray_score, used_modalities)

        return self._threshold_fusion(avail_scores, used_modalities, face_score, xray_score)

    # ---- threshold fusion ----
    def _threshold_fusion(
        self,
        scores: List[float],
        modalities_used: List[str],
        face_score: Optional[float],
        xray_score: Optional[float],
    ) -> Dict[str, Any]:
        final_score = float(np.mean(scores))
        risk_level = _risk_level(final_score, self.low, self.high)
        explanation = self._generate_threshold_explanation(
            risk_level, final_score, modalities_used, face_score, xray_score
        )
        return {
            "overall_risk": risk_level,
            "combined": explanation,
            "modalities_used": modalities_used,
            "final_score": final_score,
            "score": final_score,
            "fusion_method": "threshold",
            "confidence": final_score,
            "thresholds_used": {"low": self.low, "high": self.high},
        }

    # ---- discrete fusion ----
    def _discrete_fusion(
        self,
        face_score: Optional[float],
        xray_score: Optional[float],
        modalities_used: List[str],
    ) -> Dict[str, Any]:
        def buckets(s: Optional[float]) -> tuple[bool, bool, bool]:
            if s is None:
                return (False, False, False)
            s = float(s)
            very_high = s >= 0.80
            high = s >= self.high
            moderate = (s >= self.low) and (s < self.high)
            return very_high, high, moderate

        vh_c = h_c = m_c = 0
        scores: List[float] = []

        if face_score is not None:
            scores.append(float(face_score))
            vh, h, m = buckets(face_score)
            vh_c += int(vh)
            h_c += int(h)
            m_c += int(m)
        if xray_score is not None:
            scores.append(float(xray_score))
            vh, h, m = buckets(xray_score)
            vh_c += int(vh)
            h_c += int(h)
            m_c += int(m)

        if len(modalities_used) == 2:
            if vh_c >= 1 or h_c == 2:
                risk = "high"
                msg = "High risk: Strong PCOS indicators detected across modalities"
            elif h_c == 1 or m_c >= 1:
                risk = "moderate"
                msg = "Moderate risk: Mixed indicators across facial and X-ray analysis"
            else:
                risk = "low"
                msg = "Low risk: Minimal PCOS indicators in both modalities"
        else:
            s = face_score if face_score is not None else xray_score
            risk = _risk_level(s, self.low, self.high)
            mod_name = "facial analysis" if "face" in modalities_used else "X-ray analysis"
            if risk == "high":
                msg = f"High risk: Strong PCOS indicators in {mod_name}"
            elif risk == "moderate":
                msg = f"Moderate risk: Some PCOS indicators in {mod_name}"
            else:
                msg = f"Low risk: Minimal PCOS indicators in {mod_name}"

        final_score = float(np.mean(scores)) if scores else 0.0
        return {
            "overall_risk": risk,
            "combined": msg,
            "modalities_used": modalities_used,
            "final_score": final_score,
            "score": final_score,
            "fusion_method": "discrete",
            "confidence": final_score,
            "high_risk_count": h_c,
            "very_high_risk_count": vh_c,
            "moderate_risk_count": m_c,
            "thresholds_used": {"low": self.low, "high": self.high, "very_high": 0.80},
        }

    # ---- explanation helper ----
    def _generate_threshold_explanation(
        self,
        risk_level: str,
        final_score: float,
        modalities_used: List[str],
        face_score: Optional[float],
        xray_score: Optional[float],
    ) -> str:
        pct = final_score * 100.0

        if len(modalities_used) == 2:
            face_risk = _risk_level(face_score, self.low, self.high)
            xray_risk = _risk_level(xray_score, self.low, self.high)

            if risk_level == "high":
                if face_risk == "high" and xray_risk == "high":
                    return f"High risk ({pct:.1f}%): Both facial and X-ray analysis show strong PCOS indicators"
                if face_risk == "high" or xray_risk == "high":
                    dom = "facial" if face_risk == "high" else "X-ray"
                    return f"High risk ({pct:.1f}%): {dom} analysis shows strong PCOS indicators"
                return f"High risk ({pct:.1f}%): Combined analysis indicates elevated PCOS probability"
            if risk_level == "moderate":
                return f"Moderate risk ({pct:.1f}%): Mixed indicators across facial and X-ray modalities"
            return f"Low risk ({pct:.1f}%): Both modalities show minimal PCOS indicators"

        mod_name = "facial analysis" if "face" in modalities_used else "X-ray analysis"
        if risk_level == "high":
            return f"High risk ({pct:.1f}%): {mod_name} shows strong PCOS indicators"
        if risk_level == "moderate":
            return f"Moderate risk ({pct:.1f}%): {mod_name} shows some PCOS indicators"
        return f"Low risk ({pct:.1f}%): {mod_name} shows minimal PCOS indicators"

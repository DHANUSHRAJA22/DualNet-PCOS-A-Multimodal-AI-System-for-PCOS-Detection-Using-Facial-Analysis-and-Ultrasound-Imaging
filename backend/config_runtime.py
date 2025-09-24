"""
Runtime configuration for production robustness

Handles gender mapping, ensemble outlier filtering, and X-ray model loading fallbacks
without requiring code changes - all configurable via environment variables.
"""

import os

# Gender mapping & thresholds
GENDER_LABELS_FILE = os.getenv("GENDER_LABELS_FILE", "backend/models/face/gender.labels.txt")
# Explicitly state the model's output index for "male" (0 or 1). If None, auto-calibrate.
GENDER_MALE_INDEX = os.getenv("GENDER_MALE_INDEX")
GENDER_CONF_THRESHOLD = float(os.getenv("GENDER_CONF_THRESHOLD", "0.60"))  # gating threshold
GENDER_AUTOCALIBRATE = os.getenv("GENDER_AUTOCALIBRATE", "true").lower() == "true"
# Persisted calibration file
GENDER_CALIBRATION_CACHE = os.getenv("GENDER_CALIBRATION_CACHE", "backend/models/face/.gender_calibration.json")

# Ensemble robustness
ENSEMBLE_TRIM_RATIO = float(os.getenv("ENSEMBLE_TRIM_RATIO", "0.10"))  # trim lowest/highest 10%
ENSEMBLE_MAD_K = float(os.getenv("ENSEMBLE_MAD_K", "3.0"))             # MAD cutoff for outliers
ENSEMBLE_MODEL_DENYLIST = set([m.strip() for m in os.getenv("ENSEMBLE_MODEL_DENYLIST", "").split(",") if m.strip()])
ENSEMBLE_MODEL_ALLOWLIST = set([m.strip() for m in os.getenv("ENSEMBLE_MODEL_ALLOWLIST", "").split(",") if m.strip()])

# X-ray loader tweaks
XRAY_COMPILE = os.getenv("XRAY_COMPILE", "false").lower() == "true"
XRAY_ALLOW_FALLBACK = os.getenv("XRAY_ALLOW_FALLBACK", "true").lower() == "true"
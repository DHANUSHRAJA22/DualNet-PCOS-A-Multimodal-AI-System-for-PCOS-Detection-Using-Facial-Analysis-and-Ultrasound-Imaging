# backend/managers/face_manager.py
"""
Face analysis manager for gender detection and PCOS classification.

- Robust model loading & validation
- Detect/avoid double-preprocessing (Lambda/Rescaling/Normalization)
- Correct per-backbone preprocessing
- Stable ensemble payload: {'method','score','models_used','weights'}
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

from fastapi import UploadFile

from config import (  # noqa: E402
    FACE_MODELS_DIR,
    UPLOADS_DIR,
    get_available_face_models,
    get_ensemble_weights,
    get_risk_level,
    load_model_labels,
    normalize_weights,
    settings,
)
from utils.validators import get_safe_filename, validate_image  # noqa: E402

logger = logging.getLogger(__name__)

UNRELIABLE_MODELS = {
    "pcos_vgg16",
    "vgg16_weights_tf_dim_ordering_tf_kernels",
    "resnet50_weights_tf_dim_ordering_tf_kernels",
    "pcos_resnet50",
}

# ---------------- TF helpers ----------------


@tf.function(reduce_retracing=True)
def _forward(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    return model(x, training=False)


def _load_model(path: str) -> Optional[tf.keras.Model]:
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        logger.error(f"[Face] Load failed for {path}: {e}")
        return None


def _sanity_predict(model: tf.keras.Model, path: str) -> bool:
    try:
        shp = model.input_shape
        H, W, C = (int(shp[1]), int(shp[2]), int(shp[3])) if shp and len(shp) >= 4 else (224, 224, 3)
        dummy = np.random.random((1, H, W, C)).astype("float32")
        _ = model.predict(dummy, verbose=0)
        return True
    except Exception as e:
        logger.error(f"[Face] Validation failed for {path}: {e}")
        return False


# ---------------- Gender utils ----------------


def _read_gender_labels(p: Path) -> List[str]:
    try:
        if p.exists():
            s = p.read_text(encoding="utf-8").strip()
            if s.startswith("["):
                arr = json.loads(s)
                if isinstance(arr, list) and len(arr) >= 2:
                    return [str(x).strip().lower() for x in arr]
            lines = [ln.strip().lower() for ln in s.splitlines() if ln.strip()]
            if len(lines) >= 2:
                return lines
    except Exception as e:
        logger.warning(f"[Face] gender labels read error: {e}")
    return ["female", "male"]


def _gender_map(labels: List[str]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for i, lab in enumerate(labels):
        if "female" in lab:
            m["female"] = i
        if "male" in lab:
            m["male"] = i
    if "female" not in m:
        m["female"] = 0
    if "male" not in m:
        m["male"] = 1 if m["female"] == 0 else 0
    return m


# ---------------- Preprocess detection ----------------


def _first_layers(model: tf.keras.Model, k: int = 3):
    try:
        return model.layers[: min(k, len(model.layers))]
    except Exception:
        return []


def _has_built_in_preproc(model: tf.keras.Model) -> bool:
    """Heuristic: if first 3 layers include Lambda/Rescaling/Normalization or names hint at preprocessing."""
    for lyr in _first_layers(model, 3):
        name = (getattr(lyr, "name", "") or "").lower()
        cls = lyr.__class__.__name__.lower()
        if cls in {"lambda", "rescaling", "normalization"}:
            return True
        if any(tok in name for tok in ("preprocess", "rescale", "norm")):
            return True
    return False


def _pick_app_preproc(model_name: str) -> Callable[[np.ndarray], np.ndarray]:
    n = model_name.lower()
    try:
        if "efficientnet-b" in n:
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

            return eff_pre  # 0..255 -> [-1,1]
        if "resnet50" in n:
            from tensorflow.keras.applications.resnet50 import preprocess_input as rn_pre

            return rn_pre  # RGB->BGR + mean sub
        if "vgg16" in n:
            from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre

            return vgg_pre  # RGB->BGR + mean sub
    except Exception:
        pass
    return lambda a: a / 255.0  # fallback


# ---------------- Manager ----------------


class FaceManager:
    def __init__(self) -> None:
        self.gender_model: Optional[tf.keras.Model] = None
        self.gender_labels: List[str] = []
        self.gender_mapping: Dict[str, int] = {}

        # name -> dict(meta)
        self.pcos_models: Dict[str, Dict[str, Any]] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.loading_warnings: List[str] = []

        logger.info("[Face] Loading gender + PCOS models...")
        self._load_gender()
        self._load_pcos()
        logger.info(f"[Face] Ready. gender={self.gender_model is not None}, pcos_models={len(self.pcos_models)}")

    # ---- health helpers

    def can_lazy_load_gender(self) -> bool:
        return (FACE_MODELS_DIR / settings.GENDER_MODEL).exists()

    def can_lazy_load_pcos(self) -> bool:
        avail = get_available_face_models()
        return any(k not in UNRELIABLE_MODELS for k in avail)

    def get_model_status(self) -> Dict[str, Any]:
        out = {
            "gender": {"loaded": self.gender_model is not None, "labels": self.gender_labels},
            "pcos_models": {},
        }
        for n, d in self.pcos_models.items():
            out["pcos_models"][n] = {
                "loaded": True,
                "path": d["path"],
                "weight": d["weight"],
                "input_shape": d.get("input_hw"),
                "pos_idx": d.get("pos_idx"),
                "preproc": d.get("preproc_name"),
            }
        return out

    def get_loading_warnings(self) -> List[str]:
        return list(self.loading_warnings)

    # ---- warmup

    async def warmup(self) -> None:
        try:
            if self.gender_model is not None:
                self.gender_model.predict(np.zeros((1, 224, 224, 3), "float32"), verbose=0)
            for d in self.pcos_models.values():
                H, W = d.get("input_hw", (224, 224))
                d["model"].predict(np.zeros((1, H, W, 3), "float32"), verbose=0)
        except Exception:
            pass

    # ---- loading

    def _load_gender(self) -> None:
        p = FACE_MODELS_DIR / settings.GENDER_MODEL
        if not p.exists():
            self.loading_warnings.append(f"Gender model not found: {p.name}")
            return
        m = _load_model(str(p))
        if not m or not _sanity_predict(m, str(p)):
            self.loading_warnings.append(f"Gender model failed: {p.name}")
            return
        labels = _read_gender_labels(FACE_MODELS_DIR / "gender.labels.txt")
        self.gender_model = m
        self.gender_labels = labels
        self.gender_mapping = _gender_map(labels)
        logger.info(f"[Face] ✅ Gender loaded. labels={labels} mapping={self.gender_mapping}")

    def _load_pcos(self) -> None:
        avail = get_available_face_models()
        logger.info(f"[Face] Found {len(avail)} candidate PCOS models")
        self.ensemble_weights = get_ensemble_weights("face") or {}

        for name, path in avail.items():
            if name in UNRELIABLE_MODELS:
                logger.info(f"[Face] Skipping unreliable model: {name}")
                continue

            m = _load_model(str(path))
            if not m or not _sanity_predict(m, str(path)):
                self.loading_warnings.append(f"PCOS model failed: {name}")
                continue

            shp = m.input_shape
            H, W = (int(shp[1]), int(shp[2])) if shp and len(shp) >= 4 else settings.FACE_IMAGE_SIZE

            labels = load_model_labels(path) or ["non_pcos", "pcos"]
            labels_l = [str(x).strip().lower() for x in labels]
            pos_idx = labels_l.index("pcos") if "pcos" in labels_l else 1

            # Detect built-in preprocessing to avoid double-preprocess
            if _has_built_in_preproc(m):
                preproc = lambda a: a  # passthrough
                preproc_name = "passthrough (built-in preproc detected)"
            else:
                preproc = _pick_app_preproc(name)
                preproc_name = getattr(preproc, "__name__", "custom_/255")

            weight = float(self.ensemble_weights.get(name, 1.0))
            self.pcos_models[name] = {
                "model": m,
                "path": str(path),
                "labels": labels,
                "pos_idx": int(pos_idx),
                "weight": weight,
                "input_hw": (H, W),
                "preproc": preproc,
                "preproc_name": preproc_name,
            }
            logger.info(
                f"[Face] ✅ Loaded {name} ({H}x{W}), labels={labels}, pos_idx={pos_idx}, "
                f"weight={weight}, preproc={preproc_name}"
            )

        if self.pcos_models:
            names = list(self.pcos_models.keys())
            norm = normalize_weights({n: self.pcos_models[n]["weight"] for n in names}, names)
            for n, w in norm.items():
                self.pcos_models[n]["weight"] = float(w)
            logger.info(f"[Face] ✅ Normalized ensemble weights: {norm}")

    # ---- preprocessing / IO

    @staticmethod
    def _input_hw(model: tf.keras.Model) -> Tuple[int, int]:
        try:
            shp = model.input_shape
            if shp and len(shp) >= 4:
                return int(shp[1]), int(shp[2])
        except Exception:
            pass
        return settings.FACE_IMAGE_SIZE

    @staticmethod
    def _load_rgb(image_bytes: bytes, size_hw: Tuple[int, int]) -> np.ndarray:
        H, W = size_hw
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype="float32")  # 0..255 float

    # ---- inference

    async def predict_gender(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.gender_model is None:
            return {"male": 0.0, "female": 1.0, "label": "female", "confidence": 1.0}

        arr = self._load_rgb(image_bytes, self._input_hw(self.gender_model)) / 255.0
        x = np.expand_dims(arr, 0)

        try:
            y = _forward(self.gender_model, tf.convert_to_tensor(x)).numpy()
        except Exception:
            y = self.gender_model.predict(x, verbose=0)

        probs = y[0] if y.ndim > 1 else y
        f_i = self.gender_mapping.get("female", 0)
        m_i = self.gender_mapping.get("male", 1)
        female = float(probs[f_i]) if f_i < len(probs) else 0.5
        male = float(probs[m_i]) if m_i < len(probs) else 0.5
        if female >= male:
            label, conf = "female", female
        else:
            label, conf = "male", male
        return {"male": male, "female": female, "label": label, "confidence": conf}

    async def process_face_image(self, file: UploadFile) -> Dict[str, Any]:
        image_bytes = await validate_image(file, max_mb=settings.MAX_UPLOAD_MB)

        # persist for UI
        fid = str(uuid.uuid4())[:8]
        safe = get_safe_filename(file.filename or "face.jpg")
        out_name = f"face-{fid}-{os.path.splitext(safe)[0]}.jpg"
        (UPLOADS_DIR / out_name).write_bytes(image_bytes)

        result: Dict[str, Any] = {
            "face_img": f"/static/uploads/{out_name}",
            "gender": None,
            "face_pred": None,
            "face_risk": "unknown",
            "score": 0.0,
            "per_model": {},
            "ensemble": None,
        }

        gender = await self.predict_gender(image_bytes)
        result["gender"] = gender

        if gender["label"] == "male" and float(gender.get("confidence", 0.0)) >= 0.75:
            result["face_pred"] = "Male face detected - PCOS analysis not applicable"
            result["face_risk"] = "not_applicable"
            result["ensemble"] = {"method": "weighted_mean", "score": 0.0, "models_used": 0, "weights": {}}
            return result

        if not self.pcos_models:
            result["face_pred"] = "No PCOS face models available"
            result["ensemble"] = {"method": "weighted_mean", "score": 0.0, "models_used": 0, "weights": {}}
            return result

        per_model: Dict[str, float] = {}
        total_w = 0.0
        weighted_sum = 0.0
        weights_out: Dict[str, float] = {}

        for name, d in self.pcos_models.items():
            try:
                H, W = d["input_hw"]
                arr = self._load_rgb(image_bytes, (H, W))  # 0..255
                preproc = d["preproc"]
                try:
                    arr_pp = preproc(arr)
                except Exception:
                    arr_pp = arr / 255.0
                x = np.expand_dims(arr_pp, 0)

                y = d["model"].predict(x, verbose=0)
                probs = y[0] if y.ndim > 1 else y
                pos = int(d.get("pos_idx", 1))
                if probs.ndim == 1 and len(probs) >= 2:
                    pcos_prob = float(probs[pos])
                else:
                    pcos_prob = float(np.squeeze(probs).astype(np.float32))
                per_model[name] = pcos_prob

                w = float(d["weight"])
                weighted_sum += pcos_prob * w
                total_w += w
                weights_out[name] = w
            except Exception as e:
                logger.error(f"[Face] PCOS model {name} failed: {e}")

        result["per_model"] = per_model

        if total_w > 0 and per_model:
            score = float(weighted_sum / total_w)
            result["score"] = score
            result["face_risk"] = get_risk_level(score)
            result["face_pred"] = (
                "High PCOS risk detected"
                if result["face_risk"] == "high"
                else "Moderate PCOS indicators detected"
                if result["face_risk"] == "moderate"
                else "Low PCOS risk detected"
            )
            result["ensemble"] = {
                "method": "weighted_mean",
                "score": score,
                "models_used": int(len(per_model)),
                "weights": weights_out,
            }
        else:
            result["face_pred"] = "Analysis unavailable - no models could score the image"
            result["ensemble"] = {"method": "weighted_mean", "score": 0.0, "models_used": 0, "weights": {}}

        return result

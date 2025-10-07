# backend/managers/xray_manager.py
"""
X-ray analysis manager for YOLO detection and PCOS classification.

- Validates every Keras model at its own input size (FIX: no PIL decode for dummy)
- Strong H5 compatibility loader (JSON repair, by_name weights)
- YOLO preview explicitly rendered and saved
- Per-model preprocessing (ResNet/VGG use Caffe-style)
- Robust ensemble (drops outliers) + YOLO→probability + fused ensemble
- Health-check helpers and warmup()
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402
from fastapi import UploadFile  # noqa: E402
from keras.models import model_from_json  # noqa: E402
from tensorflow.keras.applications import (  # noqa: E402
    resnet50 as resnet50_app,
    vgg16 as vgg16_app,
)

from config import (  # type: ignore
    UPLOADS_DIR,
    XRAY_MODELS_DIR,
    YOLO_MODELS_DIR,
    get_available_xray_models,
    get_ensemble_weights,
    get_risk_level,
    load_model_labels,
    normalize_weights,
    settings,
)
from ensemble import robust_weighted_ensemble  # type: ignore
from utils.validators import get_safe_filename, validate_image  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------
# Low-level helpers
# -----------------------

def _model_input_hw(model: tf.keras.Model, default_hw: Tuple[int, int] = (224, 224)) -> Tuple[int, int]:
    try:
        shp = getattr(model, "input_shape", None)
        if shp is None:
            return default_hw
        if len(shp) >= 4:
            return int(shp[1]), int(shp[2])
        if len(shp) == 3:
            return int(shp[0]), int(shp[1])
    except Exception:
        pass
    return default_hw


def _ensure_probs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 0:
        p = 1.0 / (1.0 + np.exp(-float(y)))
        return np.array([1.0 - p, p], dtype="float32")
    if y.ndim >= 2:
        y = y[0]
    if y.size == 1:
        v = float(y[0]) if hasattr(y, "__getitem__") else float(y)
        p = 1.0 / (1.0 + np.exp(-v))
        return np.array([1.0 - p, p], dtype="float32")
    s = float(np.sum(y))
    if s <= 0.0 or s > 1.0001 or np.any(y < 0.0) or np.any(y > 1.0):
        e = np.exp(y - np.max(y))
        y = e / np.sum(e)
    return y.astype("float32")


@tf.function(reduce_retracing=True)
def _compiled_forward(model, x):
    return model(x, training=False)


def _choose_preprocess(model_name: str):
    """Pick the correct preprocessing for the backbone."""
    n = model_name.lower()
    if "resnet" in n:
        return resnet50_app.preprocess_input  # expects 0..255 RGB array
    if "vgg" in n or "detector" in n:
        return vgg16_app.preprocess_input     # expects 0..255 RGB array

    # Fallback: [-1, 1] like EfficientNet if you add other backbones later
    def _fallback(x: np.ndarray) -> np.ndarray:
        return (x / 127.5) - 1.0
    return _fallback


def _image_to_array(image_bytes: bytes, hw: Tuple[int, int]) -> np.ndarray:
    """Return a NHWC float32 array in the 0..255 range (no normalization)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((hw[1], hw[0]), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype="float32")  # 0..255
    return np.expand_dims(arr, 0)


def _try_load_full_model(path: str) -> Optional[tf.keras.Model]:
    try:
        return tf.keras.models.load_model(path, compile=False)
    except TypeError as e:
        if "Unrecognized keyword arguments" in str(e) or "batch_shape" in str(e):
            logger.warning(f"Keras version mismatch for {path}; will attempt fallback")
            return None
        raise
    except Exception as e:
        logger.warning(f"Full model load failed for {path}: {e}")
        return None


def _repair_and_load_from_h5(path: str) -> Optional[tf.keras.Model]:
    """Repair JSON config inside H5 and rebuild model."""
    try:
        with h5py.File(path, "r") as f:
            if "model_config" not in f.attrs:
                return None
            cfg = f.attrs["model_config"]
            if isinstance(cfg, (bytes, bytearray)):
                cfg = cfg.decode("utf-8")
            d = json.loads(cfg)

        def strip_batch_shape(obj):
            if isinstance(obj, dict):
                if obj.get("class_name") == "InputLayer":
                    c = obj.get("config", {})
                    if "batch_shape" in c:
                        bs = c["batch_shape"]
                        if bs and len(bs) > 1:
                            c["input_shape"] = bs[1:]
                        c.pop("batch_shape", None)
                for _, v in obj.items():
                    strip_batch_shape(v)
            elif isinstance(obj, list):
                for v in obj:
                    strip_batch_shape(v)

        strip_batch_shape(d)
        m = model_from_json(json.dumps(d))
        m.load_weights(path, by_name=True, skip_mismatch=True)
        logger.info(f"[xray] Loaded with repaired JSON config: {path}")
        return m
    except Exception as e:
        logger.warning(f"[xray] JSON repair load failed for {path}: {e}")
        return None


def _rebuild_arch_from_name(name: str, input_hw: Tuple[int, int]) -> tf.keras.Model:
    H, W = input_hw
    inp = (H, W, 3)
    lname = name.lower()
    if "resnet" in lname:
        base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        out = tf.keras.layers.Dense(2, activation="softmax")(x)
        return tf.keras.Model(base.input, out)
    if "vgg" in lname or "detector" in lname:
        base = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=inp)
        x = tf.keras.layers.Flatten()(base.output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        out = tf.keras.layers.Dense(2, activation="softmax")(x)
        return tf.keras.Model(base.input, out)
    # fallback
    base = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=inp)
    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(base.input, out)


def _safe_load_h5(path: Path) -> Optional[tf.keras.Model]:
    m = _try_load_full_model(str(path))
    if m:
        logger.info(f"[xray] Loaded full model: {path}")
        return m
    m = _repair_and_load_from_h5(str(path))
    if m:
        return m
    try:
        input_hw = tuple(settings.XRAY_IMAGE_SIZE)
        rebuilt = _rebuild_arch_from_name(path.stem, input_hw)
        rebuilt.load_weights(str(path), by_name=True, skip_mismatch=True)
        logger.info(f"[xray] Rebuilt arch + loaded weights: {path} (input_hw={input_hw})")
        return rebuilt
    except Exception as e:
        logger.error(f"[xray] Could not rebuild+load weights for {path}: {e}")
        return None


def _validate_model(m: tf.keras.Model, name: str) -> bool:
    """
    Validate by feeding a constant dummy image WITHOUT going through PIL.
    The previous version tried to open raw bytes with PIL (no PNG/JPEG header),
    which caused: 'cannot identify image file <_io.BytesIO ...>'.
    """
    try:
        H, W = _model_input_hw(m, default_hw=tuple(settings.XRAY_IMAGE_SIZE))
        # Constant mid-gray in the expected 0..255 range
        x = np.full((1, H, W, 3), 127.0, dtype="float32")
        prep = _choose_preprocess(name)
        x = prep(x.copy())
        y = m.predict(x, verbose=0)
        y = _ensure_probs(y)
        return y.ndim == 1 and y.size >= 2 and np.all(np.isfinite(y))
    except Exception as e:
        logger.error(f"X-ray model validation failed for {name}: {e}")
        return False


# -----------------------
# XrayManager
# -----------------------

class XrayManager:
    def __init__(self) -> None:
        self.yolo_model = None
        self.yolo_weights_path = Path(YOLO_MODELS_DIR) / settings.YOLO_MODEL  # e.g., "bestv8.pt"
        self.pcos_models: Dict[str, Dict[str, Any]] = {}
        self.models_unavailable: List[Dict[str, Any]] = []
        self.loading_warnings: List[str] = []
        self.ensemble_weights: Dict[str, float] = {}
        self.model_status: Dict[str, Dict[str, Any]] = {
            "yolo": {"loaded": False, "available": False, "error": None},
            "xray": {"loaded": False, "available": False, "error": None},
        }
        self._load_all()

    # --- health + warmup -----------------------------------------------------

    def can_lazy_load_yolo(self) -> bool:
        return self.yolo_weights_path.exists()

    def can_lazy_load_pcos(self) -> bool:
        try:
            if any(Path(XRAY_MODELS_DIR).glob("*.h5")):
                return True
            return bool(get_available_xray_models())
        except Exception:
            return False

    async def warmup(self):
        # YOLO warmup
        if self.yolo_model is not None:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                _ = self.yolo_model(dummy)
            except Exception:
                pass
        # Keras warmup
        for d in self.pcos_models.values():
            H, W = d.get("input_hw", (224, 224))
            x = np.zeros((1, H, W, 3), dtype="float32")
            try:
                _ = d["model"].predict(x, verbose=0)
            except Exception:
                pass

    # --- loading -------------------------------------------------------------

    def _load_all(self) -> None:
        logger.info("Loading X-ray analysis models...")
        self._load_yolo()
        self._load_xray_pcos()
        logger.info(
            "X-ray manager initialized. YOLO detection: %s, PCOS models loaded: %d",
            bool(self.yolo_model),
            len(self.pcos_models),
        )

    def _load_yolo(self) -> None:
        self.model_status["yolo"]["available"] = self.yolo_weights_path.exists()
        if not self.yolo_weights_path.exists():
            self.model_status["yolo"]["error"] = "YOLO model not found"
            self.loading_warnings.append(f"YOLO model not found: {self.yolo_weights_path}")
            logger.warning(f"YOLO model not found: {self.yolo_weights_path}")
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self.yolo_model = YOLO(str(self.yolo_weights_path))
            self.model_status["yolo"]["loaded"] = True
            logger.info(f"Loaded YOLO model: {self.yolo_weights_path}")
        except Exception as e:
            self.model_status["yolo"]["error"] = str(e)
            self.loading_warnings.append(f"YOLO load failed: {e}")
            logger.error(f"YOLO load failed: {e}")

    def _load_xray_pcos(self) -> None:
        available = get_available_xray_models()
        logger.info(f"Available X-ray PCOS models: {list(available.keys())}")
        if not available:
            self.model_status["xray"]["available"] = False
            self.loading_warnings.append("No X-ray PCOS models found")
            return

        self.ensemble_weights = get_ensemble_weights("xray")
        for name, path in available.items():
            try:
                m = _safe_load_h5(path)
                if not m or not _validate_model(m, f"xray:{name}"):
                    raise RuntimeError("load/validation failed")
                in_hw = _model_input_hw(m, default_hw=tuple(settings.XRAY_IMAGE_SIZE))
                self.pcos_models[name] = {
                    "model": m,
                    "path": str(path),
                    "weight": float(self.ensemble_weights.get(name, 1.0)),
                    "labels": load_model_labels(path),
                    "input_hw": in_hw,
                    "preprocess": _choose_preprocess(name),
                }
                logger.info("✅ Loaded X-ray model %s: input_hw=%s", name, in_hw)
            except Exception as e:
                logger.error("❌ X-ray model %s failed: %s", name, e)
                self.models_unavailable.append({"name": name, "path": str(path), "reason": str(e)})

        if self.pcos_models:
            cur = {n: self.pcos_models[n]["weight"] for n in self.pcos_models}
            norm = normalize_weights(cur, list(self.pcos_models.keys()))
            for n, w in norm.items():
                self.pcos_models[n]["weight"] = w
            logger.info("Normalized X-ray weights: %s", norm)

        self.model_status["xray"]["loaded"] = bool(self.pcos_models)
        self.model_status["xray"]["available"] = bool(available)

    # --- YOLO detection ------------------------------------------------------

    async def detect_objects(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.yolo_model:
            return {"detections": [], "found_labels": [], "yolo_vis": None}

        temp_id = str(uuid.uuid4())[:8]
        temp_path = UPLOADS_DIR / f"temp_yolo_{temp_id}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        try:
            results = self.yolo_model(str(temp_path))
            detections: List[Dict[str, Any]] = []
            found: List[str] = []
            yolo_vis = None

            if results and len(results) > 0:
                r0 = results[0]
                if hasattr(r0, "boxes") and r0.boxes is not None:
                    boxes = r0.boxes
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        label = r0.names.get(cls, f"class_{cls}") if hasattr(r0, "names") else f"class_{cls}"
                        detections.append({"box": xyxy.tolist(), "conf": conf, "label": label})
                        if label not in found:
                            found.append(label)

                # visualization
                try:
                    vis_img = r0.plot()   # BGR numpy
                    vis_img = vis_img[:, :, ::-1]  # to RGB
                    vis_path = UPLOADS_DIR / f"yolo_vis_{temp_id}.jpg"
                    Image.fromarray(vis_img).save(vis_path, format="JPEG", quality=90)
                    yolo_vis = f"/static/uploads/{vis_path.name}"
                except Exception:
                    yolo_vis = None

            return {"detections": detections, "found_labels": found, "yolo_vis": yolo_vis}
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # --- Convert YOLO detections → PCOS probability -------------------------

    def _yolo_pcos_probability(self, detections: List[Dict[str, Any]]) -> float:
        """
        Heuristic:
          - keep only classes in settings.YOLO_PCOS_CLASSES (e.g., ["cyst"])
          - score = sigmoid(alpha * sum(confidence * class_weight))
        """
        if not detections:
            return 0.0

        cls_weights = getattr(settings, "YOLO_CLASS_WEIGHTS", {}) or {}
        pcos_classes = set(getattr(settings, "YOLO_PCOS_CLASSES", ["cyst"]))
        alpha = float(getattr(settings, "YOLO_SIGMOID_ALPHA", 2.2))

        s = 0.0
        for d in detections:
            lbl = str(d.get("label", "")).lower()
            if any(k in lbl for k in pcos_classes):
                w = float(cls_weights.get(lbl, 1.0))
                s += float(d.get("conf", 0.0)) * w

        if s <= 0.0:
            return 0.0

        n = sum(1 for d in detections if any(k in str(d.get("label", "")).lower() for k in pcos_classes))
        s = s / max(1.0, n)

        prob = 1.0 / (1.0 + np.exp(-alpha * s))
        return float(max(0.0, min(1.0, prob)))

    # --- Keras ensemble ------------------------------------------------------

    async def predict_pcos_ensemble(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.pcos_models:
            return {
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "labels": ["non_pcos", "pcos"],
            }

        items: List[Dict[str, Any]] = []
        per_model: Dict[str, float] = {}

        for name, d in self.pcos_models.items():
            try:
                H, W = d.get("input_hw", tuple(settings.XRAY_IMAGE_SIZE))
                x_raw = _image_to_array(image_bytes, (H, W))   # 0..255
                x = d["preprocess"](x_raw.copy())              # Caffe/VGG style where needed
                try:
                    y = _compiled_forward(d["model"], x).numpy()
                except Exception:
                    y = d["model"].predict(x, verbose=0)
                p = _ensure_probs(y)
                pcos_prob = float(p[1]) if p.size >= 2 else float(p.squeeze())
                pcos_prob = max(0.0, min(1.0, pcos_prob))
                per_model[name] = pcos_prob
                items.append({"name": name, "score": pcos_prob, "weight": float(d["weight"])})
            except Exception as e:
                logger.error("X-ray PCOS inference failed for %s: %s", name, e)

        if not items:
            return {
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "labels": ["non_pcos", "pcos"],
            }

        final_score, kept, excluded = robust_weighted_ensemble(items, zscore_clip=2.5)
        used_weights = {k["name"]: k["used_weight"] for k in kept}

        return {
            "per_model": per_model,
            "ensemble_score": float(final_score),
            "ensemble": {
                "method": "robust_weighted_average",
                "score": float(final_score),
                "models_used": len(kept),
                "weights_used": used_weights,
                "excluded": excluded,
            },
            "labels": ["non_pcos", "pcos"],
        }

    # --- YOLO-only fallbacks -------------------------------------------------

    def _assess_from_yolo(self, labels: List[str]) -> str:
        if not labels:
            return "No significant structures detected in X-ray"
        pcos_keywords = ["cyst", "enlarged_ovary", "multiple_follicles", "polycystic"]
        found = [l for l in labels if any(k in l.lower() for k in pcos_keywords)]
        if found:
            return f"PCOS-related structures detected: {', '.join(found)}"
        return f"Anatomical structures detected: {', '.join(labels)}"

    def _yolo_risk(self, labels: List[str]) -> str:
        if not labels:
            return "unknown"
        pcos_keywords = ["cyst", "enlarged_ovary", "multiple_follicles", "polycystic"]
        n = sum(1 for l in labels if any(k in l.lower() for k in pcos_keywords))
        if n >= 2:
            return "high"
        if n == 1:
            return "moderate"
        return "low"

    # --- top-level handler ---------------------------------------------------

    async def process_xray_image(self, file: UploadFile) -> Dict[str, Any]:
        image_bytes = await validate_image(file, max_mb=settings.MAX_UPLOAD_MB)

        file_id = str(uuid.uuid4())[:8]
        safe = get_safe_filename(file.filename)
        stem, _ = os.path.splitext(safe)  # avoid double .jpg.jpg
        out_name = f"xray-{file_id}-{stem}.jpg"
        out_path = UPLOADS_DIR / out_name
        with open(out_path, "wb") as f:
            f.write(image_bytes)

        try:
            result: Dict[str, Any] = {
                "xray_img": f"/static/uploads/{out_name}",
                "detections": [],
                "found_labels": [],
                "yolo_vis": None,
                "xray_pred": None,
                "xray_risk": "unknown",
                "per_model": {},
                "ensemble_score": 0.0,
                "ensemble": {"method": "none", "score": 0.0, "models_used": 0},
                "models_unavailable": self.models_unavailable,
                "models_used": [],
            }

            # YOLO detection
            det = await self.detect_objects(image_bytes)
            result.update(det)

            # Keras ensemble
            keras_score = None
            if self.pcos_models:
                pcos = await self.predict_pcos_ensemble(image_bytes)
                result["per_model"] = pcos["per_model"]
                result["models_used"] = list(pcos["per_model"].keys())
                result["ensemble"] = pcos["ensemble"]
                keras_score = float(pcos["ensemble_score"])

            # YOLO probability
            yolo_prob = self._yolo_pcos_probability(result["detections"])
            if self.yolo_model:
                # surface YOLO as a pseudo-model for the per-model panel
                result["per_model"]["Yolo Bestv8"] = yolo_prob
                result["models_used"].append("Yolo Bestv8")

            # fuse
            if keras_score is not None and self.yolo_model:
                w = float(getattr(settings, "YOLO_WEIGHT", 0.30))
                fused = (1.0 - w) * keras_score + w * yolo_prob
                result["ensemble_score"] = fused
                risk = get_risk_level(fused)
            elif keras_score is not None:
                result["ensemble_score"] = keras_score
                risk = get_risk_level(keras_score)
            else:
                # YOLO-only
                risk = self._yolo_risk(result["found_labels"])
                result["ensemble_score"] = yolo_prob

            result["xray_risk"] = risk
            if risk == "high":
                result["xray_pred"] = "PCOS symptoms detected in X-ray"
            elif risk == "moderate":
                result["xray_pred"] = "Moderate PCOS indicators in X-ray"
            else:
                result["xray_pred"] = (
                    "No significant PCOS symptoms detected in X-ray"
                    if (keras_score is not None or yolo_prob < 0.5)
                    else self._assess_from_yolo(result["found_labels"])
                )

            return result
        except Exception as e:
            logger.error(f"X-ray processing failed: {e}")
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    # --- status helpers ------------------------------------------------------

    def get_loading_warnings(self) -> List[str]:
        return self.loading_warnings

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        status = dict(self.model_status)
        status["pcos_models"] = {}
        for n, d in self.pcos_models.items():
            status["pcos_models"][n] = {
                "loaded": True,
                "path": d["path"],
                "weight": d["weight"],
                "input_hw": d.get("input_hw"),
            }
        return status

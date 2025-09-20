"""
# PCOS Analyzer Backend API

Production-ready FastAPI backend for AI-powered PCOS screening using ensemble deep learning models.

## 🚀 Features

- **Multi-modal Analysis**: Facial recognition + X-ray morphological analysis
- **Ensemble Models**: EfficientNet, ResNet, VGG (face) + YOLOv8, ViT (X-ray)
- **Flexible Architecture**: Modular design ready for real model integration
- **Production Ready**: Comprehensive error handling, logging, and validation
- **Self-Documenting**: Pydantic models with OpenAPI/Swagger documentation
- **Containerization Ready**: Docker-friendly configuration

## 📋 Requirements

```bash
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pillow==10.1.0
numpy==1.24.3
pydantic==2.5.0
aiofiles==23.2.1

# TODO: Add when integrating real models
# tensorflow==2.13.0
# torch==2.0.1
# ultralytics==8.0.196
# opencv-python==4.8.1.78
# transformers==4.33.2
```

## 🏃‍♂️ Quick Start

### Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

### Production

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with gunicorn (recommended)
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "models": {
    "face_ensemble": {
      "loaded": true,
      "models": ["efficientnet_b0", "resnet50", "vgg16"],
      "count": 3
    },
    "xray_ensemble": {
      "loaded": true,
      "models": ["yolov8", "vision_transformer"],
      "count": 2
    },
    "ensemble_predictor": {
      "loaded": true,
      "method": "soft_voting",
      "weights": {"face": 0.6, "xray": 0.4}
    }
  }
}
```

### PCOS Prediction

```bash
POST /predict
```

**Request (multipart/form-data):**
- `face_img` (optional): Facial image file (JPEG/PNG/WebP, max 5MB)
- `xray_img` (optional): X-ray image file (JPEG/PNG/WebP, max 5MB)

**cURL Examples:**

```bash
# Face image only
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@face_photo.jpg"

# X-ray image only  
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "xray_img=@xray_scan.jpg"

# Both images (recommended)
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@face_photo.jpg" \
  -F "xray_img=@xray_scan.jpg"
```

**JavaScript Example:**

```javascript
const formData = new FormData();
formData.append('face_img', faceImageFile);
formData.append('xray_img', xrayImageFile);

const response = await fetch('http://127.0.0.1:5000/predict', {
  method: 'POST',
  body: formData
});

const results = await response.json();
console.log('Analysis results:', results);
```

**Response Format:**

```json
{
  "ok": true,
  "timestamp": "2024-01-15T10:30:45.123456",
  "analysis_id": "analysis_1705312245",
  
  // Face analysis results (if face_img provided)
  "face_pred": "No significant PCOS indicators in facial analysis",
  "face_scores": [0.75, 0.25],
  "face_confidence": 0.75,
  "face_models": {
    "efficientnet_b0": {
      "model_name": "efficientnet_b0_v1.0.0",
      "scores": [0.8, 0.2],
      "prediction": "No PCOS detected",
      "confidence": 0.8,
      "processing_time_ms": 45.2
    },
    "resnet50": {
      "model_name": "resnet50_v1.0.0",
      "scores": [0.7, 0.3],
      "prediction": "No PCOS detected", 
      "confidence": 0.7,
      "processing_time_ms": 52.1
    },
    "vgg16": {
      "model_name": "vgg16_v1.0.0",
      "scores": [0.75, 0.25],
      "prediction": "No PCOS detected",
      "confidence": 0.75,
      "processing_time_ms": 38.7
    }
  },
  "face_img": "/static/uploads/face_abc123_photo.jpg",
  "face_feature_importance": {
    "facial_regions": {
      "forehead": 0.65,
      "cheeks": 0.42,
      "chin": 0.38,
      "jawline": 0.71
    },
    "skin_features": {
      "texture": 0.58,
      "pigmentation": 0.33,
      "hair_growth": 0.29
    }
  },
  
  // X-ray analysis results (if xray_img provided)
  "xray_pred": "X-ray analysis shows normal ovarian morphology",
  "xray_scores": [0.82, 0.18],
  "xray_confidence": 0.82,
  "xray_models": {
    "yolov8": {
      "model_name": "yolov8_v8.0.0",
      "scores": [0.85, 0.15],
      "detected_objects": ["ovary_left", "ovary_right", "uterus"],
      "confidence": 0.89,
      "prediction": "Normal ovarian morphology",
      "processing_time_ms": 125.3,
      "model_type": "detection"
    },
    "vision_transformer": {
      "model_name": "vision_transformer_v1.0.0",
      "scores": [0.79, 0.21],
      "confidence": 0.79,
      "prediction": "No PCOS patterns detected",
      "processing_time_ms": 89.7,
      "model_type": "classification"
    }
  },
  "yolo_vis": "/static/results/yolo_vis_xray_abc123.jpg",
  "found_labels": ["ovary_left", "ovary_right", "uterus"],
  "xray_img": "/static/uploads/xray_abc123_scan.jpg",
  "xray_feature_importance": {
    "anatomical_regions": {
      "left_ovary": 0.73,
      "right_ovary": 0.68,
      "uterus": 0.45,
      "pelvic_cavity": 0.52
    },
    "morphological_features": {
      "cyst_count": 0.34,
      "ovarian_volume": 0.67,
      "follicle_distribution": 0.41
    }
  },
  
  // Combined analysis (if both images provided)
  "combined": "Combined analysis indicates low PCOS risk (confidence: 68.4%). Both facial and imaging analysis show minimal concerning indicators. Regular monitoring recommended.",
  "combined_confidence": 0.684,
  "ensemble_weights": {
    "face": 0.6,
    "xray": 0.4
  },
  
  // Metadata
  "total_processing_time_ms": 342.8
}
```

## 🏗 Architecture

### Model Ensembles

**Face Analysis Pipeline:**
- **EfficientNetB0**: Lightweight, efficient feature extraction optimized for mobile deployment
- **ResNet50**: Deep residual learning for complex pattern recognition
- **VGG16**: Classical CNN architecture providing baseline comparison

**X-ray Analysis Pipeline:**
- **YOLOv8**: State-of-the-art object detection for anatomical structure identification
- **Vision Transformer (ViT)**: Attention-based analysis for morphological pattern recognition

### Ensemble Methods

- **Soft Voting**: Averages probability outputs from all models (default)
- **Weighted Average**: Simple weighted combination of modality scores
- **Configurable Weights**: Face analysis (60%) + X-ray analysis (40%) by default

### Risk Classification

- **Low Risk**: 0-30% PCOS probability
- **Moderate Risk**: 30-70% PCOS probability  
- **High Risk**: 70-100% PCOS probability

## 📁 Project Structure

```
backend/
├── app.py                    # FastAPI application with endpoints
├── config.py                # Configuration and settings
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   ├── ensemble.py          # Ensemble prediction logic
│   ├── face_models.py       # Facial analysis models
│   └── xray_models.py       # X-ray analysis models
├── utils/
│   ├── __init__.py
│   ├── image_processing.py  # Image preprocessing utilities
│   └── validators.py        # Input validation functions
└── static/
    ├── uploads/             # Uploaded and processed images
    └── results/             # Generated visualizations and outputs
```

## 🔧 Adding Real Models

### 1. Face Models Integration

```python
# In models/face_models.py - BaseFaceModel.load_model()

import tensorflow as tf

async def load_model(self) -> None:
    if self.model_name == 'efficientnet_b0':
        self.model = tf.keras.models.load_model('models/weights/efficientnet_b0_face.h5')
    elif self.model_name == 'resnet50':
        self.model = tf.keras.models.load_model('models/weights/resnet50_face.h5')
    elif self.model_name == 'vgg16':
        self.model = tf.keras.models.load_model('models/weights/vgg16_face.h5')
    
    self.loaded = True

async def predict(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
    # Real inference
    prediction = self.model.predict(processed_image['array'])
    scores = prediction[0].tolist()
    
    return {
        "model_name": f"{self.model_name}_v{self.version}",
        "scores": scores,
        "prediction": "PCOS detected" if scores[1] > 0.5 else "No PCOS detected",
        "confidence": float(max(scores)),
        "processing_time_ms": processing_time
    }
```

### 2. X-ray Models Integration

```python
# In models/xray_models.py - BaseXrayModel.load_model()

from ultralytics import YOLO
import tensorflow as tf

async def load_model(self) -> None:
    if self.model_name == 'yolov8':
        self.model = YOLO('models/weights/yolov8_pcos.pt')
    elif self.model_name == 'vision_transformer':
        self.model = tf.keras.models.load_model('models/weights/vit_xray.h5')
    
    self.loaded = True

# For YOLO detection
async def _predict_detection(self, processed_image: Dict[str, Any], start_time: datetime):
    results = self.model(processed_image['yolo_path'])
    detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
    confidence_scores = results[0].boxes.conf.tolist()
    
    # Generate visualization
    visualization_path = await self._save_yolo_visualization(results, processed_image)
    
    return {
        "detected_objects": detected_objects,
        "visualization_path": visualization_path,
        # ... rest of response
    }
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and results
RUN mkdir -p static/uploads static/results models/weights

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  pcos-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PCOS_ENV=production
      - PCOS_API_HOST=0.0.0.0
      - PCOS_API_PORT=5000
    volumes:
      - ./models/weights:/app/models/weights:ro
      - ./static:/app/static
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
export PCOS_API_HOST=0.0.0.0
export PCOS_API_PORT=5000
export PCOS_ENV=production

# File Upload Limits
export PCOS_MAX_FILE_SIZE_MB=5

# Ensemble Configuration
export PCOS_ENSEMBLE_METHOD=soft_voting
export PCOS_FACE_WEIGHT=0.6
export PCOS_XRAY_WEIGHT=0.4

# Directory Paths
export PCOS_UPLOAD_DIR=/app/static/uploads
export PCOS_RESULTS_DIR=/app/static/results
export PCOS_MODEL_DIR=/app/models/weights
```

## 🧪 Testing

### API Testing

```bash
# Health check
curl http://127.0.0.1:5000/health

# Test prediction with sample images
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_images/sample_face.jpg" \
  -F "xray_img=@test_images/sample_xray.jpg"

# Test with only face image
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_images/sample_face.jpg"
```

### Model Integration Testing

```python
# Test individual model loading
from models.face_models import FaceModelEnsemble
from models.xray_models import XrayModelEnsemble

face_ensemble = FaceModelEnsemble()
await face_ensemble.load_models()
assert face_ensemble.is_loaded()

xray_ensemble = XrayModelEnsemble()
await xray_ensemble.load_models()
assert xray_ensemble.is_loaded()
```

## 🔒 Security & Production

### File Upload Security
- **File type validation**: JPEG, PNG, WebP only
- **Size limits**: 5MB per file (configurable)
- **Content type checking**: MIME type validation
- **Safe filename generation**: Prevents path traversal attacks

### Error Handling
- **Structured logging**: JSON logs in production
- **Global exception handler**: Catches unhandled errors
- **Model failure resilience**: Continues with available models
- **Detailed error messages**: For debugging and monitoring

### Performance
- **Async processing**: Non-blocking file operations
- **Efficient image processing**: PIL with optimized resampling
- **Memory management**: Proper cleanup of temporary files
- **Processing time tracking**: Performance monitoring

## 🚀 Model Integration Checklist

When adding your real models:

### Face Models
- [ ] Replace `# TODO` in `BaseFaceModel.load_model()`
- [ ] Add proper preprocessing in `ImageProcessor.process_face_image()`
- [ ] Implement attention map generation for feature importance
- [ ] Add model-specific normalization (ImageNet standards)
- [ ] Configure proper input shapes and batch processing

### X-ray Models  
- [ ] Replace `# TODO` in `BaseXrayModel.load_model()`
- [ ] Implement YOLO visualization generation
- [ ] Add ViT preprocessing and attention extraction
- [ ] Configure object detection classes and confidence thresholds
- [ ] Implement morphological feature extraction

### Ensemble Logic
- [ ] Tune ensemble weights based on validation performance
- [ ] Add cross-validation for ensemble method selection
- [ ] Implement uncertainty quantification
- [ ] Add model performance monitoring and drift detection

## 📊 Monitoring & Logging

### Production Logging

```python
# Configure structured logging for production
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if os.getenv("PCOS_ENV") == "production" else "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## 📈 Performance Metrics

The API tracks and returns:
- **Individual model processing times**
- **Total ensemble processing time**
- **Memory usage per prediction**
- **Model confidence scores**
- **Feature importance weights**

## 🔬 Research & Development

This backend is designed for:
- **Medical research**: Comprehensive logging and traceability
- **Model experimentation**: Easy model swapping and ensemble tuning
- **Clinical validation**: Structured outputs for medical evaluation
- **Scalability**: Ready for high-throughput deployment

## 📄 License

Educational and research use only. Not for medical diagnosis or treatment.

---

**Project by DHANUSH RAJA (21MIC0158)**
"""
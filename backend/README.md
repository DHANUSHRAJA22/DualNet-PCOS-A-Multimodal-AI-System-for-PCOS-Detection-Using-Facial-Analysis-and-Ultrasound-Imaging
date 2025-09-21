# PCOS Analyzer Backend

Production-ready FastAPI backend for AI-powered PCOS screening using ensemble deep learning models.

## 🚀 Features

- **Multi-modal Analysis**: Facial recognition + X-ray morphological analysis
- **Ensemble Models**: Multiple TensorFlow models with weighted averaging
- **Gender Gating**: Male faces skip PCOS analysis with clear messaging
- **YOLO Detection**: Object detection for X-ray ROI classification
- **Production Ready**: Comprehensive error handling, logging, and validation
- **CORS Support**: Configured for frontend integration
- **Static File Serving**: Automatic image serving and cleanup

## 📋 Requirements

```bash
# Core dependencies (see requirements.txt)
fastapi==0.110.0
uvicorn[standard]==0.27.1
pydantic==2.6.1
tensorflow==2.15.1
keras==2.15.0
numpy==1.26.4
pillow==10.4.0
ultralytics==8.3.30
python-multipart==0.0.9
aiofiles==23.2.1
httpx==0.27.2
```

## 🏃‍♂️ Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Your Model Files

Place your trained models in these exact locations:

```
backend/models/
├── face/
│   ├── gender_classifier.h5      # Gender detection model
│   ├── face_model_A.h5           # Face PCOS model 1 (VGG16)
│   └── face_model_B.h5           # Face PCOS model 2 (ResNet50)
├── xray/
│   ├── xray_model_A.h5           # X-ray PCOS model 1
│   └── xray_model_B.h5           # X-ray PCOS model 2
└── yolo/
    └── bestv8.pt                 # YOLO detection model
```

### 3. Start the Server

```bash
# Development
uvicorn app:app --reload --port 5000

# Production
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

### 4. Test the API

```bash
# Health check
curl http://127.0.0.1:5000/health

# Test prediction
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_face.jpg" \
  -F "xray_img=@test_xray.jpg"
```

## 📡 API Endpoints

### GET /health

Returns model availability status:

```json
{
  "status": "ok",
  "models": {
    "gender": true,
    "face": true,
    "yolo": true,
    "xray": true
  }
}
```

### POST /predict

Main prediction endpoint accepting multipart form data:

**Request:**
- `face_img` (optional): Face image file
- `xray_img` (optional): X-ray image file

**Response:**
```json
{
  "ok": true,
  "face_pred": "No significant PCOS indicators in facial analysis",
  "face_scores": [0.3, 0.25],
  "face_img": "/static/uploads/face-abc123.jpg",
  "face_risk": "low",
  "xray_pred": "PCOS symptoms detected in X-ray",
  "xray_img": "/static/uploads/xray-xyz789.jpg",
  "yolo_vis": "/static/uploads/yolo-def456.jpg",
  "found_labels": ["cyst", "ovary"],
  "xray_risk": "high",
  "combined": "Moderate risk: Mixed indicators across modalities.",
  "overall_risk": "moderate",
  "message": "ok"
}
```

### POST /predict-legacy

Identical to `/predict` - provided for backward compatibility.

### GET /img-proxy?url=...

Safe CORS proxy for external images from whitelisted hosts.

## 🔧 Configuration

### Environment Variables

```bash
# Server
export HOST=127.0.0.1
export PORT=5000
export DEBUG=true

# File handling
export MAX_UPLOAD_MB=5
export STATIC_TTL_SECONDS=3600

# CORS (comma-separated)
export ALLOWED_ORIGINS="http://localhost:5173,http://localhost:8080"
```

### Model Configuration

Edit `config.py` to adjust:
- Model file paths and names
- Input sizes for each model
- Ensemble weights
- Risk thresholds

## 🏗 Architecture

### Face Pipeline
1. **Gender Detection**: `gender_classifier.h5` → male/female classification
2. **Gender Gating**: Skip PCOS analysis for males
3. **PCOS Ensemble**: Multiple models → weighted average → risk classification

### X-ray Pipeline
1. **YOLO Detection**: `bestv8.pt` → find ROIs → generate overlay
2. **ROI Classification**: Crop detected regions → classify each ROI
3. **Fallback**: If no detections → classify full image
4. **Ensemble**: Weighted average across models → risk classification

### Final Fusion
- Combine face and X-ray scores
- Generate overall risk assessment
- Provide human-readable explanation

## 🛡 Security & Validation

- **File Type Validation**: JPEG/PNG/WebP only
- **Size Limits**: 5MB per file (configurable)
- **Safe Filenames**: Sanitized to prevent path traversal
- **CORS Protection**: Configurable allowed origins
- **Image Proxy**: Whitelisted hosts only

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p static/uploads models/face models/xray models/yolo

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
      - HOST=0.0.0.0
      - PORT=5000
      - DEBUG=false
    volumes:
      - ./models:/app/models:ro
      - ./static:/app/static
    restart: unless-stopped
```

## 🧪 Testing

### Manual Testing Scripts

```bash
# Test with face image only
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_images/female_face.jpg"

# Test with X-ray only
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "xray_img=@test_images/xray_sample.jpg"

# Test with both images
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_images/female_face.jpg" \
  -F "xray_img=@test_images/xray_sample.jpg"

# Test male face (should skip PCOS analysis)
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_images/male_face.jpg"
```

### Expected Behaviors

1. **Male Face**: `face_pred` contains skip message, `face_risk` = "unknown"
2. **Female Face**: Normal PCOS analysis with scores and risk level
3. **X-ray with Detections**: YOLO overlay saved, ROI classification performed
4. **X-ray without Detections**: Full image classification fallback
5. **Missing Models**: Graceful degradation, clear error messages
6. **Invalid Files**: 400 errors with descriptive messages

## 📊 Monitoring & Logging

### Log Levels
- **INFO**: Normal operations, model loading, predictions
- **WARNING**: Missing models, degraded performance
- **ERROR**: Processing failures, validation errors

### Metrics Tracked
- Processing times per modality
- Model availability and load status
- File upload statistics
- Error rates and types

## 🔍 Troubleshooting

### Common Issues

1. **Models Not Loading**
   - Check file paths in `config.py`
   - Verify TensorFlow/Keras versions
   - Check file permissions

2. **CORS Errors**
   - Verify `ALLOWED_ORIGINS` configuration
   - Check that frontend origin is included
   - Ensure proper headers in responses

3. **File Upload Failures**
   - Check file size limits
   - Verify MIME type validation
   - Ensure upload directory permissions

4. **YOLO Issues**
   - Verify Ultralytics installation
   - Check YOLO model file format
   - Ensure proper image preprocessing

### Debug Mode

Set `DEBUG=true` to enable:
- Detailed error messages in responses
- Verbose logging
- Development-friendly settings

## 📄 License

Educational and research use only. Not for medical diagnosis or treatment.

---

**Project by DHANUSH RAJA (21MIC0158)**
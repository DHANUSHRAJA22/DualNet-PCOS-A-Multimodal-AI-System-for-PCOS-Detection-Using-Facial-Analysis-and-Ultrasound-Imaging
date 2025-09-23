# Multimodal PCOS Analyzer - Full Stack Application

A comprehensive, production-ready full-stack application for PCOS screening analysis using AI-powered ensemble inference with automatic model discovery.

## 🚀 Ensemble Production Features

- **Automatic Model Discovery**: Discovers all `.h5` models in `models/face/` and `models/xray/` directories
- **Dynamic Ensemble Inference**: Configurable ensemble with automatic weight normalization
- **Per-Model Transparency**: Individual model scores and ensemble metadata exposed
- **ROI-Level Analysis**: YOLO detection → crop → classify → ensemble per region
- **Flexible Fusion Modes**: Threshold-based and discrete fusion strategies
- **Dynamic Label Loading**: Class labels loaded from corresponding `.labels.txt` files
- **Rich Debug Information**: Complete processing insights with model weights and ROI details
- **Production Configuration**: Environment-based settings for deployment flexibility
- **Backward Compatibility**: Legacy endpoint maintains existing client support

## 🚀 Features

### Frontend (React + Vite + TypeScript)
- **Dual Image Upload**: Support for both facial images and uterus X-rays
- **Camera Capture**: Built-in camera functionality for real-time image capture
- **Sample Images**: Pre-loaded demo images for testing
- **Responsive Design**: Mobile-first approach with accessibility compliance
- **Ensemble Results Display**: Per-model breakdowns and ROI analysis visualization
- **Smooth Animations**: Framer Motion powered interactions
- **Backend Status**: Real-time health monitoring with API configuration
- **Enhanced Results Display**: Ensemble metadata, model weights, and debug information

### Backend (FastAPI + TensorFlow + YOLO) - Ensemble Production Ready
- **Automatic Model Discovery**: Loads all available models from configured directories
- **Ensemble Inference**: Weighted averaging across multiple model architectures
- **Gender Gating**: Male faces skip PCOS analysis with clear messaging
- **ROI-Level Ensemble**: Per-region analysis with individual model contributions
- **Configurable Fusion**: Threshold and discrete fusion modes
- **Rich Metadata**: Complete ensemble details, weights, and processing insights
- **Environment Configuration**: All settings configurable via environment variables

## 🏃‍♂️ Quick Start (Production)

### Model Setup

Place your trained models in these directories:

```
backend/models/
├── face/
│   ├── gender_classifier.h5           # Gender detection model
│   ├── vgg16_pcos.h5                 # Face PCOS model (VGG16)
│   ├── vgg16_pcos.labels.txt         # Class labels for VGG16
│   ├── resnet50_pcos.h5              # Face PCOS model (ResNet50)
│   ├── resnet50_pcos.labels.txt      # Class labels for ResNet50
│   ├── efficientnetb0_pcos.h5        # Face PCOS model (EfficientNetB0)
│   └── efficientnetb0_pcos.labels.txt # Class labels for EfficientNetB0
├── xray/
│   ├── vgg16_xray.h5                 # X-ray PCOS model (VGG16)
│   ├── vgg16_xray.labels.txt         # Class labels for VGG16
│   ├── resnet50_xray.h5              # X-ray PCOS model (ResNet50)
│   ├── resnet50_xray.labels.txt      # Class labels for ResNet50
│   ├── efficientnetb0_xray.h5        # X-ray PCOS model (EfficientNetB0)
│   └── efficientnetb0_xray.labels.txt # Class labels for EfficientNetB0
└── yolo/
    └── bestv8.pt                     # YOLO detection model
```

### Label Files Format

Create `.labels.txt` files with class names (JSON array format):
```
["non_pcos", "pcos"]
```

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd pcos-analyzer

# Build and run backend with Docker
cd backend
docker build -t pcos-backend .
docker run -p 5000:5000 -v ./models:/app/models:ro pcos-backend

# Build frontend
cd ../
npm install
npm run build
# Serve dist/ with your preferred web server
```

### Environment Variables

```bash
# Backend (.env)
USE_ENSEMBLE=true
FUSION_MODE=threshold  # or "discrete"
ALLOWED_ORIGINS=http://localhost:5173,https://your-frontend.com
DEBUG=false
MAX_UPLOAD_MB=5
HOST=0.0.0.0
PORT=5000

# Ensemble weights (optional - will auto-normalize)
ENSEMBLE_WEIGHT_VGG16_PCOS=0.2
ENSEMBLE_WEIGHT_RESNET50_PCOS=0.2
ENSEMBLE_WEIGHT_EFFICIENTNETB0_PCOS=0.2
ENSEMBLE_WEIGHT_EFFICIENTNETB1_PCOS=0.2
ENSEMBLE_WEIGHT_EFFICIENTNETB2_PCOS=0.1
ENSEMBLE_WEIGHT_EFFICIENTNETB3_PCOS=0.1

# Frontend
VITE_API_BASE=https://your-backend.com
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Your trained model files (see Model Setup above)

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Add your model files (see Model Setup above)
# Models will be automatically discovered

# Start the backend server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

```bash
# Install dependencies (from root directory)
npm install

# Start the development server
npm run dev
```

### 3. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000 (proxied via Vite in dev)
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Quick Health Check

Verify everything is working:

```bash
# Test backend health
curl http://localhost:8000/health

# Should return JSON with status: "healthy" and model availability
```

If models fail to load due to Keras version mismatches, the system will automatically fall back to weights-only loading and continue operating with available models.
## 🔧 Configuration

### Backend Configuration

Edit `config.py` or use environment variables:
- `USE_ENSEMBLE`: Enable/disable ensemble inference
- `FUSION_MODE`: Choose fusion strategy ("threshold" or "discrete")
- Model weights: `ENSEMBLE_WEIGHT_{MODEL_NAME}`
- Risk thresholds and file upload limits

### Frontend Configuration

The frontend automatically detects the backend URL:
1. Query parameter: `?api=https://your-backend.com`
2. Environment variable: `VITE_API_BASE`
3. Default: Vite proxy to localhost:8000

## 📡 API Endpoints (Ensemble Enhanced)

### POST /predict (New Structured Format)
Enhanced prediction endpoint with ensemble metadata:

```json
{
  "ok": true,
  "modalities": [
    {
      "type": "face",
      "label": "PCOS indicators detected",
      "scores": [0.3, 0.7],
      "risk": "high",
      "per_model": {"vgg16_pcos": 0.65, "resnet50_pcos": 0.75, "efficientnetb0_pcos": 0.80},
      "ensemble": {
        "method": "weighted_average",
        "score": 0.73,
        "models_used": 3,
        "weights_used": {"vgg16_pcos": 0.33, "resnet50_pcos": 0.33, "efficientnetb0_pcos": 0.34}
      }
    },
    {
      "type": "xray",
      "per_roi": [
        {
          "roi_id": 0,
          "box": [100, 150, 200, 250],
          "per_model": {"vgg16_xray": 0.75, "resnet50_xray": 0.85},
          "ensemble": {"method": "weighted_average", "score": 0.80}
        }
      ],
      "detections": [{"box": [100,150,200,250], "conf": 0.85, "label": "cyst"}]
      }
    }
  ],
  "final": {
    "risk": "high",
    "confidence": 0.75,
    "explanation": "High risk: Both facial and X-ray analysis indicate PCOS symptoms",
    "fusion_mode": "threshold"
  },
  "warnings": [],
  "processing_time_ms": 1250.5,
  "debug": {
    "filenames": ["face.jpg", "xray.jpg"],
    "models_used": ["vgg16_pcos", "resnet50_pcos", "vgg16_xray"],
    "weights": {"face": {"vgg16_pcos": 0.33}, "xray": {"vgg16_xray": 0.5}},
    "roi_boxes": [{"roi_id": 0, "box": [100,150,200,250]}],
    "fusion_mode": "threshold",
    "use_ensemble": true
  }
}
```

### POST /predict-legacy (Backward Compatible)
Returns flat format for existing frontends:

```json
{
  "ok": true,
  "face_pred": "PCOS indicators detected",
  "face_scores": [0.3, 0.7],
  "face_risk": "high",
  "combined": "High risk: Multiple indicators detected",
  "overall_risk": "high"
}
```

### POST /predict-file
Single file upload with type parameter:
- `file`: Image file to analyze
- `type`: Analysis type ('face' or 'xray')

### GET /health
Returns enhanced model availability status with ensemble model details:

```json
{
  "status": "healthy",
  "models": {
    "gender": {
      "status": "loaded",
      "file_exists": true,
      "lazy_loadable": true
    },
    "face_vgg16_pcos": {
      "status": "loaded",
      "file_exists": true,
      "lazy_loadable": true,
      "path": "/app/models/face/vgg16_pcos.h5",
      "version": "weight_0.33"
    },
    "xray_resnet50_xray": {
      "status": "loaded",
      "file_exists": true,
      "lazy_loadable": true,
      "path": "/app/models/xray/resnet50_xray.h5",
      "version": "weight_0.50"
    }
  },
  "uptime_seconds": 1234.5,
  "version": "3.0.0"
}
```

### GET /img-proxy?url=...
Safe CORS proxy for external images

### Static Files
- `/static/uploads/`: Uploaded images and YOLO visualizations

## 🧪 Testing

### Automated Testing

```bash
# Run comprehensive test suite
cd backend && python test_api.py

# Run with pytest
cd backend && pytest test_api.py -v

# Run cURL examples
cd backend && ./curl_examples.sh
```

### Manual Testing

```bash
# Test health endpoint
curl http://127.0.0.1:5000/health

# Test structured response
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_face.jpg" | jq '.debug'

# Test legacy compatibility
curl -X POST "http://127.0.0.1:5000/predict-legacy" \
  -F "face_img=@test_face.jpg"
```

## 🏗 Architecture

### Ensemble Face Analysis Pipeline
1. **Gender Detection**: Classify male/female
2. **Gender Gating**: Skip PCOS analysis for males
3. **Model Discovery**: Load all available face models from directory
4. **Per-Model Analysis**: Run each discovered model individually
5. **Ensemble Fusion**: Weighted average with normalized weights → risk classification
6. **Metadata Collection**: Store per-model scores, weights, and ensemble details

### Ensemble X-ray Analysis Pipeline
1. **YOLO Detection**: Find ROIs → generate overlay
2. **Model Discovery**: Load all available X-ray models from directory
3. **Per-ROI Analysis**: Crop and classify each detected region with all models
4. **ROI-Level Ensemble**: Combine models for each ROI individually
5. **Global Ensemble**: Aggregate ROI ensembles for final prediction
6. **Fallback**: Full image ensemble classification if no detections
7. **Metadata Collection**: Store detections, ROI details, and ensemble info

### Risk Classification
- **Low Risk**: <0.33 probability
- **Moderate Risk**: 0.33-0.66 probability  
- **High Risk**: ≥0.66 probability
- **Threshold Fusion**: Probability-based risk bands
- **Discrete Fusion**: Rule-based multi-modal assessment

## 🐳 Production Deployment

### Docker Deployment

```bash
# Build and run backend
cd backend
docker build -t pcos-backend .
docker run -p 5000:5000 -v ./models:/app/models:ro pcos-backend

# Build frontend
npm run build
# Serve dist/ with your preferred web server
```

### Docker Compose

```yaml
version: '3.8'
services:
  pcos-api:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - HOST=0.0.0.0
      - PORT=5000
      - DEBUG=false
      - USE_ENSEMBLE=true
      - FUSION_MODE=threshold
      - MAX_UPLOAD_MB=5
      - ALLOWED_ORIGINS=https://your-frontend.com
    volumes:
      - ./backend/models:/app/models:ro
      - ./backend/static:/app/static
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```
### Environment Setup

```bash
# Production backend
export DEBUG=false
export HOST=0.0.0.0
export PORT=5000
export USE_ENSEMBLE=true
export FUSION_MODE=threshold
export ALLOWED_ORIGINS=https://your-frontend.com,https://your-domain.com
export MAX_UPLOAD_MB=5

# Production frontend
export VITE_API_BASE=https://your-backend.com
npm run build
```

### Heroku Deployment

```bash
# Deploy backend to Heroku
cd backend
heroku create your-pcos-backend
heroku config:set ALLOWED_ORIGINS=https://your-frontend.com
heroku config:set DEBUG=false
heroku config:set PORT=8000
git push heroku main

# Deploy frontend to Netlify/Vercel
npm run build
# Upload dist/ folder
```

## 🛡 Security & Privacy

- **File Validation**: Strict MIME type and size checking
- **Safe Filenames**: Sanitized to prevent path traversal
- **CORS Protection**: Configurable allowed origins
- **Image Proxy**: Whitelisted hosts only
- **No Permanent Storage**: Images cleaned up automatically
- **Medical Disclaimer**: Prominently displayed
- **Input Sanitization**: All user inputs validated and sanitized
- **Error Handling**: Secure error messages without sensitive information

## 📊 Monitoring

### Backend Monitoring
- Health endpoint for model status
- Structured logging with timestamps
- Processing time tracking
- Error rate monitoring
- Ensemble performance tracking
- Per-model contribution analysis
- ROI-level analysis metrics

### Frontend Monitoring
- Backend connectivity status
- Real-time health checks
- User-friendly error messages
- Ensemble performance metrics
- Rich result visualization with model breakdowns

## 🔍 Troubleshooting

### Common Issues

1. **Backend Status: Unreachable**
   - Check if backend is running on port 5000
   - Verify CORS configuration
   - For sandbox environments, use `?api=` parameter

2. **Models Not Loading**
   - Verify model files are in correct directories (`models/face/`, `models/xray/`)
   - Check file permissions
   - Review backend logs for loading errors
   - Test lazy loading capability via health endpoint
   - Ensure `.labels.txt` files exist alongside each model file

3. **File Upload Errors**
   - Check file size (5MB limit by default)
   - Verify supported formats (JPEG/PNG/WebP)
   - Ensure proper MIME type
   - Verify no 500 errors masquerading as CORS issues

4. **Ensemble Configuration**
   - Set `USE_ENSEMBLE=false` to use single best model
   - Adjust `FUSION_MODE` between "threshold" and "discrete"
   - Configure individual model weights via `ENSEMBLE_WEIGHT_*` environment variables
   - Weights are automatically normalized to sum to 1.0

5. **Label Loading Issues**
   - Ensure `.labels.txt` files exist alongside each model file
   - Check JSON format: `["non_pcos", "pcos"]`
   - Review backend logs for label loading errors
   - Each model can have different labels if needed

6. **Docker Issues**
   - Ensure models directory is mounted: `-v ./models:/app/models:ro`
   - Check environment variables are set correctly
   - Verify port mapping: `-p 5000:5000`
   - Ensure all model files and labels are present in mounted directory

## 👨‍💻 Author

**DHANUSH RAJA (21MIC0158)**

---

## 🚀 Ready to Deploy!

This full-stack application is production-ready. Simply:

1. **Add your trained models** to the `models/face/` and `models/xray/` directories
2. **Create corresponding `.labels.txt` files** for each model
3. **Start the backend** with `uvicorn app:app --port 5000`
4. **Start the frontend** with `npm run dev`
5. **Access the application** at http://localhost:5173

The application will automatically handle:
- Automatic model discovery and loading
- Ensemble predictions with configurable weights
- Per-model transparency and ROI-level analysis
- File validation and security checks
- Comprehensive error handling and logging
- Rich debugging capabilities with ensemble metadata

Perfect for production deployment with Docker, automatic model discovery, and comprehensive ensemble insights!
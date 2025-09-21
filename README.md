# Multimodal PCOS Analyzer - Full Stack Application

A comprehensive, production-ready full-stack application for PCOS screening analysis using AI-powered facial recognition and X-ray analysis.

## 🚀 Features

### Frontend (React + Vite + TypeScript)
- **Dual Image Upload**: Support for both facial images and uterus X-rays
- **Camera Capture**: Built-in camera functionality for real-time image capture
- **Sample Images**: Pre-loaded demo images for testing
- **Responsive Design**: Mobile-first approach with accessibility compliance
- **Real-time Analysis**: Integration with FastAPI backend for AI-powered screening
- **Smooth Animations**: Framer Motion powered interactions
- **Backend Status**: Real-time health monitoring with API configuration
- **Rich Results Display**: Per-model scores, ROI analysis, and ensemble details
- **Debug Information**: Development insights and processing metadata

### Backend (FastAPI + TensorFlow + YOLO) - Production Ready
- **Multi-modal Analysis**: Facial recognition + X-ray morphological analysis
- **Gender Gating**: Male faces skip PCOS analysis with clear messaging
- **Ensemble Models**: Multiple TensorFlow models with weighted averaging
- **YOLO Detection**: Object detection for X-ray ROI classification
- **Per-Model Scores**: Detailed breakdown of individual model predictions
- **ROI Processing**: Region-of-interest analysis with bounding box details
- **Structured Responses**: Rich API responses with comprehensive metadata
- **Production Ready**: Comprehensive error handling, logging, and validation
- **CORS Support**: Configured for frontend integration
- **Static File Serving**: Automatic image serving and cleanup
- **File Validation**: Size limits (5MB), MIME type checking, and security validation
- **JSON Serialization**: Proper NumPy type conversion for API responses
- **Dynamic Label Loading**: Class labels loaded from .labels.txt files
- **Risk Thresholds**: Configurable probability bands (<0.33/0.33-0.66/>0.66)
- **Fallback Classification**: Full image analysis when no ROIs detected
- **Legacy Compatibility**: /predict-legacy endpoint for existing clients

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Your trained model files (see Model Setup below)

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Add your model files (see Model Setup section)
# Place models in backend/models/ with exact filenames

# Start the backend server
uvicorn app:app --reload --port 5000
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
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/health

## 📁 Model Setup

Place your trained models in these exact locations:

```
backend/models/
├── face/
│   ├── gender_classifier.h5      # Gender detection model
│   ├── face_model_A.h5           # Face PCOS model 1 (VGG16)
│   ├── face_model_B.h5           # Face PCOS model 2 (ResNet50)
│   └── pcos_detector_158.labels.txt  # Face class labels
├── xray/
│   ├── xray_model_A.h5           # X-ray PCOS model 1
│   ├── xray_model_B.h5           # X-ray PCOS model 2
│   └── xray_classifier.labels.txt    # X-ray class labels
└── yolo/
    └── bestv8.pt                 # YOLO detection model
```

**Important**: Use these exact filenames. The backend is configured to load models from these specific paths.

### Label Files Format

Create `.labels.txt` files with class names:
```
# Option 1: JSON format
["non_pcos", "pcos"]

# Option 2: Plain text (one per line)
non_pcos
pcos
```
## 🔧 Configuration

### Backend Configuration

Edit `backend/config.py` to adjust:
- Model file paths and weights
- Risk thresholds
- File upload limits
- CORS origins

### Frontend Configuration

The frontend automatically detects the backend URL:
1. Query parameter: `?api=https://your-backend.com`
2. Environment variable: `VITE_API_BASE`
3. Default: Vite proxy to localhost:5000

### Environment Variables

```bash
# Backend
export HOST=127.0.0.1
export PORT=5000
export DEBUG=true
export MAX_UPLOAD_MB=5
export STATIC_TTL_SECONDS=3600
export ALLOWED_ORIGINS="http://localhost:5173,http://localhost:8080"

# Frontend
export VITE_API_BASE=http://localhost:5000
```

## 📡 API Endpoints

### GET /health
Returns enhanced model availability status with lazy-loading capability and detailed model information

```json
{
  "status": "healthy",
  "models": {
    "gender": {
      "status": "loaded",
      "file_exists": true,
      "lazy_loadable": true
    }
  },
  "uptime_seconds": 1234.5,
  "version": "2.0.0"
}
```

### POST /predict
Enhanced prediction endpoint with structured response:
- `face_img` (optional): Face image file
- `xray_img` (optional): X-ray image file

**Response Structure:**
```json
{
  "ok": true,
  "modalities": [
    {
      "type": "face",
      "label": "PCOS indicators detected",
      "scores": [0.3, 0.7],
      "risk": "high",
      "original_img": "/static/uploads/face-123.jpg",
      "gender": {"male": 0.1, "female": 0.9, "label": "female"},
      "per_model": {"vgg16": 0.65, "resnet50": 0.75},
      "ensemble": {
        "method": "weighted_average",
        "score": 0.7,
        "models_used": 2,
        "weights_used": {"vgg16": 0.5, "resnet50": 0.5}
      }
    },
    {
      "type": "xray",
      "label": "PCOS symptoms detected",
      "scores": [0.8],
      "risk": "high",
      "original_img": "/static/uploads/xray-456.jpg",
      "visualization": "/static/uploads/yolo-789.jpg",
      "found_labels": ["cyst", "ovary"],
      "detections": [
        {"box": [100, 150, 200, 250], "conf": 0.85, "label": "cyst"}
      ],
      "per_roi": [
        {
          "roi_id": 0,
          "box": [100, 150, 200, 250],
          "per_model": {"xray_a": 0.75, "xray_b": 0.85},
          "ensemble": {"method": "weighted_average", "score": 0.8, "models_used": 2}
        }
      ],
      "per_model": {"xray_a": 0.75, "xray_b": 0.85},
      "ensemble": {"method": "weighted_average", "score": 0.8, "models_used": 2}
    }
  ],
  "final": {
    "overall_risk": "high",
    "confidence": 0.75,
    "explanation": "High risk: Multiple indicators detected across modalities"
  },
  "warnings": [],
  "processing_time_ms": 1250.5,
  "debug": {
    "face_processing": {"filename": "face.jpg", "models_used": ["vgg16", "resnet50"]},
    "xray_processing": {"filename": "xray.jpg", "detections_count": 2, "roi_count": 1}
  }
}
```

### POST /predict-legacy
Backward compatibility endpoint returning flat structure for existing frontends

### POST /predict-file
Single file upload endpoint with type parameter:
- `file`: Image file to analyze
- `type`: Analysis type ('face' or 'xray')

Returns standard response format with structured data.
### GET /img-proxy?url=...
Safe CORS proxy for external images

### Static Files
- `/static/uploads/`: Uploaded images and YOLO visualizations

## 🏗 Architecture

### Face Analysis Pipeline
1. **Gender Detection**: Classify male/female
2. **Gender Gating**: Skip PCOS analysis for males
3. **Per-Model Analysis**: Run each face model individually
4. **Ensemble Fusion**: Weighted average → risk classification
5. **Metadata Collection**: Store per-model scores and ensemble details

### X-ray Analysis Pipeline
1. **YOLO Detection**: Find ROIs → generate overlay
2. **Per-ROI Analysis**: Crop and classify each detected region
3. **ROI Ensemble**: Combine models for each ROI
4. **Global Ensemble**: Average across all ROIs
5. **Fallback**: Full image classification if no detections
6. **Metadata Collection**: Store detections, ROI details, and ensemble info

### Risk Classification
- **Low Risk**: < 0.33 probability (configurable)
- **Moderate Risk**: 0.33-0.66 probability (configurable)
- **High Risk**: ≥ 0.66 probability (configurable)

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
  -F "face_img=@test_face.jpg" | jq '.modalities[0].per_model'

# Test with face image only
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_face.jpg"

# Test single file endpoint
curl -X POST "http://127.0.0.1:5000/predict-file?type=face" \
  -F "file=@test_face.jpg"
# Test with X-ray only
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "xray_img=@test_xray.jpg"

# Test with both images
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_face.jpg" \
  -F "xray_img=@test_xray.jpg"

# Test legacy compatibility
curl -X POST "http://127.0.0.1:5000/predict-legacy" \
  -F "face_img=@test_face.jpg"
```

### Frontend Testing

1. Visit http://localhost:5173
2. Check "Backend Status" shows "Online"
3. Upload face and/or X-ray images
4. Verify results display correctly
5. Test sample images functionality
6. Test camera capture (requires HTTPS in production)
7. Check per-model breakdowns in results
8. Verify ROI analysis for X-ray images

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
      - MAX_UPLOAD_MB=5
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
export ALLOWED_ORIGINS="https://your-frontend.com"
export MAX_UPLOAD_MB=5

# Production frontend
export VITE_API_BASE=https://your-backend.com
npm run build
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
- Per-model performance tracking
- ROI analysis metrics

### Frontend Monitoring
- Backend connectivity status
- Real-time health checks
- User-friendly error messages
- Performance metrics
- Rich result visualization

## 🔍 Troubleshooting

### Common Issues

1. **Backend Status: Unreachable**
   - Check if backend is running on port 5000
   - Verify CORS configuration
   - For sandbox environments, use `?api=` parameter

2. **Models Not Loading**
   - Verify model files are in correct locations
   - Check file permissions
   - Review backend logs for loading errors
    - Test lazy loading capability via health endpoint

3. **File Upload Errors**
   - Check file size (5MB limit by default)
   - Verify supported formats (JPEG/PNG/WebP)
   - Ensure proper MIME type
   - Verify no 500 errors masquerading as CORS issues

4. **File Upload Failures**
   - Check file size (5MB limit)
   - Verify supported formats (JPEG/PNG/WebP)
   - Ensure proper MIME type

## 📄 License

Educational and research use only. Not for medical diagnosis or treatment.



7. **Label Loading Issues**
   - Ensure .labels.txt files exist in model directories
   - Review backend logs for label loading errors
## 👨‍💻 Author

**DHANUSH RAJA (21MIC0158)**

---

## 🚀 Ready to Deploy!

This full-stack application is production-ready. Simply:

1. **Add your trained models** to the specified paths
2. **Create label files** for your model classes
3. **Start the backend** with `uvicorn app:app --port 5000`
4. **Start the frontend** with `npm run dev`
5. **Access the application** at http://localhost:5173

The application will automatically handle:
- Dynamic model and label loading
- Ensemble predictions with per-model breakdowns
- ROI processing with bounding box analysis
- File validation and security checks
- Comprehensive error handling and logging
- Rich debugging capabilities and performance monitoring

Perfect for production deployment with Docker, comprehensive testing, and full API documentation!
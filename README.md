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

### Backend (FastAPI + TensorFlow + YOLO)
- **Multi-modal Analysis**: Facial recognition + X-ray morphological analysis
- **Gender Gating**: Male faces skip PCOS analysis with clear messaging
- **Ensemble Models**: Multiple TensorFlow models with weighted averaging
- **YOLO Detection**: Object detection for X-ray ROI classification
- **Production Ready**: Comprehensive error handling, logging, and validation
- **CORS Support**: Configured for frontend integration
- **Static File Serving**: Automatic image serving and cleanup

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
│   └── face_model_B.h5           # Face PCOS model 2 (ResNet50)
├── xray/
│   ├── xray_model_A.h5           # X-ray PCOS model 1
│   └── xray_model_B.h5           # X-ray PCOS model 2
└── yolo/
    └── bestv8.pt                 # YOLO detection model
```

**Important**: Use these exact filenames. The backend is configured to load models from these specific paths.

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

# Frontend
export VITE_API_BASE=http://localhost:5000
```

## 📡 API Endpoints

### GET /health
Returns model availability status

### POST /predict
Main prediction endpoint accepting multipart form data:
- `face_img` (optional): Face image file
- `xray_img` (optional): X-ray image file

### POST /predict-legacy
Backward compatibility endpoint (same as /predict)

### GET /img-proxy?url=...
Safe CORS proxy for external images

### Static Files
- `/static/uploads/`: Uploaded images and YOLO visualizations

## 🏗 Architecture

### Face Analysis Pipeline
1. **Gender Detection**: Classify male/female
2. **Gender Gating**: Skip PCOS analysis for males
3. **PCOS Ensemble**: Multiple models → weighted average → risk classification

### X-ray Analysis Pipeline
1. **YOLO Detection**: Find ROIs → generate overlay
2. **ROI Classification**: Classify each detected region
3. **Fallback**: Full image classification if no detections
4. **Ensemble**: Weighted average → risk classification

### Risk Classification
- **Low Risk**: < 33% probability
- **Moderate Risk**: 33-66% probability  
- **High Risk**: ≥ 66% probability

## 🧪 Testing

### Manual Testing

```bash
# Test health endpoint
curl http://127.0.0.1:5000/health

# Test with face image only
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_face.jpg"

# Test with X-ray only
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "xray_img=@test_xray.jpg"

# Test with both images
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "face_img=@test_face.jpg" \
  -F "xray_img=@test_xray.jpg"
```

### Frontend Testing

1. Visit http://localhost:5173
2. Check "Backend Status" shows "Online"
3. Upload face and/or X-ray images
4. Verify results display correctly
5. Test sample images functionality
6. Test camera capture (requires HTTPS in production)

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

### Environment Setup

```bash
# Production backend
export DEBUG=false
export HOST=0.0.0.0
export PORT=5000
export ALLOWED_ORIGINS="https://your-frontend.com"

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

## 📊 Monitoring

### Backend Monitoring
- Health endpoint for model status
- Structured logging with timestamps
- Processing time tracking
- Error rate monitoring

### Frontend Monitoring
- Backend connectivity status
- Real-time health checks
- User-friendly error messages
- Performance metrics

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

3. **CORS Errors**
   - Ensure frontend origin is in `ALLOWED_ORIGINS`
   - Check that backend returns proper headers
   - Verify no 500 errors masquerading as CORS issues

4. **File Upload Failures**
   - Check file size (5MB limit)
   - Verify supported formats (JPEG/PNG/WebP)
   - Ensure proper MIME type

## 📄 License

Educational and research use only. Not for medical diagnosis or treatment.

## 👨‍💻 Author

**DHANUSH RAJA (21MIC0158)**

---

## 🚀 Ready to Deploy!

This full-stack application is production-ready. Simply:

1. **Add your trained models** to the specified paths
2. **Start the backend** with `uvicorn app:app --port 5000`
3. **Start the frontend** with `npm run dev`
4. **Access the application** at http://localhost:5173

The application will automatically handle model loading, ensemble predictions, file management, and provide a complete PCOS screening interface!
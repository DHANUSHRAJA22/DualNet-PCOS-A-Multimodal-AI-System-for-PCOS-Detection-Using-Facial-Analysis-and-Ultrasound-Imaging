# Changelog

## [Fixed] - React Router Error Boundary + Robust Keras Loaders

### Frontend Changes

#### Fixed
- **AppErrorBoundary.tsx**: Added proper React Router error boundary using `isRouteErrorResponse` instead of undefined `isStructuredResponse`
- **routes.tsx**: Wired error boundary to all routes with proper `errorElement` configuration
- **Results.tsx**: Added guards for missing results state and improved structured response handling
- **SampleImages.tsx**: Fixed sample image loading with proper success feedback
- **Index.tsx**: Switched to structured `/predict` endpoint instead of legacy format

#### Enhanced
- **vite.config.ts**: Updated proxy configuration for better localhost compatibility
- Error handling now shows friendly boundaries instead of "Unexpected Application Error"
- Navigation errors properly handled with status codes and user-friendly messages

### Backend Changes

#### Fixed
- **managers/face_manager.py**: Added robust model loading with weights-only fallback for Keras version mismatches
- **managers/xray_manager.py**: Implemented comprehensive fallback system for X-ray model loading
- **app.py**: Enhanced warning propagation and logging for model loading issues

#### Added
- `load_with_weights_fallback()` function for X-ray models with architecture detection
- `load_face_model_with_fallback()` function for face models with proper fallback handling
- Support for ResNet50, VGG16, EfficientNet, and custom detector architectures
- Graceful degradation when models fail to load (continues with available models)
- Consistent input shapes to reduce TensorFlow retracing warnings

#### Enhanced
- Model validation with consistent dummy input shapes
- Comprehensive error logging and warning collection
- Weights-only loading with `skip_mismatch=True` for robustness
- Better handling of "Unrecognized keyword arguments: ['batch_shape']" errors

### Documentation
- **README.md**: Updated quick start with proper ports, proxy info, and health check commands
- Added troubleshooting section for Keras version mismatches

### Technical Details

#### Error Boundary Fix
- Replaced undefined `isStructuredResponse` with proper `isRouteErrorResponse` from react-router-dom
- Added comprehensive error type narrowing (route errors, Error instances, unknown errors)
- Proper error element wiring at route level

#### Keras Model Loading Robustness
- **Problem**: Keras 2↔3 serialization incompatibilities causing "batch_shape" and layer mismatch errors
- **Solution**: Code-only fallback that rebuilds architecture and loads weights without modifying .h5 files
- **Fallback Strategy**:
  1. Try normal `load_model()` first
  2. If Keras version mismatch detected, rebuild architecture in code
  3. Load weights with `by_name=True, skip_mismatch=True`
  4. If all fails, exclude model from ensemble but continue operation

#### API Contract Preservation
- All existing endpoints maintain same response structure
- Structured `/predict` endpoint now primary, `/predict-legacy` maintained for compatibility
- No breaking changes to frontend-backend contract
- Graceful degradation ensures API always returns valid responses

### Testing Recommendations

1. **Health Check**: `curl http://localhost:8000/health` should show model status
2. **Face Only**: Upload face photo → should get structured response with gender detection
3. **X-ray Only**: Upload X-ray → should work even if some models fail to load
4. **Combined**: Upload both → should combine results properly
5. **Error Boundary**: Visit invalid route → should show friendly error page
6. **Sample Images**: Click sample images → should load and analyze correctly

### References
- React Router error boundaries: https://reactrouter.com/en/main/route/error-element
- Keras version compatibility: https://github.com/tensorflow/tensorflow/issues/
- TensorFlow retracing optimization: https://www.tensorflow.org/guide/function
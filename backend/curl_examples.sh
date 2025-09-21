#!/bin/bash

# PCOS Analyzer API - Comprehensive cURL Examples
# Demonstrates all endpoints with various scenarios including per-model and ROI details

set -e

BASE_URL="http://127.0.0.1:5000"
TEST_DIR="test_images"

echo "🧪 PCOS Analyzer API - cURL Examples"
echo "===================================="

# Check if server is running
echo "📡 Checking server health..."
if ! curl -s "$BASE_URL/health" > /dev/null; then
    echo "❌ Server not running at $BASE_URL"
    echo "Start the server with: uvicorn app:app --reload --port 5000"
    exit 1
fi

echo "✅ Server is running"

# Create test images directory if it doesn't exist
mkdir -p "$TEST_DIR"

# Test health endpoint with detailed output
echo ""
echo "🏥 Testing enhanced health endpoint..."
curl -s "$BASE_URL/health" | jq '.'

# Test 1: Structured prediction with both images
echo ""
echo "🔬 Test 1: Structured prediction (both images)..."
if [ -f "$TEST_DIR/female_face.jpg" ] && [ -f "$TEST_DIR/xray_sample.jpg" ]; then
    echo "Using real test images..."
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/female_face.jpg" \
        -F "xray_img=@$TEST_DIR/xray_sample.jpg" | jq '.'
else
    echo "⚠️  Real test images not found. Please add test images to $TEST_DIR/"
    echo "Expected files: female_face.jpg, xray_sample.jpg"
fi

# Test 2: Legacy prediction endpoint
echo ""
echo "🔄 Test 2: Legacy prediction endpoint..."
if [ -f "$TEST_DIR/female_face.jpg" ]; then
    curl -s -X POST "$BASE_URL/predict-legacy" \
        -F "face_img=@$TEST_DIR/female_face.jpg" | jq '.'
else
    echo "⚠️  No female face test image found"
fi

# Test 3: Face-only prediction (female)
echo ""
echo "👩 Test 3: Face-only prediction (female)..."
if [ -f "$TEST_DIR/female_face.jpg" ]; then
    echo "Response structure with per-model details:"
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/female_face.jpg" | jq '.modalities[0] | {type, risk, per_model, ensemble}'
else
    echo "⚠️  No female face test image found at $TEST_DIR/female_face.jpg"
fi

# Test 4: Face-only prediction (male)
echo ""
echo "👨 Test 4: Face-only prediction (male - should skip PCOS analysis)..."
if [ -f "$TEST_DIR/male_face.jpg" ]; then
    echo "Should show gender detection and skip PCOS analysis:"
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/male_face.jpg" | jq '.modalities[0] | {type, label, gender, warnings: .warnings}'
else
    echo "⚠️  No male face test image found at $TEST_DIR/male_face.jpg"
fi

# Test 5: X-ray-only prediction with ROI details
echo ""
echo "🩻 Test 5: X-ray-only prediction (with ROI analysis)..."
if [ -f "$TEST_DIR/xray_sample.jpg" ]; then
    echo "Response structure with ROI and detection details:"
    curl -s -X POST "$BASE_URL/predict" \
        -F "xray_img=@$TEST_DIR/xray_sample.jpg" | jq '.modalities[0] | {type, risk, detections: (.detections | length), per_roi: (.per_roi | length), per_model, ensemble}'
else
    echo "⚠️  No X-ray test image found at $TEST_DIR/xray_sample.jpg"
fi

# Test 6: Error handling - no images
echo ""
echo "❌ Test 6: Error handling (no images)..."
echo "Should return 400 error:"
curl -s -X POST "$BASE_URL/predict" | jq '.'

# Test 7: Error handling - invalid file type
echo ""
echo "❌ Test 7: Error handling (invalid file type)..."
echo "This is not an image" > /tmp/test_invalid.txt
echo "Should return 400 error for invalid file type:"
curl -s -X POST "$BASE_URL/predict" \
    -F "face_img=@/tmp/test_invalid.txt" | jq '.'
rm -f /tmp/test_invalid.txt

# Test 8: Image proxy
echo ""
echo "🖼️  Test 8: Image proxy..."
echo "Testing CORS proxy with Pexels image:"
curl -s "$BASE_URL/img-proxy?url=https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg" \
    -o /dev/null -w "Status: %{http_code}, Size: %{size_download} bytes, Time: %{time_total}s\n"

# Test 9: Debug information
echo ""
echo "🐛 Test 9: Debug information in structured response..."
if [ -f "$TEST_DIR/female_face.jpg" ]; then
    echo "Debug info from structured response:"
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/female_face.jpg" | jq '.debug'
else
    echo "⚠️  No test image for debug info test"
fi

# Test 10: Performance timing
echo ""
echo "⏱️  Test 10: Performance timing..."
if [ -f "$TEST_DIR/female_face.jpg" ] && [ -f "$TEST_DIR/xray_sample.jpg" ]; then
    echo "Processing time comparison:"
    echo "Face only:"
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/female_face.jpg" | jq '.processing_time_ms'
    echo "X-ray only:"
    curl -s -X POST "$BASE_URL/predict" \
        -F "xray_img=@$TEST_DIR/xray_sample.jpg" | jq '.processing_time_ms'
    echo "Both modalities:"
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/female_face.jpg" \
        -F "xray_img=@$TEST_DIR/xray_sample.jpg" | jq '.processing_time_ms'
else
    echo "⚠️  Need both test images for performance comparison"
fi

echo ""
echo "✅ cURL examples completed!"
echo ""
echo "📝 Notes:"
echo "   - Add test images to $TEST_DIR/ for complete testing:"
echo "     • female_face.jpg - Clear frontal female face photo"
echo "     • male_face.jpg - Clear frontal male face photo"
echo "     • xray_sample.jpg - Pelvic X-ray image"
echo "   - Check server logs for detailed processing information"
echo "   - Visit http://127.0.0.1:5000/docs for interactive API documentation"
echo "   - The structured response includes per-model scores and ROI details"
echo "   - Legacy endpoint maintains backward compatibility"
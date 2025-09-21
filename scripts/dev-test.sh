#!/bin/bash

# PCOS Analyzer Backend Test Script
# Tests all major endpoints with sample data

set -e

BASE_URL="http://127.0.0.1:5000"
TEST_DIR="test_images"

echo "🧪 PCOS Analyzer Backend Test Suite"
echo "=================================="

# Check if server is running
echo "📡 Checking server health..."
if ! curl -s "$BASE_URL/health" > /dev/null; then
    echo "❌ Server not running at $BASE_URL"
    echo "Start the server with: uvicorn app:app --reload --port 5000"
    exit 1
fi

echo "✅ Server is running"

# Test health endpoint
echo ""
echo "🏥 Testing health endpoint..."
curl -s "$BASE_URL/health" | jq '.'

# Create test images directory if it doesn't exist
mkdir -p "$TEST_DIR"

# Test 1: Face image only (female)
echo ""
echo "👩 Test 1: Female face image only..."
if [ -f "$TEST_DIR/female_face.jpg" ]; then
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/female_face.jpg" | jq '.'
else
    echo "⚠️  No female face test image found at $TEST_DIR/female_face.jpg"
    echo "   Create test images or use sample images from the frontend"
fi

# Test 2: Face image only (male)
echo ""
echo "👨 Test 2: Male face image only..."
if [ -f "$TEST_DIR/male_face.jpg" ]; then
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/male_face.jpg" | jq '.'
else
    echo "⚠️  No male face test image found at $TEST_DIR/male_face.jpg"
fi

# Test 3: X-ray image only
echo ""
echo "🩻 Test 3: X-ray image only..."
if [ -f "$TEST_DIR/xray_sample.jpg" ]; then
    curl -s -X POST "$BASE_URL/predict" \
        -F "xray_img=@$TEST_DIR/xray_sample.jpg" | jq '.'
else
    echo "⚠️  No X-ray test image found at $TEST_DIR/xray_sample.jpg"
fi

# Test 4: Both images
echo ""
echo "🔬 Test 4: Both face and X-ray images..."
if [ -f "$TEST_DIR/female_face.jpg" ] && [ -f "$TEST_DIR/xray_sample.jpg" ]; then
    curl -s -X POST "$BASE_URL/predict" \
        -F "face_img=@$TEST_DIR/female_face.jpg" \
        -F "xray_img=@$TEST_DIR/xray_sample.jpg" | jq '.'
else
    echo "⚠️  Missing test images for combined test"
fi

# Test 5: Legacy endpoint
echo ""
echo "🔄 Test 5: Legacy endpoint compatibility..."
if [ -f "$TEST_DIR/female_face.jpg" ]; then
    curl -s -X POST "$BASE_URL/predict-legacy" \
        -F "face_img=@$TEST_DIR/female_face.jpg" | jq '.'
else
    echo "⚠️  No test image for legacy endpoint test"
fi

# Test 6: Error handling - no images
echo ""
echo "❌ Test 6: Error handling (no images)..."
curl -s -X POST "$BASE_URL/predict" | jq '.'

# Test 7: Image proxy
echo ""
echo "🖼️  Test 7: Image proxy..."
curl -s "$BASE_URL/img-proxy?url=https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg" \
    -o /dev/null -w "Status: %{http_code}, Size: %{size_download} bytes\n"

echo ""
echo "✅ Test suite completed!"
echo ""
echo "📝 Notes:"
echo "   - Add test images to $TEST_DIR/ for complete testing"
echo "   - Check server logs for detailed processing information"
echo "   - Visit http://127.0.0.1:5000/docs for interactive API documentation"
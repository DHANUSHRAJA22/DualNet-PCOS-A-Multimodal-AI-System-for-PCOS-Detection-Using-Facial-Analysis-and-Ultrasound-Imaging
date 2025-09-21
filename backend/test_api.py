#!/usr/bin/env python3
"""
Comprehensive test suite for PCOS Analyzer API endpoints

Tests all endpoints with various scenarios including per-model and ROI details,
structured vs legacy responses, and error handling.

Usage:
    python test_api.py
    pytest test_api.py -v
"""

import pytest
import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np
import io

BASE_URL = "http://127.0.0.1:5000"

@pytest.fixture
def test_images():
    """Create test images for testing"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test face image
    face_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    face_img = Image.fromarray(face_array)
    face_path = test_dir / "test_face.jpg"
    face_img.save(face_path, 'JPEG')
    
    # Create a simple test X-ray image
    xray_array = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
    xray_img = Image.fromarray(xray_array)
    xray_path = test_dir / "test_xray.jpg"
    xray_img.save(xray_path, 'JPEG')
    
    return {
        'face_img': str(face_path),
        'xray_img': str(xray_path)
    }

def test_health_endpoint():
    """Test enhanced health endpoint"""
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "status" in data
    assert "models" in data
    assert "uptime_seconds" in data
    assert "version" in data
    
    # Check model status structure
    for model_name, model_info in data["models"].items():
        assert "status" in model_info
        assert "file_exists" in model_info
        assert "lazy_loadable" in model_info

def test_structured_prediction_face_only(test_images):
    """Test structured prediction with face image only"""
    if not os.path.exists(test_images['face_img']):
        pytest.skip("Test face image not available")
    
    with open(test_images['face_img'], 'rb') as f:
        files = {'face_img': f}
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structured response format
    assert data["ok"] is True
    assert "modalities" in data
    assert "final" in data
    assert "warnings" in data
    assert "processing_time_ms" in data
    assert "debug" in data
    
    # Check modalities
    assert len(data["modalities"]) >= 1
    face_modality = next((m for m in data["modalities"] if m["type"] == "face"), None)
    assert face_modality is not None
    assert "label" in face_modality
    assert "risk" in face_modality
    assert face_modality["risk"] in ["low", "moderate", "high", "unknown"]

def test_structured_prediction_xray_only(test_images):
    """Test structured prediction with X-ray image only"""
    if not os.path.exists(test_images['xray_img']):
        pytest.skip("Test X-ray image not available")
    
    with open(test_images['xray_img'], 'rb') as f:
        files = {'xray_img': f}
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structured response format
    assert data["ok"] is True
    assert "modalities" in data
    
    # Check X-ray modality
    xray_modality = next((m for m in data["modalities"] if m["type"] == "xray"), None)
    assert xray_modality is not None
    assert "label" in xray_modality
    assert "risk" in xray_modality

def test_structured_prediction_both_images(test_images):
    """Test structured prediction with both images"""
    if not (os.path.exists(test_images['face_img']) and os.path.exists(test_images['xray_img'])):
        pytest.skip("Test images not available")
    
    with open(test_images['face_img'], 'rb') as face_f, \
         open(test_images['xray_img'], 'rb') as xray_f:
        files = {
            'face_img': face_f,
            'xray_img': xray_f
        }
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structured response format
    assert data["ok"] is True
    assert len(data["modalities"]) == 2
    
    # Check final result
    assert "final" in data
    assert "overall_risk" in data["final"]
    assert "confidence" in data["final"]
    assert "explanation" in data["final"]

def test_legacy_prediction(test_images):
    """Test legacy prediction endpoint"""
    if not os.path.exists(test_images['face_img']):
        pytest.skip("Test face image not available")
    
    with open(test_images['face_img'], 'rb') as f:
        files = {'face_img': f}
        response = requests.post(f"{BASE_URL}/predict-legacy", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check legacy response format
    assert data["ok"] is True
    # Should have at least some legacy fields
    legacy_fields = ["face_pred", "face_scores", "combined", "overall_risk"]
    assert any(field in data for field in legacy_fields)

def test_predict_file_endpoint(test_images):
    """Test single file prediction endpoint"""
    if not os.path.exists(test_images['face_img']):
        pytest.skip("Test face image not available")
    
    with open(test_images['face_img'], 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/predict-file?type=face", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check standard response format
    assert data["ok"] is True
    assert "message" in data
    assert "data" in data

def test_error_handling_no_files():
    """Test error handling when no files provided"""
    response = requests.post(f"{BASE_URL}/predict", timeout=5)
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data

def test_error_handling_invalid_file_type():
    """Test error handling for invalid file type"""
    # Create a text file
    test_file = "test_invalid.txt"
    with open(test_file, 'w') as f:
        f.write("This is not an image")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'face_img': f}
            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=5)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_error_handling_large_file():
    """Test error handling for files that are too large"""
    # Create a large image (simulate > 5MB)
    large_array = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    large_img = Image.fromarray(large_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    large_img.save(img_bytes, format='JPEG', quality=100)
    img_bytes.seek(0)
    
    files = {'face_img': ('large_image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(f"{BASE_URL}/predict", files=files, timeout=10)
    
    # Should return 400 for file too large
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "size" in data["detail"].lower()

def test_image_proxy():
    """Test image proxy endpoint"""
    # Test with a valid Pexels URL
    test_url = "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg"
    response = requests.get(
        f"{BASE_URL}/img-proxy",
        params={'url': test_url},
        timeout=10
    )
    
    # Should either succeed or fail gracefully
    assert response.status_code in [200, 400, 404, 500]

def test_response_json_serializable(test_images):
    """Test that all responses are JSON serializable"""
    if not os.path.exists(test_images['face_img']):
        pytest.skip("Test face image not available")
    
    with open(test_images['face_img'], 'rb') as f:
        files = {'face_img': f}
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    
    assert response.status_code == 200
    
    # Should be able to parse as JSON without errors
    data = response.json()
    
    # Should be able to serialize back to JSON
    json_str = json.dumps(data)
    assert isinstance(json_str, str)
    
    # Should be able to parse back
    reparsed = json.loads(json_str)
    assert reparsed == data

if __name__ == "__main__":
    """Run tests manually"""
    print("🧪 PCOS Analyzer API Test Suite")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server not running at", BASE_URL)
        print("Start the server with: uvicorn app:app --reload --port 5000")
        exit(1)
    
    # Run tests with pytest
    pytest.main([__file__, "-v"])
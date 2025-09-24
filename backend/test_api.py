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

BASE_URL = "http://127.0.0.1:8000"

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
    
    print(f"✅ Health check passed: {data['status']}")
    print(f"   Models available: {len(data['models'])}")

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
    assert "debug" in data
    
    # Check modalities
    assert len(data["modalities"]) >= 1
    face_modality = next((m for m in data["modalities"] if m["type"] == "face"), None)
    assert face_modality is not None
    assert "label" in face_modality
    assert "risk" in face_modality
    assert face_modality["risk"] in ["low", "moderate", "high", "unknown"]
    
    print(f"✅ Face prediction successful")
    print(f"   Risk: {face_modality['risk']}")
    if face_modality.get('per_model'):
        print(f"   Per-model scores: {face_modality['per_model']}")

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
    
    print(f"✅ Legacy prediction successful")
    print(f"   Overall risk: {data.get('overall_risk', 'N/A')}")

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
    
    print(f"✅ Single file prediction successful")

def test_error_handling_no_files():
    """Test error handling when no files provided"""
    response = requests.post(f"{BASE_URL}/predict", timeout=5)
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data or "ok" in data
    
    print(f"✅ No files error handled correctly")

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
    assert "detail" in data or "ok" in data
    
    print(f"✅ Large file error handled correctly")

def test_gender_detection(test_images):
    """Test gender detection functionality"""
    if not os.path.exists(test_images['face_img']):
        pytest.skip("Test face image not available")
    
    with open(test_images['face_img'], 'rb') as f:
        files = {'face_img': f}
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    
    if data["ok"] and data.get("modalities"):
        face_modality = next((m for m in data["modalities"] if m["type"] == "face"), None)
        if face_modality and face_modality.get("gender"):
            gender = face_modality["gender"]
            assert "male" in gender
            assert "female" in gender
            assert "label" in gender
            assert gender["label"] in ["male", "female"]
            
            print(f"✅ Gender detection working")
            print(f"   Detected: {gender['label']} ({gender.get('confidence', 0):.3f})")

def test_ensemble_metadata(test_images):
    """Test that ensemble metadata is properly included"""
    if not os.path.exists(test_images['face_img']):
        pytest.skip("Test face image not available")
    
    with open(test_images['face_img'], 'rb') as f:
        files = {'face_img': f}
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check debug information
    assert "debug" in data
    debug_info = data["debug"]
    
    if "models_used" in debug_info:
        print(f"✅ Ensemble metadata present")
        print(f"   Models used: {debug_info['models_used']}")
        if "weights" in debug_info:
            print(f"   Weights: {debug_info['weights']}")

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
        print("Start the server with: uvicorn app:app --reload --port 8000")
        exit(1)
    
    # Create test images
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create simple test images
    face_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    face_img = Image.fromarray(face_array)
    face_path = test_dir / "test_face.jpg"
    face_img.save(face_path, 'JPEG')
    
    xray_array = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
    xray_img = Image.fromarray(xray_array)
    xray_path = test_dir / "test_xray.jpg"
    xray_img.save(xray_path, 'JPEG')
    
    test_images = {
        'face_img': str(face_path),
        'xray_img': str(xray_path)
    }
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_endpoint()),
        ("Face Prediction", lambda: test_structured_prediction_face_only(test_images)),
        ("Legacy Compatibility", lambda: test_legacy_prediction(test_images)),
        ("Single File", lambda: test_predict_file_endpoint(test_images)),
        ("Error Handling - No Files", lambda: test_error_handling_no_files()),
        ("Error Handling - Large File", lambda: test_error_handling_large_file()),
        ("Gender Detection", lambda: test_gender_detection(test_images)),
        ("Ensemble Metadata", lambda: test_ensemble_metadata(test_images))
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
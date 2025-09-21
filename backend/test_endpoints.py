#!/usr/bin/env python3
"""
Comprehensive test suite for PCOS Analyzer API endpoints

Tests all endpoints with various scenarios including per-model and ROI details,
structured vs legacy responses, and error handling.

Usage:
    python test_endpoints.py
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:5000"

def test_health_endpoint():
    """Test enhanced health endpoint"""
    print("🏥 Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Models: {len(data['models'])} configured")
            print(f"   Uptime: {data['uptime_seconds']:.1f}s")
            print(f"   Version: {data['version']}")
            
            # Check model details
            for model_name, model_info in data['models'].items():
                status_icon = "✅" if model_info['lazy_loadable'] else "❌"
                print(f"   {status_icon} {model_name}: {model_info['status']} (lazy_loadable: {model_info['lazy_loadable']})")
                
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

def test_structured_prediction(files: Dict[str, str]):
    """Test new structured prediction endpoint"""
    print("\n🔬 Testing structured prediction endpoint...")
    
    try:
        # Prepare files for upload
        upload_files = {}
        for field_name, file_path in files.items():
            if os.path.exists(file_path):
                upload_files[field_name] = open(file_path, 'rb')
        
        if not upload_files:
            print("⚠️  No test files found, skipping structured prediction test")
            return
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            files=upload_files,
            timeout=30
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Close files
        for f in upload_files.values():
            f.close()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Structured prediction successful ({processing_time:.0f}ms)")
            print(f"   Overall risk: {data['final']['overall_risk']}")
            print(f"   Confidence: {data['final']['confidence']:.3f}")
            print(f"   Processing time: {data['processing_time_ms']:.0f}ms")
            print(f"   Modalities: {len(data['modalities'])}")
            print(f"   Warnings: {len(data['warnings'])}")
            
            # Check modality details
            for modality in data['modalities']:
                print(f"   📊 {modality['type'].title()} Analysis:")
                print(f"      Risk: {modality['risk']}")
                print(f"      Scores: {modality['scores']}")
                
                if modality.get('per_model'):
                    print(f"      Per-model scores: {modality['per_model']}")
                
                if modality.get('ensemble'):
                    ensemble = modality['ensemble']
                    print(f"      Ensemble: {ensemble['method']} ({ensemble['models_used']} models)")
                
                if modality.get('per_roi'):
                    print(f"      ROIs: {len(modality['per_roi'])} regions analyzed")
                    for roi in modality['per_roi'][:2]:  # Show first 2 ROIs
                        print(f"        ROI {roi['roi_id']}: {roi['ensemble']['score']:.3f} score")
            
            # Check debug info
            if data.get('debug'):
                print(f"   🐛 Debug info available: {list(data['debug'].keys())}")
            
        else:
            print(f"❌ Structured prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text[:200]}")
                
    except Exception as e:
        print(f"❌ Structured prediction error: {str(e)}")

def test_legacy_prediction(files: Dict[str, str]):
    """Test legacy prediction endpoint"""
    print("\n🔄 Testing legacy prediction endpoint...")
    
    try:
        # Prepare files for upload
        upload_files = {}
        for field_name, file_path in files.items():
            if os.path.exists(file_path):
                upload_files[field_name] = open(file_path, 'rb')
        
        if not upload_files:
            print("⚠️  No test files found, skipping legacy prediction test")
            return
        
        response = requests.post(
            f"{BASE_URL}/predict-legacy",
            files=upload_files,
            timeout=30
        )
        
        # Close files
        for f in upload_files.values():
            f.close()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Legacy prediction successful")
            print(f"   Overall risk: {data.get('overall_risk', 'N/A')}")
            print(f"   Face prediction: {data.get('face_pred', 'N/A')}")
            print(f"   X-ray prediction: {data.get('xray_pred', 'N/A')}")
            print(f"   Combined: {data.get('combined', 'N/A')}")
            
        else:
            print(f"❌ Legacy prediction failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Legacy prediction error: {str(e)}")

def test_error_handling():
    """Test error handling scenarios"""
    print("\n❌ Testing error handling...")
    
    # Test no files
    try:
        response = requests.post(f"{BASE_URL}/predict", timeout=5)
        if response.status_code == 400:
            print("✅ No files error handled correctly")
        else:
            print(f"⚠️  Unexpected response for no files: {response.status_code}")
    except Exception as e:
        print(f"❌ No files test error: {str(e)}")
    
    # Test invalid file type (if we have a text file)
    test_file = "test_invalid.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("This is not an image")
        
        with open(test_file, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/predict",
                files={'face_img': f},
                timeout=5
            )
        
        os.remove(test_file)
        
        if response.status_code == 400:
            print("✅ Invalid file type error handled correctly")
        else:
            print(f"⚠️  Unexpected response for invalid file: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Invalid file test error: {str(e)}")

def test_image_proxy():
    """Test image proxy endpoint"""
    print("\n🖼️  Testing image proxy...")
    
    try:
        # Test with a valid Pexels URL
        test_url = "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg"
        response = requests.get(
            f"{BASE_URL}/img-proxy",
            params={'url': test_url},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Image proxy successful ({len(response.content)} bytes)")
        else:
            print(f"❌ Image proxy failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Image proxy error: {str(e)}")

def create_test_images():
    """Create simple test images if they don't exist"""
    from PIL import Image
    import numpy as np
    
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test face image
    face_path = test_dir / "test_face.jpg"
    if not face_path.exists():
        # Create a simple colored rectangle as test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(face_path, 'JPEG')
        print(f"📁 Created test face image: {face_path}")
    
    # Create a simple test X-ray image
    xray_path = test_dir / "test_xray.jpg"
    if not xray_path.exists():
        # Create a simple grayscale-like image
        img_array = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(xray_path, 'JPEG')
        print(f"📁 Created test X-ray image: {xray_path}")
    
    return {
        'face_img': str(face_path),
        'xray_img': str(xray_path)
    }

def main():
    """Run all tests"""
    print("🧪 PCOS Analyzer API Test Suite")
    print("=" * 50)
    
    # Test health endpoint first
    test_health_endpoint()
    
    # Create test images if needed
    try:
        test_files = create_test_images()
    except ImportError:
        print("⚠️  PIL not available, using existing test files if any")
        test_files = {
            'face_img': 'test_images/test_face.jpg',
            'xray_img': 'test_images/test_xray.jpg'
        }
    
    # Test prediction endpoints
    test_structured_prediction(test_files)
    test_legacy_prediction(test_files)
    
    # Test individual modalities
    print("\n👩 Testing face-only prediction...")
    test_structured_prediction({'face_img': test_files['face_img']})
    
    print("\n🩻 Testing X-ray-only prediction...")
    test_structured_prediction({'xray_img': test_files['xray_img']})
    
    # Test error handling
    test_error_handling()
    
    # Test image proxy
    test_image_proxy()
    
    print("\n" + "=" * 50)
    print("✅ Test suite completed!")
    print("\n💡 Tips:")
    print("   - Check server logs for detailed processing information")
    print("   - Visit http://127.0.0.1:5000/docs for interactive API documentation")
    print("   - Use the frontend at http://localhost:5173 for visual testing")

if __name__ == "__main__":
    main()
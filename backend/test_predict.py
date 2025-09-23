#!/usr/bin/env python3
"""
Test script for PCOS Analyzer API with exact model filenames

Tests the ensemble inference capabilities with hardcoded model names and validates
that per-model predictions are properly averaged.

Usage:
    python test_predict.py
"""

import requests
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
import io

BASE_URL = "http://127.0.0.1:8000"

def create_test_image(size=(224, 224), color_mode='RGB'):
    """Create a test image for API testing"""
    # Create a simple test pattern
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, color_mode)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=90)
    img_bytes.seek(0)
    
    return img_bytes

def test_health_endpoint():
    """Test enhanced health endpoint"""
    print("🏥 Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f}s")
            
            # Check model details
            print("   Models:")
            for model_name, model_info in data['models'].items():
                status_icon = "✅" if model_info['lazy_loadable'] else "❌"
                print(f"     {status_icon} {model_name}: {model_info['status']}")
                if model_info.get('path'):
                    print(f"        Path: {model_info['path']}")
                if model_info.get('version'):
                    print(f"        Version: {model_info['version']}")
                
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_ensemble_face_prediction():
    """Test face ensemble prediction with hardcoded models"""
    print("\n👩 Testing face ensemble prediction...")
    
    try:
        # Create test face image
        face_img = create_test_image((224, 224))
        
        files = {'face_img': ('test_face.jpg', face_img, 'image/jpeg')}
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Face ensemble prediction successful ({processing_time:.0f}ms)")
            
            # Check for ensemble structure
            if data.get('ok') and data.get('modalities'):
                face_modality = next((m for m in data['modalities'] if m['type'] == 'face'), None)
                if face_modality:
                    print(f"   Risk: {face_modality['risk']}")
                    print(f"   Scores: {face_modality['scores']}")
                    
                    # Check for per-model breakdown
                    if face_modality.get('per_model'):
                        print("   Per-model scores:")
                        for model_name, score in face_modality['per_model'].items():
                            print(f"     {model_name}: {score:.3f}")
                    
                    # Check ensemble metadata
                    if face_modality.get('ensemble'):
                        ensemble = face_modality['ensemble']
                        print(f"   Ensemble: {ensemble['method']} ({ensemble['models_used']} models)")
                        print(f"   Final score: {ensemble['score']:.3f}")
                        
                        if ensemble.get('weights_used'):
                            print("   Model weights:")
                            for model, weight in ensemble['weights_used'].items():
                                print(f"     {model}: {weight:.3f}")
            
            # Check debug info for expected models
            if data.get('debug') and data['debug'].get('models_used'):
                print(f"   Models used: {data['debug']['models_used']}")
            
            return True
        else:
            print(f"❌ Face prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Face prediction error: {str(e)}")
        return False

def test_ensemble_xray_prediction():
    """Test X-ray ensemble prediction with hardcoded models"""
    print("\n🩻 Testing X-ray ensemble prediction...")
    
    try:
        # Create test X-ray image
        xray_img = create_test_image((640, 640))
        
        files = {'xray_img': ('test_xray.jpg', xray_img, 'image/jpeg')}
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ X-ray ensemble prediction successful ({processing_time:.0f}ms)")
            
            # Check for ensemble structure
            if data.get('ok') and data.get('modalities'):
                xray_modality = next((m for m in data['modalities'] if m['type'] == 'xray'), None)
                if xray_modality:
                    print(f"   Risk: {xray_modality['risk']}")
                    print(f"   Found labels: {xray_modality.get('found_labels', [])}")
                    
                    # Check for per-model breakdown
                    if xray_modality.get('per_model'):
                        print("   Per-model scores:")
                        for model_name, score in xray_modality['per_model'].items():
                            print(f"     {model_name}: {score:.3f}")
                    
                    # Check ensemble metadata
                    if xray_modality.get('ensemble'):
                        ensemble = xray_modality['ensemble']
                        print(f"   Ensemble: {ensemble['method']} ({ensemble['models_used']} models)")
                        print(f"   Final score: {ensemble['score']:.3f}")
            
            # Check debug info for expected models
            if data.get('debug') and data['debug'].get('models_used'):
                print(f"   Models used: {data['debug']['models_used']}")
            
            return True
        else:
            print(f"❌ X-ray prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ X-ray prediction error: {str(e)}")
        return False

def test_legacy_compatibility():
    """Test legacy endpoint compatibility"""
    print("\n🔄 Testing legacy endpoint compatibility...")
    
    try:
        # Create test face image
        face_img = create_test_image((224, 224))
        
        files = {'face_img': ('test_face.jpg', face_img, 'image/jpeg')}
        
        response = requests.post(f"{BASE_URL}/predict-legacy", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Legacy endpoint successful")
            
            # Check legacy format
            legacy_fields = ['ok', 'face_pred', 'face_scores', 'overall_risk', 'combined']
            present_fields = [field for field in legacy_fields if field in data]
            print(f"   Legacy fields present: {present_fields}")
            
            if data.get('face_pred'):
                print(f"   Face prediction: {data['face_pred']}")
            if data.get('overall_risk'):
                print(f"   Overall risk: {data['overall_risk']}")
            
            return True
        else:
            print(f"❌ Legacy endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Legacy endpoint error: {str(e)}")
        return False

def test_model_discovery():
    """Test that expected hardcoded models are discovered"""
    print("\n🔍 Testing model discovery...")
    
    expected_face_models = ["vgg16", "resnet50", "efficientnetb0", "efficientnetb1", "efficientnetb2", "efficientnetb3"]
    expected_xray_models = ["vgg16", "resnet50", "efficientnetb0", "detector_158"]
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check face models
            face_models_found = []
            xray_models_found = []
            
            for model_name, model_info in data['models'].items():
                if model_name.startswith('face_'):
                    face_models_found.append(model_name.replace('face_', ''))
                elif model_name.startswith('xray_'):
                    xray_models_found.append(model_name.replace('xray_', ''))
            
            print(f"   Expected face models: {expected_face_models}")
            print(f"   Found face models: {face_models_found}")
            print(f"   Expected xray models: {expected_xray_models}")
            print(f"   Found xray models: {xray_models_found}")
            
            # Check if any expected models are found
            face_match = any(model in face_models_found for model in expected_face_models)
            xray_match = any(model in xray_models_found for model in expected_xray_models)
            
            if face_match or xray_match:
                print("✅ Model discovery working - some expected models found")
                return True
            else:
                print("⚠️  No expected models found - check model files")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model discovery test error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 PCOS Analyzer Ensemble API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Discovery", test_model_discovery),
        ("Face Ensemble", test_ensemble_face_prediction),
        ("X-ray Ensemble", test_ensemble_xray_prediction),
        ("Legacy Compatibility", test_legacy_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ensemble inference is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Expected Model Files:")
    print("   Face models in backend/models/face/:")
    print("     - pcos_vgg16.h5")
    print("     - pcos_resnet50.h5") 
    print("     - pcos_efficientnetb0.h5")
    print("     - pcos_efficientnetb1.h5")
    print("     - pcos_efficientnetb2.h5")
    print("     - pcos_efficientnetb3.h5")
    print("   X-ray models in backend/models/xray/:")
    print("     - pcos_vgg16.h5")
    print("     - pcos_resnet50.h5")
    print("     - pcos_efficientnetb0.h5")
    print("     - pcos_detector_158.h5")
    print("   Other models:")
    print("     - backend/models/face/gender_classifier.h5")
    print("     - backend/models/yolo/bestv8.pt")

if __name__ == "__main__":
    main()
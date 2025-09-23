#!/usr/bin/env python3
"""
Test script for PCOS Analyzer API ensemble predictions

Tests the ensemble inference capabilities with sample requests and validates
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
                
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_ensemble_face_prediction():
    """Test face ensemble prediction"""
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
    """Test X-ray ensemble prediction"""
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

def test_combined_ensemble_prediction():
    """Test combined face + X-ray ensemble prediction"""
    print("\n🔬 Testing combined ensemble prediction...")
    
    try:
        # Create test images
        face_img = create_test_image((224, 224))
        xray_img = create_test_image((640, 640))
        
        files = {
            'face_img': ('test_face.jpg', face_img, 'image/jpeg'),
            'xray_img': ('test_xray.jpg', xray_img, 'image/jpeg')
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        processing_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Combined ensemble prediction successful ({processing_time:.0f}ms)")
            
            # Check final result
            if data.get('final'):
                final = data['final']
                print(f"   Overall risk: {final['risk']}")
                print(f"   Confidence: {final['confidence']:.3f}")
                print(f"   Fusion mode: {final['fusion_mode']}")
                print(f"   Explanation: {final['explanation'][:100]}...")
            
            # Check modalities
            if data.get('modalities'):
                print(f"   Modalities processed: {len(data['modalities'])}")
                for modality in data['modalities']:
                    print(f"     {modality['type']}: {modality['risk']} risk")
                    if modality.get('ensemble'):
                        print(f"       Ensemble score: {modality['ensemble']['score']:.3f}")
            
            # Check debug info
            if data.get('debug'):
                debug = data['debug']
                print(f"   Debug info:")
                print(f"     Models used: {debug.get('models_used', [])}")
                print(f"     Fusion mode: {debug.get('fusion_mode')}")
                print(f"     Use ensemble: {debug.get('use_ensemble')}")
            
            return True
        else:
            print(f"❌ Combined prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Combined prediction error: {str(e)}")
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

def validate_ensemble_averaging():
    """Validate that ensemble averaging works correctly"""
    print("\n🧮 Validating ensemble averaging logic...")
    
    # This would require access to individual model outputs
    # For now, we'll just check that ensemble scores are reasonable
    try:
        face_img = create_test_image((224, 224))
        files = {'face_img': ('test_face.jpg', face_img, 'image/jpeg')}
        
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('modalities'):
                face_modality = next((m for m in data['modalities'] if m['type'] == 'face'), None)
                if face_modality and face_modality.get('per_model') and face_modality.get('ensemble'):
                    per_model = face_modality['per_model']
                    ensemble_score = face_modality['ensemble']['score']
                    
                    # Check that ensemble score is within reasonable range of individual scores
                    individual_scores = list(per_model.values())
                    if individual_scores:
                        min_score = min(individual_scores)
                        max_score = max(individual_scores)
                        
                        if min_score <= ensemble_score <= max_score:
                            print("✅ Ensemble score is within expected range")
                            print(f"   Individual scores: {[f'{s:.3f}' for s in individual_scores]}")
                            print(f"   Ensemble score: {ensemble_score:.3f}")
                            return True
                        else:
                            print(f"⚠️  Ensemble score ({ensemble_score:.3f}) outside range [{min_score:.3f}, {max_score:.3f}]")
                            return False
            
            print("⚠️  Could not validate ensemble averaging - insufficient data")
            return False
        else:
            print(f"❌ Validation request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 PCOS Analyzer Ensemble API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Face Ensemble", test_ensemble_face_prediction),
        ("X-ray Ensemble", test_ensemble_xray_prediction),
        ("Combined Ensemble", test_combined_ensemble_prediction),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Ensemble Validation", validate_ensemble_averaging)
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
    
    print("\n💡 Tips:")
    print("   - Ensure your models are placed in the correct directories:")
    print("     • models/face/vgg16_pcos.h5, resnet50_pcos.h5, etc.")
    print("     • models/xray/vgg16_xray.h5, resnet50_xray.h5, etc.")
    print("     • models/yolo/bestv8.pt")
    print("   - Check backend logs for detailed model loading information")
    print("   - Verify ensemble weights are configured correctly in config.py")

if __name__ == "__main__":
    main()
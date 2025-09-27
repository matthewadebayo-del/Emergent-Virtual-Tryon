#!/usr/bin/env python3
"""
Production Test Script
Tests all critical components after deployment
"""

import requests
import base64
import json
import sys
from pathlib import Path

def test_health_endpoint():
    """Test basic health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_pose_detection():
    """Test PrecisePoseDetector import and basic functionality"""
    try:
        import sys
        sys.path.append('/opt/virtualfit/backend')
        
        from src.core.precise_pose_detector import PrecisePoseDetector
        from src.core.customer_image_analyzer import CustomerImageAnalyzer
        
        # Test initialization
        detector = PrecisePoseDetector()
        analyzer = CustomerImageAnalyzer()
        
        print("✅ PrecisePoseDetector and CustomerImageAnalyzer imported successfully")
        return True
    except Exception as e:
        print(f"❌ Pose detection test failed: {e}")
        return False

def test_virtual_tryon_api():
    """Test virtual try-on API with sample data"""
    try:
        # Create a simple test image (1x1 pixel base64)
        test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        data = {
            "user_image_base64": test_image,
            "product_id": "test-product",
            "use_stored_measurements": "false"
        }
        
        response = requests.post(
            "http://localhost:8000/tryon", 
            data=data,
            timeout=30
        )
        
        if response.status_code in [200, 422]:  # 422 is expected for invalid test image
            print("✅ Virtual try-on API endpoint accessible")
            return True
        else:
            print(f"❌ Virtual try-on API failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Virtual try-on API test error: {e}")
        return False

def test_dependencies():
    """Test critical dependencies"""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import torch
        import transformers
        
        print("✅ All critical dependencies imported successfully")
        print(f"   - OpenCV: {cv2.__version__}")
        print(f"   - MediaPipe: {mp.__version__}")
        print(f"   - NumPy: {np.__version__}")
        print(f"   - PyTorch: {torch.__version__}")
        return True
    except Exception as e:
        print(f"❌ Dependency test failed: {e}")
        return False

def main():
    """Run all production tests"""
    print("🧪 Running VirtualFit Production Tests...\n")
    
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Dependencies", test_dependencies),
        ("Pose Detection", test_pose_detection),
        ("Virtual Try-On API", test_virtual_tryon_api),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Production deployment is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the logs and fix issues before going live.")
        sys.exit(1)

if __name__ == "__main__":
    main()
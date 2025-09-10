#!/usr/bin/env python3
"""
Test script for backend endpoints including HEIC conversion and rendering pipeline
"""

import requests
import base64
import json
from PIL import Image
import io
import tempfile
import os

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (512, 512), color='white')
    return img

def test_heic_endpoint():
    """Test the HEIC conversion endpoint"""
    print("🧪 Testing HEIC Conversion Endpoint")
    print("=" * 50)
    
    try:
        test_image = create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        jpeg_bytes = img_bytes.getvalue()
        
        # Convert to base64
        test_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        
        url = "https://virtualfit-backend-kkohgb7xuq-uc.a.run.app/api/v1/convert-heic"
        payload = {"heic_base64": test_base64}
        
        print(f"📤 Sending request to {url}")
        print(f"   Payload size: {len(json.dumps(payload))} bytes")
        
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "jpeg_base64" in result:
                print(f"✅ HEIC conversion successful")
                print(f"   Output size: {len(result['jpeg_base64'])} characters")
                return True
            else:
                print(f"❌ Invalid response format: {result}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ HEIC endpoint test failed: {str(e)}")
        return False

def test_rendering_endpoint():
    """Test the rendering pipeline endpoint"""
    print("\n🧪 Testing Rendering Pipeline Endpoint")
    print("=" * 50)
    
    try:
        url = "https://virtualfit-backend-kkohgb7xuq-uc.a.run.app/api/v1/test-rendering-pipeline"
        
        print(f"📤 Sending request to {url}")
        
        response = requests.get(url, timeout=60)
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Rendering endpoint responded")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   File size: {result.get('file_size', 0)} bytes")
            print(f"   Is placeholder: {result.get('is_placeholder', True)}")
            print(f"   Blender available: {result.get('blender_available', False)}")
            
            if result.get('status') == 'success' and result.get('file_size', 0) > 15000:
                print("✅ Rendering pipeline produces valid output")
                return True
            else:
                print("⚠️ Rendering pipeline may have issues")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Rendering endpoint test failed: {str(e)}")
        return False

def test_blender_status():
    """Test the Blender status endpoint"""
    print("\n🧪 Testing Blender Status Endpoint")
    print("=" * 50)
    
    try:
        url = "https://virtualfit-backend-kkohgb7xuq-uc.a.run.app/api/v1/debug-blender-status"
        
        print(f"📤 Sending request to {url}")
        
        response = requests.get(url, timeout=30)
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Blender status endpoint responded")
            print(f"   Available: {result.get('available', False)}")
            print(f"   Version: {result.get('version', 'unknown')}")
            
            return result.get('available', False)
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Blender status test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Backend Endpoint Tests")
    print("=" * 70)
    
    heic_success = test_heic_endpoint()
    
    rendering_success = test_rendering_endpoint()
    
    blender_success = test_blender_status()
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print(f"HEIC Conversion: {'✅ PASS' if heic_success else '❌ FAIL'}")
    print(f"Rendering Pipeline: {'✅ PASS' if rendering_success else '❌ FAIL'}")
    print(f"Blender Status: {'✅ PASS' if blender_success else '❌ FAIL'}")
    
    overall_success = heic_success and rendering_success and blender_success
    print(f"Overall: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    exit(0 if overall_success else 1)

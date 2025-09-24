"""
Test script for production virtual try-on system
Tests all components and capabilities
"""

import asyncio
import base64
import io
import requests
from PIL import Image
import numpy as np

def create_test_image(width=512, height=512, color='blue'):
    """Create a test image"""
    img = Image.new('RGB', (width, height), color=color)
    
    # Add some simple content
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], outline='white', width=3)
    draw.text((width//2-50, height//2), "TEST IMAGE", fill='white')
    
    # Convert to base64
    with io.BytesIO() as buffer:
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
    
    return base64.b64encode(img_bytes).decode('utf-8')

def test_server_health():
    """Test server health endpoint"""
    try:
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            data = response.json()
            print("[OK] Server health check passed")
            print(f"Status: {data['status']}")
            print(f"Components: {data['components']}")
            return True
        else:
            print(f"[ERROR] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Could not connect to server: {e}")
        return False

def test_server_capabilities():
    """Test server capabilities"""
    try:
        response = requests.get("http://localhost:8001/debug")
        if response.status_code == 200:
            data = response.json()
            print("[OK] Server capabilities:")
            print(f"Full 3D Pipeline: {data['capabilities']['full_3d_pipeline']}")
            print(f"AI Enhancement: {data['capabilities']['ai_enhancement']}")
            print(f"Hybrid Processing: {data['capabilities']['hybrid_processing']}")
            print(f"Libraries: {data['libraries']}")
            return True
        else:
            print(f"[ERROR] Capabilities check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Could not check capabilities: {e}")
        return False

def test_user_registration():
    """Test user registration"""
    try:
        user_data = {
            "email": "test@virtualfit.com",
            "password": "testpass123",
            "full_name": "Test User"
        }
        
        response = requests.post("http://localhost:8001/api/register", json=user_data)
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] User registration successful")
            return data["access_token"]
        elif response.status_code == 400:
            # User might already exist, try login
            return test_user_login()
        else:
            print(f"[ERROR] Registration failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Registration error: {e}")
        return None

def test_user_login():
    """Test user login"""
    try:
        login_data = {
            "email": "test@virtualfit.com",
            "password": "testpass123"
        }
        
        response = requests.post("http://localhost:8001/api/login", json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] User login successful")
            return data["access_token"]
        else:
            print(f"[ERROR] Login failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Login error: {e}")
        return None

def test_measurements_save(token):
    """Test saving measurements"""
    try:
        measurements = {
            "height": 175.0,
            "weight": 70.0,
            "chest": 95.0,
            "waist": 80.0,
            "hips": 100.0,
            "shoulder_width": 48.0
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            "http://localhost:8001/api/measurements", 
            json=measurements,
            headers=headers
        )
        
        if response.status_code == 200:
            print("[OK] Measurements saved successfully")
            return True
        else:
            print(f"[ERROR] Measurements save failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Measurements save error: {e}")
        return False

def test_virtual_tryon_fallback(token):
    """Test virtual try-on with fallback processing"""
    try:
        # Create test images
        user_image = create_test_image(color='lightblue')
        garment_image = create_test_image(color='red')
        
        # Test data
        tryon_data = {
            "user_image_base64": user_image,
            "garment_image_base64": garment_image,
            "processing_mode": "fallback"
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            "http://localhost:8001/api/virtual-tryon",
            data=tryon_data,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] Virtual try-on (fallback) successful")
            print(f"Processing method: {data['processing_method']}")
            print(f"Confidence: {data['confidence']}")
            print(f"Features used: {data['features_used']}")
            
            # Save result image for inspection
            result_image_data = base64.b64decode(data['result_image_base64'])
            with open('test_result_fallback.jpg', 'wb') as f:
                f.write(result_image_data)
            print("[OK] Result image saved as 'test_result_fallback.jpg'")
            
            return True
        else:
            print(f"[ERROR] Virtual try-on failed: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Virtual try-on error: {e}")
        return False

def test_virtual_tryon_hybrid(token):
    """Test virtual try-on with hybrid processing"""
    try:
        # Create test images
        user_image = create_test_image(color='lightgreen')
        garment_image = create_test_image(color='purple')
        
        # Test data
        tryon_data = {
            "user_image_base64": user_image,
            "garment_image_base64": garment_image,
            "processing_mode": "hybrid"
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            "http://localhost:8001/api/virtual-tryon",
            data=tryon_data,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] Virtual try-on (hybrid) successful")
            print(f"Processing method: {data['processing_method']}")
            print(f"Confidence: {data['confidence']}")
            
            # Save result image for inspection
            result_image_data = base64.b64decode(data['result_image_base64'])
            with open('test_result_hybrid.jpg', 'wb') as f:
                f.write(result_image_data)
            print("[OK] Result image saved as 'test_result_hybrid.jpg'")
            
            return True
        else:
            print(f"[ERROR] Virtual try-on (hybrid) failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Virtual try-on (hybrid) error: {e}")
        return False

def test_virtual_tryon_full_3d(token):
    """Test virtual try-on with full 3D processing"""
    try:
        # Create test images
        user_image = create_test_image(color='yellow')
        garment_image = create_test_image(color='navy')
        
        # Test data
        tryon_data = {
            "user_image_base64": user_image,
            "garment_image_base64": garment_image,
            "processing_mode": "full_3d"
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            "http://localhost:8001/api/virtual-tryon",
            data=tryon_data,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] Virtual try-on (full 3D) successful")
            print(f"Processing method: {data['processing_method']}")
            print(f"Confidence: {data['confidence']}")
            
            # Save result image for inspection
            result_image_data = base64.b64decode(data['result_image_base64'])
            with open('test_result_full_3d.jpg', 'wb') as f:
                f.write(result_image_data)
            print("[OK] Result image saved as 'test_result_full_3d.jpg'")
            
            return True
        else:
            print(f"[ERROR] Virtual try-on (full 3D) failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Virtual try-on (full 3D) error: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 60)
    print("PRODUCTION VIRTUAL TRY-ON SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Server Health
    print("\n1. Testing server health...")
    if not test_server_health():
        print("[FAIL] Server health check failed. Exiting.")
        return False
    
    # Test 2: Server Capabilities
    print("\n2. Testing server capabilities...")
    if not test_server_capabilities():
        print("[FAIL] Server capabilities check failed.")
        return False
    
    # Test 3: User Registration/Login
    print("\n3. Testing user authentication...")
    token = test_user_registration()
    if not token:
        print("[FAIL] User authentication failed. Exiting.")
        return False
    
    # Test 4: Measurements
    print("\n4. Testing measurements save...")
    if not test_measurements_save(token):
        print("[FAIL] Measurements save failed.")
    
    # Test 5: Virtual Try-On (Fallback)
    print("\n5. Testing virtual try-on (fallback mode)...")
    if not test_virtual_tryon_fallback(token):
        print("[FAIL] Virtual try-on (fallback) failed.")
    
    # Test 6: Virtual Try-On (Hybrid)
    print("\n6. Testing virtual try-on (hybrid mode)...")
    if not test_virtual_tryon_hybrid(token):
        print("[FAIL] Virtual try-on (hybrid) failed.")
    
    # Test 7: Virtual Try-On (Full 3D)
    print("\n7. Testing virtual try-on (full 3D mode)...")
    if not test_virtual_tryon_full_3d(token):
        print("[FAIL] Virtual try-on (full 3D) failed.")
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST COMPLETE")
    print("=" * 60)
    print("\nCheck the generated test result images:")
    print("- test_result_fallback.jpg")
    print("- test_result_hybrid.jpg") 
    print("- test_result_full_3d.jpg")
    
    return True

if __name__ == "__main__":
    run_comprehensive_test()
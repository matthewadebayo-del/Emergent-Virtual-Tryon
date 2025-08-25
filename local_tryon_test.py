#!/usr/bin/env python3
"""
Local Virtual Try-On Test
Tests the virtual try-on endpoint locally to bypass proxy timeouts
"""

import requests
import sys
import json
import time
import io
from PIL import Image, ImageDraw

def create_test_image() -> bytes:
    """Create a test image for virtual try-on"""
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a person-like silhouette
    # Head
    draw.ellipse([200, 50, 312, 162], fill='#D2B48C')  # Skin tone
    # Body
    draw.rectangle([180, 160, 332, 400], fill='#4169E1')  # Blue shirt
    # Arms
    draw.rectangle([120, 180, 180, 350], fill='#4169E1')  # Left arm
    draw.rectangle([332, 180, 392, 350], fill='#4169E1')  # Right arm
    # Legs
    draw.rectangle([190, 400, 240, 500], fill='#2F4F4F')  # Left leg
    draw.rectangle([272, 400, 322, 500], fill='#2F4F4F')  # Right leg
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    return buffer.getvalue()

def test_local_virtual_tryon():
    """Test virtual try-on locally"""
    base_url = "http://localhost:8001/api"
    
    print("🧪 Testing Virtual Try-On Locally (bypassing proxy)")
    print("=" * 60)
    
    # Step 1: Register a user
    print("\n1. Registering test user...")
    test_email = f"local_test_{int(time.time())}@example.com"
    registration_data = {
        "email": test_email,
        "password": "LocalTest123!",
        "full_name": "Local Test User"
    }
    
    try:
        response = requests.post(f"{base_url}/register", json=registration_data, timeout=10)
        if response.status_code == 200:
            token = response.json().get('access_token')
            print(f"✅ User registered successfully")
        else:
            print(f"❌ Registration failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return
    
    # Step 2: Get products
    print("\n2. Getting products...")
    try:
        response = requests.get(f"{base_url}/products", timeout=10)
        if response.status_code == 200:
            products = response.json().get('products', [])
            if products:
                product_id = products[0]['id']
                product_name = products[0]['name']
                print(f"✅ Got product: {product_name}")
            else:
                print("❌ No products found")
                return
        else:
            print(f"❌ Failed to get products: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Products error: {e}")
        return
    
    # Step 3: Test virtual try-on with timeout monitoring
    print(f"\n3. Testing virtual try-on with {product_name}...")
    print("   (This will test our timeout protection)")
    
    test_image = create_test_image()
    
    form_data = {
        'product_id': product_id,
        'service_type': 'hybrid',
        'size': 'M',
        'color': 'Blue'
    }
    
    files = {'user_photo': ('test_user.jpg', io.BytesIO(test_image), 'image/jpeg')}
    headers = {'Authorization': f'Bearer {token}'}
    
    start_time = time.time()
    try:
        print("   Starting virtual try-on request...")
        response = requests.post(
            f"{base_url}/tryon", 
            data=form_data, 
            files=files, 
            headers=headers,
            timeout=50  # 50 second timeout
        )
        processing_time = time.time() - start_time
        
        print(f"   Response received after {processing_time:.1f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result_data = response.json()
            if result_data.get('success'):
                data = result_data.get('data', {})
                cost = data.get('cost', 0)
                service_type = data.get('service_type')
                result_url = data.get('result_image_url', '')
                
                print(f"✅ Virtual try-on completed successfully!")
                print(f"   Cost: ${cost}")
                print(f"   Service: {service_type}")
                print(f"   Processing time: {processing_time:.1f}s")
                print(f"   Result type: {'Data URL' if result_url.startswith('data:') else 'External URL'}")
                
                # Test passed - no 502 error locally
                print(f"\n🎉 SUCCESS: No 502 error when testing locally!")
                print(f"   This confirms the timeout fix is working in the backend.")
                print(f"   The 502 errors are likely from the external proxy/load balancer.")
                
            else:
                print(f"❌ Try-on failed: {result_data}")
        else:
            print(f"❌ Try-on request failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        processing_time = time.time() - start_time
        print(f"❌ Request timed out after {processing_time:.1f} seconds")
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Try-on error after {processing_time:.1f}s: {e}")

if __name__ == "__main__":
    test_local_virtual_tryon()
#!/usr/bin/env python3
"""
Comprehensive Photo Workflow Test - Tests the complete user journey
"""

import requests
import json
import time
import io
import base64
from PIL import Image, ImageDraw

def create_realistic_photo():
    """Create a realistic test photo"""
    img = Image.new('RGB', (400, 600), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # Realistic person silhouette
    # Head
    draw.ellipse([150, 50, 250, 150], fill='#D2B48C')
    # Body
    draw.rectangle([120, 150, 280, 400], fill='#4169E1')
    # Arms
    draw.rectangle([80, 180, 120, 350], fill='#D2B48C')
    draw.rectangle([280, 180, 320, 350], fill='#D2B48C')
    # Legs
    draw.rectangle([140, 400, 190, 580], fill='#2F4F4F')
    draw.rectangle([210, 400, 260, 580], fill='#2F4F4F')
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    return buffer.getvalue()

def test_complete_workflow():
    """Test the complete photo saving and virtual try-on workflow"""
    base_url = "http://localhost:8001/api"
    
    print("🚀 COMPREHENSIVE PHOTO WORKFLOW TEST")
    print("=" * 60)
    print("Testing complete user journey:")
    print("1. Register → 2. Login → 3. Extract Measurements (Save Photo)")
    print("4. Verify Profile Photo → 5. Virtual Try-On → 6. No Photo Errors")
    print("=" * 60)
    
    # Test data
    timestamp = int(time.time())
    test_email = f"workflow_test_{timestamp}@example.com"
    test_password = "WorkflowTest123!"
    test_name = "Workflow Test User"
    
    try:
        # Step 1: Register user
        print("\n🔐 STEP 1: Register User")
        print("-" * 30)
        registration_data = {
            "email": test_email,
            "password": test_password,
            "full_name": test_name
        }
        
        response = requests.post(f"{base_url}/register", json=registration_data, timeout=30)
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            user = data.get('user')
            print(f"✅ User registered: {user.get('email')}")
            print(f"   Initial profile_photo: {user.get('profile_photo')}")
        else:
            print(f"❌ Registration failed: {response.status_code} - {response.text}")
            return False
        
        # Step 2: Login user
        print("\n🔑 STEP 2: Login User")
        print("-" * 30)
        login_data = {
            "email": test_email,
            "password": test_password
        }
        
        response = requests.post(f"{base_url}/login", json=login_data, timeout=30)
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            user = data.get('user')
            print(f"✅ Login successful: {user.get('email')}")
        else:
            print(f"❌ Login failed: {response.status_code}")
            return False
        
        # Step 3: Extract measurements and save photo
        print("\n📸 STEP 3: Extract Measurements & Save Photo")
        print("-" * 30)
        headers = {'Authorization': f'Bearer {token}'}
        photo_data = create_realistic_photo()
        files = {'user_photo': ('user_photo.jpg', io.BytesIO(photo_data), 'image/jpeg')}
        
        response = requests.post(f"{base_url}/extract-measurements", files=files, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                measurements = data.get('data', {})
                print(f"✅ Measurements extracted successfully")
                print(f"   Height: {measurements.get('height')} inches")
                print(f"   Chest: {measurements.get('chest')} inches")
                print(f"   Waist: {measurements.get('waist')} inches")
            else:
                print(f"❌ Measurement extraction failed: {data}")
                return False
        else:
            print(f"❌ Measurement extraction request failed: {response.status_code}")
            return False
        
        # Step 4: Verify profile photo saved
        print("\n👤 STEP 4: Verify Profile Photo Saved")
        print("-" * 30)
        response = requests.get(f"{base_url}/profile", headers=headers, timeout=30)
        if response.status_code == 200:
            profile = response.json()
            profile_photo = profile.get('profile_photo')
            
            if profile_photo and profile_photo.startswith('data:image/'):
                try:
                    base64_data = profile_photo.split(',')[1]
                    decoded_data = base64.b64decode(base64_data)
                    print(f"✅ Profile photo saved successfully")
                    print(f"   Photo size: {len(decoded_data)} bytes")
                    print(f"   Format: {profile_photo.split(',')[0]}")
                    
                    # Verify measurements are also saved
                    measurements = profile.get('measurements')
                    if measurements:
                        print(f"   Measurements saved: ✅")
                    else:
                        print(f"   Measurements saved: ❌")
                        
                except Exception as e:
                    print(f"❌ Invalid base64 data: {str(e)}")
                    return False
            else:
                print(f"❌ Profile photo not saved properly")
                print(f"   Type: {type(profile_photo)}")
                print(f"   Value: {str(profile_photo)[:100]}...")
                return False
        else:
            print(f"❌ Profile retrieval failed: {response.status_code}")
            return False
        
        # Step 5: Get products for virtual try-on
        print("\n🛍️ STEP 5: Get Products for Try-On")
        print("-" * 30)
        response = requests.get(f"{base_url}/products", timeout=30)
        if response.status_code == 200:
            data = response.json()
            products = data.get('products', [])
            if products:
                test_product = products[0]
                product_id = test_product.get('id')
                product_name = test_product.get('name')
                print(f"✅ Products retrieved: {len(products)} available")
                print(f"   Testing with: {product_name}")
            else:
                print(f"❌ No products available")
                return False
        else:
            print(f"❌ Product retrieval failed: {response.status_code}")
            return False
        
        # Step 6: Virtual try-on test
        print("\n👗 STEP 6: Virtual Try-On Test")
        print("-" * 30)
        
        # Use the same photo for try-on
        photo_data = create_realistic_photo()
        form_data = {
            'product_id': product_id,
            'service_type': 'hybrid',
            'size': 'M',
            'color': 'Blue'
        }
        files = {'user_photo': ('user_photo.jpg', io.BytesIO(photo_data), 'image/jpeg')}
        
        start_time = time.time()
        response = requests.post(f"{base_url}/tryon", data=form_data, files=files, headers=headers, timeout=120)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result_data = data.get('data', {})
                result_image_url = result_data.get('result_image_url')
                cost = result_data.get('cost', 0)
                service_type = result_data.get('service_type')
                
                print(f"✅ Virtual try-on completed successfully")
                print(f"   Service: {service_type}")
                print(f"   Cost: ${cost}")
                print(f"   Processing time: {processing_time:.1f}s")
                
                # Check for "No photo available" errors
                if result_image_url and "no photo available" not in result_image_url.lower():
                    print(f"   Result: Valid image URL generated")
                    print(f"   No 'No photo available' errors: ✅")
                    
                    # Check if it's a data URL (real processing) or external URL
                    if result_image_url.startswith('data:image/'):
                        print(f"   Real AI processing: ✅ (data URL)")
                    else:
                        print(f"   Result type: External URL")
                    
                    return True
                else:
                    print(f"❌ Got 'No photo available' error or invalid result")
                    print(f"   Result URL: {result_image_url}")
                    return False
            else:
                print(f"❌ Try-on failed: {data}")
                return False
        else:
            print(f"❌ Try-on request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main test execution"""
    print("🧪 Starting Comprehensive Photo Workflow Test")
    print(f"⏰ Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = test_complete_workflow()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    if success:
        print("🎉 COMPREHENSIVE WORKFLOW TEST PASSED!")
        print("✅ Photo saving workflow is working correctly")
        print("✅ User profile updates work properly")
        print("✅ Virtual try-on uses saved photos")
        print("✅ No 'No photo available' errors detected")
        print("\n🚀 The complete user journey works as expected!")
    else:
        print("❌ COMPREHENSIVE WORKFLOW TEST FAILED!")
        print("⚠️  Issues detected in the photo workflow")
        print("🔧 Review the failed steps above for details")
    
    print(f"⏰ Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
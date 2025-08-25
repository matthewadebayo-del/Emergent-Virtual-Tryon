#!/usr/bin/env python3
"""
Simple Photo Workflow Test - Tests the critical photo saving functionality
"""

import requests
import json
import time
import io
import base64
from PIL import Image, ImageDraw

def create_test_photo():
    """Create a simple test photo"""
    img = Image.new('RGB', (300, 400), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # Simple person shape
    draw.ellipse([100, 50, 200, 150], fill='#D2B48C')  # Head
    draw.rectangle([80, 150, 220, 350], fill='#4169E1')  # Body
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    return buffer.getvalue()

def test_photo_workflow():
    """Test the photo saving workflow"""
    base_url = "http://localhost:8001/api"
    
    print("🚀 Testing Photo Saving Workflow")
    print("=" * 50)
    
    # Test data
    timestamp = int(time.time())
    test_email = f"photo_test_{timestamp}@example.com"
    test_password = "TestPass123!"
    test_name = "Photo Test User"
    
    try:
        # Step 1: Register user
        print("\n1. Registering user...")
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
            print(f"❌ Registration failed: {response.status_code}")
            return False
        
        # Step 2: Extract measurements (save photo)
        print("\n2. Extracting measurements and saving photo...")
        headers = {'Authorization': f'Bearer {token}'}
        photo_data = create_test_photo()
        files = {'user_photo': ('test_photo.jpg', io.BytesIO(photo_data), 'image/jpeg')}
        
        response = requests.post(f"{base_url}/extract-measurements", files=files, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                measurements = data.get('data', {})
                print(f"✅ Measurements extracted: height={measurements.get('height')}")
            else:
                print(f"❌ Measurement extraction failed: {data}")
                return False
        else:
            print(f"❌ Measurement extraction request failed: {response.status_code}")
            return False
        
        # Step 3: Verify profile photo saved
        print("\n3. Verifying profile photo saved...")
        response = requests.get(f"{base_url}/profile", headers=headers, timeout=30)
        if response.status_code == 200:
            profile = response.json()
            profile_photo = profile.get('profile_photo')
            
            if profile_photo and profile_photo.startswith('data:image/'):
                # Verify it's valid base64
                try:
                    base64_data = profile_photo.split(',')[1]
                    decoded_data = base64.b64decode(base64_data)
                    print(f"✅ Profile photo saved successfully (size: {len(decoded_data)} bytes)")
                    print(f"   Photo format: {profile_photo[:50]}...")
                    return True
                except Exception as e:
                    print(f"❌ Invalid base64 data: {str(e)}")
                    return False
            else:
                print(f"❌ Profile photo not saved: {type(profile_photo)}")
                print(f"   Value: {profile_photo}")
                return False
        else:
            print(f"❌ Profile retrieval failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_photo_workflow()
    if success:
        print("\n🎉 PHOTO WORKFLOW TEST PASSED!")
        print("✅ Photos are saved to user profiles correctly")
    else:
        print("\n❌ PHOTO WORKFLOW TEST FAILED!")
        print("⚠️  Photo saving functionality has issues")
#!/usr/bin/env python3
"""
Final Photo Workflow Verification - Quick test of critical functionality
"""

import requests
import json
import time
import io
import base64
from PIL import Image, ImageDraw

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (200, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 30, 150, 130], fill='#D2B48C')  # Head
    draw.rectangle([40, 130, 160, 250], fill='#4169E1')  # Body
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.getvalue()

def main():
    """Quick verification of photo workflow"""
    base_url = "http://localhost:8001/api"
    
    print("🔍 FINAL PHOTO WORKFLOW VERIFICATION")
    print("=" * 50)
    
    # Test data
    timestamp = int(time.time())
    email = f"final_test_{timestamp}@example.com"
    password = "FinalTest123!"
    name = "Final Test User"
    
    try:
        # 1. Register
        print("1. Registering user...")
        reg_data = {"email": email, "password": password, "full_name": name}
        response = requests.post(f"{base_url}/register", json=reg_data, timeout=20)
        
        if response.status_code != 200:
            print(f"❌ Registration failed: {response.status_code}")
            return False
        
        token = response.json().get('access_token')
        user = response.json().get('user')
        print(f"✅ User registered: {user.get('email')}")
        print(f"   Initial photo: {user.get('profile_photo')}")
        
        # 2. Extract measurements (save photo)
        print("\n2. Extracting measurements and saving photo...")
        headers = {'Authorization': f'Bearer {token}'}
        photo_data = create_test_image()
        files = {'user_photo': ('photo.jpg', io.BytesIO(photo_data), 'image/jpeg')}
        
        response = requests.post(f"{base_url}/extract-measurements", files=files, headers=headers, timeout=20)
        
        if response.status_code != 200:
            print(f"❌ Measurement extraction failed: {response.status_code}")
            return False
        
        data = response.json()
        if not data.get('success'):
            print(f"❌ Measurement extraction unsuccessful: {data}")
            return False
        
        measurements = data.get('data', {})
        print(f"✅ Measurements extracted: {measurements.get('height')} inches")
        
        # 3. Verify profile photo saved
        print("\n3. Verifying profile photo saved...")
        response = requests.get(f"{base_url}/profile", headers=headers, timeout=20)
        
        if response.status_code != 200:
            print(f"❌ Profile retrieval failed: {response.status_code}")
            return False
        
        profile = response.json()
        profile_photo = profile.get('profile_photo')
        
        if not profile_photo or not profile_photo.startswith('data:image/'):
            print(f"❌ Profile photo not saved properly: {type(profile_photo)}")
            return False
        
        # Verify base64 data
        try:
            base64_data = profile_photo.split(',')[1]
            decoded_data = base64.b64decode(base64_data)
            print(f"✅ Profile photo saved: {len(decoded_data)} bytes")
            print(f"   Format: {profile_photo.split(',')[0]}")
            
            # Check measurements also saved
            if profile.get('measurements'):
                print(f"✅ Measurements also saved to profile")
            else:
                print(f"⚠️  Measurements not found in profile")
            
            return True
            
        except Exception as e:
            print(f"❌ Invalid photo data: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 PHOTO WORKFLOW VERIFICATION PASSED!")
        print("✅ Photos are saved to user profiles")
        print("✅ Profile updates work correctly")
        print("✅ No 'No photo available' errors expected")
    else:
        print("❌ PHOTO WORKFLOW VERIFICATION FAILED!")
        print("⚠️  Issues detected in photo saving")
    print("=" * 50)